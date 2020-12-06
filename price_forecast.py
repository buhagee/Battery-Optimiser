import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import custom_module as cm

from keras.models import Model, load_model
from keras.layers.convolutional import Conv1D
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

data = pd.read_csv('combined_data.csv', parse_dates=True)
data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"])
data.index = data["SETTLEMENTDATE"]
data.drop(columns="SETTLEMENTDATE", inplace=True)
#print(data.head(5))
#print(data.tail(5))
data = data["2016-01-01":"2019-07-01"]
X_test = data["2019-01-01":"2019-07-01"]
data = cm.replace_outliers(data, 'RRP5MIN', 3)
data = cm.replace_outliers(data, "RESIDUAL_DEMAND", 3)
scaler = MinMaxScaler()
data["RRP5MIN"] = scaler.fit_transform(data["RRP5MIN"].values.reshape(-1,1))
data["RESIDUAL_DEMAND"] = scaler.fit_transform(data["RESIDUAL_DEMAND"].values.reshape(-1,1))
train = data.copy()
lags = 864
for i in range(1,lags+1):
    train["lp_{}".format(i)] = train["RRP5MIN"].shift(i)
    train["ld_{}".format(i)] = train["RESIDUAL_DEMAND"].shift(i)
train.dropna(inplace=True)
#print(train.head(5))
#print(train.tail(5))

prelim_features = train.drop('RRP5MIN', axis=1)
prelim_labels = pd.DataFrame(train['RRP5MIN'])

# format labels to 24 hour output range
for i in range(0, 288):
    prelim_labels['t_{}'.format(i)] = prelim_labels['RRP5MIN'].shift(-i)
prelim_labels.drop('RRP5MIN', axis=1, inplace=True)
print(prelim_labels.head())

print("DISCRETIZATION OF DATA")
# apply one-day discretization to the data
labels = prelim_labels[prelim_labels.index.minute == 0]
labels = labels[labels.index.hour == 0]
features = prelim_features[prelim_features.index.minute == 0]
features = features[features.index.hour == 0]

features_train = features[:'2018']
features_test = features['2019':'2019-07-01']
labels_train = labels[:'2018']

print(features_train.iloc[0])
print(features_test.iloc[0])

samples_train = len(features_train)
samples_test = len(features_test)
print(samples_train)
print(samples_test)

timesteps = 864

# convert pandas data frames to numpy ndarrays
features_train = features_train.to_numpy().reshape(samples_train, 2*timesteps + 1, 1)
features_test = features_test.to_numpy().reshape(samples_test, 2*timesteps + 1, 1)
labels_train = labels_train.to_numpy()

# check for correct data shape
print(features_train.shape, labels_train.shape)

# split into training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size=0.2, random_state=7)



###################### DESIGNING THE NN ###################
##########################################################
## 1D Convolution layer to avoid overfitting
## 3 layers of LSTM considering the complexity of the dataset

# Initialising
rnn = Sequential()
# Adding Conv1D Layer
rnn.add(Conv1D(64, kernel_size=288, strides=288, padding='valid', input_shape=(X_train.shape[1],1)))
# Add LSTM layer 1st
rnn.add(LSTM(50, recurrent_activation='relu', return_sequences=True))
rnn.add(Dropout(0.1))
# Add LSTM layer 2nd
rnn.add(LSTM(50, recurrent_activation='relu'))
rnn.add(Dropout(0.1))
rnn.add(Dense(units=288))
rnn.compile(optimizer='adam', loss='mse')

# train the model and calculate the performance on the test set
results, hist = cm.train_predict_evaluate(rnn, X_train, X_valid, y_train, y_valid, features_test,
                                       X_test.to_numpy().flatten(), X_test.index, scaler, 32, 200,
                                       'simple_neural_network.hdf5', verbose=1)

"""
f, ax = plt.subplots(figsize=(12, 6))
results.loc['2010-01-01':'2010-01-04'].plot(ax=ax);
ax.set_ylabel('Day-Ahead price in $/MWh', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
ax.set_title('Short Term predictive Performance', fontsize=14);

cm.quantify_performance(results['prediction'], results['true values'])
"""
