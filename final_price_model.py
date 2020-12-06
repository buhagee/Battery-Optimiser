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
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input
import tensorflow as tf



data = pd.read_csv("combined_data.csv", parse_dates=True)
# Setting the data to requirement
data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"])
data.index = data["SETTLEMENTDATE"]
data.drop(columns=["SETTLEMENTDATE"], inplace=True)



# The Real Test Data Set To Test Later #
X_test = data["2019-01-01":"2019-06-30"].copy()
X_test = X_test["RRP5MIN"]

# replace outliers by outlier threshold
data = cm.replace_outliers(data, 'RRP5MIN', 4)

# Scaling the RRP between 0 and 1 as required by the NN
feature_scaler = MinMaxScaler()
data["RESIDUAL_DEMAND"] = feature_scaler.fit_transform(pd.DataFrame(data["RESIDUAL_DEMAND"]))

# scale price data to 0-1 range
label_scaler = MinMaxScaler()
data['RRP5MIN'] = label_scaler.fit_transform(data['RRP5MIN'].values.reshape(-1, 1))

train = data.loc["2017":].copy()
#yearly_lags_creater = data.loc["2016":].copy()
#weekly_lags_creater = data.loc["2016-12-25":].copy()

# include time lags of timeseries data for 3 days = 864
# We will use 3 days data to identify patterns to predict the next day

# Creating Daily lags
j = 1
for i in range(1, 289):
    train["d_l_{}".format(j)] = train["RRP5MIN"].shift(i)
    train["d_avg_p_{}".format(j)] = train["avg_price"].shift(i)
    train["d_avg_d_{}".format(j)] = train["avg_demand"].shift(i)

    j += 1
"""
# Creating Week ago lags
j = 1
for i in range(1, 289):
    train["w_l_{}".format(j)] = weekly_lags_creater["RRP5MIN"].shift(-i)
    j += 1

'''
# Creating Month ago lags
j = 1
for i in range(1728,2016):
    train["w_l_{}".format(j)] = train["RRP5MIN"].shift(i)
    j++

'''
# Creating year ago lags
j = 1
for i in range(1, 289):
    train["y_l_{}".format(j)] = yearly_lags_creater["RRP5MIN"].shift(-i)
    j += 1
"""
# Drop NANS
train.dropna(inplace=True)


# create feature and label dataframes
prelim_features = train.drop(['RRP5MIN', 'RESIDUAL_DEMAND'], axis=1)
prelim_labels = pd.DataFrame(train[['RRP5MIN']])


# format labels to 24 hour output range
for i in range(0, 288):
    prelim_labels['t_{}'.format(i)] = prelim_labels['RRP5MIN'].shift(-i)
prelim_labels.drop(['RRP5MIN'], axis=1, inplace=True)

# apply one-day discretization to the data
labels = prelim_labels[prelim_labels.index.minute == 0]
labels = labels[labels.index.hour == 0]
features = prelim_features[prelim_features.index.minute == 0]
features = features[features.index.hour == 0]

features_train = features[:'2018']
features_test = features['2019':'2019-06-30']
labels_train = labels[:'2018']
#print(features_train.iloc[0])
#print(labels_train.iloc[0])


samples_train = len(features_train)
samples_test = len(features_test)
print(samples_test)
print(samples_train)

print(features_train.shape)
timesteps = 288

# convert pandas data frames to numpy ndarrays
features_train = features_train.to_numpy().reshape(samples_train, timesteps, 3)
features_test = features_test.to_numpy().reshape(samples_test, timesteps, 3)
labels_train = labels_train.to_numpy()

print(features_train.head())
# check for correct data shape
features_train.shape, labels_train.shape


# split into training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size=0.2, random_state=7)

###################### DESIGNING THE NN ###################
##########################################################
## 1D Convolution layer to avoid overfitting
## 3 layers of LSTM considering the complexity of the dataset
# Initialising
rnn = Sequential()
# Adding Conv1D Layer
rnn.add(Conv1D(128, kernel_size=288, strides=288, padding='valid', input_shape=(X_train.shape[1],3)))
# Add LSTM layer 1st
rnn.add(LSTM(200, recurrent_activation='sigmoid', return_sequences=True))
rnn.add(Dropout(0.1))
# Add LSTM layer 2nd
rnn.add(LSTM(128, recurrent_activation='sigmoid'))
rnn.add(Dropout(0.1))
rnn.add(Dense(units=288))
rnn.compile(optimizer='adam', loss='mae') ################ CHANGED LOSS TO MSE #################

#model.compile(loss='mae', optimizer='adam')
checkpoint = ModelCheckpoint('./models/multidim_timeseries_testing.hdf5', save_best_only=True)

hist = rnn.fit(X_train, y_train,
                 validation_data=(X_valid, y_valid),
                 callbacks=[checkpoint],
                 verbose=1, batch_size=32, epochs=160)

best = load_model('./models/multidim_timeseries_testing.hdf5')
# pred = best.predict([input_test, input_test[:, :, 3]])
pred = best.predict(features_test)
#pred = scaler.inverse_transform(pred.flatten().reshape(-1, 1))

pred = label_scaler.inverse_transform(pred.flatten().reshape(-1, 1))

results = pd.DataFrame({'prediction':pred.flatten(), 'true values':X_test}, index=X_test.index)

cm.plot_chart(results.loc['2019-01-01':'2019-01-03'], xlab='Time', ylab='Day Ahead price in $/MWh ($)', title='Short Term Predictive Performance', legend=True)
