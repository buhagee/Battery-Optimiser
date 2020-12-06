import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import custom_module as cm
#import optimizer_module as om
from datetime import date
import holidays

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
from keras.layers import concatenate


def create_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(64, input_shape=dim, activation="relu"))
    model.add(Dense(32, activation="relu"))
    # check to see if the regression node should be added
    return model


def create_conv_lstm(dim):
    conv_input_layer = Input(batch_shape=dim)
    x = Conv1D(64, kernel_size=280, strides=280, padding='valid')(conv_input_layer)
    x = Dropout(0.1)(x)
    x = LSTM(64, recurrent_activation='relu')(x)
    x = Dense(32 , activation='relu')(x)
    model = Model(inputs=[conv_input_layer], outputs=[x])
    return model



def price_prediction(dataset):
    # Loading the dataset
    data = pd.read_csv(dataset, parse_dates=True, index_col=0)
    data = data["2016":]
    print(data.columns)
    # Load this file for saving time
    # Saving the data file so we can reload with the features made again to reduce time
    # data.to_csv('final.csv')

    """
    # Setting the data to index
    data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"])
    data.index = data["SETTLEMENTDATE"]
    data.drop(columns="SETTLEMENTDATE", inplace=True)
    """

    # replace outliers by outlier threshold
    data = cm.replace_outliers(data, 'RRP5MIN', 4)


    # Seperating the test dataset for testing purposes in evaluation
    X_test = data["2019-01-01":"2019-06-30"].copy()
    X_test = X_test["RRP5MIN"]

    """
    aus_holidays = holidays.CountryHoliday('AUS', prov='SA')
    
    data["public holiday"] = 0
    for i in range(len(data)):
        if (data.index[i] in aus_holidays):
            print(data.index[i])
            data["public holiday"][i] = 1
    """

    # Scaling the RRP between 0 and 1 as required by the NN
    features = ['RESIDUAL_DEMAND', 'AVG_PRICE', 'DIFF_PRICE']
    feature_scaler = MinMaxScaler()
    for i in features:
        data[i] = feature_scaler.fit_transform(pd.DataFrame(data[i]))

    # scale price data to 0-1 range
    label_scaler = MinMaxScaler()
    data['RRP5MIN'] = label_scaler.fit_transform(data['RRP5MIN'].values.reshape(-1, 1))

    train = data['2016-12-25 00:00:00':].copy()

    # include time lags of timeseries data for last day i.e. 288 data points at 5 minutes granularity
    # Also 80 lags of same day previous week

    # Creating Daily lags
    for i in range(1, 201):
        train["price_l_{}".format(i)] = train["DIFF_PRICE"].shift(i)
        train["demand_l_{}".format(i)] = train["RESIDUAL_DEMAND"].shift(i)
        train["avgPrice_l_{}".format(i)] = train["AVG_PRICE"].shift(i)

    # Creating Week ago lags
    j = 1
    size = 2016
    for i in range(size, size - 80, -1):
        train["w_price_l_{}".format(j)] = train["DIFF_PRICE"].shift(i)
        train["w_demand_l_{}".format(j)] = train["RESIDUAL_DEMAND"].shift(i)
        train["w_avgPrice_l_{}".format(j)] = train["AVG_PRICE"].shift(i)
        j += 1


    # Drop NANS
    train.dropna(inplace=True)


    ########### THIS IS FOR MULTILAYER PERCEPTRON PURPOSES
    train1 = data[['hour', 'weekday', 'month', 'business hour', 'public holiday', 'RRP5MIN']]
    train1 = train1["2017":]


    # Scaling the categorical variables using the same scaler used for LSTM variables
    cont = ['hour', 'weekday', 'month', 'business hour', 'public holiday']
    for i in cont:
        train1[i] = feature_scaler.transform(pd.DataFrame(train1[i]))

    features1 = train1[train1.index.minute == 0]
    features1 = features1[features1.index.hour == 0]

    # Seperating training and test data for Multi-Layer Perceptron Network
    features_train1 = features1[:'2018']
    features_test1 = features1['2019':'2019-06-30']

    # Reshaping the features and test data to NP-Array as per Keras input requirement
    features_train1 = features_train1.to_numpy().reshape(features_train1.shape[0], features_train1.shape[1])
    features_test1 = features_test1.to_numpy().reshape(features_test1.shape[0], features_test1.shape[1])

    #################### PROCESSING THE DATA FOR LSTM NETWORK ###################
    # create feature and label dataframes
    prelim_features = train.drop(['RRP5MIN', 'RESIDUAL_DEMAND', 'AVG_PRICE', 'DIFF_PRICE', 'hour', 'weekday', 'month', 'business hour', 'public holiday'], axis=1)
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

    samples_train = len(features_train)
    samples_test = len(features_test)
    timesteps = 280

    # convert pandas data frames to numpy ndarrays
    features_train = features_train.to_numpy().reshape(samples_train, timesteps, 3)
    features_test = features_test.to_numpy().reshape(samples_test, timesteps, 3)
    labels_train = labels_train.to_numpy()

    # Creating the 2 models
    mlp = create_mlp((features_train1.shape[1],))
    lstm = create_conv_lstm((None, features_train.shape[1], 3))

    # Merging the 2 networks into a bigger network
    combinedInput = concatenate([mlp.output, lstm.output])

    # Mapping the bigger Network to the output layer to predict one-day ahead i.e. 288 intervals
    x = Dense(32, activation="relu")(combinedInput)
    x = Dense(288, activation="sigmoid")(x)
    model = Model(inputs=[mlp.input, lstm.input], outputs=x)

    # Compiling the model with Mean Absolute Error as loss function and Adam as optimizer
    model.compile(loss='mae', optimizer='adam')
    checkpoint = ModelCheckpoint('./models/multidim_timeseries_testing.hdf5', save_best_only=True)

    # Training the Model
    hist = model.fit(x=[features_train1, features_train], y=labels_train,
                     verbose=1, batch_size=50, epochs=160)

    # Making Predictions on the Testing Data
    pred = model.predict([features_test1, features_test])

    # Inverse scaling the predictions and re-shaping it to 1D output vector
    pred = label_scaler.inverse_transform(pred.flatten().reshape(-1, 1))

    # Combining Predictions and True Values in results dataframe
    results = pd.DataFrame({'prediction':pred.flatten(), 'true values':X_test}, index=X_test.index)

    return results
    # Plot of predictions against Actuals
    #cm.plot_chart(results["2019-01-01":"2019-01-10"], legend=True)


    # Quantifying Performance using MAE, MSE, RMSE
    #cm.quantify_performance(results)

