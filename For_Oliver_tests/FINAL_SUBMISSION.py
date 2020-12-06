import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import custom_module as cm
import optimizer_module as om
from datetime import date
import holidays

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import tensorflow as tf
from keras.layers import concatenate
import datetime

def feature_engineering_input(historic_input):
    historic_data = pd.DataFrame.from_dict(historic_input)

    # Setting the data to index
    historic_data["SETTLEMENTDATE"] = pd.to_datetime(historic_data["SETTLEMENTDATE"])
    curr_date = historic_data["SETTLEMENTDATE"].tail(1)
    curr_date += datetime.timedelta(days=1)
    historic_data.index = historic_data["SETTLEMENTDATE"]
    historic_data.drop(columns="SETTLEMENTDATE", inplace=True)



    # Avg Price of last 1 hour i.e. 12 data points at 5 minutes granularity
    historic_data["AVG_PRICE_DIFF"] = pd.DataFrame(cm.average_hours(historic_data["RRP5MIN"]))

    # Differencing the average price and creating a differenced price variable
    historic_data["AVG_PRICE_DIFF"] = cm.period_difference(historic_data["AVG_PRICE_DIFF"])
    historic_data["DIFF_PRICE"] = cm.period_difference(historic_data["RRP5MIN"])

    historic_data['RESIDUAL_DEMAND'] = historic_data['DEMAND'] - historic_data['WIND'] - historic_data['WIND']
    historic_data.drop(columns=["DEMAND","WIND","WIND"], inplace=True)

    time = pd.date_range(curr_date, periods=288, freq='5min')
    columns = ["RRP5MIN", "AVG_PRICE_DIFF", "DIFF_PRICE", "RESIDUAL_DEMAND"]

    #create dummy dataframe 1 day ahead
    dummy_data = pd.DataFrame(index=time, columns=columns)

    #append to historic data
    historic_data = historic_data.append(dummy_data)

    # Creating Daily lags
    for i in range(1, 201):
        historic_data["price_l_{}".format(i)] = historic_data["DIFF_PRICE"].shift(i)
        historic_data["demand_l_{}".format(i)] = historic_data["RESIDUAL_DEMAND"].shift(i)
        historic_data["avgPrice_l_{}".format(i)] = historic_data["AVG_PRICE"].shift(i)

    # Creating Daily lags
    for i in range(255, 325):
        historic_data["price_l_{}".format(i)] = historic_data["DIFF_PRICE"].shift(i)
        historic_data["demand_l_{}".format(i)] = historic_data["RESIDUAL_DEMAND"].shift(i)
        historic_data["avgPrice_l_{}".format(i)] = historic_data["AVG_PRICE"].shift(i)

    # Creating Week ago lags
    j = 1
    size = 2016
    for i in range(size, size - 65, -1):
        historic_data["w_price_l_{}".format(j)] = historic_data["DIFF_PRICE"].shift(i)
        historic_data["w_demand_l_{}".format(j)] = historic_data["RESIDUAL_DEMAND"].shift(i)
        historic_data["w_avgPrice_l_{}".format(j)] = historic_data["AVG_PRICE"].shift(i)
        j += 1


    # Generate 'hour', 'weekday' and 'month' features
    historic_data['hour'] = 0
    historic_data['weekday'] = 0
    historic_data['month'] = 0
    for i in range(len(historic_data)):
        position = historic_data.index[i]
        historic_data['hour'][i] = position.hour
        historic_data['weekday'][i] = position.weekday()
        historic_data['month'][i] = position.month

    # MAKING FEATURES
    # Generate 'business hour' feature. 7am-7pm business hours
    historic_data["business hour"] = 0
    for i in range(len(historic_data)):
        position = historic_data.index[i]
        hour = position.hour
        if ((hour > 7 and hour < 12) or (hour > 14 and hour < 19)):
            historic_data["business hour"][i] = 2
        elif (hour >= 12 and hour <= 14):
            historic_data["business hour"][i] = 1
        else:
            historic_data["business hour"][i] = 0

    # Generate 'weekend' feature
    for i in range(len(historic_data)):
        position = historic_data.index[i]
        weekday = position.weekday()
        if (weekday == 6):
            historic_data['weekday'][i] = 2
        elif (weekday == 5):
            historic_data['weekday'][i] = 1
        else:
            historic_data['weekday'][i] = 0

    aus_holidays = holidays.CountryHoliday('AUS', prov='SA')

    historic_data["public holiday"] = 0
    for i in range(len(historic_data)):
        if (historic_data.index[i] in aus_holidays):
            historic_data["public holiday"][i] = 1

    return historic_data

def train_price_predictor():

    data = pd.read_csv("final.csv", parse_dates=True, index_col=0)
    # replace outliers by outlier threshold
    data = cm.replace_outliers(data, 'RRP5MIN', 4)

    # Scaling the RRP between 0 and 1 as required by the NN
    features = ['RESIDUAL_DEMAND', 'AVG_PRICE_DIFF', 'DIFF_PRICE']
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

    # Creating Daily lags
    for i in range(255, 325):
        train["price_l_{}".format(i)] = train["DIFF_PRICE"].shift(i)
        train["demand_l_{}".format(i)] = train["RESIDUAL_DEMAND"].shift(i)
        train["avgPrice_l_{}".format(i)] = train["AVG_PRICE"].shift(i)

    # Creating Week ago lags
    j = 1
    size = 2016
    for i in range(size, size - 65, -1):
        train["w_price_l_{}".format(j)] = train["DIFF_PRICE"].shift(i)
        train["w_demand_l_{}".format(j)] = train["RESIDUAL_DEMAND"].shift(i)
        train["w_avgPrice_l_{}".format(j)] = train["AVG_PRICE"].shift(i)
        j += 1
    # Drop NANS
    train.dropna(inplace=True)
    train.head(5)
    train = train["2017":]

    #################### PROCESSING THE DATA FOR MLP NETWORK ###################
    ########### THIS IS FOR MULTILAYER PERCEPTRON PURPOSES ###################
    train1 = data[['hour', 'weekday', 'month', 'business hour', 'public holiday']]
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
    timesteps = 335

    # convert pandas data frames to numpy ndarrays
    features_train = features_train.to_numpy().reshape(samples_train, timesteps, 3)
    features_test = features_test.to_numpy().reshape(samples_test, timesteps, 3)
    labels_train = labels_train.to_numpy()

    #############################################################################
    ########### CONCATENATE THE 2 NN & COMPILE THEM TO FORM BIGGER NN ##########
    ###########################################################################

    # Creating the 2 models
    mlp_shape = (features_train1.shape[1],)
    lstm_shape = (None, features_train.shape[1], 3)

    # mlp = cm.create_mlp((features_train1.shape[1],))
    # lstm = cm.create_conv_lstm((None, features_train.shape[1], 3))

    mlp = cm.create_mlp(mlp_shape)
    lstm = cm.create_conv_lstm(lstm_shape)

    # Merging the 2 networks into a bigger network
    combinedInput = concatenate([mlp.output, lstm.output])

    # Mapping the bigger Network to the output layer to predict one-day ahead i.e. 288 intervals
    x = Dense(32, activation="relu")(combinedInput)
    x = Dense(288, activation="sigmoid")(x)
    model = Model(inputs=[mlp.input, lstm.input], outputs=x)

    # Compiling the model with Mean Absolute Error as loss function and Adam as optimizer
    model.compile(loss='mse', optimizer='adam')

    model.save('price_predictor.h5')  # creates a HDF5 file 'my_model.h5'

    return



