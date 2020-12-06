# Created 1st Oct 2020
# The custom module includes all functions to aid with the Neural Network implementation
# It also supports Plotting analysis


# Importing Required Libraries

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from keras.models import Model, load_model, Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
#from keras.regularizers import l1_l2

from sklearn.metrics import mean_absolute_error, mean_squared_error
import json



###########################################################
####################### SECTION 1 #########################
######### CUSTOM FUNCTIONS TO PREPROCESS THE DATA ########
######### AND ALSO FEATURE ENGINEERING FUNCTIONS ########
########################################################

# This function replaces the outliers in the data to ignore noise
def replace_outliers(data, column, tolerance):

    """Replace outliers out of 75% + tolerance * IQR or 25% - tolerance * IQR by these thresholds"""
    
    tol = tolerance
    data_prep = data.copy(deep=True)
    
    # calculate quantiles and inter-quantile range of the data
    q75 = data_prep[column].quantile(.75)
    q25 = data_prep[column].quantile(.25)
    IQR = q75 - q25

    # values larger (smaller) than q75 (q25) plus 'tol' times IQR get replaced by that value
    data_prep[column] = data_prep[column].apply(lambda x: q75 + tol * IQR if (x > q75 + tol * IQR) else x)
    data_prep[column] = data_prep[column].apply(lambda x: q25 - tol * IQR if (x < q75 - tol * IQR) else x)
    
    return data_prep


# This function runs the Dickey_Fuller test to test whether the series is stationary
def is_Stationary(data):
    series = data.values
    result = adfuller(series)
    if(result[1] <= 0.05):
        print("THE SERIES IS STATIONARY")
    else:
        print("THE SERIES IS NOT STATIONARY")

    print("-------------------")
    print("  Test Statistics  ")
    print("-------------------")

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# This function plots the ACF and PACF plots to find useful lags for time series
def serial_corr(data, lags):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    plt.tight_layout()
    plt.show()


# 
def subtract_years(dt, years):
    try:
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(year=dt.year-years)
    except ValueError:
        dt = str(dt)
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(year=dt.year-years, day=dt.day-1)
    return dt

# This function smoothes the curve by taking hourly average
def average_hours(data_column):
    new_column = data_column.rolling(min_periods=1, window=12).mean()
    return new_column

# This function takes the difference of a series to make it stationary
def period_difference(data_column):
    new_column = data_column.diff()
    return new_column

#################################################################
#################### END OF SECTION #1 #########################
###############################################################


###################################################################
####################### SECTION 2 ################################
################## NEURAL NETWORK MODELS ########################
######### MULTI-LAYER PERCEPTRON & LSTM MODELS CREATION ########
###############################################################

def create_mlp(dim, dense1=64, dense2=32):
    '''This Model takes the input shape and creates the MLP model'''
    # define our MLP network
    model = Sequential()
    model.add(Dense(dense1, input_shape=dim, activation="relu"))
    model.add(Dense(dense2, activation="relu"))
    return model

# Creates the Conv-LSTM model to perform Time-Series analysis
def create_conv_lstm(dim, dense1=64, dense2=64, dense3=32, activation_func='sigmoid',
dropout=0.1):
    '''This Model takes the input dimension and creates the LSTM model'''
    conv_input_layer = Input(batch_shape=dim)

    x = Conv1D(dense1, kernel_size=335, strides=335, padding='valid')(conv_input_layer)
    x = Dropout(dropout)(x)
    x = LSTM(dense2, recurrent_activation=activation_func)(x)
    x = Dense(dense3 , activation=activation_func)(x)
    model = Model(inputs=[conv_input_layer], outputs=[x])
    return model



#################################################################
#################### END OF SECTION #1 #########################
###############################################################


####################################################################
####################### SECTION 3 #################################
################## TRAINING THE NEURAL NETWROK ###################
####### EVALUATION ON TESTING DATA AND QUANTIFABLE METRICS ######
################################################################



# This function trains the model and validates it and outputs the predictions on best model
def train_predict_evaluate(model, X_train1, X_train, y_train, X_test1, X_test, y_test, test_index, scaler, 
                           batch_size, epochs, verbose=0):
    
    """Fit model to training data. We will validate on the sample from training and use the best model"""
    
    # train model, save best keep best performer on validation set
    #checkpoint = ModelCheckpoint('./models/' + filename, save_best_only=True)
# Training the Model
    hist = model.fit(x=[X_train1, X_train], y=y_train, 
                    verbose=verbose, batch_size=batch_size, epochs=epochs)
    # load best model
    #best = load_model('./models/' + filename)

    # Making Predictions on the Testing Data
    pred = model.predict([X_test1, X_test])
    
    # transform back to original data scale
    pred = scaler.inverse_transform(pred.flatten().reshape(-1, 1))
    results = pd.DataFrame({'prediction':pred.flatten(), 'true values':y_test}, index=test_index)
    
    return results, hist
    
# This function plots the loss over the training period
def training_evaluation(hist):
    f, ax = plt.subplots()
    pd.DataFrame(hist.history).plot(figsize=(12, 6), ax=ax)
    ax.set_title('Training and Validation Error over time', fontsize=16)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    ax.set_xlabel('Training Epoch', fontsize=14)

# This Function for Quantifying errors as MAE, MSE, RMSE
def quantify_performance(results):
    print('MAE: ', mean_absolute_error(y_pred=results['prediction'], y_true=results['true values']))
    print('MSE: ', mean_squared_error(y_pred=results['prediction'], y_true=results['true values']))
    print('RMSE: ', sqrt(mean_squared_error(y_pred=results['prediction'], y_true=results['true values'])))


#################################################################
#################### END OF SECTION #3 #########################
###############################################################



############################################
###### Custom Function for plotting #######
##########################################

# Plots any chart 
def plot_chart(data, xlab='Time', ylab='Price ($)', title=None, legend=False):
    ax = data.plot(figsize=(12, 5), legend=legend)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_title(title, fontsize=14)



############################################
################ Extras ###################
##########################################

# Creates the MLP model to perform regression on the categorical variables
# def create_mlp(dim):
#     '''This Model takes the input shape and creates the MLP model'''
#     # define our MLP network
#     model = Sequential()
#     model.add(Dense(64, input_shape=dim, activation="relu"))
#     model.add(Dense(32, activation="relu"))
#     # check to see if the regression node should be added
#     return model

# # Creates the Conv-LSTM model to perform Time-Series analysis
# def create_conv_lstm(dim):
#     '''This Model takes the input dimension and creates the LSTM model'''
#     conv_input_layer = Input(batch_shape=dim)

#     x = Conv1D(64, kernel_size=335, strides=335, padding='valid')(conv_input_layer)
#     x = Dropout(0.1)(x)
#     x = LSTM(64, recurrent_activation='relu')(x)
#     x = Dense(32 , activation='relu')(x)
#     model = Model(inputs=[conv_input_layer], outputs=[x])
#     return model