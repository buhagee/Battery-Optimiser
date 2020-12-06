import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

from keras.models import Model, load_model, Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2

from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import datetime


###############################################
######### CUSTOM FUNC TO REPLACE OUTLIERS ####
#############################################
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


######################################################
######### CUSTOM FUNC TO TRAIN AND EVALUATE MODEL ###
####################################################

def train_predict_evaluate(model, X_train, X_valid, y_train, y_valid, X_test, y_test, test_index, scaler,
                           batch_size, epochs, filename='models.hdf5', verbose=0):
    """Fit models to training data. We will validate on the sample from training and use the best models"""

    # train models, save best keep best performer on validation set
    checkpoint = ModelCheckpoint('./models/' + filename, save_best_only=True)
    hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), callbacks=[checkpoint],
                     verbose=verbose, batch_size=batch_size, epochs=epochs)

    # load best models
    best = load_model('./models/' + filename)

    # predict for test set
    pred = best.predict(X_test)

    # transform back to original data scale
    pred = scaler.inverse_transform(pred.flatten().reshape(-1, 1))
    results = pd.DataFrame({'prediction': pred.flatten(), 'true values': y_test}, index=test_index)

    return results, hist


############################################
###### Custom Function for plotting #######
##########################################

def plot_chart(data, xlab='Time', ylab='Price ($)', title=None, legend=False):
    ax = data.plot(figsize=(12, 5), legend=legend)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_title(title, fontsize=14)
    plt.show()

#################################################################
## Custom Function for plotting Evaluation of traing results ###
###############################################################

def training_evaluation(hist):
    f, ax = plt.subplots()
    pd.DataFrame(hist.history).plot(figsize=(12, 6), ax=ax)
    ax.set_title('Training and Validation Error over time', fontsize=16)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    ax.set_xlabel('Training Epoch', fontsize=14);


def quantify_performance(results):
    print('MAE: ', mean_absolute_error(y_pred=results['prediction'], y_true=results['true values']))
    print('MSE: ', mean_squared_error(y_pred=results['prediction'], y_true=results['true values']))
    print('RMSE: ', sqrt(mean_squared_error(y_pred=results['prediction'], y_true=results['true values'])))


#################################################################
## Feature Engineering functions ###
###############################################################

def subtract_years(dt, years):
    try:
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(year=dt.year-years)
    except ValueError:
        dt = str(dt)
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(year=dt.year-years, day=dt.day-1)
    return dt

def average_hours(data_column):
    new_column = data_column.rolling(min_periods=1, window=12).mean()
    return new_column

def period_difference(data_column):
    new_column = data_column.diff()
    return new_column


