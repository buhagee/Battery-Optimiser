import seaborn as sns
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd



#############################################
######### CUSTOM FUNC TO REPLACE OUTLIERS
############################################
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

#############################################
######### CUSTOM FUNC TO REPLACE OUTLIERS
############################################

def train_predict_evaluate(model, X_train, X_valid, y_train, y_valid, X_test, real_test, real_test_index, scaler, 
                           batch_size, epochs, filename='model.hdf5', verbose=0):
    
    """Fit model to training data. Use best performant on validation data to predict for the test set. 
    Evaluate on the test set and return results as dataframes"""
    
    # train model, save best keep best performer on validation set
    checkpoint = ModelCheckpoint('./models/' + filename, save_best_only=True)
    hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), callbacks=[checkpoint], 
                     verbose=verbose, batch_size=batch_size, epochs=epochs)
    
    # load best model
    best = load_model('./models/' + filename)

    # predict for test set
    pred = best.predict(X_test)
    
    
    # transform back to original data scale
    results = pd.DataFrame(scaler.inverse_transform(pred), columns=["Prediction"])
    results.index = real_test_index
    results = pd.concat([real_test, results], axis=1)
    
    return results, hist

##################################
## Custom Function for plotting
#################################

def plot_chart(data, xlab='Time', ylab='Price ($)', title=None, legend=False):
    ax = data.plot(figsize=(12, 5), legend=legend)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_title(title, fontsize=14)