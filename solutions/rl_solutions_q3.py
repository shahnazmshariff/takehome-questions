import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import EarlyStopping
import keras

def initial_plot(df):

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),df['IPG3113N'])
    plt.xticks(range(0,df.shape[0],100),df['observation_date'].loc[::100],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Candy Production',fontsize=18)
    plt.savefig('candy_production_data.png')
    plt.show()

def generate_stats_on_dataset(df):
    print df.describe()
    # prints :
    ##
    #         IPG3113N
    # count  548.000000
    # mean   100.662524
    # std     18.052931
    # min     50.668900
    # 25%     87.862475
    # 50%    102.278550
    # 75%    114.691900
    # max    139.915300
    ##

def normalize_data(df):
    df_mean = df['IPG3113N'].mean()
    df_max = df['IPG3113N'].max()
    df_min = df['IPG3113N'].min()

    index_data = (df['IPG3113N'] - df_mean) / (df_max - df_min)

    return index_data

def denormalize_data(df, series):
    df_mean = df['IPG3113N'].mean()
    df_max = df['IPG3113N'].max()
    df_min = df['IPG3113N'].min()
    return series * (df_max - df_min) + df_mean

def transform_data(index_data, window_size):
    result = []
    window_size = window_size + 1

    for i in range(len(index_data) - window_size + 1):
        result.append(index_data[i: i + window_size])

    result = np.array(result)

    return result

def split_training_test_data(result, predict):

    # 90% for training and rest for testing
    cut_off_range = int(round(0.9 * result.shape[0]))

    train = result[:cut_off_range, :]

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[cut_off_range:, :-1]
    y_test = result[cut_off_range:, -1]

    # need to transform or reshape the data to use Keras LSTM model - numpy array of 3 dimensions (No. of training
    # samples, window size and number of features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    if predict:
    ## last 36 data points to predict the unseen value i.e., to predict the candy production index for Sept 2017
        x_test1 = result[511:, 1:37]
        x_test1 = np.reshape(x_test1, (x_test1.shape[0], x_test1.shape[1], 1))
        print x_test1.shape, "Shape of test dataset to predict next value in the series"
        return x_test1

    return [x_train, y_train, x_test, y_test]


def verify_shape_of_datasets(x_train, y_train, x_test, y_test):
    print("X_train", x_train.shape)
    print("Y_train", y_train.shape)
    print("X_test", x_test.shape)
    print("Y_test", y_test.shape)

def build_model(layers):
    dropout = 0.25
    inputs = keras.Input(shape=(layers[1], layers[0]))
    #using training=True to calculate prediction uncertainty
    x = keras.layers.LSTM(36, recurrent_dropout=dropout, return_sequences=True)(inputs, training=True)
    x = keras.layers.Dropout(dropout)(x, training=True)
    x = keras.layers.LSTM(128)(x, training=True)
    x = keras.layers.Dropout(dropout)(x, training=True)
    outputs = keras.layers.Dense(layers[3])(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    return model

def run_model(model, x_train, y_train, x_test, y_test):

    epochs = 50
    seed = 7
    np.random.seed(seed)
    batch = 32

    # early callback to ensure proper training and avoiding overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience = 5)

    # fit model with batch size and epochs and set the validation split to 10%
    history = model.fit(x_train, y_train, batch_size=batch, nb_epoch=epochs, validation_split=0.1, callbacks=[early_stopping])

    score = model.evaluate(x_test, y_test, verbose=0)

    print ("Test loss of the model: " + str(score))

    return history


def plot_validation_loss(history):
    fig = plt.figure(facecolor='white', figsize=(8, 8))
    spacing = 0.01
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')
    plt.title('Model train vs Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_vs_validation_loss.png')
    plt.show()


def plot_predictive_interval(y_test, y_test_pred, z):
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    # confidence interval of 90%
    interval = z * rmse

    plt.plot(y_test_pred, 'r', label='predictions')
    plt.plot(y_test, 'g', label='labels')
    plt.plot(y_test + interval, 'b')
    plt.plot(y_test - interval, 'b')
    plt.ylabel('index_value')
    plt.xlabel('time steps')
    plt.title('Labels & Predictions')
    plt.legend()
    plt.savefig('confidence_int_90.png')
    plt.show()


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, i.e. 1 step ahead
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def plot_results(predicted_data, true_data):
    tick_spacing = 0.05
    fig = plt.figure(facecolor='white', figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.plot(true_data, label='True Data', color = 'blue')
    plt.plot(predicted_data, label='Prediction', color = 'green')
    plt.legend()
    plt.savefig('prediction_result.png')
    plt.show()

def perf_stats(y_pred, y_test):
    # For normalised values
    forecast_errors = [y_test[i] - y_pred[i] for i in range(len(y_test))]

    ## Mean forecast error or bias
    mean_forecast_error = np.mean(forecast_errors)
    print('Normalized Bias: %s' % mean_forecast_error)

    ## RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print('Normalized RMSE: %f' % rmse)


    # Denormalizing the data
    y_test = denormalize_data(df, y_test)
    y_pred = denormalize_data(df, y_pred)

    forecast_errors = [y_test[i] - y_pred[i] for i in range(len(y_test))]

    ## Mean forecast error or forecast bias (using actual values)
    mean_forecast_error = np.mean(forecast_errors)
    print('Actual Bias: %s' % mean_forecast_error)

    ## RMSE (using actual values)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print('Actual RMSE: %f' % rmse)

def predict_next_value(model, df, x_test1):
    T = 1000  # Do 1000 predictions to estimate uncertainty
    predictions = np.array([[model.predict(x_test1)] for _ in range(T)])

    # mean of all predictions is a good approximation of the final prediction,
    # while the standard deviation (of all predictions) is a good measure of the uncertainty of the model for
    # that particular input
    denormalized_predictions = np.array([denormalize_data(df,val) for val in predictions])

    pred_mean = denormalized_predictions.mean(axis=0)
    pre_std = denormalized_predictions.std(axis=0)

    print pred_mean, "predicting next value (unseen) based on mean value from 1000 iterations"
    print pre_std, "uncertainty in prediction (given by the std. deviation for 1000 iterations)"


if __name__ == '__main__':
    df = pandas.read_csv('~/RLSolutions/candy_production.csv')

    #prints the stats such as count, mean, min, max etc (optional)
    generate_stats_on_dataset(df)

    # save initial plot (optional)
    initial_plot(df)

    # get normalized data
    index_data = normalize_data(df)

    # pass window_size = 36 and normalized data to obtain a list of list
    #each list within the main list contains 37 data points
    result = transform_data(index_data, 36)

    # split_datasets contains a list of x_train, y_train, x_test, y_test
    split_datasets = split_training_test_data(result, predict=False)

    x_train = split_datasets[0]
    y_train = split_datasets[1]

    x_test = split_datasets[2]
    y_test = split_datasets[3]

    # print the dimensions/shape of train and test datasets (optional)
    verify_shape_of_datasets(x_train, y_train, x_test, y_test)

    ####### TRAINING AND VALIDATION PHASE #######

    # Build model with 2 hidden layers with 36 and 128 lstm cells respectively
    model = build_model([1, 36, 128, 1])
    print model.summary()

    history = run_model(model, x_train, y_train, x_test, y_test)

    # plot train vs. validation loss to understand the fit of the model
    plot_validation_loss(history)

    ############ TESTING PHASE ###############

    y_pred = predict_point_by_point(model, x_test)
    plot_results(y_pred, y_test) # plot prediction results
    perf_stats(y_pred, y_test) # get basic performance stats (RMSE, forecast bias) for LSTM model evaluation

    y_test_actual = denormalize_data(df, y_test)
    y_pred_actual = denormalize_data(df, y_pred)

    # z = 1.64 for 90% predictive interval
    plot_predictive_interval(y_test_actual, y_pred_actual, 1.64)

    ########## PREDICT NEXT VALUE IN SERIES ###########

    x_test1 = split_training_test_data(result, predict=True)
    predict_next_value(model, df, x_test1) # predicts the next value in the timeseries i.e. value for Sep 2017



