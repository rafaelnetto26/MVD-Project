import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score


def plot_results(y_true, y_pred):
    fig = pyplot.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(y_true, label='True Data')
    pyplot.plot(y_pred, label='Prediction')
    pyplot.legend()
    pyplot.show()

def _calc_re(y_true, y_pred):
    """
    Sum/Mean Relative Error）
    """
    return [((y_true - y_pred) / y_pred).sum().values, ((y_true - y_pred) / y_pred).mean().values]

def model_evaluation(y_true, y_pred):
    metrics = _cal_metrics(y_true, y_pred)
    for (k, v) in metrics.items():
        print(k + ": " + str(v))




def _cal_metrics(y_true, y_pred):
    """
    Calcule o valor de cada indicador
    """
    re = _calc_re(y_true, y_pred)
    metrics = {
        "explained_variance_score":
            explained_variance_score(y_true, y_pred),
        "mean_absolute_error":
            mean_absolute_error(y_true, y_pred),
        "mean_squared_error":
            mean_squared_error(y_true, y_pred),
        "median_absolute_error":
            median_absolute_error(y_true, y_pred),
        "r2_score":
            r2_score(y_true, y_pred),
        "sum_relative_error":
            re[0],
        "mean_relative_error":
            re[1]
    }

    return metrics




def moving_test_window_preds(n_future_preds):
    ''' n_future_preds - Represents the number of future predictions we want to make
                         This coincides with the number of windows that we will move forward
                         on the test data
    '''
    preds_moving = []  # Use this to store the prediction made on each test window
    moving_test_window = [test_X[0, :].tolist()]  # Creating the first test window
    moving_test_window = np.array(moving_test_window)  # Making it an numpy array

    for i in range(n_future_preds):
        preds_one_step = model.predict(moving_test_window)  # Note that this is already a scaled prediction so no need to rescale this
        preds_moving.append(preds_one_step[0, 0])  # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1, 1, 1)  # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:, 1:, :], preds_one_step),axis=1)  # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end
    preds_moving = np.reshape(preds_moving,(-1,1))
    preds_moving = scaler.inverse_transform(preds_moving)

    return preds_moving


series = pd.read_csv('stock.csv', header=None )
print(series.head())
print(series.shape)

print(series.describe())

pyplot.figure(figsize=(20,6))
pyplot.plot(series.values)
pyplot.show()



#pyplot.figure(figsize=(20,6))
#pyplot.plot(series.values[:50])
#pyplot.show()

# normalize features -

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(series.values)
series = pd.DataFrame(scaled)

window_size = 60

series_s = series.copy()
for i in range(window_size):
    series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)

series.dropna(axis=0, inplace=True)
print(series.head())
print(series.shape)
nrow = round(0.8*series.shape[0])
train = series.iloc[:nrow, :]
test = series.iloc[nrow:,:]
train = shuffle(train)
train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]
train_X = train_X.values
train_y = train_y.values
test_X = test_X.values
test_y = test_y.values
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)


train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)








# Define the LSTM model
model = Sequential()
model.add(LSTM(input_shape = (window_size,1), output_dim= 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("tanh"))
model.compile(loss="mse", optimizer="adam")
model.summary()

start = time.time()
model.fit(train_X,train_y,batch_size=240,nb_epoch=5,validation_split=0.35)
print("> Compilation Time : ", time.time() - start)

# Doing a prediction on all the test data at once
preds = model.predict(test_X)
preds = scaler.inverse_transform(preds)
test_y = test_y.reshape(test_y.shape[0],1)
actuals = scaler.inverse_transform(test_y)
#actuals = test_y

print(mean_squared_error(actuals,preds))
plot_results(actuals,preds)
#preds_moving = moving_test_window_preds(50)
#plot_results(actuals,preds_moving)
model_evaluation(pd.DataFrame(actuals), pd.DataFrame(preds))







