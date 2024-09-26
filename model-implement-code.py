# -*- coding: utf-8 -*-


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting

from sklearn.preprocessing import MinMaxScaler
from rnn_utils import mae, mse, rmse, mape, evaluate # helper evaluation functions
from keras import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU

FILE_PATH = "/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"
df = pd.read_csv(FILE_PATH, sep=";", parse_dates={'ds':['Date', 'Time']}, na_values=['nan', '?'], infer_datetime_format=True,low_memory=False)
df.head()





## Attribute Information

# 1. ds: Date in format dd/mm/yyyy
# 2. time: time in format hh:mm:ss
# 3. globalactivepower: household global minute-averaged active power (in kilowatt)
# 4. globalreactivepower: household global minute-averaged reactive power (in kilowatt)
# 5. voltage: minute-averaged voltage (in volt)
# 6. global_intensity: household global minute-averaged current intensity (in ampere)
# 7. submetering1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
# 8. submetering2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
# 9. submetering3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.


print(f"Missing values: {df.isnull().sum().any()}")
# imputation with the columns means
for j in range(0,8):
  df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
# checking for missing values
print(f"Missing values: {df.isnull().sum().any()}")

"""To simplify the problem, we just wish to forecast the future household electricity consumption, we will do some pre-processing to frame this problem properly:
- Downsampling from minute-average activate power to daily active power for households
- Imputation missing values with column mean
- MinMax Normalization to preserve variable distributions for our Recurrent Neural Networks


"""

df_resample = df.resample('D', on='ds').sum()
df_resample.rename(columns={"Global_active_power":"y"}, inplace=True)
df_resample = df_resample[['y']]
df_resample.head()

"""# 2. Pre-processing

Here we have some helper functions that help to create some simple lagged features to add into our model. You incorporate more complicated time-series features in your own work.

Recurrent Neural Networks can take in additional features as a 3-D array for input, where the three dimensions of this input are `sample`, `time_steps` and `features`:

1. Samples - One sequence is one sample. A batch is comprised of one or more samples.
2. Time Steps - One time step is one point of observation in the sample.
3. Features - One feature is one observation at a time step.

This means that the input layer expects a 3D array of data when fitting the model and when making predictions, even if specific dimensions of the array contain a single value, e.g. one sample or one feature.
"""

def create_lags(df, days=7):
    # create lagged data for features
    for i in range(days):
        df["Lag_{lag}".format(lag=i+1)] = df['y'].shift(i+1)
    return df

def create_features(X, time_steps=1, n_features=7):
    # create 3d dataset for input
    cols, names = list(), list()
    for i in range(1, time_steps+1):
        cols.append(X.shift(-time_steps))
        names += [name + "_" + str(i) for name in X.columns]
        agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    agg.dropna(inplace=True)
    agg = agg.values.reshape(agg.shape[0], time_steps, n_features)
    return agg

def create_dataset(df, yhat):
    # yhat needs to be scaled
    preds = pd.DataFrame(yhat.flatten())
    temp = pd.concat([df.iloc[:,0], preds])
    temp.columns = ['y']
    date_idx = pd.date_range(start='2006-12-23', periods=temp.shape[0])
    temp.set_index(date_idx, inplace=True)
    return temp

"""We preprocess by normalizing all variables first, taking care to avoid data leakage by using our MinMaxScaler on training data only"""

chosen = df_resample.copy()
chosen = create_lags(chosen)
chosen.dropna(inplace=True)

# Fit scaler on training data only to prevent data leakage
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_x = scaler.fit(chosen.iloc[:1096,1:])
scaler_y = scaler.fit(chosen.iloc[:1096,0].values.reshape(-1,1))

x_scaled = scaler_x.transform(chosen.iloc[:,1:])
y_scaled = scaler_y.transform(chosen.loc[:,['y']])

scaled = np.hstack((x_scaled, y_scaled))
scaled = pd.DataFrame(scaled, index=chosen.index, columns=chosen.columns)
print(scaled.shape)
scaled.head()

"""### Train-val-test split

We a simple train-test split for illustration purposes, where we predict for values from `2010-06-01` onwards for the test set.

Train - `2006-12-23` - `2009-12-22`  
Val - `2009-12-23` - `2010-05-31`  
Test - `2010-06-01` - `2010-11-26`
"""

train = scaled[:1096]
val = scaled[1096:1256]
test = scaled[1256:]
x_train = train.drop(["y"],axis=1)
y_train = train["y"]
x_val = val.drop(["y"],axis=1)
y_val = val["y"]
x_test = test.drop(["y"],axis=1)
y_test = test["y"]

x_train_np = create_features(x_train, 7, 7)
x_val_np = create_features(x_val, 7, 7)
x_test_np = create_features(x_test, 7, 7)
#print(x_train_np.shape, x_val_np.shape, x_test_np.shape)
y_test = y_test[:x_test_np.shape[0]]
y_train = y_train[:x_train_np.shape[0]]
y_val = y_val[:x_val_np.shape[0]]
#print(y_train.shape, y_val.shape, y_test.shape)

"""# 3. Forecasting with Recurrent Neural Networks
Here's a helper function to help us train our RNNs, LSTMs, GRUs, where we then forecast with them to get the normalized predictions
"""

def fit_model(m, units, x_train_np, x_val_np, verbose=False):
    model = Sequential()
    model.add(m (units = units, return_sequences = True, input_shape = [x_train_np.shape[1], x_train_np.shape[2]]))
    #model.add(Dropout(0.2))
    model.add(m (units = units))
    #model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    # Compile Model
    model.compile(loss='mse', optimizer='adam')
    # Fit Model
    history = model.fit(x_train_np, y_train, epochs=50, batch_size=70,
                        validation_data=(x_val_np, y_val), verbose=False, shuffle=False)
    return model

RNN_model = fit_model(SimpleRNN, 64, x_train_np, x_val_np)
LSTM_model = fit_model(LSTM, 64, x_train_np, x_val_np)
GRU_model = fit_model(GRU, 64, x_train_np, x_val_np)



RNN_preds = RNN_model.predict(x_test_np)
LSTM_preds = LSTM_model.predict(x_test_np)
GRU_preds = GRU_model.predict(x_test_np)

"""## 3.1 RNN"""

resultsDict = {}

rnn_preds = scaler_y.inverse_transform(RNN_preds)
y_test_actual = scaler_y.inverse_transform(pd.DataFrame(y_test))
resultsDict['RNN'] = evaluate(y_test_actual, rnn_preds)
evaluate(y_test_actual, rnn_preds)

plt.figure(figsize=(18,8))
plt.plot(rnn_preds, "r-", label="Predicted")
plt.plot(y_test_actual, label="Actual")
plt.title('RNN')
plt.legend()
plt.grid(True)
plt.savefig('1 - RNN.jpg', dpi=200)
plt.show()

"""## 3.2 LSTM"""

lstm_preds = scaler_y.inverse_transform(LSTM_preds)
y_test_actual = scaler_y.inverse_transform(pd.DataFrame(y_test))
resultsDict['LSTM'] = evaluate(y_test_actual, lstm_preds)
evaluate(y_test_actual, lstm_preds)



"""## 3.3 GRU"""

gru_preds = scaler_y.inverse_transform(GRU_preds)
y_test_actual = scaler_y.inverse_transform(pd.DataFrame(y_test))
resultsDict['GRU'] = evaluate(y_test_actual, gru_preds)
evaluate(y_test_actual, gru_preds)



"""# 4. Rolling Forecast with RNN, LSTM, GRU
Instead of forecasting out a very long sequence out `2010-06-01` to `2010-11-23`, 175 days between them is a medium-length sequence. Anecdotally, I can handle up to 300-length sequences with LSTM and GRU, but should experiment with a rolling forecasting schemes to see if it handles the potential vanishing gradient problem.

By rolling, we mean that we train on initial train set, predict next month. Expand training window to include the predictions from next month, then repeat the following cycle until we have our desired 175 prediction window:
1. Predict one month ahead
2. Create features based on predictions
3. Expand training window to include the predictions

This means that the maximum output sequence is 30-length long and can be readily handled without vanishing gradient problems.
"""

chosen = df_resample.copy()
chosen = create_lags(chosen)
chosen.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler_x = scaler.fit(chosen.iloc[:1096,1:])
scaler_y = scaler.fit(chosen.iloc[:1096,0].values.reshape(-1,1))

x_scaled = scaler_x.transform(chosen.iloc[:,1:])
y_scaled = scaler_y.transform(chosen.loc[:,['y']])

scaled = np.hstack((x_scaled, y_scaled))
scaled = pd.DataFrame(scaled, index=chosen.index, columns=chosen.columns)

train = scaled[:1078]
val = scaled[1078:1256]
test = scaled[1256:]

x_train = train.drop(["y"],axis=1)
y_train = train["y"]
x_val = val.drop(["y"],axis=1)
y_val = val["y"]
x_test = test.drop(["y"],axis=1)
y_test = test["y"]

"""Recreate the dataset, and write some helper functions for preprocessing, forecasting"""

## Helper Function
i = 0
def train_test_split(df, i=0):
    chosen = create_lags(df)
    chosen.dropna(inplace=True)
    x_scaled = scaler_x.transform(chosen.iloc[:,1:])
    y_scaled = scaler_y.transform(chosen.loc[:,['y']])

    scaled = np.hstack((x_scaled, y_scaled))
    scaled = pd.DataFrame(scaled, index=chosen.index, columns=chosen.columns)

    train = scaled[:1078+i]
    val = scaled[1078+i:1256+i]
    test = scaled[1256+i:]

    x_train = train.drop(["y"],axis=1)
    y_train = train["y"]
    x_val = val.drop(["y"],axis=1)
    y_val = val["y"]
    x_test = test.drop(["y"],axis=1)
    y_test = test["y"]

    n_features = len(x_train.columns)
    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = train_test_split(df_resample, i)
print(x_test.shape)

"""Use a simple for loop here of train, predict, re-train, predict our forecast"""

TIME_STEPS, N_FEATURES = 7, 7
rnn, lstm, gru = list(), list(), list()

for i in range(0, len(x_test), 30):
    temp = df_resample.copy()
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_split(temp, i)

    x_train_np = create_features(x_train, TIME_STEPS, N_FEATURES)
    x_val_np = create_features(x_val, TIME_STEPS, N_FEATURES)
    x_test_np = create_features(x_test, TIME_STEPS, N_FEATURES)
    #print(x_train_np.shape, x_val_np.shape, x_test_np.shape)
    y_test = y_test[:x_test_np.shape[0]]
    y_train = y_train[:x_train_np.shape[0]]
    y_val = y_val[:x_val_np.shape[0]]
    #print(y_train.shape, y_val.shape, y_test.shape)

    if y_test.shape[0] != 0:
        RNN_model = fit_model(SimpleRNN, 64, x_train_np, x_val_np)
        LSTM_model = fit_model(LSTM, 64, x_train_np, x_val_np)
        GRU_model = fit_model(GRU, 64, x_train_np, x_val_np)

        RNN_preds = RNN_model.predict(x_test_np)
        yhat_actual = scaler_y.inverse_transform(RNN_preds)
        rnn.extend(yhat_actual.flatten()[:30])
        LSTM_preds = LSTM_model.predict(x_test_np)
        yhat_actual = scaler_y.inverse_transform(LSTM_preds)
        lstm.extend(yhat_actual.flatten()[:30])
        GRU_preds = GRU_model.predict(x_test_np)
        yhat_actual = scaler_y.inverse_transform(GRU_preds)
        gru.extend(yhat_actual.flatten()[:30])

"""Because we are using the first 7 inputs as sequence to create features, then dropping the NA values, we have to first do:
```
y_test_actual[7:]
```
to enforce the lengths of the test values and the predictions to be of equal length
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model

# Temporal Focus Attention mechanism
class TempFocusAttention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(TempFocusAttention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # This will learn attention weights for each input step
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer="glorot_uniform", trainable=True)
        super(TempFocusAttention, self).build(input_shape)

    def call(self, inputs):
        # Alignment scores between input and the focus
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[2, 0]) + self.b)
        score = tf.tensordot(score, self.u, axes=[2, 0])
        
        # Softmax over time steps to get the attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Reshape attention weights to be compatible with LSTM output
        attention_weights = tf.expand_dims(attention_weights, axis=-1)

        # Multiply attention weights with LSTM output
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        if self.return_sequences:
            return context_vector, attention_weights
        return context_vector

# Define the LSTM model with Temporal Focus Attention
def fit_model(LSTM_units, batch_size, x_train, x_val):
    input_shape = x_train.shape[1:]
    
    inputs = Input(shape=input_shape)
    
    # LSTM layer
    lstm_out = LSTM(LSTM_units, return_sequences=True)(inputs)
    
    # Apply Temporal Focus Attention
    context_vector, attention_weights = TempFocusAttention(return_sequences=True)(lstm_out)
    
    # Dense layer to generate final output
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(context_vector)
    
    # Model definition
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    history = model.fit(x_train, batch_size=batch_size, epochs=10, validation_data=(x_val, None))
    
    return model, history

# Example call to fit the model
LSTM_model, history = fit_model(64, 32, x_train_np, x_val_np)
