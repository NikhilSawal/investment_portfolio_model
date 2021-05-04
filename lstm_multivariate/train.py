import numpy as np
import pandas as pd

import load_data as ld
import get_clean_data as gcd
import feature_engineering as fe

import matplotlib.pyplot as plt
from math import sqrt

import tensorflow as ten
ten.random.set_seed(221)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Load and Clean data
df = ld.get_data('2021-01-01', '2021-04-30', 'Uber Technologies, Inc. (UBER)')
df = gcd.clean_data(df)

object_columns = list(df.select_dtypes(include=['object']).columns)
float_columns = list(df.select_dtypes(include=['float64']).columns)

object_indexes = [df.columns.get_loc(i) for i in object_columns]
float_indexes = [df.columns.get_loc(i) for i in float_columns]

# Prepare data for train-val-test split
X_total = df.copy()
X_total = X_total.values

# Perform train-val-test split
X_train, X_val, X_test = fe.train_val_test_split(X_total)

# Get feature scaled data
train_set_scaled = fe.fit_transformations(X_train, float_indexes, object_indexes)[0]
val_set_scaled = fe.apply_transformations(X_train, X_val, float_indexes, object_indexes)
test_set_scaled = fe.apply_transformations(X_train, X_test, float_indexes, object_indexes)

# Create training data structure with 20 timestamps as input and 10 as output
X_train = []
y_train = []
for i in range(20, train_set_scaled.shape[0]-10):
    X_train.append(train_set_scaled[i-20:i, :])
    y_train.append(train_set_scaled[i:i+10, -1])

X_train, y_train = np.array(X_train), np.array(y_train)

# Create validation data structure with 20 timestamps as input and 10 as output
val_set_scaled = np.append(train_set_scaled[train_set_scaled.shape[0]-20:,0:], val_set_scaled, axis=0)

X_val = []
y_val = []
for i in range(20, val_set_scaled.shape[0]-10):
    X_val.append(val_set_scaled[i-20:i, :])
    y_val.append(val_set_scaled[i:i+10, -1])

X_val, y_val = np.array(X_val), np.array(y_val)

# Train and fit model
# Initialize the RNN
regressor = Sequential()

# Adding first LSTM layer and Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# Adding second layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding third layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding forth layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding output layer
regressor.add(Dense(units=10))

# Compile RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting RNN
regressor.fit(X_train, y_train,
              epochs=100, batch_size=30,
              validation_data=(X_val, y_val))

# Save Model
regressor.save('/Users/nikhilsawal/OneDrive/investment_portfolio_model/lstm_multivariate/models/multivar_lstm')
