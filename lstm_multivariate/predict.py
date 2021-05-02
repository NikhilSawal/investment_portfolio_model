import datetime
import numpy as np
import pandas as pd
import get_data as gd
import get_moving_avg as gma
import text_features as tf

import matplotlib.pyplot as plt
from statistics import stdev
from math import sqrt

import tensorflow as ten
ten.random.set_seed(221)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', 8000)
pd.set_option('display.max_columns', 500)

df = gd.get_data()
df.iloc[:,1:4] = df.iloc[:,1:4].fillna(0.0)
df.iloc[:,4:6] = df.iloc[:,4:6].fillna('{None}')
df.iloc[:,6:] = df.iloc[:,6:].fillna(0.0)

# Perform feature engineering
df.loc[:,'top_3_news'] = tf.lambda_nltk_news(df, 'top_3_news')
df.loc[:,'news_source'] = tf.lambda_nltk_news(df, 'news_source')

df = df[['date', 'delta_price', 'delta_price_perc', 'top_3_news',
         'news_source', 'snp_500', 'snp_500_delta', 'snp_500_delta_perc',
         'dow_30', 'dow_30_delta', 'dow_30_delta_perc', 'nasdaq',
         'nasdaq_delta', 'nasdaq_delta_perc', 'price']]

object_columns = list(df.select_dtypes(include=['object']).columns)
float_columns = list(df.select_dtypes(include=['float64']).columns)

object_indexes = [df.columns.get_loc(i) for i in object_columns]
float_indexes = [df.columns.get_loc(i) for i in float_columns]

X_total = df.copy()
X_total = X_total.values

# Perform train test split
tsv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tsv.split(X_total):
    X, X_test = X_total[train_index], X_total[test_index]

# Perform train validation split
for train_index, val_index in tsv.split(X):
    X_train, X_val = X[train_index], X[val_index]

# Feature Engineering
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer

# Fit transformations & apply them

# Numeric data
mmscl = MinMaxScaler(feature_range=(0,1))
num_x_train = mmscl.fit_transform(X_train[:, float_indexes])
num_x_val = mmscl.transform(X_val[:, float_indexes])
num_x_test = mmscl.transform(X_test[:, float_indexes])

################
## Approach 1 ##
################

# Text data
tfidf = TfidfVectorizer()

# Top 3 news
text_x_train_3 = tfidf.fit_transform(X_train[:, 3]).toarray()
text_x_val_3 = tfidf.transform(X_val[:, 3]).toarray()
text_x_test_3 = tfidf.transform(X_test[:, 3]).toarray()

# News source
text_x_train_4 = tfidf.fit_transform(X_train[:, 4]).toarray()
text_x_val_4 = tfidf.transform(X_val[:, 4]).toarray()
text_x_test_4 = tfidf.transform(X_test[:, 4]).toarray()

train_set_scaled = np.concatenate((text_x_train_3, text_x_train_4, num_x_train), axis=1)
val_set_scaled = np.concatenate((text_x_val_3, text_x_val_4, num_x_val), axis=1)
test_set_scaled = np.concatenate((text_x_test_3, text_x_test_4, num_x_test), axis=1)


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

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

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

# Create test data structure with 20 timestamps as input and 10 as output
test_set_scaled = np.append(val_set_scaled[val_set_scaled.shape[0]-20:,0:], test_set_scaled, axis=0)

X_test = []
y_test = []
for i in range(20, test_set_scaled.shape[0]-10):
    X_test.append(test_set_scaled[i-20:i, :])
    y_test.append(test_set_scaled[i:i+10, -1])

X_test, y_test = np.array(X_test), np.array(y_test)

# Make Predictions
predicted_stock_price = regressor.predict(X_test)

inv_df = num_x_test[num_x_test.shape[0]-10:, :]
inv_df[:, -1] = predicted_stock_price[147]
inv_df = mmscl.inverse_transform(inv_df)

predicted_stock_price = inv_df[:, -1]

inv_df = num_x_test[num_x_test.shape[0]-10:, :]
inv_df[:, -1] = y_test[147]
inv_df = mmscl.inverse_transform(inv_df)

real_stock_price = inv_df[:, -1]

print(sqrt(mean_squared_error(predicted_stock_price, real_stock_price)))
# Visualize the results
plt.plot(real_stock_price, color='red', label='Real stock prices')
plt.plot(predicted_stock_price, color='blue', label='Predicted stock prices')
plt.legend()
plt.show()

# 1.79
# 0.2720538303665242
# 0.2720538303665242








#
#
#
#
#
#
#
#
#
# # ############################################################
# # # print(df.columns)
# # #
# # # # 'date', 'price', 'delta_price', 'delta_price_perc', 'top_3_news',
# # # # 'news_source', 'snp_500', 'snp_500_delta', 'snp_500_delta_perc',
# # # # 'dow_30', 'dow_30_delta', 'dow_30_delta_perc', 'nasdaq', 'nasdaq_delta',
# # # # 'nasdaq_delta_perc'
# # #
# # # # plt.plot(df.nasdaq)
# # # # plt.show()
# # # #
# # # # plt.plot(df.nasdaq_delta)
# # # # plt.show()
# # # #
# # # # plt.plot(df.nasdaq_delta_perc)
# # # # plt.show()
# # #
# # # # print([i for i in df.nasdaq_delta_perc if i > 20])
# # # # print(df.nasdaq_delta_perc.mean())
# # # # print(df.nasdaq_delta_perc.median())
# # # # print([i for i in df.nasdaq_delta])
# # # # print(df.nasdaq_delta.mean())
# # # # print(3*stdev(df.nasdaq_delta.fillna(0.0)))
# # # #
# # # delta_mean = df.nasdaq_delta.mean().round(2)
# # # delta_stdev = 3*stdev(df.nasdaq_delta.fillna(0.0))
# # #
# # # new_nas_del = [delta_mean if i > delta_stdev or str(i) == 'nan' else i for i in df.nasdaq_delta]
# # #
# # #
# # #
# # #
# # # # plt.plot(new_nas_del)
# # # # plt.show()
# # #
# # # print(df.isna().sum())
# # #
# # # price_sma = gma.sma(df, 'price', 9)
# # # price_ema = gma.ema(df, 'price', 25)
# # #
# # # # plt.plot(df.price[9:])
# # # # plt.plot(price_sma[9:])
# # # # # plt.plot(price_ema[25:])
# # # # plt.show()
# # #
# # # # print(df.price[25:50])
# # #
# # # while df['price'].isna():
# # #     df.loc[:, 'price'] = gma.sma(df, 'price', 9)
# # #
# # # print(df.price.isna().sum())
