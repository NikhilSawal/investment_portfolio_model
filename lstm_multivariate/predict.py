import keras
import numpy as np
import pandas as pd

import load_data as ld
import get_clean_data as gcd
import feature_engineering as fe

import matplotlib.pyplot as plt
from math import sqrt

# Load train data
df_train = ld.get_data('2021-01-01', '2021-04-30', 'Uber Technologies, Inc. (UBER)')
df_train = gcd.clean_data(df_train)

# Prepare data for train-val-test split
X_total_train = df_train.copy()
X_total_train = X_total_train.values

# Perform train-val-test split
X_train, X_val, X_test = fe.train_val_test_split(X_total_train)

# Load and Clean data
df = ld.get_data('2021-04-28', '2021-04-30', 'Uber Technologies, Inc. (UBER)')
df = gcd.clean_data(df)

object_columns = list(df.select_dtypes(include=['object']).columns)
float_columns = list(df.select_dtypes(include=['float64']).columns)

object_indexes = [df.columns.get_loc(i) for i in object_columns]
float_indexes = [df.columns.get_loc(i) for i in float_columns]

# Prepare data for train-val-test split
X_total = df.copy()
X_total = X_total.values

test_set_scaled, num_x, MinMaxSc = fe.apply_transformations(X_train, X_total, float_indexes, object_indexes)

model = keras.models.load_model('/Users/nikhilsawal/OneDrive/investment_portfolio_model/lstm_multivariate/models/multivar_lstm')

X_test = []
for i in range(0, test_set_scaled.shape[0]-20):
    X_test.append(test_set_scaled[i:i+20, :])

X_test = np.array(X_test)

# Make Predictions
predicted_stock_price = model.predict(X_test)

# Inverse transformations
inv_df = num_x[num_x.shape[0]-10:, :]
inv_df[:, -1] = predicted_stock_price[6]
inv_df = MinMaxSc.inverse_transform(inv_df)
print(inv_df[:, -1])
