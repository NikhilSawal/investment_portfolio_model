import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def train_val_test_split(data):

    # Perform train test split
    tsv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tsv.split(data):
        X, X_test = data[train_index], data[test_index]

    # Perform train validation split
    for train_index, val_index in tsv.split(X):
        X_train, X_val = X[train_index], X[val_index]

    return X_train, X_val, X_test

def fit_transformations(data, float_indexes, object_indexes):

    # Numeric data
    mmscl = MinMaxScaler(feature_range=(0,1))
    num_x_train = mmscl.fit_transform(data[:, float_indexes])

    # Top 3 news
    tfidf_1 = TfidfVectorizer()
    text_x_train_3 = tfidf_1.fit_transform(data[:, object_indexes[0]]).toarray()

    # News Source
    tfidf_2 = TfidfVectorizer()
    text_x_train_4 = tfidf_2.fit_transform(data[:, object_indexes[1]]).toarray()

    train_set_scaled = np.concatenate((text_x_train_3, text_x_train_4, num_x_train), axis=1)

    return train_set_scaled, mmscl, tfidf_1, tfidf_2


def apply_transformations(train_df, fit_df, float_indexes, object_indexes):

    # Get scaling
    MinMaxSc = fit_transformations(train_df, float_indexes, object_indexes)[1]
    tfidf_1 = fit_transformations(train_df, float_indexes, object_indexes)[2]
    tfidf_2 = fit_transformations(train_df, float_indexes, object_indexes)[3]

    num_x = MinMaxSc.transform(fit_df[:, float_indexes])
    text_x_3 = tfidf_1.transform(fit_df[:, object_indexes[0]]).toarray()
    text_x_4 = tfidf_2.transform(fit_df[:, object_indexes[1]]).toarray()

    scaled_set = np.concatenate((text_x_3, text_x_4, num_x), axis=1)

    return scaled_set, num_x, MinMaxSc
