import numpy as np
import pandas as pd

def data_loader(train, test, test_y):
    '''Load csv data.

    Args:
        - train: path to training set
        - test: path to testing set excludint target feature
        - test_y: path to target feature of testing set

    Returns:
        - X_tr: (Numpy ndarray) training set excluding target feature
        - y_tr: (Numpy array) target feature of training set
        - no: (Numpy array) numbers of testing instances
        - X_ts: (Numpy ndarray) testing set excludint target feature
        - y_ts: (Numpy array) target feature of testing set
    '''
    df_tr = pd.read_csv(train)
    X_tr = df_tr.drop(['No', 'Target'], axis=1).to_numpy()
    y_tr = df_tr['Target'].to_numpy()

    df_ts = pd.read_csv(test)
    no = df_ts['No'].to_numpy()
    X_ts = df_ts.drop('No', axis=1).to_numpy()

    if test_y != None:
        y_ts = pd.read_csv(test_y).drop('No', axis=1).to_numpy()
    else:
        y_ts = None

    return X_tr, y_tr, no, X_ts, y_ts
