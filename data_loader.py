import numpy as np
import pandas as pd

def DataLoader(train, test):
    '''Load csv data.

    Args:
        - train: path to the training set
        - test: path to the testing set

    Returns:
        - X_tr: (Numpy ndarray) training set excluding the target feature
        - y_tr: (Numpy array) the target feature of the training set
        - no: (Numpy array) keys to testing instances
        - X_ts: (Numpy ndarray) testing set
    '''
    df_tr = pd.read_csv(train)
    X_tr = df_tr.drop(['No', 'Target'], axis=1).to_numpy()
    y_tr = df_tr['Target'].to_numpy()

    df_ts = pd.read_csv(test)
    no = df_ts['No'].to_numpy()
    X_ts = df_ts.drop('No', axis=1).to_numpy()

    return X_tr, y_tr, no, X_ts
