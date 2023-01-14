import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import constants
import config


def unify_lable(y):
    _y = []
    for i in y:
        if i == 1:
            _y.append(0)
        else:
            _y.append(1)

    return np.array(_y)

def getFeatures(_path):
    df = pd.read_csv(_path)
    cols = df.columns.values.tolist()
    # cols.remove('Unnamed: 0')
    # cols.remove('INHOSPTIAL_ID')
    cols.remove('ZHENGHOU1')

    X = df[cols]
    return X.columns

def split(_path):
    df = pd.read_csv(_path)
    cols = df.columns.values.tolist()
    # cols.remove('Unnamed: 0')
    # cols.remove('INHOSPTIAL_ID')
    cols.remove('ZHENGHOU1')

    X = df[cols]
    y = df['ZHENGHOU1']

    _x = X.to_numpy()
    y = unify_lable(y)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    return X_train,X_test,y_train,y_test

if __name__ == '__main__':
    csv_path = config.PATH
    split(csv_path)