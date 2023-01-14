OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
from itertools import count
import pandas as pd
import numpy as np
import sys
from typing import Counter, Iterable
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
import constants
import utils.misc as misc
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/analysis/imbalance/')
import SMOTE_Bagging
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# 汇总表-未做特征选择
PATH = constants.MERGE_CSV_PATH

def read_data():
    df = pd.read_csv(PATH)
    

    del df['INHOSPTIAL_ID']
    return df

def rf(data, n_estimators=10, base_estimator=DecisionTreeClassifier()):
    foreast = []; proba = []
    for i in range(n_estimators):
        foreast.append(base_estimator)
        _data = data[i]
        y_pred_proba = foreast[i].fit(_data[0:200,: -1], _data[0:200,-1]).predict_proba(_data[200:-1,: -1])
        proba.append(y_pred_proba)

    classes_ = []
    size = len(proba[0])

    result = []
    for i in range(n_estimators): # 遍历每个分类器的结果
        classes_.append(np.argmax(proba[i], axis=1))
    classes_ = np.array(classes_)

    for j in range(size):
        count_0, count_1 = 0, 0
        for i in range(n_estimators):
            if classes_[i][j] == 0:
                count_0 += 1
            else:
                count_1 += 1
        if count_0 > count_1:
            result.append(1)
        else:
            result.append(2)

    print(classification_report(result,_data[200:-1,-1]))

if __name__ == '__main__':
    df = read_data()
    imbalanced_data = df.to_numpy()
    print(imbalanced_data.shape)

    n_estimators = 10
    data = SMOTE_Bagging.SMOTE_Bagging(n_estimators,imbalanced_data)

    rf(data=data, n_estimators = 10)