import pandas as pd
import numpy as np
from typing import Counter
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
import constants
import utils.misc as misc
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/analysis/imbalance/')

# 汇总表-未做特征选择
PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
# PATH = constants.SELECTION_MERGE_CSV_PATH
# SMOTE 采样
# TO_PATH = constants.SMOTE_MERGE_CSV_PATH
# SMOTE Borderline1 采样
# TO_PATH = constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH
TO_PATH = constants.RANDOM_OVER_SAMPLER_CSV_PATH

def read_data():
    df = pd.read_csv(PATH)

    del df['INHOSPTIAL_ID']
    return df

if __name__ == '__main__':
    df = read_data()
    data = df.to_numpy()
    X = data[:,:-1]
    y = data[:,-1]

    print(sorted(Counter(y).items()))
    X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))

    cols = df.columns.values.tolist()
    balanced_data_arr2 = np.column_stack((X_resampled, y_resampled))
    to_df = pd.DataFrame(data=balanced_data_arr2, columns=cols)
    print(to_df)

    # # 导出文件
    to_df.to_csv(TO_PATH, index=False)

# X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
#                             n_redundant=0, n_repeated=0, n_classes=3,
#                             n_clusters_per_class=1,
#                             weights=[0.01, 0.05, 0.94],
#                             class_sep=0.8, random_state=0)
# X_resampled, y_resampled = SMOTE().fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))

# clf_smote = LinearSVC().fit(X_resampled, y_resampled)
# X_resampled, y_resampled = ADASYN().fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))

# clf_adasyn = LinearSVC().fit(X_resampled, y_resampled)