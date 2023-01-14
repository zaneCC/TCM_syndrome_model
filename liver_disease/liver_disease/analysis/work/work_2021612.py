OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
from itertools import count
import pandas as pd
import numpy as np
import sys
from typing import Counter, Iterable
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants
import utils.misc as misc
from sklearn.datasets import make_classification 

# 汇总表-未做特征选择
PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
# PATH = constants.SELECTION_MERGE_CSV_PATH

# 其他数据集
# PATH = '/Users/zhouzhan/Documents/to_github/NLP/Python/codes/sklearn/SymptomAnalysis/diabetes_data_upload.csv'

# example of stratified k-fold cross-validation with an imbalanced dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# generate 2 class dataset
# X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)



# X_arr = X.to_numpy()

# 从两类中，分别重采样20个样本（共40个样本）建立一个分类器，建立（5个）分类器后，采用平均权值的方法
from sklearn.ensemble import RandomForestClassifier

class RF():

    def read_data(self):
        df = pd.read_csv(PATH)
        cols = df.columns.values.tolist()

        cols.remove('INHOSPTIAL_ID')
        cols.remove('ZHENGHOU1')
        self.X = df[cols]
        self.y = df['ZHENGHOU1']

    def read_other(self):
        dataset = pd.read_csv(PATH)
        
        dataset['Gender'] = dataset['Gender'].map({'Male':1,'Female':2})
        dataset['class'] = dataset['class'].map({'Positive':1,'Negative':2})
        dataset['Polyuria'] = dataset['Polyuria'].map({'Yes':1,'No':2})
        dataset['Polydipsia'] = dataset['Polydipsia'].map({'Yes':1,'No':2})
        dataset['sudden weight loss'] = dataset['sudden weight loss'].map({'Yes':1,'No':2})
        dataset['weakness'] = dataset['weakness'].map({'Yes':1,'No':2})
        dataset['Polyphagia'] = dataset['Polyphagia'].map({'Yes':1,'No':2})
        dataset['Genital thrush'] = dataset['Genital thrush'].map({'Yes':1,'No':2})
        dataset['visual blurring'] = dataset['visual blurring'].map({'Yes':1,'No':2})
        dataset['Itching'] = dataset['Itching'].map({'Yes':1,'No':2})
        dataset['Irritability'] = dataset['Irritability'].map({'Yes':1,'No':2})
        dataset['delayed healing'] = dataset['delayed healing'].map({'Yes':1,'No':2})
        dataset['partial paresis'] = dataset['partial paresis'].map({'Yes':1,'No':2})
        dataset['muscle stiffness'] = dataset['muscle stiffness'].map({'Yes':1,'No':2})
        dataset['Alopecia'] = dataset['Alopecia'].map({'Yes':1,'No':2})
        dataset['Obesity'] = dataset['Obesity'].map({'Yes':1,'No':2})
        self.X = dataset[['Polydipsia','sudden weight loss','partial paresis','Irritability','Polyphagia','Age','visual blurring']]
        self.y = dataset['class']

    # Generates toy dataset for binary classification with shape x = [5000, 20] 
    def generate_data(self): 
        self.X, self.y = make_classification(n_samples=5000, n_features=20, n_classes=2, weights=[0.95, 0.05]) 
        

    def __init__(self, n_estimators = 100, base_estimator=DecisionTreeClassifier()):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.foreast = []
        
        # self.read_data()
        # self.read_other()
        self.generate_data()

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size = 0.2,random_state=0)
        # print('train:',Counter(self.y_train),'test:',Counter(self.y_test))

    def shuffle_data(self, X, y):
        """ Random shuffle of the samples in X and y """
        # if seed:
        #     np.random.seed(seed)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    '''
    对训练集随机打乱，针对每一类随机重采样num个样本
    return l = {1:[X_train_1,y_train_1],2:[X_train_2,y_train_2]}
    ''' 
    def split(self, num=15):
        types = set(self.y_train)
        # 打乱数据
        # _X, _y = self.shuffle_data(self.X_train.to_numpy(), self.y_train.to_numpy())
        _X, _y = self.shuffle_data(self.X_train, self.y_train)


        # 每个类别重采样相同个数的样本
        _ret_X_1, _ret_y_1, _ret_X_2, _ret_y_2 = [], [], [], []
        for key,value in enumerate(_X):
            if _y[key] == 1 and len(_ret_y_1) < num:
                _ret_y_1.append(1)
                _ret_X_1.append(value)
            elif _y[key] == 2 and len(_ret_y_2) < num :
                _ret_y_2.append(2)
                _ret_X_2.append(value)
            

        l = {1:[_ret_X_1, _ret_y_1],2:[_ret_X_2, _ret_y_2]}
        return l

    def fit(self):
        proba = []
        for i in range(self.n_estimators):
            # 建立一个模型
            self.foreast.append(self.base_estimator)
            l = self.split()
            # 拼接成分类器可用数据
            _X_train, _y_train = [],[]
            for j in l:
                _X_train.extend(l[j][0])
                _y_train.extend(l[j][1])
            _X_train, _y_train = self.shuffle_data(np.array(_X_train), np.array(_y_train))
            # fit
            y_pred_proba = self.foreast[i].fit(_X_train, _y_train).predict_proba(self.X_test)
            proba.append(y_pred_proba)

        # print(proba)
        # 求票数最多的作为最终分类结果
        classes_ = []
        size = len(proba[0])
        

        result = []
        for i in range(self.n_estimators): # 遍历每个分类器的结果
            classes_.append(np.argmax(proba[i], axis=1))
        classes_ = np.array(classes_)
        # print(classes_)

        for j in range(size):
            count_0, count_1 = 0, 0
            for i in range(self.n_estimators):
                if classes_[i][j] == 0:
                    count_0 += 1
                else:
                    count_1 += 1
            if count_0 > count_1:
                result.append(1)
            else:
                result.append(2)
        return result
    
    def report(self, pre):
        print(classification_report(pre,self.y_test))
        print(pre,self.y_test.to_numpy())

    # def roc(self):
    #     y_pred_proba = i.fit(self.X_train, self.y_train).predict_proba(self.X_test)
    #     fpr,tpr,threshold = roc_curve(self.y_test, y_pred_proba[:,1], pos_label=2) # 计算真正率和假正率

if __name__ == '__main__':
    # rf = RF(base_estimator=SVC(kernel='linear',random_state=0, probability=True))
    rf = RF()

    pre = rf.fit()
    # rf.report(pre)
