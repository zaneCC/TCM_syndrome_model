OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'

import pandas as pd
import numpy as np
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold # 通过方差提取特征
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# 汇总表
# PATH = constants.MERGE_CSV_PATH
# 汇总表-SMOTE过采样后
# PATH = constants.SMOTE_MERGE_CSV_PATH
# SMOTE Borderline1 采样
PATH = constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH

TO_PATH = constants.SELECTION_MERGE_CSV_PATH


class Selection_v2021428():

    def __init__(self):
        df = pd.read_csv(PATH)
        self.cols = df.columns.values.tolist()
        
        # print(df['Unnamed: 0'])
        # cols.remove('Unnamed: 0')
        # self.cols.remove('INHOSPTIAL_ID')
        self.cols.remove('ZHENGHOU1')
        self.X = df[self.cols]
        self.y = df['ZHENGHOU1']
        self.df = df


    def split(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        return X_train,X_test,y_train,y_test

    # lasso 特征选择
    def lasso(self):
        X_train,X_test,y_train,y_test = self.split(self.X, self.y)

        model = Lasso(alpha=0.05, fit_intercept=False)
        model.fit(X_train, y_train)

        print(model.coef_.ravel())

        return model

    # 选择特征
    def selection(self, model):
        count = 0
        for index, row in self.df.iteritems():
            if index == 'INHOSPTIAL_ID' or index == 'ZHENGHOU1':
                # count += 1
                continue
            if model.coef_.ravel()[count] > 0.00001 or model.coef_.ravel()[count] < -0.00001:
                count += 1
                print(index, count)
                continue
            self.df.drop(index,axis=1,inplace=True)
            count += 1

    # 使用逻辑回归+L1正则选取特征
    def l1Selection(self):
        # C=0.1,
        # 腹胀,双下肢水肿,乏力,黑便,腹痛,身目黄染,恶心,腹泻,呕吐,口干
        # selector = SelectFromModel(LogisticRegression(penalty="l1", C=0.8, solver='saga')).fit(self.X, self.y)
        selector = SelectFromModel(LinearSVC(penalty="l1", C=0.2, dual=False)).fit(self.X, self.y)

        r = selector.get_support()

        # new_df = pd.DataFrame({'INHOSPTIAL_ID':self.df['INHOSPTIAL_ID'],'ZHENGHOU1':self.df['ZHENGHOU1']})
        new_df = pd.DataFrame({'ZHENGHOU1':self.df['ZHENGHOU1']})
        for key,value in enumerate(r):
            if value:
                new_df[self.cols[key]] = self.df[self.cols[key]]
                print(self.cols[key])
        return new_df

    def l1_l2Selection(self):
        # 带L1和L2惩罚项的逻辑回归作为基模型的特征选择
        # 参数threshold为权值系数之差的阈值
        X_new = SelectFromModel(LR(threshold=0.01, C=0.5)).fit(self.X, self.y)
        r = X_new.get_support()
        # new_df = pd.DataFrame({'INHOSPTIAL_ID':self.df['INHOSPTIAL_ID'],'ZHENGHOU1':self.df['ZHENGHOU1']})
        new_df = pd.DataFrame({'ZHENGHOU1':self.df['ZHENGHOU1']})
        for key,value in enumerate(r):
            if value:
                new_df[self.cols[key]] = self.df[self.cols[key]]
                print(self.cols[key])
        return new_df

    def KBest(self):
        print('-------------各个症状与诊断结果进行卡方检验')
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        best_feature = SelectKBest(score_func=chi2,k=10)
        # 各个症状与诊断结果进行卡方检验
        fit = best_feature.fit(self.X,self.y)
        # 得出卡方检验分数
        dataset_scores = pd.DataFrame(fit.scores_)
        dataset_cols = pd.DataFrame(self.X.columns)

        featurescores = pd.concat([dataset_cols,dataset_scores],axis=1)
        featurescores.columns=['column','scores']
        # 输出贡献最大的十个症状
        print(featurescores.nlargest(10,'scores'))
        # 画出图形显示
        # featureview=pd.Series(fit.scores_, index=self.X.columns)
        # featureview.plot(kind='barh')
        return featurescores.nlargest(10,'scores')

    def KBestDF(self, fList):
        return self.df[fList]

    def PCA(self):
        pca=PCA(n_components=10)
        newX=pca.fit_transform(self.X)
        X = pca.inverse_transform(newX)
        print(X)

    def featureSelectionByVariance(self):
        print('-------------通过方差提取特征')

        feature_high_variance = VarianceThreshold(threshold=(0.5*(1-0.5)))
        falls=feature_high_variance.fit(self.X)

        dataset_scores1 = pd.DataFrame(falls.variances_)
        dat1 = pd.DataFrame(self.X.columns)

        high_variance = pd.concat([dataset_scores1,dat1],axis=1)
        high_variance.columns=['variance','cols']
        print(high_variance[high_variance['variance']>0.2])


    # 输出 CSV 文件
    def toCSV(self,path,df=None):
        if df is None:
            self.df.to_csv(path,index=False)
        else:
            df.to_csv(path,index=False)

class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        # 权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight,
                                    random_state=random_state, solver=solver, max_iter=max_iter,
                                    multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        # 使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
    def fit(self, X, y, sample_weight=None):
        # 训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)
        cntOfRow, cntOfCol = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                # L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    # 对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        # 在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self


if __name__  == '__main__':
    sele = Selection_v2021428()
    # 特征选择：Lasso
    # model = sele.lasso()
    # sele.selection(model)
    # sele.toCSV(TO_PATH)

    # 逻辑回归+l1正则
    # retDF = sele.l1Selection()
    # sele.toCSV(TO_PATH,retDF)

    # retDF = sele.l1_l2Selection()
    # sele.toCSV(TO_PATH,retDF)

    # 特征选择：卡方检验
    # featurescores = sele.KBest()
    # fList = list(featurescores['column'].values)
    # fList.append('ZHENGHOU1')
    # retDF = sele.KBestDF(fList)
    # sele.toCSV(TO_PATH,retDF)

    # 特征选择：PCA
    sele.PCA()


