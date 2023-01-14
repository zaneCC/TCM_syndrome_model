import numpy as np
import pandas as pd
import sys

from xgboost import plot_importance

sys.path.append(r'/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants,config
import analysis.models.mlearn_models as models
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import config
import analysis.models.mlearn_models as models
import utils.data_utils as data_utils

''' 症状-证
汇总表-未做特征选择:    MERGE_CSV_PATH
汇总表-特征选择后:     SELECTION_MERGE_CSV_PATH
人工选择特征:         SYMP_MAIN_ACC_DIAGNOSIS_PATH

SMOTE:              ANALYSIS_SMOTE_MERGE_CSV_PATH
SMOTE_Borderline1:  ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH
SMOTE_D:            ANALYSIS_SMOTE_D_MERGE_CSV_PATH
SMOTE_BORDERLINE_D: ANALYSIS_SMOTE_Borderline_D_CSV_PATH
随机过采样:           ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH
'''
''' 舌象-证
汇总表-舌象:          MERGE_CSV_DIA_TONGUE_PATH
随机过采样:           TUE_RANDOM_OVER_SAMPLER_CSV_PATH
SMOTE:              TUE_SMOTE_MERGE_CSV_PATH
SMOTE Borderline1:  TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH
SMOTE_D:            TUE_SMOTE_D_MERGE_CSV_PATH
SMOTE_BORDERLINE_D: TUE_SMOTE_Borderline_D_CSV_PATH
'''
PATH = constants.TUE_SMOTE_MERGE_CSV_PATH


class VarAnalysis():

    def __init__(self):
        # 设置字体
        # mpl.rcParams['font.sans-serif'] = [u'simHei']
        # mpl.rcParams['axes.unicode_minus'] = False

        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        # 读数据
        self.read_data()
        
    def read_data(self,path=PATH):
        config.PATH = path # 重要变量分析，不同采样获取其最佳模型参数，需修改当前数据集
        self.df = pd.read_csv(path)
        cols = self.df.columns.values.tolist()
        # cols.remove('INHOSPTIAL_ID')
        cols.remove('ZHENGHOU1')

        X = self.df[cols] 
        y = self.df['ZHENGHOU1']
        y = data_utils.unify_lable(y)
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
        self.path = path

    # 决策树
    def get_DTC_important_var(self,top=20, k=50):
        importances = np.zeros(len(self.X_test.columns))
        s = 0
        for i in range(k):
            # clf = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split)
            clf = models.decisionTree()
            clf.fit(self.X_train,self.y_train)
            test_result = clf.predict(self.X_test)
            importances += clf.feature_importances_
            # print(clf.feature_importances_.sum())

        importances = importances / k
        # print(importances)
        # 模型评估
        print('决策树准确度：')
        print(metrics.classification_report(self.y_test,test_result))

        features = list(self.X_test.columns)
        indices = np.argsort(importances)[::-1]
        top_indices = indices[0:top]
        print(top_indices)
        top_features = len(top_indices)
        return top_features, importances, top_indices, features

    # 随机森林
    def get_RFC_important_var(self, top=20, k=50):

        importances = np.zeros(len(self.X_test.columns))

        for i in range(k):
            # rfc = RandomForestClassifier(criterion='entropy',max_depth=7,max_features=0.6,min_samples_split=8,n_estimators=20)
            rfc = models.randomForestClassifier()
            rfc.fit(self.X_train,self.y_train)
            # test_result = rfc.predict(self.X_test)
            f_importance = rfc.feature_importances_

            # xgboost = models.xgboost()
            # xgboost.fit(self.X_train,self.y_train)
            # f_importance = xgboost.feature_importances_
            importances += f_importance
            # 模型评估
            # print('随机森林准确度：')
            # print(metrics.classification_report(self.y_test,test_result))

        importances = (importances / k) * 1000
        print(importances)

        features = list(self.X_test.columns)
        indices = np.argsort(importances)[::-1]
        top_indices = indices[0:top]
        print(top_indices)
        top_features = len(top_indices)
        return top_features, importances, top_indices, features
        
    def get_path(self,title,type=0):
        p_path = ''
        t_path = ''
        if type == 0:
            title = '决策树-' + title
        elif type == 1:
            title = '随机森林-' + title

        if config.IS_SHE:
            os_path = constants.OS_PATH + '/output/特征贡献度/舌象'
        else:
            os_path = constants.OS_PATH + '/output/特征贡献度'

        if self.path == constants.ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH or self.path == constants.TUE_ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH: # 随机过采样
            p_path = os_path + '/随机过采样/'+ title + '.png'
            t_path = os_path + '/随机过采样/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_MERGE_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_MERGE_CSV_PATH: # SMOTE
            p_path = os_path + '/SMOTE/'+ title +'.png'
            t_path = os_path + '/SMOTE/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH: # SMOTE Borderline1
            p_path = os_path + '/SMOTE_BORDERLINE/'+ title +'.png'
            t_path = os_path + '/SMOTE_BORDERLINE/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_D_MERGE_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_D_MERGE_CSV_PATH: # SMOTE_D
            p_path = os_path + '/SMOTE_D/'+ title +'.png'
            t_path = os_path + '/SMOTE_D/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_Borderline_D_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_Borderline_D_CSV_PATH: # SMOTE_BORDERLINE_D
            p_path = os_path + '/SMOTE_BORDERLINE_D/'+ title +'.png'
            t_path = os_path + '/SMOTE_BORDERLINE_D/'+ title + '.txt'
        return p_path, t_path

    def show_important_var(self,top_features,importances,top_indices,features, key, type,title='特征贡献度'):
        # plt.figure()
        ax = plt.subplot(1,1,key+1)
        # ax.set_title(title)

        # plt.title(title)
        rects = ax.bar(range(top_features), importances[top_indices], color=[(32/255,119/255,180/255)], align="center")
        plt.xticks(range(top_features), [features[i] for i in top_indices], rotation='90')
        plt.xlim([-1, top_features])

        plt.ylim(0, 170)

        # 给每个柱子上面添加标注
        for b in rects:  # 遍历每个柱子
            height = int(b.get_height())
            ax.annotate('{}'.format(height),
                        # xy控制的是，标注哪个点，x=x坐标+width/2, y=height，即柱子上平面的中间
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3),  # 文本放置的位置，如果有textcoords，则表示是针对xy位置的偏移，否则是图中的固定位置
                        textcoords="offset points",  # 两个选项 'offset pixels'，'offset pixels'
                        va='bottom', ha='center'  # 代表verticalalignment 和horizontalalignment，控制水平对齐和垂直对齐。
                        )


        # plt.tick_params(axis='x', which='major', labelsize=8)
        # plt.tight_layout()

        p_path, t_path = self.get_path(title,type)
        # 保存图片
        plt.savefig(p_path)
        # plt.show()

        # 写入文件-各个特征的重要度
        try:
            f = open(t_path, 'w')
            for i in top_indices:
                s = "{0} - {1:.3f}".format(features[i], importances[i]/5)
                f.write(s+'\n')
        except FileNotFoundError:
            os.mknod(t_path)

    def show(self):
        plt.tight_layout()
        plt.show()
        
    # 所有采样方法的平均重要度特征排名-前top位-写入文件
    def get_top_important_var_by_file(self,regs,value,top=10):
        # 初始化计数字典
        dic = {}
        for i in self.X_test.columns:
            dic[i] = 0
        if value == '决策树':
            type = 0
        elif value == '随机森林':
            type = 1
        # 迭代所有算法采样后特征重要性文件
        for key,(title, path) in enumerate(regs):
            self.read_data(path)
            p_path, t_path = self.get_path(title, type)

            with open(t_path) as f:
                for line1 in f:
                    line1 = line1.strip()
                    strlist = line1.split(' - ')
                    strlist[1] = float(strlist[1])
                    dic[strlist[0]] = dic[strlist[0]] + strlist[1]
        # 获得 top 重要特征
        sorted_dic = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[0:top]

        keys = []
        values = []
        for i in sorted_dic:
            keys.append(i[0])
            values.append(i[1])

        # 展示
        plt.figure()
        plt.title('所有采样-'+value+'-特征重要性top'+str(top))
        plt.bar(range(len(keys)), values, color="g", align="center")
        plt.xticks(range(len(keys)), keys, rotation='90')
        # plt.xlim([-1, keys])

        # 写入文件
        path = constants.OS_PATH + '/output/特征贡献度/所有采样-'+value+'-特征重要性top'+str(top)+'.txt'
        try:
            f = open(path, 'w')
            r = 0
            for s in sorted_dic[0:10]:
                r += s[1]
                f.write(str(s)+'\n')
            f.write("总贡献度："+str(r)+'\n')
        except FileNotFoundError:
            os.mknod(path)


    def show_xgboost_var_importance(self):
        path = config.PATH
        df = pd.read_csv(path)
        cols = df.columns.values.tolist()
        cols.remove('ZHENGHOU1')

        X = df[cols]
        y = df['ZHENGHOU1']

        _x = X.to_numpy()
        y = data_utils.unify_lable(y)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

        xgboost = models.xgboost()
        model = xgboost.fit(X_train,y_train)

        f_importance = model.feature_importances_
        f_importance = (f_importance * 1000).astype(int)

        # fig, ax = plt.subplots(figsize=(15, 10))

        features = list(X.columns)
        indices = np.argsort(f_importance)[::-1]
        top = 20
        top_indices = indices[0:top]
        print(top_indices)
        top_features = len(top_indices)

        important_features = [features[i] for i in top_indices]
        titles = ['随机过采样', 'SMOTE', 'SMOTE-BORDERLINE', 'SMOTE-D']
        # plot_importance(dict(zip(important_features, f_importance)), title='SMOTE')

        # titles = ['随机过采样', 'SMOTE', 'SMOTE-BORDERLINE', 'SMOTE-D']
        plot_importance(model,title='SMOTE-D')
        plt.show()

    def show_rf_var_importance(self):
        path = config.PATH
        df = pd.read_csv(path)
        cols = df.columns.values.tolist()
        cols.remove('ZHENGHOU1')

        X = df[cols]
        y = df['ZHENGHOU1']

        _x = X.to_numpy()
        y = data_utils.unify_lable(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        rf = models.randomForestClassifier()
        model = rf.fit(X_train, y_train)
        f_importance = model.feature_importances_
        f_importance = (f_importance * 1000).astype(int)

        # fig, ax = plt.subplots(figsize=(15, 10))

        features = list(X.columns)
        indices = np.argsort(f_importance)[::-1]
        top = 20
        top_indices = indices[0:top]
        print(top_indices)
        top_features = len(top_indices)

        important_features = [features[i] for i in top_indices]
        titles = ['随机过采样','SMOTE','SMOTE-BORDERLINE','SMOTE-D']
        plot_importance(dict(zip(important_features,f_importance)),title='SMOTE-D')

        plt.show()


if __name__ == '__main__':
    va = VarAnalysis()
    
    # 决策树    随机森林
    value = '随机森林'
    if config.IS_SHE:
        regs = [
            # ('随机过采样', constants.TUE_ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH),
            ('SMOTE', constants.TUE_ANALYSIS_SMOTE_MERGE_CSV_PATH),
            # ('SMOTE', constants.TUE_ANALYSIS_SMOTE_MERGE_CSV_PATH_EN),

            # ('SMOTE_Borderline', constants.TUE_ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH),
            # ('SMOTE_D', constants.TUE_ANALYSIS_SMOTE_D_MERGE_CSV_PATH)
            # ('SMOTE_BORDERLINE_D', constants.TUE_ANALYSIS_SMOTE_Borderline_D_CSV_PATH),
        ]
    else:
        regs = [
                ('随机过采样',constants.ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH),
                ('SMOTE',constants.ANALYSIS_SMOTE_MERGE_CSV_PATH),
                ('SMOTE_Borderline',constants.ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH),
                ('SMOTE_D',constants.ANALYSIS_SMOTE_D_MERGE_CSV_PATH)
                # ('SMOTE_BORDERLINE_D',constants.ANALYSIS_SMOTE_Borderline_D_CSV_PATH),
                ]

    for key,(name, path) in enumerate(regs):
        va.read_data(path)
        # 生成特征贡献度文件
        # 决策树
        if value == '决策树':
            top_features, importances, top_indices, features = va.get_DTC_important_var()
            va.show_important_var(top_features, importances, top_indices, features, key, type=0,title=name)
        elif value == '随机森林':
            top_features, importances, top_indices, features = va.get_RFC_important_var()
            va.show_important_var(top_features, importances, top_indices, features, key, type=1,title=name)

    va.show()
    # va.show_xgboost_var_importance()
    # va.show_rf_var_importance()

    # 获得所有样本重要特征
    # va.get_top_important_var_by_file(regs=regs,value=value)
    # va.show()

    # top_features, importances, top_indices, features = va.get_RFC_important_var()
    # va.show_important_var(top_features, importances, top_indices, features, title='特征贡献度-随机森林-随机过采样')