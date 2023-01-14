import shap
from sklearn.svm import SVC

import models.mlearn_models as models
import utils.data_utils as data_utils
import config
import matplotlib.pyplot as plt
import sklearn.svm as svm

def get_waterfall(path=config.PATH):
    # 解决画图中文乱码问题
    # from pylab import *
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    X_train,X_test,y_train,y_test = data_utils.split(path)

    # xgboost = models.xgboost()
    # model = xgboost.fit(X_train,y_train)
    # explainer = shap.Explainer(model)
    # shap_values = explainer(X_train)
    #
    # # visualize the first prediction's explanation
    # # shap.plots.waterfall(shap_values[0])
    # shap.plots.beeswarm(shap_values)

    # svm = models.SVM()
    # param = {'C': 20.427613294134296}
    # svm = SVC(C=20.427613294134296, probability=True)
    # model = svm.fit(X_train, y_train)
    # svm_explainer = shap.KernelExplainer(svm.predict, X_test)
    # svm_shap_values = svm_explainer.shap_values(X_test)
    # shap.plots.beeswarm(svm_shap_values)

if __name__ == '__main__':
    get_waterfall()