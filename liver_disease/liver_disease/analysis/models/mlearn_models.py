
# 逻辑回归
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import sys
sys.path.append(r'/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/')
import config
import analysis.models.params as params
import analysis.models.model_constants as model_constants

CURRENT_PATH = config.PATH
CURRENT_TYPE = model_constants.TYPE_F1_PRE


# 逻辑回归
def logisiticRegression():
    lg =LogisticRegressionCV(multi_class="multinomial", solver="newton-cg")
    return lg

# 支持向量机
def SVM():
    param = params.get_svm_params(CURRENT_PATH, CURRENT_TYPE)
    if param is None:
        sv = svm.SVC(probability=True)
    else:
        sv =svm.SVC(**param, probability=True)
    return sv

# TODO 运行前改参
# 决策树
def decisionTree():
    param = params.get_dt_params(CURRENT_PATH, CURRENT_TYPE)
    if param is None:
        dt = DecisionTreeClassifier()
    else:
        dt = DecisionTreeClassifier(**param)
    return dt

# TODO 运行前改参
# 随机森林
def randomForestClassifier():
    param = params.get_rf_params(CURRENT_PATH, CURRENT_TYPE)
    if param is None:
        rfc = RandomForestClassifier()
    else:
        rfc = RandomForestClassifier(**param)
    return rfc

# TODO 运行前改参
# adaboost
def adaboostClassifier():
    return AdaBoostClassifier(algorithm='SAMME.R' ,learning_rate=0.6 ,n_estimators=60)

def xgboost():
    param = params.get_xgboost_params(CURRENT_PATH,CURRENT_TYPE)
    if param is None:
        return xgb.XGBClassifier( use_label_encoder=False)
    else:
        return xgb.XGBClassifier(**param, use_label_encoder=False)


Logisitic_RegressionCV = 'Logisitic RegressionCV'
SVC = 'SVC'
Random_Forest = 'Random Forest'
Decision_Tree = 'Decision Tree'
XGBOOST = 'XGBoost'
MODELS_NAME = [Logisitic_RegressionCV, SVC, Random_Forest,Decision_Tree, XGBOOST]
