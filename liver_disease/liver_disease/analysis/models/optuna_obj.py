import pandas as pd
import numpy as np
import os
import sys

from sklearn.metrics import f1_score, precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

sys.path.append(r'/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/')
import constants
import analysis.models.mlearn_models as models
import utils.data_utils as data_utils
import optuna
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb

def split():
    path = config.PATH
    df = pd.read_csv(path)
    cols = df.columns.values.tolist()
    # cols.remove('Unnamed: 0')
    # cols.remove('INHOSPTIAL_ID')
    cols.remove('ZHENGHOU1')

    X = df[cols].to_numpy()
    y = df['ZHENGHOU1'].to_numpy()
    y = data_utils.unify_lable(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    y_train = y_train
    y_test = y_test
    return X_train, y_train, X_test, y_test
# {'booster': 'gbtree', 'lambda': 3.369992849615612e-08, 'alpha': 1.4450195138500373e-06,
#      'subsample': 0.8901580563888312, 'colsample_bytree': 0.7594984417638513, 'max_depth': 9, 'min_child_weight': 2,
#      'eta': 0.4538840438869842, 'gamma': 4.283265811681698e-06, 'grow_policy': 'depthwise'}
def xgboost(trial):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # "booster": trial.suggest_categorical("booster", ["gbtree"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    return xgb.XGBClassifier(**param,use_label_encoder=False)


CURRENT_MODEL = models.XGBOOST
F1 = 'f1'
AUC = 'auc'
CURRENT_TAG = F1
def objective(trial):

    X_train, y_train, X_test, y_test = split()

    if CURRENT_MODEL == models.Logisitic_RegressionCV: # Logisitic RegressionCV
        # lrg_penalty = trial.suggest_categorical('penalty',['l1','l2'])
        # if lrg_penalty == 'l1':
        #     lrg_solver = 'liblinear'
        # else:
        #     lrg_solver = trial.suggest_categorical('solver',['liblinear','lbfgs','newton-cg','sag'])
        #
        # classifier_obj = LogisticRegressionCV( solver=lrg_solver, penalty=lrg_penalty)
        pass # 不需要调参
    elif CURRENT_MODEL == models.SVC: # SVC
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    elif CURRENT_MODEL == models.Random_Forest: # Random Forest
        rf_max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        rf_n_estimators = trial.suggest_int("n_estimators", 5, 20, log=True)
        rf_criterion = trial.suggest_categorical("criterion", ["entropy", "gini"])
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators, criterion=rf_criterion
        )

    elif CURRENT_MODEL == models.Decision_Tree: # Decision Tree
        dt_criterion = trial.suggest_categorical('criterion',["entropy", "gini"])
        dt_max_depth = trial.suggest_int('max_depth',2,32,log=True)
        dt_min_samples_split = trial.suggest_int('min_samples_split',2,20,step=2)
        classifier_obj = DecisionTreeClassifier(criterion=dt_criterion ,max_depth=dt_max_depth, min_samples_split = dt_min_samples_split)

    elif CURRENT_MODEL == models.XGBOOST: # XGBoost
        classifier_obj = xgboost(trial)

    if CURRENT_TAG == F1:
        classifier_obj.fit(X_train, y_train)
        pre = classifier_obj.predict(X_test)
        f1 = f1_score(y_test, pre, average=None)
        precision = precision_score(y_test, pre, average=None)
        return f1[0],f1[1],precision[0],precision[1]
    else:
        if CURRENT_MODEL == models.Random_Forest or CURRENT_MODEL == models.Decision_Tree or CURRENT_MODEL == models.XGBOOST:  # 这两个分类器没有 decision_function 方法
            y_pred_proba = classifier_obj.fit(X_train, y_train).predict_proba(X_test)
            fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)  # 计算真正率和假正率
        else:
            y_pred_proba = classifier_obj.fit(X_train, y_train).decision_function(X_test)
            fpr, tpr, threshold = roc_curve(y_test, y_pred_proba, pos_label=1)  # 计算真正率和假正率

        roc_auc = auc(fpr, tpr)
        return roc_auc

if __name__ == "__main__":
    # path = constants.TUE_SMOTE_MERGE_CSV_PATH
    # df = pd.read_csv(path)
    # cols = df.columns.values.tolist()
    # # cols.remove('Unnamed: 0')
    # # cols.remove('INHOSPTIAL_ID')
    # cols.remove('ZHENGHOU1')
    #
    # X = df[cols].to_numpy()
    # y = df['ZHENGHOU1'].to_numpy()
    # data = (X,y)

    # obj = Obj(data)

    if CURRENT_TAG == AUC:
        study = optuna.create_study(directions=["maximize"])
    else:
        study = optuna.create_study(directions=["maximize","maximize","maximize","maximize"])

    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=1000)
    print(study.best_trials)