import constants
import analysis.models.model_constants as model_constants

# SVM
SVM_OVER_SAMPLER_PARAMS=[
# 症状&舌脉象
# TYPE_F1_PRE
    {'C': 113722158.65263128 },
# TYPE_AUC
    {'C': 7.884479277826943},
# 症状
# TYPE_F1_PRE
    {'C': 10.17169488377493},
# TYPE_AUC
    {'C': 184.12149138904286}
    ]

SVM_SMOTE_PARAMS = [
    {'C': 20.427613294134296},
    {'C': 1.3122654687210835},
    {'C': 355579489.48038155},
    {'C': 431.52296341648497}
]
SVM_SMOTE_BORDERLINE1_PARAMS = [
    {'C': 5415708639.990359},
    {'C': 9976509030.539387},
    {'C': 44.180363026819236},
    {'C': 101.32847865130577}
]
SVM_SMOTE_D_PARAMS = [
    {'C': 8.736810045184805 },
    {'C': 1.6040637815548981},
    {'C': 45.690480259326606},
    {'C': 33.88633207494328}
]

# RF
RF_OVER_SAMPLER_PARAMS = [
    # 症状&舌脉象
    # TYPE_F1_PRE
    {'max_depth': 28, 'n_estimators': 20, 'criterion': 'entropy'},
    # TYPE_AUC
    {'max_depth': 21, 'n_estimators': 13, 'criterion': 'gini'},
    # 症状
    {'max_depth': 26, 'n_estimators': 8, 'criterion': 'gini'},
    {'max_depth': 21, 'n_estimators': 8, 'criterion': 'gini'}
]
RF_SMOTE_PARAMS = [
    {'max_depth': 20, 'n_estimators': 8, 'criterion': 'gini'},
    {'max_depth': 21, 'n_estimators': 19, 'criterion': 'entropy'},
    {'max_depth': 22, 'n_estimators': 16, 'criterion': 'entropy'},
    {'max_depth': 18, 'n_estimators': 18, 'criterion': 'entropy'}
]
RF_SMOTE_BORDERLINE1_PARAMS = [
    {'max_depth': 27, 'n_estimators': 9, 'criterion': 'gini'},
    {'max_depth': 30, 'n_estimators': 10, 'criterion': 'gini'},
    {'max_depth': 32, 'n_estimators': 10, 'criterion': 'gini'},
    {'max_depth': 28, 'n_estimators': 15, 'criterion': 'gini'}
]
RF_SMOTE_D_PARAMS = [
    {'max_depth': 23, 'n_estimators': 14, 'criterion': 'entropy'},
    {'max_depth': 32, 'n_estimators': 17, 'criterion': 'entropy'},
    {'max_depth': 13, 'n_estimators': 6, 'criterion': 'gini'},
    {'max_depth': 30, 'n_estimators': 17, 'criterion': 'entropy'}
]
# DT
DT_OVER_SAMPLER_PARAMS = [
# 症状&舌脉象
{'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 2},
{'criterion': 'gini', 'max_depth': 32, 'min_samples_split': 8},
# 症状
{'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 4},
{'criterion': 'entropy', 'max_depth': 29, 'min_samples_split': 8}
]
DT_SMOTE_PARAMS = [
# 症状&舌脉象
{'criterion': 'gini', 'max_depth': 22, 'min_samples_split': 2},
{'criterion': 'gini', 'max_depth': 16, 'min_samples_split': 4},
# 症状
{'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 16},
{'criterion': 'entropy', 'max_depth': 28, 'min_samples_split': 2}
]
DT_SMOTE_BORDERLINE1_PARAMS = [
# 症状&舌脉象
{'criterion': 'gini', 'max_depth': 24, 'min_samples_split': 4},
{'criterion': 'gini', 'max_depth': 28, 'min_samples_split': 2},
# 症状
{'criterion': 'gini', 'max_depth': 19, 'min_samples_split': 6},
{'criterion': 'gini', 'max_depth': 19, 'min_samples_split': 2}
]
DT_SMOTE_D_PARAMS = [
# 症状&舌脉象
{'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 6},
{'criterion': 'entropy', 'max_depth': 21, 'min_samples_split': 12},
# 症状
{'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 8},
{'criterion': 'entropy', 'max_depth': 26, 'min_samples_split': 20}
]
# XGBoost
XGBOOST_OVER_SAMPLER_PARAMS = [
# 症状&舌脉象
# {'booster': 'gbtree', 'lambda': 0.19704487916214167, 'alpha': 2.093847357224429e-08,
#  'subsample': 0.8759815293981179, 'colsample_bytree': 0.552977595566911, 'max_depth': 9,
#  'min_child_weight': 2, 'eta': 0.4684859156660726, 'gamma': 0.030018973492410807, 'grow_policy': 'depthwise'},
{'booster': 'gbtree', 'lambda': 0.014744274727130389, 'alpha': 0.00036686945469780255,
 'subsample': 0.5555013992195952, 'colsample_bytree': 0.5592811234836909, 'max_depth': 7,
 'min_child_weight': 2, 'eta': 0.39664487930190834, 'gamma': 1.3995485710205678e-08, 'grow_policy': 'depthwise'},
{'booster': 'dart', 'lambda': 5.712025830112571e-08, 'alpha': 0.021184676003902867,
    'subsample': 0.9015855999686154, 'colsample_bytree': 0.24374656182912582, 'max_depth': 7,
    'min_child_weight': 2, 'eta': 0.9952157917931137, 'gamma': 0.0014989524392155474,
    'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'tree',
    'rate_drop': 8.68809834753798e-06, 'skip_drop': 1.3620523416984143e-07},
# 症状
{'booster': 'dart', 'lambda': 4.778860809484166e-07, 'alpha': 8.389108509496219e-05,
 'subsample': 0.6046044289049408, 'colsample_bytree': 0.9144086733688235, 'max_depth': 7,
 'min_child_weight': 2, 'eta': 0.13930597019662755, 'gamma': 5.414548741045133e-08,
 'grow_policy': 'lossguide', 'sample_type': 'weighted', 'normalize_type': 'forest',
 'rate_drop': 7.021041759657993e-06, 'skip_drop': 4.440635632214902e-06},
{'booster': 'dart', 'lambda': 6.519895853480133e-07, 'alpha': 5.457972085962283e-08,
 'subsample': 0.9152221264926876, 'colsample_bytree': 0.8253039545422858, 'max_depth': 7,
 'min_child_weight': 2, 'eta': 0.9993629934784735, 'gamma': 1.8090657427778115e-05,
 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'tree',
 'rate_drop': 0.002510594425119743, 'skip_drop': 0.11326398161999948}
]
XGBOOST_SMOTE_PARAMS = [
# 症状&舌脉象
{'booster': 'gbtree', 'lambda': 3.369992849615612e-08, 'alpha': 1.4450195138500373e-06,
     'subsample': 0.8901580563888312, 'colsample_bytree': 0.7594984417638513, 'max_depth': 9, 'min_child_weight': 2,
     'eta': 0.4538840438869842, 'gamma': 4.283265811681698e-06, 'grow_policy': 'depthwise'},
{'booster': 'gblinear', 'lambda': 1.2981643589951566e-08, 'alpha': 0.025060444875515935,
             'subsample': 0.8144462739426666, 'colsample_bytree': 0.502435521745772},
# 症状
{'booster': 'dart', 'lambda': 0.4077539301647922, 'alpha': 4.796179556114411e-06, 'subsample': 0.7769661263545271,
 'colsample_bytree': 0.6302475785258441, 'max_depth': 3, 'min_child_weight': 3, 'eta': 0.5535020431204466,
 'gamma': 0.010994866746320676, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'tree',
 'rate_drop': 1.6161147038382376e-07, 'skip_drop': 0.0007520447790170259},
{'booster': 'gblinear', 'lambda': 8.630636412065469e-07, 'alpha': 0.004828692424257938,
 'subsample': 0.4325802285231406, 'colsample_bytree': 0.5459444129467454}
]
XGBOOST_SMOTE_BORDERLINE1_PARAMS = [
# 症状&舌脉象
{'booster': 'gbtree', 'lambda': 0.914538072483275, 'alpha': 1.8652684277036476e-07,
 'subsample': 0.8999976011446285, 'colsample_bytree': 0.7271143131930315, 'max_depth': 9,
 'min_child_weight': 2, 'eta': 0.17797764080875952, 'gamma': 0.009323091055556482, 'grow_policy': 'depthwise'},
{'booster': 'gbtree', 'lambda': 8.832863215352322e-05, 'alpha': 0.0029490031472444896,
    'subsample': 0.9790727032229547, 'colsample_bytree': 0.8129539552905347, 'max_depth': 9,
    'min_child_weight': 6, 'eta': 0.5546283949833964, 'gamma': 0.00018408651570821504, 'grow_policy': 'lossguide'},
# 症状
{'booster': 'dart', 'lambda': 8.309754526130136e-06, 'alpha': 6.859650367730424e-05,
 'subsample': 0.9758394846058291, 'colsample_bytree': 0.9991354255033817, 'max_depth': 5,
 'min_child_weight': 4, 'eta': 2.2290549063015877e-05, 'gamma': 6.7933164880917255e-06,
 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 0.20272249411422238,
 'skip_drop': 0.6318460743538985},
{'booster': 'gblinear', 'lambda': 1.731577211363274e-06, 'alpha': 0.00410323261392637,
 'subsample': 0.7901107590545371, 'colsample_bytree': 0.32784786347134776}
]
XGBOOST_SMOTE_D_PARAMS = [
# 症状&舌脉象
{'booster': 'gbtree', 'lambda': 0.016162497239068596, 'alpha': 0.005223776157313879,
              'subsample': 0.7118845749597382, 'colsample_bytree': 0.45835384576306376, 'max_depth': 7,
              'min_child_weight': 2, 'eta': 0.30301527084809826, 'gamma': 0.4401296769899615,
              'grow_policy': 'lossguide'},
{'booster': 'gbtree', 'lambda': 6.3613063490166845e-06, 'alpha': 0.0012803748439886838,
    'subsample': 0.9475524177321181, 'colsample_bytree': 0.983958990723794, 'max_depth': 7,
    'min_child_weight': 5, 'eta': 0.9928978687905468, 'gamma': 0.0008203572059634942, 'grow_policy': 'lossguide'},
# 症状
{'booster': 'gblinear', 'lambda': 1.6399216017031773e-05, 'alpha': 0.006839915505343473,
 'subsample': 0.4723916286684567, 'colsample_bytree': 0.981222068358717},
{'booster': 'gblinear', 'lambda': 1.2252926899255686e-06, 'alpha': 0.0028114505001042064,
 'subsample': 0.25868410396282515, 'colsample_bytree': 0.4896502933874423}
]

def get_svm_params(path,type):
    if type == model_constants.TYPE_F1_PRE:
        # 症状&舌脉象
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return SVM_OVER_SAMPLER_PARAMS[0]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return SVM_SMOTE_PARAMS[0]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return SVM_SMOTE_BORDERLINE1_PARAMS[0]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return SVM_SMOTE_D_PARAMS[0]
        # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return SVM_OVER_SAMPLER_PARAMS[2]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return SVM_SMOTE_PARAMS[2]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return SVM_SMOTE_BORDERLINE1_PARAMS[2]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return SVM_SMOTE_D_PARAMS[2]
    else:
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return SVM_OVER_SAMPLER_PARAMS[1]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return SVM_SMOTE_PARAMS[1]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return SVM_SMOTE_BORDERLINE1_PARAMS[1]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return SVM_SMOTE_D_PARAMS[1]
            # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return SVM_OVER_SAMPLER_PARAMS[3]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return SVM_SMOTE_PARAMS[3]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return SVM_SMOTE_BORDERLINE1_PARAMS[3]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return SVM_SMOTE_D_PARAMS[3]

def get_rf_params(path,type):
    ''' 随机森林 '''
    if type == model_constants.TYPE_F1_PRE:
        # 症状&舌脉象
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return RF_OVER_SAMPLER_PARAMS[0]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return RF_SMOTE_PARAMS[0]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return RF_SMOTE_BORDERLINE1_PARAMS[0]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return RF_SMOTE_D_PARAMS[0]
        # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return RF_OVER_SAMPLER_PARAMS[2]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return RF_SMOTE_PARAMS[2]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return RF_SMOTE_BORDERLINE1_PARAMS[2]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return RF_SMOTE_D_PARAMS[2]
    else:
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return RF_OVER_SAMPLER_PARAMS[1]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return RF_SMOTE_PARAMS[1]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return RF_SMOTE_BORDERLINE1_PARAMS[1]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return RF_SMOTE_D_PARAMS[1]
            # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return RF_OVER_SAMPLER_PARAMS[3]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return RF_SMOTE_PARAMS[3]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return RF_SMOTE_BORDERLINE1_PARAMS[3]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return RF_SMOTE_D_PARAMS[3]

def get_dt_params(path,type):
    ''' 决策树 '''
    if type == model_constants.TYPE_F1_PRE:
        # 症状&舌脉象
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return DT_OVER_SAMPLER_PARAMS[0]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return DT_SMOTE_PARAMS[0]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return DT_SMOTE_BORDERLINE1_PARAMS[0]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return DT_SMOTE_D_PARAMS[0]
        # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return DT_OVER_SAMPLER_PARAMS[2]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return DT_SMOTE_PARAMS[2]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return DT_SMOTE_BORDERLINE1_PARAMS[2]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return DT_SMOTE_D_PARAMS[2]
    else:
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return DT_OVER_SAMPLER_PARAMS[1]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return DT_SMOTE_PARAMS[1]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return DT_SMOTE_BORDERLINE1_PARAMS[1]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return DT_SMOTE_D_PARAMS[1]
            # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return DT_OVER_SAMPLER_PARAMS[3]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return DT_SMOTE_PARAMS[3]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return DT_SMOTE_BORDERLINE1_PARAMS[3]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return DT_SMOTE_D_PARAMS[3]

def get_xgboost_params(path,type):
    if type == model_constants.TYPE_F1_PRE:
        # 症状&舌脉象
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return XGBOOST_OVER_SAMPLER_PARAMS[0]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_PARAMS[0]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_BORDERLINE1_PARAMS[0]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_D_PARAMS[0]
        # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return XGBOOST_OVER_SAMPLER_PARAMS[2]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_PARAMS[2]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_BORDERLINE1_PARAMS[2]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_D_PARAMS[2]
    else:
        if path == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
            return XGBOOST_OVER_SAMPLER_PARAMS[1]
        elif path == constants.TUE_SMOTE_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_PARAMS[1]
        elif path == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_BORDERLINE1_PARAMS[1]
        elif path == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_D_PARAMS[1]
            # 症状
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:
            return XGBOOST_OVER_SAMPLER_PARAMS[3]
        elif path == constants.SMOTE_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_PARAMS[3]
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_BORDERLINE1_PARAMS[3]
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:
            return XGBOOST_SMOTE_D_PARAMS[3]
