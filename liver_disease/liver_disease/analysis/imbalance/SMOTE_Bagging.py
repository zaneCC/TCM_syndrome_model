from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from base_sampler import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

ROOT_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/analysis/imbalance/'

# SMOTE
# 使用K-近邻方法产生新样本
def make_sample_SMOTE(old_feature_data, diff):
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nns = NearestNeighbors(n_neighbors=6).fit(old_feature_data).kneighbors(old_feature_data, return_distance=False)[:,1:]
    # 随机产生diff个随机数（范围在0-少数类样本数），作为之后产生新样本的选取的样本下标值
    samples_indices = np.random.randint(low=0, high=np.shape(old_feature_data)[0], size=diff)
    # 随机产生diff个随机数作为之后产生新样本的间距值
    steps = np.random.uniform(size=diff)
    cols = np.mod(samples_indices, nns.shape[1])
    reshaped_feature = np.zeros((diff, old_feature_data.shape[1]))
    for i, (col, step) in enumerate(zip(cols, steps)):
        row = samples_indices[i]
        reshaped_feature[i] = old_feature_data[row] - step * (old_feature_data[row] - old_feature_data[nns[row, col]])

    return reshaped_feature

# 处于多数类与少数类边缘的样本
def in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data):
    nn_m = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data)
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nnm_x = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data).kneighbors(old_feature_data,
                                                                                    return_distance=False)[:,1:]
    nn_label = (imbalanced_label_data[nnm_x] != old_label_data).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return np.bitwise_and(n_maj >= (nn_m.n_neighbors - 1) / 2, n_maj < nn_m.n_neighbors - 1)

# SMOTE-Borderline1
# 产生少数类新样本的方法
# def make_sample_SMOTE_Borderline1(imbalanced_data_arr2, diff):
#     # 将数据集分开为少数类数据和多数类数据
#     # minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
#     imbalanced_featured_data = imbalanced_data_arr2[:, : -1]
#     imbalanced_label_data = imbalanced_data_arr2[:, -1]
#     # 原始少数样本的特征集
#     old_feature_data = minor_data_arr2[:, : -1]
#     # 原始少数样本的标签值
#     old_label_data = minor_data_arr2[0][-1]
#     danger_index = in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data)
#     # 少数样本中噪音集合，也就是最终要产生新样本的集合
#     danger_index_data = _safe_indexing(old_feature_data, danger_index)
#     # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
#     nns = NearestNeighbors(n_neighbors=6).fit(old_feature_data).kneighbors(danger_index_data,
#                                                                            return_distance=False)[:, 1:]
#     # 随机产生diff个随机数作为之后产生新样本的选取的样本下标值
#     samples_indices = np.random.randint(low=0, high=np.shape(danger_index_data)[0], size=diff)
#     # 随机产生diff个随机数作为之后产生新样本的间距值
#     steps = np.random.uniform(size=diff)
#     cols = np.mod(samples_indices, nns.shape[1])
#     reshaped_feature = np.zeros((diff, danger_index_data.shape[1]))
#     for i, (col, step) in enumerate(zip(cols, steps)):
#         row = samples_indices[i]
#         reshaped_feature[i] = danger_index_data[row] - step * (danger_index_data[row] - old_feature_data[nns[row, col]])
#     # new_min_feature_data = np.vstack((reshaped_feature, old_feature_data))
#     return reshaped_feature

def SMOTE_Bagging(n_estimators, imbalanced_data_arr2):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    print('多数类：', len(major_data_arr2), '少数类：', len(minor_data_arr2))
    min = 10; max = 100
    b = min # b取值范围：[10,100]
    ret_data = []
    for i in range(n_estimators):
        # 选取少数类样本数： Ni * b%
        minor_data_num = int(minor_data_arr2.shape[0] * (b/100))
        # 选取插值样本数： Nc - Ni * b%
        minor_generate_num = major_data_arr2.shape[0] - minor_data_num
        # 去掉类别的样本
        minor_data = minor_data_arr2[:, : -1]
        # 少数类样本的标签
        old_label_data = minor_data_arr2[0][-1]

        # 随机选取 minor_data_num 个少数类样本
        samples_indices = np.random.randint(low=0, high=np.shape(minor_data_arr2)[0], size=minor_data_num)
        minor_old_data = np.zeros((minor_data_num, minor_data.shape[1]))
        for key, value in enumerate(samples_indices):
            minor_old_data[key] = minor_data[value]

        # 使用 SMOTE 生成少数类样本集
        new_feature_data = make_sample_SMOTE(minor_data, minor_generate_num)
        new_minor_data = np.vstack((new_feature_data, minor_old_data))
        new_labels_data = np.array([old_label_data] * np.shape(major_data_arr2)[0])

        new_minor_data_arr2 = np.column_stack((new_minor_data, new_labels_data))
        # 将少数类与多数类拼接起来
        balanced_data_arr2 = concat_and_shuffle_data(new_minor_data_arr2, major_data_arr2)
        ret_data.append(balanced_data_arr2)
        if n_estimators > max:
            b += n_estimators / max
        else:
            b += max / n_estimators

    
    return ret_data # 返回每个分类器的样本



if __name__ == '__main__':
    imbalanced_data = np.load(ROOT_PATH + 'imbalanced_train_data_arr2.npy')
    
    # 使用多棵决策树进行投票，得到最终分类结果
    foreast = []; proba = []
    n_estimators = 10  #  决策树个数
    base_estimator=DecisionTreeClassifier()

    data = SMOTE_Bagging(n_estimators,imbalanced_data)
    
    for i in range(n_estimators):
        foreast.append(base_estimator)
        _data = data[i]
        y_pred_proba = foreast[i].fit(_data[0:800,: -1], _data[0:800,-1]).predict_proba(_data[800:-1,: -1])
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
            result.append(0)
        else:
            result.append(1)

    # print(result)

    print(classification_report(result,_data[800:-1,-1]))
