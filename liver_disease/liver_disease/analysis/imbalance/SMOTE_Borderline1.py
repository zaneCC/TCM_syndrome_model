from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing

from base_sampler import *
import numpy as np

OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
ROOT_PATH = OS_WINDOWS_PATH + '/analysis/imbalance/'

# 处于多数类与少数类边缘的样本
def in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data):
    nn_m = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data)
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nnm_x = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data).kneighbors(old_feature_data,
                                                                                    return_distance=False)[:,1:]
    nn_label = (imbalanced_label_data[nnm_x] != old_label_data).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return np.bitwise_and(n_maj >= (nn_m.n_neighbors - 1) / 2, n_maj < nn_m.n_neighbors - 1)


# 产生少数类新样本的方法
def make_sample(imbalanced_data_arr2, diff):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    imbalanced_featured_data = imbalanced_data_arr2[:, : -1]
    imbalanced_label_data = imbalanced_data_arr2[:, -1]
    # 原始少数样本的特征集
    old_feature_data = minor_data_arr2[:, : -1]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][-1]
    danger_index = in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data)
    # 少数样本中噪音集合，也就是最终要产生新样本的集合
    danger_index_data = _safe_indexing(old_feature_data, danger_index)
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nns = NearestNeighbors(n_neighbors=6).fit(old_feature_data).kneighbors(danger_index_data,
                                                                           return_distance=False)[:, 1:]
    # 随机产生diff个随机数作为之后产生新样本的选取的样本下标值
    samples_indices = np.random.randint(low=0, high=np.shape(danger_index_data)[0], size=diff)
    # 随机产生diff个随机数作为之后产生新样本的间距值
    steps = np.random.uniform(size=diff)
    cols = np.mod(samples_indices, nns.shape[1])
    reshaped_feature = np.zeros((diff, danger_index_data.shape[1]))
    for i, (col, step) in enumerate(zip(cols, steps)):
        row = samples_indices[i]
        reshaped_feature[i] = danger_index_data[row] - step * (danger_index_data[row] - old_feature_data[nns[row, col]])
    new_min_feature_data = np.vstack((reshaped_feature, old_feature_data))
    return new_min_feature_data


# 对不平衡的数据集imbalanced_data_arr2进行Border-SMOTE采样操作，返回平衡数据集
# :param imbalanced_data_arr2: 非平衡数据集
# :return: 平衡后的数据集
def Border_SMOTE(imbalanced_data_arr2):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    # print(minor_data_arr2.shape)
    # 计算多数类数据和少数类数据之间的数量差,也是需要过采样的数量
    diff = major_data_arr2.shape[0] - minor_data_arr2.shape[0]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][-1]
    # 使用K近邻方法产生的新样本特征集
    new_feature_data = make_sample(imbalanced_data_arr2, diff)
    # 使用K近邻方法产生的新样本标签数组
    new_labels_data = np.array([old_label_data] * np.shape(major_data_arr2)[0])
    # 将类别标签数组合并到少数类样本特征集，构建出新的少数类样本数据集
    new_minor_data_arr2 = np.column_stack((new_feature_data, new_labels_data))
    # print(new_minor_data_arr2[:,-1])
    # 将少数类数据集和多数据类数据集合并，并对样本数据进行打乱重排，
    balanced_data_arr2 = concat_and_shuffle_data(new_minor_data_arr2, major_data_arr2)
    return balanced_data_arr2


# 测试
if __name__ == '__main__':
    imbalanced_data = np.load(ROOT_PATH + 'imbalanced_train_data_arr2.npy')
    print(imbalanced_data.shape)
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data)
    print(minor_data_arr2.shape)
    print(major_data_arr2.shape)
    # 测试Border_SMOTE方法
    balanced_data_arr2 = Border_SMOTE(imbalanced_data)
    print(balanced_data_arr2.shape)