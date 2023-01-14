
import numpy
from sklearn.ensemble import RandomForestClassifier

import config
from utils import data_utils
import constants
import pandas as pd
import analysis.models.mlearn_models as models
from sklearn.metrics import classification_report, accuracy_score, precision_score
import analysis.imbalance.imbalanced_process as imbalanced_process

class RFAnalysis():

    def init(self):
        self.l_one_rule, self.r_one_rule = [], []
        self.tree_results = []
        self.results = []  # 所有树的规则

    def __init__(self):
        self.init()

    def save_decision_rules(self,rf, csv_path, to_path):
        features = data_utils.getFeatures(csv_path)

        with open(to_path, 'w') as f:
            for tree_idx, est in enumerate(rf.estimators_):
                tree = est.tree_
                assert tree.value.shape[1] == 1  # no support for multi-output

                f.write('TREE: {}'.format(tree_idx) + '\n')
                print('TREE: {}'.format(tree_idx))
                iterator = enumerate(
                    zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
                for node_idx, data in iterator:
                    left, right, feature, th, value = data

                    class_idx = numpy.argmax(value[0])

                    # 写入文件
                    if left == -1 and right == -1:
                        print('{} LEAF: return class={}'.format(node_idx, class_idx))
                        f.write('LEAF:' + str(node_idx) + ',' + str(class_idx) + '\n')
                    else:
                        print(
                            '{} NODE: if feature[{}] < {} then next={} else next={}'.format(node_idx, features[feature],
                                                                                            th,
                                                                                            left, right))
                        f.write('NODE:' + str(node_idx) + ',' + str(features[feature]) + ',' + str(left) + ',' + str(
                            right) + '\n')
                f.write("#\n") # 每棵树以"#"结束

    def left_tree(self,tree, left,top_feature):  # 左边：规则
        self.r_one_rule.append(top_feature+':0')
        line = tree[int(left)]

        if line.find("LEAF") != -1:  # 叶子节点
            l = line.split(",")
            value = l[-1]
            if len(self.r_one_rule) > 0: # 没有右边的值，就不加
                self.r_one_rule.append(value)
                _rule = self.r_one_rule.copy()
                self.tree_results.append(_rule)
                del self.r_one_rule[-1]
                del self.r_one_rule[-1]


        if line.find('NODE') != -1:  # 继续遍历
            l = line.split(",")
            feature = l[1]
            _left = l[2]
            _right = l[3]
            # 遍历左子树
            self.left_tree(tree, _left,feature)
            # 遍历右子树
            self.right_tree(tree, _right, feature)

    def right_tree(self,tree, right, top_feature):  # 右边：规则

        if top_feature+':0' in self.r_one_rule:
            self.r_one_rule.remove(top_feature+':0')

        self.r_one_rule.append(top_feature+':1')
        line = tree[int(right)]

        if line.find("LEAF") != -1:  # 叶子节点
            l = line.split(",")
            value = l[-1]
            self.r_one_rule.append(value)
            _rule = self.r_one_rule.copy()
            self.tree_results.append(_rule)
            # del self.r_one_rule[-1]
            del self.r_one_rule[-1]
            del self.r_one_rule[-1]

        if line.find('NODE') != -1:  # 继续遍历
            l = line.split(",")
            feature = l[1]
            _left = l[2]
            _right = l[3]
            # 遍历左子树
            self.left_tree(tree, _left,feature)
            # 遍历右子树
            self.right_tree(tree, _right, feature)

    def read_decision_rules(self,path):
        trees = []
        rules = []
        with open(path, 'r') as f:
            for line in f:
                if line.find('#') != -1:
                    trees.append(rules)
                    rules = []
                else:
                    if line.find('TREE:') != -1:
                        continue
                    rules.append(line)


        for i, tree in enumerate(trees):  # 遍历每棵树
            self.tree_results = []  # 一棵树的所有规则

            root = tree[0]
            # print(root)
            l = root.split(",")
            feature = l[1]
            left = l[2]
            right = l[3]

            self.left_tree(tree, left,feature)
            self.r_one_rule = []
            self.right_tree(tree, right, feature)

            self.results.append(self.tree_results)


    def save_rules(self, path):
        l = []
        with open(path, 'w') as f:
            for i, tree in enumerate(self.results):
                for j, value in enumerate(tree):
                    if (len(value) <= 2):
                        continue
                    l.append(value)
                    print(value)
                    for w,k in enumerate(value):
                        if w != 0:
                            f.write(',')
                        f.write(k)
        print(len(l))

    def filter_rules(self,rules_path):
        """ 规则去重 """
        rules = []
        with open(rules_path, 'r') as f:
            for line in f:
                rules.append(line)

        rules_copy = rules.copy()
        for k,v in enumerate(rules):
            r = [i for i,x in enumerate(rules) if x is v]
            print(r)

    def get_rule_frequency_error(self,csv_path,rules_path,save_path):
        """ 计算每条规则频率和误差，并保存在：save_path 中 """
        rules = [] # rules:字典:{'尿黄':0}
        _id = 0
        with open(rules_path, 'r') as f:
            for line in f:
                rule = {}
                l = line.split(",")
                label = l[-1].replace('\n', '')
                rule['id'] = _id
                for i in l[:-1]:
                    block = i.split(":")
                    key = block[0]
                    value = block[1]
                    rule[key] = value
                rule['label'] = label
                rules.append(rule)
                _id += 1
        # print(rules)

        df = pd.read_csv(csv_path)
        df_len = len(df)
        df[constants.ZHENGHOU1] = data_utils.unify_lable(df[constants.ZHENGHOU1])
        for i, rule in enumerate(rules):
            rule['frequency1'] = 0
            rule['error1'] = 0
            for row in df.itertuples():
                is_true = True # 是否有满足规则的样本
                for k, value in enumerate(rule):
                    if value == 'frequency1' or value == 'id' or value == 'error1':
                        continue

                    if value == 'label':
                        row_value = int(getattr(row, constants.ZHENGHOU1))
                        r = int(rule[value])
                        if row_value != r:
                            rule['error1'] = rule['error1'] + 1
                        continue

                    row_value = int(getattr(row, value))
                    r = int(rule[value])
                    if row_value != r:
                        is_true = False
                        break
                if is_true:
                    rule['frequency1'] = rule['frequency1'] + 1 # 满足规则样本数加一
            rule['frequency2'] = rule['frequency1'] / df_len

            if rule['frequency1'] > 0:
                rule['error2'] = rule['error1'] / rule['frequency1']
                # print(rule['id'],', ',rule['frequency1'])

        # print(len(rules))

        count = 0
        for i, rule in enumerate(rules):
            if rule['frequency1'] != 0:
                count += 1
        print(count)

        # 存储频率不为0的规则
        with open(save_path, 'w') as f:
            for i, rule in enumerate(rules):
                if rule['frequency1'] == 0:
                    continue
                for k, value in enumerate(rule):
                    block = value+":"+str(rule[value])
                    f.write(block)
                    if value != 'error2':
                        f.write(',')
                f.write('\n')

    def get_rank_rules(self,rules_path):
        """ 获取规则排序，频率高，误差小 """
        rules = []
        with open(rules_path, 'r') as f:
            for line in f:
                rule = {}
                l = line.split(",")
                last = l[-1].replace('\n', '')
                l[-1] = last
                is_true = False
                is_true_true = False
                for i in l:
                    block = i.split(":")
                    key = block[0]
                    value = block[1]
                    # 筛选频率大于 0。01的
                    rule[key] = value
                    if key == 'frequency2' and float(value) > 0.03:
                        is_true = True
                    if key == 'error2' and is_true and float(value) < 0.05:
                        is_true_true = True
                        # 筛选长度 <= 10
                if is_true_true:
                    rules.append(rule)
        # print(rules)
        ranked_rules = sorted(rules, key=lambda i: i['frequency2'],reverse=True)
        print('------------筛选结果-----------')
        for i in ranked_rules:
            print(i)
        # print(ranked_rules[0:20])
        return ranked_rules

    def build_smote_rf_rules_process(self,k=10, resampler_path=constants.MERGE_CSV_DIA_TONGUE_PATH):
        df = pd.read_csv(resampler_path)
        del df['INHOSPTIAL_ID']
        # cols = df.columns.values.tolist()
        root_path = constants.OS_PATH + '/output/模型解释/smote_rf'
        for i in range(k):
            self.init()
            # 1. smote过采样一次，保存一个样本集，并为此训练一个随机森林模型
            data = imbalanced_process.get_SMOTE_resampled(df)
            to_df = pd.DataFrame(data=data, columns=df.columns.values.tolist())
            to_df[to_df.iloc[:, :-1] >= 0.5] = 1
            to_df[to_df.iloc[:, :-1] < 0.5] = 0
            to_path = root_path + '/smote/' + str(i) + '.csv'
            to_df.to_csv(to_path, index=False)

            estimator = models.randomForestClassifier()
            x_cols = df.columns.values.tolist()
            x_cols.remove('ZHENGHOU1')
            X = to_df[x_cols]
            y = to_df['ZHENGHOU1']
            estimator.fit(X, y)

            # 2. 获取规则集，并保存
            # 2.1 提取并存储规则集
            txt_path = root_path + '/rf_rules/temp_file/随机森林' + str(i) + '.txt'
            self.save_decision_rules(estimator, to_path, txt_path)
            # 2.2 整理规则集&保存规则集
            rf_analysis.read_decision_rules(txt_path)
            save_path = root_path + '/rf_rules/temp_file/结果'+ str(i) +'.txt'
            rf_analysis.save_rules(save_path)
            # 2.3 获取规则集
            rules_path = save_path
            save_path = root_path + '/rf_rules/temp_file/结果_频率_误差'+ str(i) +'.txt'
            rf_analysis.get_rule_frequency_error(to_path,rules_path,save_path)
            ranked_rules = rf_analysis.get_rank_rules(rules_path=save_path)
            # 2.4 保存筛选结果
            save_path = root_path + '/rf_rules/简化规则集' + str(i) + '.txt'
            with open(save_path, 'w') as f:
                for j in ranked_rules:
                    f.write(str(j)+ '\n')

    # 合并所有简化规则集
    def save_conbine_rules(self, rules_path_name, file_num, save_path):
        rules = []
        for i in range(file_num):
            path = rules_path_name + '/简化规则集'+ str(i) + '.txt'
            with open(path, 'r') as f:
                for line in f:
                    line = line.replace('\n', '')
                    dic = eval(line)
                    rules.append(dic)

        print(len(rules))

        # rules = [{'id': '231', '舌质_红绛': '1', '舌苔_薄': '0', 'label': '0', 'frequency1': '27',
        #           'error1': '0', 'frequency2': '0.0630841121495327', 'error2': '0.0'},
        #          {'id': '232', '舌质_红绛': '1', '舌苔_薄': '0', 'label': '0', 'frequency1': '27',
        #           'error1': '0', 'frequency2': '0.0630841121495327', 'error2': '0.1'},
        #         {'id': '367', '舌质_红': '1', '双下肢水肿': '0', '脉象_实': '0', '恶心': '0', '身目黄染': '0',
        #          'label': '0', 'frequency1': '17', 'error1': '0', 'frequency2': '0.0397196261682243',
        #          'error2': '0.0'},
        #         {'id': '30', '身目黄染': '1', '口苦': '0', '脉象_细': '0', '舌质_淡白': '0', '脉络_增粗': '0',
        #          '脉象_实': '0', '舌质_红': '0', '乏力': '0', '舌质_淡红': '0', '舌质_暗': '0', '舌苔_薄': '0',
        #          'label': '0', 'frequency1': '16', 'error1': '0', 'frequency2': '0.037383177570093455',
        #          'error2': '0.0'}]

        same_index = []
        error = False
        for i in range(len(rules)):
            if error:
                break
            i_keys = list(rules[i].keys())
            i_keys.remove('id')
            i_keys.remove('frequency1')
            i_keys.remove('error1')
            i_keys.remove('frequency2')
            i_keys.remove('error2')

            same_count = 0
            for j in range(len(rules)):
                if error:
                    break
                if j <= i :
                    continue
                j_keys = list(rules[j].keys())
                j_keys.remove('id')
                j_keys.remove('frequency1')
                j_keys.remove('error1')
                j_keys.remove('frequency2')
                j_keys.remove('error2')

                is_i_ok = False

                for key in i_keys:
                    if key != 'label' and (int(rules[i][key]) == 1) and (key not in j_keys): # i 中有且标记为"1"的症状，j 中不存在
                        break
                    if key != 'label' and (int(rules[i][key]) == 1) and (int(rules[j][key]) != 1): # i 中有且标记为"1"的症状，j 中标记为"0"
                        break

                    if key == 'label':
                        # 相同规则
                        is_i_ok = True

                for key_j in j_keys:
                    if not is_i_ok:
                        break

                    if key_j != 'label' and (int(rules[j][key_j]) == 1) and (key_j not in i_keys):
                        break
                    if key_j != 'label' and (int(rules[j][key_j]) == 1) and (int(rules[i][key_j]) != 1):
                        break


                    if key_j == 'label':
                        # 相同规则
                        same_index.append(j)
                        same_count += 1

                        print('相同规则')
                        print('***i')
                        print(rules[i])
                        print('---j')
                        print(rules[j])
                        # # 合并相同规则的频率、误差（取最大误差）
                        # rules[i]['frequency1'] = int(rules[i]['frequency1']) + float(rules[j]['frequency1'])
                        # rules[i]['frequency2'] = float(rules[i]['frequency2']) + float(rules[j]['frequency2'])
                        # rules[i]['error1'] = max(float(rules[i]['error1']),float(rules[j]['error1']))
                        # rules[i]['error2'] = max(float(rules[i]['error2']), float(rules[j]['error2']))

            rules[i]['same_count'] = same_count


            print('====遍历完一条====')
            print(rules[i])
        print('set(same_index):',len(set(same_index)))
        count = 0
        for k, v in enumerate(set(same_index)):
            del rules[v-count]
            count += 1
        print('count:',count,len(rules))

        # 存储规则集
        with open(save_path, 'w') as f:
            for j in rules:
                f.write(str(j) + '\n')


    def save_filter_rules(self, rules_path, save_path):
        rules = []
        with open(rules_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                dic = eval(line)
                rules.append(dic)
        print(len(rules))

        # rules = [{'id': '231', '舌质_红绛': '1', '舌苔_薄': '0', 'label': '0',
        #           'frequency1': '27', 'error1': '0', 'frequency2': '0.0630841121495327',
        #           'error2': '0.0', 'same_count': 29},
        #         {'id': '4', '身目黄染': '1', '舌质_红': '1', '舌体_瘦': '0', '舌质_淡红': '0', '恶心': '0',
        #          '脉象_数': '0', '舌质_红绛': '0', '舌质_暗': '1', 'label': '1', 'frequency1': '26',
        #          'error1': '0', 'frequency2': '0.06074766355140187', 'error2': '0.0', 'same_count': 59},
        #         {'id': '367', '舌质_红': '1', '双下肢水肿': '0', '脉象_实': '0', '恶心': '0', '身目黄染': '0',
        #          'label': '0', 'frequency1': '17', 'error1': '0', 'frequency2': '0.0397196261682243',
        #          'error2': '0.0', 'same_count': 71},
        #         {'id': '30', '身目黄染': '1', '口苦': '0', '脉象_细': '0', '舌质_淡白': '0', '脉络_增粗': '0',
        #          '脉象_实': '0', '舌质_红': '0', '乏力': '0', '舌质_淡红': '0', '舌质_暗': '0', '舌苔_薄': '0',
        #          'label': '0', 'frequency1': '16', 'error1': '0', 'frequency2': '0.037383177570093455',
        #          'error2': '0.0', 'same_count': 15}]


        result_rules = []
        for i in rules:
            i_keys = list(i.keys())
            i_keys.remove('id')
            i_keys.remove('frequency1')
            i_keys.remove('error1')
            i_keys.remove('frequency2')
            i_keys.remove('error2')
            i_keys.remove('same_count')
            i_keys.remove('label')
            # print(i_keys)

            count = 0
            for j in i_keys:
                if int(i[j]) == 1:
                    count += 1

            if count > 5:
                result_rules.append(i)

        print(len(result_rules))

    def rank_rules(self,rules_path,save_path_root):
        rules = []
        with open(rules_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                dic = eval(line)
                rules.append(dic)
        print(len(rules))
        # 通过 frequency1 从大到小排序
        sorted_rules = sorted(rules, key=lambda e: e.__getitem__('frequency1'), reverse=True)
        # 取 label=0 的 10个规则集, label=1 的 10 个规则集
        result_rules_0, result_rules_1 = [], []
        count_0, count_1 = 0, 0
        for i in sorted_rules:
            label = int(i['label'])
            if label == 0 and count_0 < 10:
                result_rules_0.append(i)
                count_0 += 1
                continue
            if label == 1 and count_1 < 10:
                result_rules_1.append(i)
                count_1 += 1
                continue
        print('-----label_0------')
        print(result_rules_0)
        print('-----label_1------')
        print(result_rules_1)

        # 存储规则集
        with open(save_path_root+'_label_0.txt', 'w') as f:
            for j in result_rules_0:
                f.write(str(j) + '\n')

        with open(save_path_root+'_label_1.txt', 'w') as f:
            for j in result_rules_1:
                f.write(str(j) + '\n')


if __name__ == '__main__':
    rf_analysis = RFAnalysis()

    # csv_path = config.PATH
    # # X_train,X_test,y_train,y_test = data_utils.split(csv_path)
    # df = pd.read_csv(csv_path)
    # cols = df.columns.values.tolist()
    # # cols.remove('Unnamed: 0')
    # # cols.remove('INHOSPTIAL_ID')
    # cols.remove('ZHENGHOU1')
    #
    # X = df[cols]
    # y = df['ZHENGHOU1']
    #
    # estimator = models.randomForestClassifier()
    # # estimator = RandomForestClassifier()
    # estimator.fit(X, y)
    # # pre = estimator.predict(X_test)
    # # precision = precision_score(y_test, pre, average=None)
    # # print(precision)
    # #
    # # 提取并存储规则集
    # # txt_path = constants.OS_PATH + '/output/模型解释/随机森林.txt'
    # # rf_analysis.save_decision_rules(estimator,csv_path,txt_path)
    #
    # # 整理规则集&保存规则集
    # # txt_path = constants.OS_PATH + '/output/模型解释/随机森林.txt'
    # # rf_analysis.read_decision_rules(txt_path)
    # # save_path = constants.OS_PATH + '/output/模型解释/结果.txt'
    # # rf_analysis.save_rules(save_path)
    #
    # # rf_analysis.filter_rules(rules_path=save_path)
    #
    # csv_path = constants.OS_PATH + '/output/模型解释/smote.csv'
    # # # 获取规则集
    # rules_path = constants.OS_PATH + '/output/模型解释/结果.txt'
    # save_path = constants.OS_PATH + '/output/模型解释/结果_频率_误差.txt'
    # rf_analysis.get_rule_frequency_error(csv_path,rules_path,save_path)
    # rf_analysis.get_rank_rules(rules_path=save_path)

    # 提取简化规则
    k = 800
    # rf_analysis.build_smote_rf_rules_process(k=k)
    # 合并重复规则
    # rules_path = constants.OS_PATH + '/output/模型解释/smote_rf/rf_rules'
    # save_path = constants.OS_PATH + '/output/模型解释/smote_rf/rf_rules/最终结果' + str(k) + '.txt'
    # rf_analysis.save_conbine_rules(rules_path,file_num=k, save_path = save_path)

    # rules_path = constants.OS_PATH + '/output/模型解释/smote_rf/rf_rules/最终结果' + str(k) + '.txt'
    # save_path = constants.OS_PATH + '/output/模型解释/smote_rf/rf_rules/最终结果' + str(k) + '_第二次.txt'
    # rf_analysis.save_filter_rules(rules_path, save_path)

    save_path = constants.OS_PATH + '/output/模型解释/smote_rf/rf_rules/最终结果' + str(k) + '.txt'
    save_path_root = constants.OS_PATH + '/output/模型解释/smote_rf/rf_rules/最终结果' + str(k)
    rf_analysis.rank_rules(save_path, save_path_root)