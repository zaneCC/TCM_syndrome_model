## sklearn

使用sklearn：LogisticRegression, SVM.SVC, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier

### 使用汇总表数据

其症状为：所有症状

地址：MERGE_CSV_PATH = "/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/汇总表.csv"


预测结果：

LogisticRegression()

0.7547169811320755

SVC(kernel='linear', random_state=0)

0.7547169811320755

DecisionTreeClassifier(max_depth=5)

0.7547169811320755

RandomForestClassifier(criterion='entropy')

0.7169811320754716

AdaBoostClassifier(n_estimators=100)

0.49056603773584906

### 使用主症-伴随症-舌脉象-分型
其症状为：主诉症状（主症-伴随症）

地址：SYMP_MAIN_ACC_TONGUE_DIAGNOSIS_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/主症-伴随症-舌脉象-分型.csv'


预测结果：

LogisticRegression()

0.7619047619047619

SVC(kernel='linear', random_state=0)

0.7619047619047619

DecisionTreeClassifier(max_depth=5)

0.7619047619047619

RandomForestClassifier(criterion='entropy')

0.7857142857142857

AdaBoostClassifier(n_estimators=100)

0.7619047619047619
