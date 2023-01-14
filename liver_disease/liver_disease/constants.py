OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'

# 公共常量

# 病人 id
INCASE_ID = 'INCASE_ID'
# 病人 id
INHOSPTIAL_ID = 'INHOSPTIAL_ID'

# 入院记录——舌脉 INCASE_FIELD_NAME
INCASE_FIELD_NAME = 'INCASE_FIELD_NAME'
# 入院记录——舌脉 INCASE_FIELD_VALUE
INCASE_FIELD_VALUE = 'INCASE_FIELD_VALUE'

# 出院记录-舌脉诊 INCASE_FILEVALUE_1
INCASE_FILEVALUE_1 = 'INCASE_FILEVALUE_1'

# 主诉症状
# 症状类型
INCASE_FIELD_NAME = 'INCASE_FIELD_NAME'
# 症状内容
TRANSFORMED_GVALUE = 'TRANSFORMED_GVALUE'

ZHENGHOU1 = 'ZHENGHOU1'
# panads index
INDEX = 'Index'

# 类别
CLASS_1 = 1
CLASS_2 = 2
CLASS_3 = 3

# --------------------- 输出文件定义
# ROOT_PATH_WINDOWS = 'E:'
# ROOT_PATH_MAC = '/Users/zhouzhan/Documents/codes/python_code'
OS_MAC_PATH = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
OS_PATH = OS_MAC_PATH
PAPER_VERSION = '论文1.0'

PATH_DIAGNOSIS = OS_PATH + '/data/分型.xlsx'
PATH_DIAGNOSIS_NORMALIZATION = OS_PATH + '/output/分型数值化.xlsx'


SYMP_RECORD_NORMALIZATION_PATH = OS_PATH + "/output/病历症状表（规范化）.xlsx"
MERGE_CSV_PATH = OS_PATH + "/output/汇总表.csv"
MERGE_CSV_DIA_TONGUE_PATH = OS_PATH + '/output/汇总表-症状-舌象.csv'
SYMP_MAIN_ACC_PATH = OS_PATH + '/output/主症-伴随症-病历表.csv'

SYMP_MAIN_ACC_TONGUE_DIAGNOSIS_PATH = OS_PATH + '/output/主症-伴随症-舌脉象-分型.csv'
SYMP_MAIN_ACC_DIAGNOSIS_PATH = OS_PATH + '/output/主症-伴随症-分型.csv'

SELECTION_MERGE_CSV_PATH = OS_PATH + "/output/特征选择后-汇总表.csv"

# 采样输出文件
''' 症状-证 '''
RANDOM_OVER_SAMPLER_CSV_PATH = OS_PATH + '/output/SMOTE/随机过采样-汇总表.csv'
SMOTE_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/SMOTE-采样-汇总表.csv'
SMOTE_BORDERLINE1_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/SMOTE_Borderline1-采样-汇总表.csv'
SMOTE_D_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/SMOTE_D-采样-汇总表.csv'
SMOTE_Borderline_D_CSV_PATH = OS_PATH + '/output/SMOTE/SMOTE_Borderline_D-采样-汇总表.csv'
''' 症状-舌象-证 '''
TUE_RANDOM_OVER_SAMPLER_CSV_PATH = OS_PATH + '/output/SMOTE/症状_舌象/随机过采样-汇总表.csv'
TUE_SMOTE_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/症状_舌象/SMOTE-采样-汇总表.csv'
TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/症状_舌象/SMOTE_Borderline1-采样-汇总表.csv'
TUE_SMOTE_D_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/症状_舌象/SMOTE_D-采样-汇总表.csv'
TUE_SMOTE_Borderline_D_CSV_PATH = OS_PATH + '/output/SMOTE/症状_舌象/SMOTE_Borderline_D-采样-汇总表.csv'

''' 症状-证-分析数据 '''
ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/随机过采样-汇总表.csv'
ANALYSIS_SMOTE_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/SMOTE-采样-汇总表.csv'
ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/SMOTE_Borderline1-采样-汇总表.csv'
ANALYSIS_SMOTE_D_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/SMOTE_D-采样-汇总表.csv'
ANALYSIS_SMOTE_Borderline_D_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/SMOTE_Borderline_D-采样-汇总表.csv'

''' 症状-舌象-证-分析数据 '''
TUE_ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/症状_舌象/随机过采样-汇总表.csv'
TUE_ANALYSIS_SMOTE_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/症状_舌象/SMOTE-采样-汇总表.csv'
TUE_ANALYSIS_SMOTE_MERGE_CSV_PATH_EN = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/症状_舌象/SMOTE-采样-汇总表(en).csv'

TUE_ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/症状_舌象/SMOTE_Borderline1-采样-汇总表.csv'
TUE_ANALYSIS_SMOTE_D_MERGE_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/症状_舌象/SMOTE_D-采样-汇总表.csv'
TUE_ANALYSIS_SMOTE_Borderline_D_CSV_PATH = OS_PATH + '/output/SMOTE/'+PAPER_VERSION+'/症状_舌象/SMOTE_Borderline_D-采样-汇总表.csv'