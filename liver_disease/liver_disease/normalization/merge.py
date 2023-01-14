# 合并成汇总表
OS_MAC_PATH = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'

import pandas as pd
import numpy as np
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants
'''
    所有症状-分型合并
'''
OS_PATH = OS_MAC_PATH
PATH_TONGUE = OS_PATH + '/output/舌象/入院记录-舌脉象-病历表.csv'
PATH_SYMP = OS_PATH + '/output/病历症状表（规范化）.xlsx'
PATH_DIAGNOSIS = OS_PATH + '/output/分型数值化.xlsx'


# 所有症状-分型合并
def mergeSymp_Diagnosis():
    sympDF = pd.read_excel(PATH_SYMP)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)

    resultDF = pd.merge(diagnosisDF,sympDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    # resultDF.drop(['ZHENGHOU2'],axis=1,inplace=True)
    # resultDF.drop(['Unnamed: 0'],axis=1,inplace=True)
    return resultDF

# 舌象-分型
def merge_dianosis_tongue():
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)
    tongueDF = pd.read_csv(PATH_TONGUE)
    resultDF = pd.merge(diagnosisDF, tongueDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)
    return resultDF

# 所有症状-分型-舌脉象合并
def mergeSymp_Diagnosis_Tongue():
    sympDF = pd.read_excel(PATH_SYMP)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)
    tongueDF = pd.read_csv(PATH_TONGUE)

    combineTongueSympDF = pd.merge(tongueDF, sympDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF = pd.merge(diagnosisDF, combineTongueSympDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    # resultDF.drop(['ZHENGHOU2'],axis=1,inplace=True)
    # resultDF.drop(['Unnamed: 0'],axis=1,inplace=True)
    return resultDF

# 主症-伴随症-分型合并
def mergeSympMainAcc_Diagnosis():
    sympDF = pd.read_csv(constants.SYMP_MAIN_ACC_PATH)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)

    resultDF = pd.merge(sympDF, diagnosisDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    return resultDF

# 主症-伴随症-舌脉象-分型合并
def mergeSympMainAcc_Tongue_Diagnosis():
    sympDF = pd.read_csv(constants.SYMP_MAIN_ACC_PATH)
    tongueDF = pd.read_csv(PATH_TONGUE)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)

    combineTongueSympDF = pd.merge(sympDF, tongueDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF = pd.merge(combineTongueSympDF, diagnosisDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    return resultDF

if __name__ == '__main__':
    # resultDF = mergeSymp_Diagnosis()
    # print(len(resultDF.columns))
    #
    # # 去掉无用症状
    # dicSym = {}
    # for index, row in resultDF.iteritems():
    #     if index == 'INHOSPTIAL_ID' or index == 'ZHENGHOU1':
    #         continue
    #     count = 0
    #     for row in resultDF.itertuples():
    #         _value = getattr(row, index)
    #         if _value == 1:
    #             count += 1
    #
    #     if count == 0 : # 无用症状
    #         print('删除：',index)
    #         del resultDF[index]
    #         continue
    #     dicSym[index] = count
    #
    # print(len(resultDF.columns))
    # resultDF.to_csv(constants.MERGE_CSV_PATH, encoding="utf-8-sig", index=False)

    # 症状-舌象-分型
    resultDF = mergeSymp_Diagnosis_Tongue()
    resultDF.to_csv(constants.MERGE_CSV_DIA_TONGUE_PATH,  encoding="utf-8-sig", index=False)