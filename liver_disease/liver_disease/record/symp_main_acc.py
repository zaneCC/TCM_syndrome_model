import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/')
import constants
import utils.re_utils as re_utils

'''
    主症状和伴随症状的提取
'''

PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/data/症状/主诉症状.xlsx'


SYMP_DICT = {1:'肝衰竭主症', 2:'肝衰竭伴随症1|肝衰竭伴随症1持续时间量|肝衰竭伴随症2'}
def main():
    df = pd.read_excel(PATH, sheet_name='Sheet1')

    sympList = []
    # 数据过滤
    for row in df.itertuples():
        _index = getattr(row, 'Index')
        _ids = getattr(row, constants.INHOSPTIAL_ID)
        _type = getattr(row, constants.INCASE_FIELD_NAME)
        _value = getattr(row, constants.TRANSFORMED_GVALUE)

        _mathResult = re_utils.rematch(_type,SYMP_DICT)
        if _mathResult == 0: # 删除与主症，伴随症无关内容
            df.drop(index=_index,inplace=True)
            continue

        if _mathResult == 1: # 主症
            _updateValue = _value+'（主症）'
            sympList.append(_updateValue)
            df[constants.TRANSFORMED_GVALUE][_index] = _updateValue
            continue
        if _mathResult == 2: # 伴随症 
            _updateValue = _value+'（伴随）'
            sympList.append(_updateValue)
            df[constants.TRANSFORMED_GVALUE][_index] = _updateValue
            continue

    # 初始化病历症状表
    symps = list(set(sympList)) # 获得所有症状列表，作为dataframe的列
    ids = list(set(df[constants.INHOSPTIAL_ID]))    # 病人id列表，作为dataframe的行
    data = np.zeros((len(ids),len(symps)), dtype=int)   # 数据初始化，所有症状作为列，病人id作为行
    resultDF = pd.DataFrame(data=data, columns=symps, index=ids) 

    # 筛选出：主症、伴随症
    for row in df.itertuples():
        _index = getattr(row, 'Index')
        _ids = getattr(row, constants.INHOSPTIAL_ID)
        _type = getattr(row, constants.INCASE_FIELD_NAME)
        _value = getattr(row, constants.TRANSFORMED_GVALUE)

        resultDF[_value][_ids] = 1 # 记录原始症状

    r = resultDF.columns.values.tolist()
    print(r)

    resultDF.to_csv(constants.SYMP_MAIN_ACC_PATH)

if __name__ == '__main__':
    main()