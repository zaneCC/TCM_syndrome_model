import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/')
import constants

PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/data/出院记录_舌脉诊.xlsx'
TO_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/出院记录-舌脉象-病历表.xlsx'
TONGUE_VEINS = ['舌体','舌下脉络','舌质','舌苔','脉象']
MATCH_TONGUE_VEINS = ['舌体','舌下脉络','舌质|舌','苔','脉象|脉']
NULL = 0

# 结果中去掉 MATCH_TONGUE_VEINS 包含的元素
def trimName(_name):
    for _pattern in MATCH_TONGUE_VEINS:
        if re.match(_pattern,_name):
            return re.sub(_pattern, '', _name)

# 匹配
def matchTongueVein(_value):
    for i,_pattern in enumerate(MATCH_TONGUE_VEINS):
        if re.match(_pattern,_value):
            return i,_value
    return  NULL

df = pd.read_excel(PATH)

# 初始化入院舌脉象表
ids = list(set(df[constants.INHOSPTIAL_ID]))
data = np.zeros((len(ids),len(TONGUE_VEINS)), dtype=int)
resultDF = pd.DataFrame(data=data, columns=TONGUE_VEINS, index=ids)

# 标记
for row in df.itertuples():
    _ids = getattr(row, constants.INHOSPTIAL_ID)
    _incaseFileValue = getattr(row, constants.INCASE_FILEVALUE_1)

    if _incaseFileValue != 0:
        _incaseFileValueList = re.split('，|,',_incaseFileValue)
        for _value in _incaseFileValueList:
            _matchValue = matchTongueVein(_value)
            if _matchValue != NULL:
                resultDF[TONGUE_VEINS[_matchValue[0]]][_ids] = trimName(_matchValue[1]) # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
        

# 保存文件
resultDF.to_excel(TO_PATH)
