# 根据肝病分型，输出分型数值化
import pandas as pd
import numpy as np
import cmath
import re
import math
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants


PATH_DIAGNOSIS = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/data/分型.xlsx'
TO_PATH = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/output/分型数值化.xlsx'

# 将所有肝病分为：阳黄证和非阳黄证，两种
ZHENGHOU1 = 'ZHENGHOU1'
# ZHENGHOU2 = 'ZHENGHOU2'
INHOSPTIAL_ID = 'INHOSPTIAL_ID'

# KIND_ZHENGHOU1 = {constants.CLASS_1:'阳黄证',constants.CLASS_2:'阴黄证',constants.CLASS_3:'阴阳黄证|阴阳黄'}
KIND_ZHENGHOU1 = {constants.CLASS_1:'阳黄证',constants.CLASS_2:'阴阳黄证|阴阳黄'}
# KIND_ZHENGHOU1 = {constants.CLASS_2:'阴阳黄证|阴阳黄',constants.CLASS_3:'阴黄证'}

# 没用
# KIND_ZHENGHOU2 = {1:'湿热瘀黄', 2:'脾虚瘀黄', 3:'湿热内蕴', 4:'阳虚血瘀', 5:'瘀毒互结', 6:'肝肾阴虚', 7:'阴虚血瘀',
#                     8:'寒湿困脾', 9:'脾虚瘀热', 10:'肝郁气滞', 11:'脾虚瘀黄兼湿热', 12:'湿热瘀黄兼阴虚'}

def getDigitizeDiagnosis(_posture, kindList):
    for _pattern in kindList:
        if re.match(kindList[_pattern], _posture):
            return _pattern
    return -1

# def getDigitizeDiagnosis(name, kindList):
#     for _pattern in kindList:
#         if kindList[_pattern] == name:
#             return _pattern
#     return -1

df = pd.read_excel(PATH_DIAGNOSIS)
del df['ZHENGHOU2']

# 遍历表，并记录
for row in df.itertuples():
    _index = getattr(row, 'Index')
    _id = getattr(row,INHOSPTIAL_ID)
    _kind1 = getattr(row, ZHENGHOU1)
    # _kind2 = getattr(row, ZHENGHOU2)
        

    if not isinstance(_kind1, str) and cmath.isnan(_kind1):
        # df[ZHENGHOU1][_index] = -1
        df.drop(index=_index,inplace=True)
    else:
        _kind1 = getDigitizeDiagnosis(str(_kind1), KIND_ZHENGHOU1)
        if _kind1 == -1:
            df.drop(index=_index,inplace=True)
        else:
            df[ZHENGHOU1][_index] = _kind1

    # if not isinstance(_kind2, str) and cmath.isnan(_kind2):
    #     df[ZHENGHOU2][_index] = 0
    # else:
    #     _kind2 = getDigitizeDiagnosis(str(_kind2), KIND_ZHENGHOU2)
    #     df[ZHENGHOU2][_index] = _kind2

df.to_excel(TO_PATH,index=False)