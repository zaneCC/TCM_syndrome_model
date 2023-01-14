# 入院_舌脉象
import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/')
import constants
import utils.re_utils as re_utils
sys.path.append(r'/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/normalization')
from normalization import tongue_record_normalization as trn

PATH = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/data/入院记录——舌脉.csv'
TO_PATH = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/output/舌象/入院记录-舌脉象-病历表.csv'
TO_PATH_1 = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/output/舌象/入院记录-舌脉象-舌体-病历表.csv'
TO_PATH_2 = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/output/舌象/入院记录-舌脉象-舌下脉络-病历表.csv'
TO_PATH_3 = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/output/舌象/入院记录-舌脉象-舌质-病历表.csv'
TO_PATH_4 = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/output/舌象/入院记录-舌脉象-舌苔-病历表.csv'
TO_PATH_5 = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/output/舌象/入院记录-舌脉象-脉象-病历表.csv'

INCASE_DICT = ['中医四诊舌体情况', '中医四诊舌下脉络情况','中医四诊舌质情况','中医四诊舌苔情况','中医四诊脉象情况']
TONGUE_VEINS = ['舌体','舌下脉络','舌质','舌苔','脉象']
# 分类前要清洗数据（把所有逗号/句号去掉），不在分类中的用0表示 胖大，边有齿痕
# 舌体种类
KIND_TONGUE_POSTURE = { '舌体_胖大齿痕':'胖大边有齿痕|胖大，边有齿痕','舌体_正常':'正常|正常正常', '舌体_瘦':'瘦', '舌体_胖大':'胖大', '舌体_齿痕':'边有齿痕|正常边有齿痕'}
# 舌质种类_舌色5种：淡红，淡白，红，绛，青紫
KIND_TONGUE_COLOR = { '舌质_暗紫红':'暗紫红','舌质_淡红':'淡红|淡红嫩|稍红', '舌质_红绛':'暗红|红暗|红绛|淡暗红',  '舌质_暗紫':'暗紫|红暗紫',
                     '舌质_红':'红|红嫩','舌质_暗':'暗|淡暗|暗淡','舌质_淡白':'淡白|淡'} 
# 脉象种类:弦、滑、细、数、弱、沉。。。（可拆分）
KIND_TONGUE_PULSE = {'脉象_细沉数弦':'细沉数弦','脉象_细数弦':'细数弦',
                    '脉象_弦滑':'弦滑|滑弦', '脉象_弦数':'弦数', '脉象_弦细':'弦细', '脉象_细弱':'细弱', 
                    '脉象_沉细':'沉细', '脉象_实弦':'实弦', '脉象_弦弱':'弦弱','脉象_弦缓':'弦缓',
                    '脉象_细':'细','脉象_弦':'弦|弦弦', '脉象_濡':'濡',  '脉象_实':'实', '脉象_滑':'滑'} 
# 舌苔种类:1,质（薄，厚，润，燥，腻，腐，剥，偏，全，真，假）;2,色（白，黄，灰黑）。。（可拆分）
KIND_TONGUE_FUR = {'舌苔_厚腻腐黄白':'厚腻腐黄白','舌苔_白腻中间有裂痕':'白腻中间有裂痕', '舌苔_湿润白腻厚':'湿润白厚腻',
                    '舌苔_薄黄厚腻':'薄黄厚腻', '舌苔_白灰黑腻':'白灰黑腻', '舌苔_干燥灰黑腻黄':'干燥灰黑腻黄','舌苔_湿润黄少':'湿润黄少', 
                    '舌苔_白湿润腻':'白湿润腻', '舌苔_厚腻湿润':'厚腻湿润', '舌苔_黄干燥少':'黄干燥少', '舌苔_湿润黄腻':'湿润黄腻',
                    '舌苔_腻湿润':'腻湿润', '舌苔_白黄腻':'白黄腻|白腻黄', '舌苔_腐黄腻':'腐黄腻',
                    '舌苔_薄黄腻':'薄黄腻|腻薄黄', '舌苔_黄腻少':'黄腻少', '舌苔_黄厚腻':'腻厚黄|黄腻厚|厚腻黄|黄厚腻.', 
                    '舌苔_少薄黄':'少薄黄', '舌苔_黄干燥':'黄干燥','舌苔_腻白少':'白腻少', '舌苔_腻白厚':'白腻厚|白厚腻',
                    '舌苔_腻白':'白腻|腻白|白微腻', '舌苔_腻黄':'黄腻|腻黄|腻略黄',
                    '舌苔_薄白':'薄白', '舌苔_白少':'少白','舌苔_白厚':'白厚','舌苔_黄少':'黄少','舌苔_黄厚':'黄厚', '舌苔_腻少':'少腻','舌苔_湿润':'湿润',
                    '舌苔_白':'白','舌苔_黄':'黄','舌苔_少':'少','舌苔_腻':'腻'} 
# 舌下脉络种类（可拆分）
KIND_SUBLINGUAL_VEIN = {'脉络_正常':'脉络正常|舌下脉络正常', '脉络_瘀点迂曲':'瘀点脉络迂曲','脉络_迂曲':'脉络迂曲', '脉络_瘀斑':'瘀斑或瘀条|瘀点',
                     '脉络_增粗':'脉络增粗|舌下脉络增粗延长|脉络增粗延长'} 

NOT_TONGUE_VEINS_DICT = {'{.'}

df = pd.read_csv(PATH)

# 初始化入院舌脉象表
ids = list(set(df[constants.INHOSPTIAL_ID]))
data = np.zeros((len(ids),len(TONGUE_VEINS)), dtype=int)
resultDF = pd.DataFrame(data=data, columns=TONGUE_VEINS, index=ids)

def getValue(_incaseName):
    for i,value in enumerate(TONGUE_VEINS):
        if TONGUE_VEINS[i] in _incaseName:
            return TONGUE_VEINS[i]

# 遍历表，并记录
for row in df.itertuples():
    _ids = getattr(row, constants.INHOSPTIAL_ID)
    _incaseName = getattr(row, constants.INCASE_FIELD_NAME)
    _incaseValue = getattr(row, constants.INCASE_FIELD_VALUE)

    # 清洗数据（把所有逗号/句号去掉）
    if '。' in _incaseValue:
        _incaseValue = str(_incaseValue).replace('。','')
    if '，' in _incaseValue:
        _incaseValue = str(_incaseValue).replace('，','')
    
    # 清洗数据（把无用数据设置0）
    _incaseValue = re_utils.rematchFilter(_incaseValue,NOT_TONGUE_VEINS_DICT)

    _value = getValue(_incaseName)
    
   
    if _value == TONGUE_VEINS[0]: # 舌体数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_POSTURE)
    if _value == TONGUE_VEINS[1]:  # 舌下脉络数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_SUBLINGUAL_VEIN)
    if _value == TONGUE_VEINS[2]:  # 舌质数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_COLOR)
    if _value == TONGUE_VEINS[3]:  # 舌苔数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_FUR)
    if _value == TONGUE_VEINS[4]:  # 脉象数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_PULSE)

    resultDF[_value][_ids] = _incaseValue # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
    
# 规范化
# resultDF = trn.combineTONGUE(resultDF)

# 保存文件：单独 & 全部
# 单独
# 舌体
# data1 = pd.get_dummies(resultDF["舌体"])
# data1.drop([0],axis=1,inplace=True)
# data1 = trn.combineSHE_TI(data1)
# data1.to_csv(TO_PATH_1)
# # 舌下脉络
# data2 = pd.get_dummies(resultDF["舌下脉络"])
# data2.drop([0],axis=1,inplace=True)
# data2 = trn.combineSHE_XIA_MAI_LUO(data2)
# data2.to_csv(TO_PATH_2)
# # 舌质
# data3 = pd.get_dummies(resultDF["舌质"])
# data3.drop([0],axis=1,inplace=True)
# data3 = trn.combineSHE_ZHI(data3)
# data3.to_csv(TO_PATH_3)
# # 舌苔
# data4 = pd.get_dummies(resultDF["舌苔"])
# data4.drop([0],axis=1,inplace=True)
# data4 = trn.combineSHE_TAI(data4)
# data4.to_csv(TO_PATH_4)
# # 脉象
# data5 = pd.get_dummies(resultDF["脉象"])
# data5.drop([0],axis=1,inplace=True)
# data5 = trn.combineSHE_MAI_XIANG(data5)
# data5.to_csv(TO_PATH_5)

# 全部
data1 = pd.read_csv(TO_PATH_1)
data2 = pd.read_csv(TO_PATH_2)
data3 = pd.read_csv(TO_PATH_3)
data4 = pd.read_csv(TO_PATH_4)
data5 = pd.read_csv(TO_PATH_5)

resultDF = pd.merge(data1,data2, how='left', on=constants.INHOSPTIAL_ID)
resultDF = pd.merge(resultDF,data3, how='left', on=constants.INHOSPTIAL_ID)
resultDF = pd.merge(resultDF,data4, how='left', on=constants.INHOSPTIAL_ID)
resultDF = pd.merge(resultDF,data5, how='left', on=constants.INHOSPTIAL_ID)
resultDF.to_csv(TO_PATH,index=False)