import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
import constants
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/normalization')
import symp_record_normalization as srn

PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/data/阴阳黄课题——入院症状.csv'
TO_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/output/病历症状表（规范化）.xlsx'

NOT_SYMPS_DICT = {'为.','日行.','经常','偶发','注意始得','短暂即过','无需药物','可不搔抓','食后半小时以上缓解',
                '不意亦得','稍察即得','1小时缓解','隐约可见','见处较少','见处较多','每日一次','呈黧黑色',
                '黄中带黑','微青','深黄','可以忍受','缠绵不愈','抓破肌肤方快','欲反复搔抓','面积大于2cm',
                '需手足裸露于外','一望可知','遍及胸腹','秋冬和阴雨天更甚'}

# 匹配是否为症状，是：返回症状名，否：返回''
def matchSYMP(_symp):
    for _pattern in NOT_SYMPS_DICT:
        if re.match(_pattern,_symp):
            return ''
    return _symp    

df = pd.read_csv(PATH)
# 清洗非症状数据
df[constants.TRANSFORMED_GVALUE] = list(map(matchSYMP,list(df[constants.TRANSFORMED_GVALUE])))
df = df.drop(df[df[constants.TRANSFORMED_GVALUE]==''].index.tolist())

# 初始化病历症状表
dupSYMPDF = df.drop_duplicates(subset=constants.TRANSFORMED_GVALUE)   # 所有症状去重
symps = list(dupSYMPDF[constants.TRANSFORMED_GVALUE]) # 获得所有症状列表，作为dataframe的列

ids = list(set(df[constants.INCASE_ID]))    # 病人id列表，作为dataframe的行
data = np.zeros((len(ids),len(symps)), dtype=int)   # 数据初始化，所有症状作为列，病人id作为行
resultDF = pd.DataFrame(data=data, columns=symps, index=ids) 

# 去重：相同病人相同症状
dupDF = df.drop_duplicates(subset=[constants.INCASE_ID,constants.TRANSFORMED_GVALUE])
# 按行遍历所有症状，记录病人症状
for row in dupDF.itertuples():
    _id = getattr(row, constants.INCASE_ID)
    _symp = getattr(row, constants.TRANSFORMED_GVALUE)
    resultDF[_symp][_id] = 1 # 记录原始症状


# 开始规范化症状
# 类似症状合并
resultDF = srn.combineSYMP(resultDF)
# 加个“身黄”症状
resultDF['身黄'] = 0

print(resultDF[resultDF['黄疸']!=0]['身目黄染'])

for row in resultDF.itertuples(): 
    resultDF = srn.doNormalization(row, resultDF)

resultDF = srn.delSYMP(resultDF)
# 规范数据，大于0的值 ，设置为1
resultDF[resultDF>0] = 1

print(resultDF.columns.values.tolist())



# 保存文件
resultDF.to_excel(TO_PATH)
