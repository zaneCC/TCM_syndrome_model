import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/')
import constants

# PATH = '/Users/zhouzhan/Documents/codes/python_code/病历症状表.xlsx'
TO_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/病历症状表（规范化）.csv'

# df = pd.read_excel(PATH)

# 合并类似症状
def combineSYMP(df):
    df['身目尿黄'] = df['身目尿黄'] + df['尿黄皮肤目黄'] + df[' 身目尿黄'] + df['随后出身目尿黄']
    del df['尿黄皮肤目黄'] 
    del df[' 身目尿黄'] 
    del df['随后出身目尿黄']

    df['计算力下降'] = df['计算力下降'] + df['计算力']
    del df['计算力']
    
    # TODO
    df['腹胀'] = df['腹胀'] + df[' 腹胀'] + df['上腹胀'] + df['腹胀食后'] + df['阵发性腹胀痛'] + df['上腹胀满'] + df['腹胀大'] + df['腹部饱满'] + df['前出腹胀大'] + df['胃脘不适'] + df['胃脘隐痛不适'] + df['胃脘部不适']+ df['上腹胀痛不适']
    del df[' 腹胀']   
    del df['上腹胀'] 
    del df['腹胀食后'] 
    del df['阵发性腹胀痛'] 
    del df['上腹胀满']
    del df['腹胀大'] 
    del df['腹部饱满'] 
    del df['前出腹胀大']

    # df['胃痛'] = df['剑突下疼痛'] + df['胃脘不适'] + df['胃脘隐痛不适'] + df['胃脘部不适']
    del df['胃脘隐痛不适'] 
    del df['胃脘部不适']
    del df['胃脘不适']
    


    # 上腹胀痛不适 = 腹胀 + 腹痛
    df['腹痛'] = df['腹痛'] + df['上腹胀痛不适'] + df['腹痛 '] + df['腹痛感不'] + df['胀痛感'] + df['脐疼痛'] + df['脐隐痛'] + df['剑突下疼痛']
    del df['上腹胀痛不适'] 
    del df['腹痛 '] 
    del df['腹痛感不']
    del df['胀痛感'] 
    del df['脐疼痛']
    del df['脐隐痛']
    del df['剑突下疼痛'] 


    df['纳差'] = df['纳差'] + df['食纳下降'] + df['食欲下降'] + df['纳少'] + df['纳食少']
    del df['食纳下降'] 
    del df['食欲下降']
    del df['纳食少']
    del df['纳少']

    df['恶心厌油'] = df['恶心厌油'] + df['恶心厌油 ']
    del df['恶心厌油 ']

    df['口干'] = df['口干'] + df['口干不欲饮'] + df['口干欲饮']
    del df['口干不欲饮'] 
    del df['口干欲饮']

    df['口苦'] = df['口苦'] + df['口苦不干']
    del df['口苦不干']

    df['口干口苦'] = df['口干口苦'] + df['口苦干'] + df['口苦口干']
    del df['口苦干'] 
    del df['口苦口干'] 

    df['乏力'] = df['乏力'] + df['极度乏力'] + df['全身乏力'] + df['双下肢乏力']
    del df['极度乏力']
    del df['全身乏力'] 
    del df['双下肢乏力']

    df['便溏'] = df['便溏'] + df['大便时溏']
    del df['大便时溏']

    df['神志欠清'] = df['神志欠清'] + df['神志模糊']
    del df['神志模糊']

    df['呕吐'] = df['呕吐'] + df['进食后呕吐不适'] + df['呕吐次']
    del df['进食后呕吐不适'] 
    del df['呕吐次']

    df['夜寐差'] = df['夜寐差'] + df['夜寐欠佳'] + df['夜寐查'] + df['夜寐欠安']
    del df['夜寐欠佳']
    del df['夜寐查']
    del df['夜寐欠安']

    df['精神差'] = df['精神差'] + df['精神欠佳'] + df['精神极差']
    del df['精神欠佳'] 
    del df['精神极差']

    df['大便稀溏'] = df['便溏'] + df['大便稀'] + df['大便偏稀'] + df['大便稀稀']
    del df['便溏'] 
    del df['大便稀'] 
    del df['大便偏稀'] 
    del df['大便稀稀'] 

    df['黑便'] = df['黑便'] + df['夹黑便']
    del df['夹黑便']

    df['肝区疼痛'] = df['肝区疼痛'] + df['右胁胀痛'] + df['右胁隐痛']
    del df['右胁胀痛'] 
    del df['右胁隐痛']

    

    return df

# print(len(df.columns.values.tolist()))

def doNormalization(row, resultDF):
    _id = getattr(row, constants.INDEX)

    _symp = getattr(row, '身目尿黄乏力')
    if _symp!=0:
        resultDF['身目尿黄'][_id] = 1
        resultDF['乏力'][_id] = 1

    _symp = getattr(row, '厌油纳差')
    if _symp!=0:
        resultDF['厌油'][_id] = 1
        resultDF['纳差'][_id] = 1

    _symp = getattr(row, '乏力恶心厌油')
    if _symp!=0:
        resultDF['乏力'][_id] = 1
        resultDF['恶心'][_id] = 1
        resultDF['厌油'][_id] = 1

    _symp = getattr(row, '恶心厌油')
    if _symp!=0:
        resultDF['厌油'][_id] = 1
        resultDF['恶心'][_id] = 1

    _symp = getattr(row, '乏力身目尿黄')
    if _symp!=0:
        resultDF['乏力'][_id] = 1
        resultDF['身目尿黄'][_id] = 1

    _symp = getattr(row, '口干口苦')
    if _symp!=0:
        resultDF['口干'][_id] = 1
        resultDF['口苦'][_id] = 1

    _symp = getattr(row, '乏力纳差')
    if _symp!=0:
        resultDF['乏力'][_id] = 1
        resultDF['纳差'][_id] = 1

    _symp = getattr(row, '黑色水样便')
    if _symp!=0:
        resultDF['黑便'][_id] = 1
        resultDF['水样便'][_id] = 1

    _symp = getattr(row, '恶心呕吐')
    if _symp!=0:
        resultDF['恶心'][_id] = 1
        resultDF['呕吐'][_id] = 1

    _symp = getattr(row, '口干欲饮口苦')
    if _symp!=0:
        resultDF['口干'][_id] = 1
        resultDF['口苦'][_id] = 1

    # 方法一：拆分
    # _symp = getattr(row, '皮肤目黄')
    # if _symp!=0:
    #     resultDF['身黄'][_id] = 1
    #     resultDF['目黄'][_id] = 1

    # _symp1 = getattr(row, '黄疸')
    # _symp2 = getattr(row, '身目尿黄')
    # if _symp1!=0 and _symp2==0:
    #     resultDF['身黄'][_id] = 1
    #     resultDF['目黄'][_id] = 1
        
    # _symp1 = getattr(row, '身目黄染')
    # _symp2 = getattr(row, '身目尿黄')
    # if _symp1!=0 and _symp2==0:
    #     resultDF['身黄'][_id] = 1
    #     resultDF['目黄'][_id] = 1

    # _symp = getattr(row, '身目尿黄')
    # if _symp!=0:
    #     resultDF['身黄'][_id] = 1
    #     resultDF['目黄'][_id] = 1
    #     resultDF['尿黄'][_id] = 1
    # end 方法一

    # 方法二：激进的合并
    _symp = getattr(row, '皮肤目黄')
    if _symp!=0:
        resultDF['身目黄染'][_id] = 1

    # _symp = getattr(row, '身目尿黄')
    # if _symp!=0:
    #     resultDF['身目黄染'][_id] = 1

    _symp = getattr(row, '黄疸')
    if _symp!=0:
        resultDF['身目黄染'][_id] = 1

    # 方法三：全部归类于身目尿黄
    # _symp = getattr(row, '皮肤目黄')
    # if _symp!=0:
    #     resultDF['身目尿黄'][_id] = 1

    # _symp = getattr(row, '黄疸')
    # if _symp!=0:
    #     resultDF['身目尿黄'][_id] = 1

    # _symp = getattr(row, '身目黄染')
    # if _symp!=0:
    #     resultDF['身目尿黄'][_id] = 1

    # 身目尿黄，拆成：身目黄染+尿黄
    _symp = getattr(row, '身目尿黄')
    if _symp!=0:
        resultDF['身目黄染'][_id] = 1
        resultDF['尿黄'][_id] = 1


    return resultDF

def delSYMP(df):
    del df['身目尿黄乏力']
    del df['厌油纳差']
    del df['乏力恶心厌油']
    del df['恶心厌油']
    del df['乏力身目尿黄']
    del df['口干口苦']
    del df['乏力纳差']
    del df['黑色水样便']
    del df['恶心呕吐']
    del df['口干欲饮口苦']

    # TODO 身目尿黄问题
    # del df['身目黄染']
    del df['身目尿黄']
    del df['皮肤目黄']
    del df['黄疸']
    del df['身黄']
    del df['目黄']
    # del df['尿黄']

    del df['身目微黄']
    del df['小便深黄']
    del df['小便短赤']

    del df['次数较多']
    del df['未予重视']
    del df['上症逐日']
    del df['大便日行']
    del df['夜']
    del df['质软']
    del df['大便']
    del df['量少']
    del df['神情']
    del df['能自行缓解']
    del df['食后尤甚']
    del df['夹白色']
    del df['影响进食']
    del df['小便']
    del df['色黄']

    return df

    

# r = df.columns.values.tolist()
# print(r)

# df.to_csv(TO_PATH)


