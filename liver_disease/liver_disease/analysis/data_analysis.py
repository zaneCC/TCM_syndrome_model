# 数据分析
OS_MAC_PATH = '/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
from numpy.core.defchararray import array
import pandas as pd
import numpy as np
import sys
from typing import Counter, Iterable
sys.path.append(r'/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants
import utils.misc as misc
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
import config

# 分型数值化
# PATH = constants.PATH_DIAGNOSIS_NORMALIZATION
# 汇总表-未做特征选择
# PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
# PATH = constants.SELECTION_MERGE_CSV_PATH


PATH = config.PATH

class DataAnalysis():


    def __init__(self):
        self.df = pd.read_csv(PATH)
        # self.df = pd.read_excel(PATH)
        # 设置字体
        # mpl.rcParams['font.sans-serif'] = [u'simHei']
        # mpl.rcParams['axes.unicode_minus'] = False

        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

        self.misc = misc.Misc()
        r = self.df['ZHENGHOU1'].value_counts()
        self.count_1 = r.values[0]
        self.count_2 = r.values[1]

        if config.IS_SHE:
            # 所有舌脉象特征
            self.top40 = ['脉象_弦','舌体_正常','尿黄','身目黄染','脉络_正常','舌质_红','恶心','厌油','腹胀','乏力',
                          '纳差','舌苔_腻','舌质_淡红','舌苔_白','舌苔_黄','脉络_增粗','脉络_迂曲','舌质_红绛','精神差',
                          '脉象_滑','舌体_胖大','夜寐差','脉象_细','呕吐','舌苔_厚','口干',
                          '腹痛','脉象_数','舌质_暗','舌苔_薄','舌体_齿痕','舌质_淡白','舌质_暗紫','口苦','脉络_瘀斑',
                          '易疲劳','发热','舌苔_少','大便稀溏','黑便']

            # self.top20 = ['舌质_淡白','舌质_暗','舌苔_薄','黑便','舌质_红','舌苔_厚','口干','舌质_红绛','舌质_暗紫','口苦','舌质_淡红','舌苔_黄',
            #      '脉络_增粗','舌体_齿痕','发热','易疲劳','腹痛','身目黄染','脉象_细','舌体_胖大']

            self.top20 = ['TongueBody_pale_white', 'TongueBody_dark', 'TongueCoating_thin', 'Melena', 'TongueBody_red', 'TongueCoating_thick', 'Dry_mouth', 'TongueBody_red_crimson',
                          'TongueBody_dark_purple', 'Bitter_taste', 'TongueBody_light_red', 'TongueCoating_yellow',
                          'SublingualVessel_thicken', 'FormOfTongue_teeth_printed', 'Fever', 'Easy_fatigue', 'Abdominal_pain', 'Yellowing_of_the_body_and_eyes', 'PulseCondition_thin', 'FormOfTongue_enlarged']
        else:
            self.top40= ['尿黄','身目黄染','恶心','厌油','腹胀','乏力','纳差','精神差','夜寐差','口干','呕吐','腹痛','口苦','大便稀溏',
                    '发热','易疲劳','黑便','腹泻','双下肢水肿','头晕','大便时干时稀','呕血','肝区疼痛','皮肤瘙痒','咳嗽','大便干',
                    '尿少','欲呕','水样便','神志欠清','鼻塞','肛门坠胀感','大便次数增多','咳痰','全身胀痛','计算力下降','大便偏干'
                    ,'排便不爽','肝功能受损','陶土样大便'
                        ]
            self.top20 = ['头晕','黑便','双下肢水肿','口干','腹泻','口苦','发热','易疲劳','腹痛','身目黄染','大便稀溏','尿黄','乏力','精神差',
                    '纳差','夜寐差','厌油','恶心','呕吐','腹胀']

    def plotSize(self):
        self.misc.plotSize(self.df['ZHENGHOU1'])
        self.misc.show()

    def countSymDiff(self):
        df = self.df
        dicSym = {}
        top = self.top40
        for index, row in df.iteritems():
            if index not in top:
                continue

            if index == 'INHOSPTIAL_ID' or index == 'ZHENGHOU1':
                continue

            count_1 = 0;count_2 = 0 # 同一症状，阳黄与阴阳黄对该症状的出现次数
            for row in df.itertuples():
                _value = getattr(row, index)
                _class = getattr(row, 'ZHENGHOU1')

                if _class == constants.CLASS_1 and _value == 1:
                    count_1 += 1
                elif _class == constants.CLASS_2 and _value == 1:
                    count_2 += 1

            c1 = count_1/self.count_1
            c2 = count_2/self.count_2
            c_sum = c1 + c2
            dicSym[index] = abs(c1-c2)/c_sum
            # dicSym[index] = {constants.CLASS_1: count_1 / count_1+count_2, constants.CLASS_2: count_2 / count_1+count_2}
            sorted_dic = sorted(dicSym.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
        print(sorted_dic[0:20])

        return sorted_dic

    # 统计症状的频次
    def countSym(self,df):
        dicSym = {}
        dicTop40 = {}
        for index, conlumns in df.iteritems():
            if index == 'INHOSPTIAL_ID' or index == 'ZHENGHOU1':
                continue
            count = 0
            count_1 = 0; count_2 = 0
            for row in df.itertuples():

                try:
                    count += 1
                    _value = getattr(row, index)
                    _class = getattr(row, 'ZHENGHOU1')

                    if _class == constants.CLASS_1 and _value == 1:
                        count_1 += 1
                    elif _class == constants.CLASS_2 and _value == 1:
                        count_2 += 1
                except Exception as e:
                    print(e)
                    print('count: ',count)


            # 统计症状频次
            dicSym[index] = {constants.CLASS_1:count_1/self.count_1,constants.CLASS_2:count_2/self.count_2}


        dic = dict(self.df.sum())
        # del dic['INHOSPTIAL_ID']
        del dic['ZHENGHOU1']
        r_dic = sorted(dic.items(), reverse=True, key=lambda kv: (kv[1], kv[0]))
        print(r_dic)
            # dicTop40[index] = Counter[self.df[index]][1]

        sortDic = sorted(dicTop40.items(), key=lambda e:e[1], reverse=True)
        print(sortDic[:40])


        print(self.top40)
        print(dicSym)
        return dicSym

    def showCountSym(self,df=None,type=0): # type=0: top40, type=1: top20+差异频度
        if df is None:
            df = self.df
        dicSym = self.countSym(df)

        plt.ylim((0, 1))

        if type == 0:
            page=8
            value_list = self.top40
            row = 2
        else:
            page=4
            value_list = self.top20
            row = 1

        total_width, n = 0.8, 2
        width = total_width / n
        # x = x - (total_width - width) / 2
        loc = 0
        for i in range(page):
            values = value_list[loc:loc+5]
            a,b = [],[]
            for val in values:
                a.append(dicSym[val][1])
                b.append(dicSym[val][2])

            size = len(values)
            x = np.arange(size)
            plt.subplot(row,4,i+1)

            b1 = plt.bar(x, np.array(a),  width=width, label='a')
            b2 = plt.bar(x + width, np.array(b), width=width, label='b')

            plt.xticks(range(size),value_list[loc:loc+5])
            plt.yticks([0,0.2,0.4,0.6,0.8])
            plt.xticks(rotation=90)
            plt.tight_layout()   #xlable坐标轴显示不全
            loc += 5

        plt.legend(handles=[b1, b2], labels=['Yang jaundice', 'Yang-Yin jaundice'], loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

        plt.show()
    

    def countSymByZHENGHOU1(self):
        for row in self.df.itertuples():
            _ZHENGHOU1 = getattr(row, 'ZHENGHOU1') 
            
    
    def countZhengHou(self):
        count = self.df['ZHENGHOU1'].groupby(self.df['ZHENGHOU2']).describe()
        # r = self.df.groupby(['ZHENGHOU1','ZHENGHOU2']).groups
        grouped = self.df.groupby(['ZHENGHOU2']).groups
        print(Counter(self.df['ZHENGHOU1']))

        # 阳黄证
        z1 = {'证型':[],'频次':[]}
        # 阴阳黄证，阴阳黄
        z2 = {'证型':[],'频次':[]}
        # 阴黄证
        z3 = {'证型':[],'频次':[]}

        for row in self.df.itertuples():
            _ZHENGHOU1 = getattr(row, 'ZHENGHOU1') 
            _ZHENGHOU2 = getattr(row, 'ZHENGHOU2') 
            if _ZHENGHOU1 == '阳黄证':
                if _ZHENGHOU2 in z1['证型']:
                    index = z1['证型'].index(_ZHENGHOU2)
                    z1['频次'][index] += 1
                else:
                    z1['证型'].append(_ZHENGHOU2)
                    z1['频次'].append(1)
            elif _ZHENGHOU1 == '阴阳黄证' or _ZHENGHOU1 == '阴阳黄':
                if _ZHENGHOU2 in z2['证型']:
                    index = z2['证型'].index(_ZHENGHOU2)
                    z2['频次'][index] += 1
                else:
                    z2['证型'].append(_ZHENGHOU2)
                    z2['频次'].append(1)
            elif _ZHENGHOU1 == '阴黄证':
                if _ZHENGHOU2 in z3['证型']:
                    index = z3['证型'].index(_ZHENGHOU2)
                    z3['频次'][index] += 1
                else:
                    z3['证型'].append(_ZHENGHOU2)
                    z3['频次'].append(1)

        print(z1)
        print(z2)
        print(z3)

    def EDA(self):
        # corrdata = self.df.corr()
        # sns.heatmap(corrdata,annot=True)

        sns.countplot(x='class',data=self.df,hue='sudden weight loss')
        plt.show()

    def getSMOTESym(self):
        retDF = self.df[self.df['tag']==1]
        print(retDF)
        self.showCountSym(retDF)

if __name__ == '__main__':
    dataAna = DataAnalysis()
    # 统计症状频次
    dicSym = dataAna.showCountSym(type=1)
    # dicSym = dataAna.countSymDiff()
    # sortDic = sorted(dicSym.items(), key=lambda e:e[1], reverse=True)
    # print(sortDic[:40])

    # 分型统计
    # dataAna.countZhengHou()

    # 分型饼图
    # dataAna.plotSize()

    # EDA
    # dataAna.EDA()

    # 获取合成采样后症状特点
    # dataAna.getSMOTESym()