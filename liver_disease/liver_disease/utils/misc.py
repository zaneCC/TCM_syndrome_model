'''
    统一绘图工具
'''
from typing import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl

class Misc():

    def __init__(self):
        # 设置字体
        # mpl.rcParams['font.sans-serif'] = [u'simHei']
        # mpl.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        

    def plotAccuLine(self,accuracies,color='r',dlabel=None):
        plt.ylim((0, 100))
        plt.ylabel('准确度（单位：百分比）')
        plt.xlabel('10折交叉验证次数')
        plt.plot(accuracies,color,label=dlabel)
        if dlabel is not None:
            plt.legend()

    def plotData(self,X,y,color='r',dlabel=None):
        plt.plot(X,y,color,label=dlabel)
        if dlabel is not None:
            plt.legend()

    # 绘制证的比例饼图
    def plotSize(self,y):
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        # labels = '阳黄证', '阴黄证', '阴阳黄证'
        labels = '阳黄证', '阴阳黄证'
        # sizes = [Counter(y==1)[True], Counter(y==2)[True], Counter(y==3)[True]]
        sizes = [Counter(y==1)[True], Counter(y==2)[True]]
        # explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        explode = (0, 0)

        fig1, ax1 = plt.subplots()
        patches,l_text,p_text = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        for t in l_text:
            t.set_size(20)
        for t in p_text:
            t.set_size(20)

    def figure(self):
        plt.figure()
    
    def plotAUC(self, fpr, tpr, roc_auc,color, label, title):
        lw = 2
        plt.plot(fpr, tpr, color=color,
                lw=lw, label=label + '(area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

        
    def show(self):
        plt.show()


# if __name__ == '__main__':
    