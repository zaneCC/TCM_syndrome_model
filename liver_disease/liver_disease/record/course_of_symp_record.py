import pandas as pd
import numpy as np
import re

PATH = '/Users/zhouzhan/Documents/codes/python_code/records/病程记录舌苔.csv'
TO_PATH = '/Users/zhouzhan/Documents/codes/python_code/病程记录-舌脉象-病历表.xlsx'
INHOSPTIAL_ID = 'INHOSPTIAL_ID'
TRACE_REC_TIME = 'TRACE_REC_TIME'
SYMP_NAME = 'SYMP_NAME'
SYMP_VALUE = 'SYMP_VALUE'
DOMAIN_NAME = 'DOMAIN_NAME'

df = pd.read_csv(PATH)
print(df)