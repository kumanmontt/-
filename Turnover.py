# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:31:38 2019

@author: DATACVG
"""

import pandas as pd
import numpy as np

df_turnover = pd.read_excel("E:/KAGGLE/turnover/Turnover.xlsx")
print(df_turnover.columns)
print(df_turnover.info())
#处理数据
'''
分析
1.ng员工没有company,g和ng要分开
2.ETL_TIME没有意义可删去
3.BU是BU_DESCR的总类，BU_DESCR可删去
4.招聘渠道空值太多可删去
'''
#df_turnover.COMPANY[df_turnover["EMPLOYEE_TYPE"]=="NG"] = 0
#df_turnover.drop(['ETL_TIME','BU_DESCR','HIRE_CHANNEL'],axis=1)
df_turnoverng = df_turnover[df_turnover["EMPLOYEE_TYPE"]=="NG"]
df_turnoverG = df_turnover[df_turnover["EMPLOYEE_TYPE"]=="G"]
#查看单个因素与离职率的关系（Tableau）
'''
分析
1.主动离职的比被动离职的多一半
2.BIZ-LINE为operation的离职的最多
3.packing&label离职的最多，infrastructure几乎没人离职
4.DDPC离职的人最多
5.P、PST员工离职的最多
6.月初离职的人最多
7.G员工离职的多
8.operation离职的最多
9.个人和薪资原因离职的较多
10.高低绩效看不出差别
11.
'''

































