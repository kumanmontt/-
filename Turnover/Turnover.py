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
1.ng员工没有company,g和ng要分开,ng员工没有company
2.ETL_TIME没有意义可删去
3.BU是BU_DESCR的总类，BU_DESCR可删去
4.招聘渠道空值太多可删去
'''
#df_turnover.COMPANY[df_turnover["EMPLOYEE_TYPE"]=="NG"] = 0
#df_turnover.drop(['ETL_TIME','BU_DESCR','HIRE_CHANNEL'],axis=1)
df_turnover = df_turnover.drop(['ETL_TIME','BU_DESCR'],axis=1)
df_turnoverng = df_turnover[df_turnover["EMPLOYEE_TYPE"]=="NG"]
df_turnoverng = df_turnoverng.drop(['COMPANY'],axis=1)
df_turnoverG = df_turnover[df_turnover["EMPLOYEE_TYPE"]=="G"]
#查看单个因素与离职率的关系（Tableau）
'''
分析
共同点
1.CONTRACT_TYPE为P、PST的离职人数最多
2.月初离职的人较多
3.SM为Jerry的离职较多
4.原因为个人原因较多 
不同点
G:
1.被动离职的比较多
2.BIZ-LINE为Opration的离职的较多
3.BU为Opration的离职的较多（BU和BIZ-LINE是否有关联需验证）   
4.BUDGET_OWNER为fisker、james、jason的离职的较多（BUDGET_OWNER和BUDGET_UNIT是否有关联需要验证）
5.COMPANY为DDPC的离职人数最多
6.ORG_LEVEL为operation的离职较多
7.服务年限为1~3个月的最多
NG:
1.主动离职的比较多
2.BIZ-LINE为Commercial的离职的较多
3.BU为Marketing Service和Packout&Lable的离职的较多（BU和BIZ-LINE是否有关联需验证）
4.BUDGET_OWNER为Chunyi、Edward、Fisker的离职的较多（BUDGET_OWNER和BUDGET_UNIT是否有关联需要验证）
5.ORG_LEVEL为profession的离职较多  
6.服务年限为1~3年的最多
'''







































