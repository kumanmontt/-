该项目是对某公司人员离职情况的分析
===
## 说明
* 从部门、离职原因、服务年限、绩效等多维度分析
* 数据已做修改，并非该公司原始数据

## 数据清洗
### 分析
* ng员工没有company,g和ng要分开
* ETL_TIME没有意义可删去
* BU是BU_DESCR的总类，BU_DESCR可删去
* 招聘渠道空值太多可删去

### 代码实现
#### 1.把g和ng分成两个DataFrame,ng员工没有company
df_turnover = pd.read_excel("E:/KAGGLE/turnover/Turnover.xlsx")<br>
df_turnover = df_turnover.drop(['ETL_TIME','BU_DESCR'],axis=1)<br>
df_turnoverng = df_turnover[df_turnover["EMPLOYEE_TYPE"]=="NG"]<br>
df_turnoverng = df_turnoverng.drop(['COMPANY'],axis=1)<br>
df_turnoverG = df_turnover[df_turnover["EMPLOYEE_TYPE"]=="G"]
#### 2.查看单个因素与离职率的关系（Tableau）
##### 共同点
* CONTRACT_TYPE为P、PST的离职人数最多
* 月初离职的人较多
* SM为Jerry的离职较多
* 原因为个人原因较多 

##### 不同点
* G:
1.被动离职的比较多<br>
2.BIZ-LINE为Opration的离职的较多<br>
3.BU为Opration的离职的较多（BU和BIZ-LINE是否有关联需验证）<br> 4.BUDGET_OWNER为fisker、james、jason的离职的较多（BUDGET_OWNER和BUDGET_UNIT是否有关联需要验证）<br>5.COMPANY为DDPC的离职人数最多<br>6.ORG_LEVEL为operation的离职较多<br>7.服务年限为1~3个月的最多

* NG:
1.主动离职的比较多<br>
2.BIZ-LINE为Commercial的离职的较多<br>
3.BU为Marketing Service和Packout&Lable的离职的较多（BU和BIZ-LINE是否有关联需验证）<br>
4.BUDGET_OWNER为Chunyi、Edward、Fisker的离职的较多（BUDGET_OWNER和BUDGET_UNIT是否有关联需要验证）<br>
5.ORG_LEVEL为profession的离职较多  
6.服务年限为1~3年的最多
