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
