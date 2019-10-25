# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:32:13 2019

@author: DATACVG
"""
'''
(1)获取数据
(2)对数据绘图，观测是否为平稳时间序列(对于非平稳时间序列要先进行d阶差分运算，化为平稳时间序列)
(3)要对平稳时间序列分别求得其自相关系数ACF 和偏自相关系数PACF，通过对自相关图和偏自相关图的分析，得到最佳的阶层 p 和阶数 q
(4)由以上得到的d、q、p，得到ARIMA模型,然后开始对得到的模型进行模型检验。
'''

#1用pandas导入和处理时序数据
#1.1导入常用的库
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6
#1.2导入时序数据
#1.3处理时序数据(转化为datetime类型)
#法一
data = pd.read_csv('E:/KAGGLE/TimeSeries/AirPassengers.csv')
print(data.head())
print('\n Data types:',data.dtypes)
data.Month = pd.to_datetime(data.Month)
data.index = data.Month

#法二
dateparse = lambda dates:pd.datetime.strptime(dates,'%Y-%m')#定义函数作用为：转化为datetime类型
data = pd.read_csv('E:/KAGGLE/Time/AirPassengers.csv',parse_dates=['Month'],index_col='Month',date_parser=dateparse)
print(data.head())
print(data.index)

#2检查时序数据的稳定性
'''
1.常量的均值：数据的均值对于时间轴是常量
2.常量的方差：数据的方差对于时间轴是常量
3.与时间独立的自协方差：数据的自协方差与时间无关
'''
from statsmodels.tsa.stattools import adfuller
#ADF检验，单位根检验。对数据或者数据的n阶差分进行平稳检验
def test_stationarity(timeseries):
    rolmean = pd.rolling_mean(timeseries,window=12)#均值
    rolstd = pd.rolling_std(timeseries, window=12)#方差
    rolcsv = pd.rolling_cov(timeseries,window=12)#协方差
    
    #法一：Rolling statistic --即每个时间段内的数据均值和标准差情况
    fig = plt.figure()
    fig.add_subplot()
    orig = plt.plot(timeseries,color='blue',label='Original')
    mean = plt.plot(rolmean,color='red',label='Rolling mean')
    std = plt.plot(rolstd,color = 'black',label='Rolling standard deviation')
    csv = plt.plot(rolcsv,color='green',label='Rolling covariance')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation & Rolling covariance')
    plt.show(block=False)
    
    #法二：Dickey-Fuller Test --在一定置信水平下，对于时序数据假设Null hyypothesis:非稳定。
          #if 通过检验值(statistic)<临界值（critical value）,则拒绝null hypothesis,即数据是稳定的；反之则是非稳定的
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries,autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key ,value in dftest[4].items():
        dfoutput['Critical value (%s)' %key] = value
        
    print (dfoutput)
    
ts = data['#Passengers']  
test_stationarity(ts)
'''
分析结果
均值和标准差都随着时间增长而增大
adf检验 第一个值比第五个中的任何一个都小就说明稳定 ，此时说明不稳定
'''   
#让时序变为稳定
'''
数据不稳定的原因有两个
1.趋势-数据随着时间变化。升高或降低
2.季节性-数据在特定的时间段内变动，比如节假日或者活动
1.1检测和去除趋势
    聚合：将时间轴缩短，以一段时间内星期/月/年作为数据值，使不同的时间段内的值差距减小
    平滑：以一个滑动窗口内的均值代替原来的值，为了使值之间的差距缩小
    多项式过滤：用一个回归模型来拟合现有数据，使数据更平滑
'''    
ts_log =np.log(ts)#缩小值域，同时保留其他信息

#将所有的时间平等看待
#使用平滑方法（MA）
moving_avg = pd.rolling_mean(ts_log,12) 
plt.plot(ts_log,color = 'blue')
plt.plot(moving_avg,color='red') #可以看出moving_average比原值平滑很多
#然后做差
ts_log_moving_avg_diff = ts_log-moving_avg
ts_log_moving_avg_diff.dropna(inplace = True)
test_stationarity(ts_log_moving_avg_diff) #在95%的置信度下，数据是稳定的

#越近的时刻越重要，引入指数加权移动平均
#衰减因子 alpha =1-exp(log(0.5)/halflife)
expweighted_avg = pd.ewma(ts_log,halflife=12)
ts_log_ewma_diff = ts_log - expweighted_avg
test_stationarity(ts_log_ewma_diff)#在99%的置信度上是稳定的（置信度更高）

'''
2检验和去除季节性
2.1差分法：以特定滞后数目的时刻的值的作差
2.2分解：对趋势和季节性分别建模在移除他们
'''  
#Differencing--差分  
ts_log_diff = ts_log - ts_log.shift()#采用特定瞬间和它前一个瞬间的不同的观察结果
plt.plot(ts_log_diff)    
ts_log_diff.dropna(inplace=True)    
test_stationarity(ts_log_diff)#在90%的置信区间是稳定的
 
#Decomposition-分解 
from statsmodels.tsa.seasonal import seasonal_decompose
def decompose(timeseries):
    #三个部分：trend(趋势部分)、seasonal(季节性部分)、residual(残留部分)
    decomposition = seasonal_decompose(timeseries)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(ts_log,label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend,label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(seasonal,label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    
    return trend,seasonal,residual
    
#消除了trend和seasonal之后，只对residual部分作为想要的时序数据进行处理
trend,seasonal,residual = decompose(ts_log)   
residual.dropna(inplace=True)
test_stationarity(residual) #均值和方差几乎无波动，可以直观上认为是稳定的数据


#对时序数据进行预测
'''
1.通过ACF,PACF进行ARIMA(p,d,q)的p,q参数估计
由前文Differencing部分已知，一阶差分后数据已经稳定，所以d=1。
所以用一阶差分化的ts_log_diff = ts_log - ts_log.shift() 作为输入。
'''
from statsmodels.tsa.stattools import acf,pacf
lag_acf = acf(ts_log_diff,nlags=20)
lag_pacf = pacf(ts_log_diff,nlags=20,method='ols')
#plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#模型1 ARIMA(2,1,0)
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log,order=(2,1,0))
results_AR =model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS:%.4f'%sum((results_AR.fittedvalues-ts_log_diff)**2))

#模型2 ARIMA(0,1,2)
model = ARIMA(ts_log,order=(0,1,2))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues,color='red')
plt.title('RRS:%.4f'%sum((results_MA.fittedvalues-ts_log_diff)**2))


#模型3 ARIMA(2,1,2) 拟合最好
model = ARIMA(ts_log,order=(2,1,2))
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RRS:%.4f'%sum((results_ARIMA.fittedvalues-ts_log_diff)**2)) 
    
#将模型代入原数据预测
#拟合的是一阶差分，是第i个月和i-1个月的ts_log差值
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy=True)    
print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.ix[0],index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log) 
plt.figure()
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE:%.4f'%np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    





















