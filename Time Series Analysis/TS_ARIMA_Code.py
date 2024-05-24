###########################################
#Demo: Testing Stationarity
###########################################
import pandas as pd
ts_data1 =pd.read_csv("D:\\Google Drive\\Training\\5. Machine Learning Python\\3.Reference\\8. Time Series Analysis in Python\\ARIMA\\Datasets\\Sales_TS_Data1.csv",header=0,parse_dates=[0], index_col='Month')
ts_data1
ts_data1.shape
ts_data1.head()
ts_data1.dtypes
ts1 = ts_data1["Sales"]
print(ts1)

#Drawing Time Series
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6 # Image length and width

#Plotting 
plt.plot(ts1)

#Test for stationarity  
# A custom function for stationarity 
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
      #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(15, 6))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
#Test for Stationarity 
test_stationarity(ts1)

#The above series is not stationary
#Let us take lag1 difference
ts1_det = ts_data1['Sales'] - ts_data1['Sales'].shift()  
ts1_det.plot(figsize=(15, 6))  
plt.plot(ts1_det)

#Test for stationarity on lag1 difference data
#Need to remove missing values
ts1_det.dropna(inplace=True)
test_stationarity(ts1_det)

###########################################
#Demo: Identification
###########################################
ts_data2 =pd.read_csv("D:\\Google Drive\\Training\\5. Machine Learning Python\\3.Reference\\8. Time Series Analysis in Python\\ARIMA\\Datasets\\Call_Volumes.csv", header=0,parse_dates=[0], index_col='Day')
Call_volume_ts=ts_data2["Calls_in_mm"]
Call_volume_ts
Call_volume_ts.head()

#Plotting 
plt.plot(Call_volume_ts)

#First lets test stationarity 
test_stationarity(Call_volume_ts)

#Plot ACF and PACF
import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(Call_volume_ts, lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(Call_volume_ts, lags=10, ax=ax2)

#ACF slowly dampening PCF cut off at 2. 
#yt = a1yt-1 + a2yt-2 + εt

###########################################
#LAB: Identification
###########################################
ts_data3 =pd.read_csv("D:\\Google Drive\\Training\\5. Machine Learning Python\\3.Reference\\8. Time Series Analysis in Python\\ARIMA\\Datasets\\web_views.csv", header=0,parse_dates=[0], index_col='Day')
web_views=ts_data3["views"]
web_views

#Plotting 
plt.plot(web_views)

#First lets test stationarity 
test_stationarity(web_views)

#Plot ACF and PACF
import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(web_views, lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(web_views, lags=10, ax=ax2)

#ACF slowly dampening PCF cut off at 1 
#yt = a1yt-1 + εt

####################################################
##Demo1: Parameter Estimation
####################################################
from statsmodels.tsa.arima_model import ARIMA
import statsmodels as sm
model = ARIMA(Call_volume_ts,order=(2, 0, 0))  
results_AR = model.fit()  
results_AR.summary()

plt.title('Fitted Vs Actual Values')
plt.plot(Call_volume_ts)
plt.plot(results_AR.fittedvalues, color='red')


####################################################
##LAB: Parameter Estimation
####################################################
from statsmodels.tsa.arima_model import ARIMA
import statsmodels as sm
model1 = ARIMA(web_views,order=(1, 0, 0))  
results_AR1 = model1.fit()  
results_AR1.summary()

plt.plot(web_views)
plt.plot(results_AR1.fittedvalues, color='red')
plt.title('Fitted Vs Actual Values')

####################################################
#####Forecasting
####################################################

#Returns:	 forecast : array
#Array of out of sample forecasts stderr : array
#Array of the standard error of the forecasts.
#conf_int : array 2d array of the confidence interval for the forecast

results_AR.forecast(steps=10)

####################################################
#####LAB: Forecasting using ARIMA
####################################################

results_AR1.forecast(steps=10)















