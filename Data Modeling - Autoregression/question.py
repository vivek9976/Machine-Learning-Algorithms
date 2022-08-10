#question 1
print("Question-1")
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import numpy as np
import scipy.stats
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg as AR
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns
data=pd.read_csv('daily_covid_cases.csv')
#part a
print("Part-a")
date=list(data['Date'])
cases=list(data['new_cases'])

# generating the x-ticks
x = [16]
for i in range(10):
    x.append(x[i] + 60)

#label for the month-year
label= ['Feb-20', 'April -20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']
plt.rcParams['figure.figsize'] = [20, 8]
plt.xticks(x, label)
# plotting the points
plt.xlabel("month")
plt.ylabel("new confirmed cases")
plt.plot(date,cases)
plt.show()
#part b
print("part b")
length=len(cases)
lag_cases=cases[:length-1]
actual_cases=cases[1:]
print(scipy.stats.pearsonr(lag_cases, actual_cases)[0])
#part c
print("part c")
#scatter plot between lag time series and gven time series
plt.scatter(actual_cases, lag_cases, c ="blue")
plt.xlabel("lag_cases")
plt.ylabel("actual_cases")  
# To show the plot
plt.show()
#part d
print("part d")
corr=[]
for i in range(6):
    lagcases=cases[:length-(i+1)]
    actualcases=cases[i+1:]
    corr.append(scipy.stats.pearsonr(lagcases, actualcases)[0])
print(corr)
x=[1,2,3,4,5,6]
plt.plot(x,corr)
plt.show()
#part e
print("part e")

#calculate autocorrelations by inbuilt function
print(sm.tsa.acf(cases,nlags=6))
#plot autocorrelation function using inbuilt function
fig = tsaplots.plot_acf(cases, lags=6)
plt.show()

#Question 2
print("question 2")
#part a
print("part a")
# Train test split
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
# Code snippet to train AR model and predict using the coefficients.
window=5;
model = AR(train, lags=window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 
print(coef)
#part b
print("partb")
# walk forward over time steps in test
history = train[len(train)-window:]

history = [history[i] for i in range(len(history))]

predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
#part b 1
print("partb1")
plt.scatter(test, predictions, c ="blue")
# To show the plot
plt.show()
#part b 2
print("partb2")
# plot lines
plt.plot(test, label = "test actual data")
plt.plot(predictions,label = "test predicted data")
plt.legend()
plt.show()
#part b3
print("part b3")
mse=mean_squared_error(test,predictions)
rmse=math.sqrt(mse)
for items in test:
    d+=items
d/=len(test)
RMSEper=(rmse/d)*100
print("RMSE percent value gor lag=5",RMSEper,"%")
#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
l=MAPE(test,predictions)
print("MAPE vapue for lag=5 between actual and predicted data",l)
#question 3
print("Question 3")
m=[1,5,10,15,25]
rrmse=[]
mape=[]
for window in m:
    
    model = AR(train, lags=window) 
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model 
    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
    mse=mean_squared_error(test,predictions)
    rmse=math.sqrt(mse)
    for items in test:
        d+=items
    d/=len(test)
    RMSEper=(rmse/d)*100
    rrmse.append(RMSEper)
    #Defining MAPE function
    def MAPE(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape
    l=MAPE(test,predictions)
    mape.append(l)
print("rmse percentage for different lag value",rrmse)
print("mape value for different lag value",mape)
#bar plot rmseper vs lag
flat=[]
for i in rrmse:
    for j in i:
        flat.append(j)

# Creating our own dataframe
datarmse = {"lag": m,
        "rmseper":flat}
 
# Now convert this dictionary type data into a pandas dataframe
# specifying what are the column names
df = pd.DataFrame(datarmse, columns=['lag', 'rmseper'])
# Defining the plotsize
plt.figure(figsize=(8, 6))
 
# Defining the x-axis, the y-axis and the data
# from where the values are to be taken
plots = sns.barplot(x="lag", y="rmseper", data=df)
 
# Setting the x-acis label and its size
plt.xlabel("lag value", size=15)
 
# Setting the y-axis label and its size
plt.ylabel("RMSE %", size=15)
 
# Finallt plotting the graph
plt.show()
#bar plot between mape value and lag value
# Creating our own dataframe
datamape = {"lag": m,
        "mape":mape}
 
# Now convert this dictionary type data into a pandas dataframe
# specifying what are the column names
df = pd.DataFrame(datamape, columns=['lag', 'mape'])
# Defining the plotsize
plt.figure(figsize=(8, 6))
 
# Defining the x-axis, the y-axis and the data
# from where the values are to be taken
plots = sns.barplot(x="lag", y="mape", data=df)
 
# Setting the x-acis label and its size
plt.xlabel("lag value", size=15)
 
# Setting the y-axis label and its size
plt.ylabel("mape value for their lag", size=15)
 
# Finallt plotting the graph
plt.show()
#question 4
print("question 4")

d=2/(math.sqrt(len(train)))

trains=[]
print(d)
for i in train:
    for j in i:
        trains.append(j)
length=len(trains)
print(length)
for i in range(100):
    lagcases=trains[:length-(i+1)]
    actualcases=trains[i+1:]
    autocorr=abs(scipy.stats.pearsonr(lagcases, actualcases)[0])
    print(autocorr)
    if autocorr<=d:
        p=i+1
        break
      
print(p)
#heuristic value for the optimal number of lags up to the condition on 
#autocorrelation such that abs(AutoCorrelation) > 2/sqrt(T) is 77
window=77;
model = AR(train, lags=window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 


# walk forward over time steps in test
history = train[len(train)-window:]

history = [history[i] for i in range(len(history))]

predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
mse=mean_squared_error(test,predictions)
rmse=math.sqrt(mse)
for items in test:
    d+=items
d/=len(test)
RMSEper=(rmse/d)*100
print("RMSE percent value gor lag=77",RMSEper,"%")
#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
l=MAPE(test,predictions)

print("mape value for lag=77",l)