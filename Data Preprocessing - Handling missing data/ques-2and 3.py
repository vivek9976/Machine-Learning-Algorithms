#question-2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math
 
data = pd.read_csv("landslide_data3_miss.csv")
#part a
df=data['stationid'].isnull().sum()
print(df)
print()
data = data.dropna(subset=['stationid'])

print(data)
#part b
a=0
for i in range(len(data.index)):
    if(data.iloc[i].isnull().sum()>=3):
        a+=1
        data.dropna()

#no of row having non values >=3 in 9 attribute
print(a)
print()
print(data)
#question 3

tt=data.isnull().sum()
#number of missing values in each attributes after question 2
print(tt)
print()
ll=data.isnull().sum().sum()
#total number of missing values in the file
print(ll)
print()
#question 4
#part a 
copydata=data.copy()
dat = pd.read_csv("landslide_data3_original.csv")

index = {} 
mm = data.copy()
names = mm.columns.values.tolist()  
names = names[2:] 
for i in names:
    index[i] = mm.loc[pd.isna(mm[i]),:].index.tolist()  
    mm[i] = mm[i].fillna(mm[i].mean()) 


data['temperature'] = data['temperature'].fillna(data['temperature'].mean())
data['humidity'] = data['humidity'].fillna(data['humidity'].mean())
data['pressure'] = data['pressure'].fillna(data['pressure'].mean())
data['rain'] = data['rain'].fillna(data['rain'].mean())
data['lightavgw/o0'] = data['lightavgw/o0'].fillna(data['lightavgw/o0'].mean())
data['lightmax'] = data['lightmax'].fillna(data['lightmax'].mean())
data['moisture'] = data['moisture'].fillna(data['moisture'].mean())
print()
print()


#part a 1


# printing mean of miss file and original file respectively
print(data['temperature'].mean(),dat['temperature'].mean())
print(data['humidity'].mean(),dat['humidity'].mean())
print(data['pressure'].mean(),dat['pressure'].mean())
print(data['rain'].mean(),dat['rain'].mean())
print(data['lightavgw/o0'].mean(),dat['lightavgw/o0'].mean())
print(data['lightmax'].mean(),dat['lightmax'].mean())
print(data['moisture'].mean(),dat['moisture'].mean())
# printing median of miss file and original file respectively
print()
print()
print(data['temperature'].median(),dat['temperature'].median())
print(data['humidity'].median(),dat['humidity'].median())
print(data['pressure'].median(),dat['pressure'].median())
print(data['rain'].median(),dat['rain'].median())
print(data['lightavgw/o0'].median(),dat['lightavgw/o0'].median())
print(data['lightmax'].median(),dat['lightmax'].median())
print(data['moisture'].median(),dat['moisture'].median())
# printing mode of miss file and original file respectively
print()
print()
print(data['temperature'].mode(),dat['temperature'].mode())
print(data['humidity'].mode(),dat['humidity'].mode())
print(data['pressure'].mode(),dat['pressure'].mode())
print(data['rain'].mode(),dat['rain'].mode())
print(data['lightavgw/o0'].mode(),dat['lightavgw/o0'].mode())
print(data['lightmax'].mode(),dat['lightmax'].mode())
print(data['moisture'].mode(),dat['moisture'].mode())
# printing standard deviation of miss file and original file respectively
print()
print()
print(data['temperature'].std(),dat['temperature'].std())
print(data['humidity'].std(),dat['humidity'].std())
print(data['pressure'].std(),dat['pressure'].std())
print(data['rain'].std(),dat['rain'].std())
print(data['lightavgw/o0'].std(),dat['lightavgw/o0'].std())
print(data['lightmax'].std(),dat['lightmax'].std())
print(data['moisture'].std(),dat['moisture'].std())
print()
#part a 2
rmse = {}  
for i in names:
    act = [] 
    pred = [] 
    for j in index[i]:
        act.append(dat.loc[j, i])
        pred.append(mm[i].mean())
   
    mse = mean_squared_error(act, pred)
    rmse[i] = round(math.sqrt(mse), 3)
print("cjhcj")
print(rmse)
#plotting graph between rmse and attribute

plt.yscale("log")
plt.bar(rmse.keys(), rmse.values())
plt.title("rmse vs Attribute")
plt.xlabel("Attribute")
plt.ylabel("rmse")
plt.xticks(fontsize=7)
plt.show()

print()
#part b interpolate

copydata['temperature'] = copydata['temperature'].fillna(copydata['temperature'].interpolate())
copydata['humidity'] = copydata['humidity'].fillna(copydata['humidity'].interpolate())
copydata['pressure'] = copydata['pressure'].fillna(copydata['pressure'].interpolate())
copydata['rain'] = copydata['rain'].fillna(copydata['rain'].interpolate())
copydata['lightavgw/o0'] = copydata['lightavgw/o0'].fillna(copydata['lightavgw/o0'].interpolate())
copydata['lightmax'] = copydata['lightmax'].fillna(copydata['lightmax'].interpolate())
copydata['moisture'] = copydata['moisture'].fillna(copydata['moisture'].interpolate())
print()
print()
#part b 1
# printing mean of miss file and original file respectively
print(copydata['temperature'].mean(),dat['temperature'].mean())
print(copydata['humidity'].mean(),dat['humidity'].mean())
print(copydata['pressure'].mean(),dat['pressure'].mean())
print(copydata['rain'].mean(),dat['rain'].mean())
print(copydata['lightavgw/o0'].mean(),dat['lightavgw/o0'].mean())
print(copydata['lightmax'].mean(),dat['lightmax'].mean())
print(copydata['moisture'].mean(),dat['moisture'].mean())
# printing median of miss file and original file respectively
print()
print()
print(copydata['temperature'].median(),dat['temperature'].median())
print(copydata['humidity'].median(),dat['humidity'].median())
print(copydata['pressure'].median(),dat['pressure'].median())
print(copydata['rain'].median(),dat['rain'].median())
print(copydata['lightavgw/o0'].median(),dat['lightavgw/o0'].median())
print(copydata['lightmax'].median(),dat['lightmax'].median())
print(copydata['moisture'].median(),dat['moisture'].median())
# printing mode of miss file and original file respectively
print()
print()
print(copydata['temperature'].mode(),dat['temperature'].mode())
print(copydata['humidity'].mode(),dat['humidity'].mode())
print(copydata['pressure'].mode(),dat['pressure'].mode())
print(copydata['rain'].mode(),dat['rain'].mode())
print(copydata['lightavgw/o0'].mode(),dat['lightavgw/o0'].mode())
print(copydata['lightmax'].mode(),dat['lightmax'].mode())
print(copydata['moisture'].mode(),dat['moisture'].mode())
# printing standard deviation of miss file and original file respectively
print()
print()
print(copydata['temperature'].std(),dat['temperature'].std())
print(copydata['humidity'].std(),dat['humidity'].std())
print(copydata['pressure'].std(),dat['pressure'].std())
print(copydata['rain'].std(),dat['rain'].std())
print(copydata['lightavgw/o0'].std(),dat['lightavgw/o0'].std())
print(copydata['lightmax'].std(),dat['lightmax'].std())
print(copydata['moisture'].std(),dat['moisture'].std())
print()

# part b 2
rmse = {}  
for i in names:
    actual = [] 
    predicted = [] 
    for j in index[i]:
        actual.append(dat.loc[j, i])
        predicted.append(copydata.loc[j, i])
    # calculating rmse of all attributes
    mse = mean_squared_error(actual, predicted)
    rmse[i] = round(math.sqrt(mse), 3)
print(rmse)
# plotting bar graph between rmse values and attributes
plt.yscale("log")
plt.bar(rmse.keys(), rmse.values())
plt.title("rmse vs Attributes")
plt.xlabel("Attributes")
plt.ylabel("rmse")
plt.xticks(fontsize=7)
plt.show()


# question 5 part a
# temp
Q1t= np.percentile(copydata['temperature'], 25, interpolation='midpoint')  
Q3t = np.percentile(copydata['temperature'], 75, interpolation='midpoint')  #
IQRt = Q3t - Q1t
Uppert = Q3t + (1.5 * IQRt) 
Lowert= Q1t - (1.5 * IQRt)  
temp = copydata[(copydata['temperature'] < Lowert) | (copydata['temperature'] > Uppert)] 
print("The outliers in temp are:", temp['temperature'].tolist())  

# plotting box plot before replacing the outlier
plt.boxplot(copydata['temperature'])
plt.title("Box Plot of temp before replacing the outlier")
plt.xlabel("temp")
plt.ylabel("values")
plt.show()

# rain
Q1r = np.percentile(copydata['rain'], 25, interpolation='midpoint')  
Q3r = np.percentile(copydata['rain'], 75, interpolation='midpoint')  
IQRr = Q3r - Q1r
Upperr = Q3r + (1.5 * IQRr)  
Lowerr = Q1r - (1.5 * IQRr)  
Rain = copydata[(copydata['rain'] < Lowerr) | (copydata['rain'] > Upperr)] 
print(Rain.index.to_list())
print(len(Rain))
print("The outliers in rain are:", (Rain['rain'].tolist()))  


plt.boxplot(copydata['rain'])
plt.yscale('log')
plt.title("Box Plot of rain before replacing the outliers")
plt.xlabel("Rain")
plt.ylabel("Values")
plt.show()

# 5 b)
# Temperature
median_T = copydata['temperature'].median()  
# replacing outliers with median
copydata.loc[copydata['temperature'] < Lowert, 'temperature'] = median_T
copydata.loc[copydata['temperature'] > Uppert, 'temperature'] = median_T
# plotting box plot after replacing the outliers with median
plt.boxplot(copydata['temperature'])
plt.title("Box Plot of temperature after replacing the outliers")
plt.xlabel("Temperature")
plt.ylabel("Values")
plt.show()

temp = copydata[(copydata['temperature'] < Lowert) | (copydata['temperature'] > Uppert)]  
print(temp.index.to_list())

# Rain
median_R = copydata['rain'].median()  
# replacing outliers with medicopydata
copydata.loc[copydata['rain'] > Upperr, 'rain'] = median_R
copydata.loc[copydata['rain'] < Lowerr, 'rain'] = median_R
# plotting box plot after replacing the outliers with median
plt.boxplot(copydata['rain'])
plt.yscale('log')
plt.title("Box Plot of rain after replacing the outliers")
plt.xlabel("Rain")
plt.ylabel("Values")
plt.show()

# listing outliers their row number and values
Q1_R = np.percentile(copydata['rain'], 25, interpolation='midpoint') 
Q3_R = np.percentile(copydata['rain'], 75, interpolation='midpoint')  
IQR_R = Q3_R - Q1_R 
print(IQR_R)
Upper_R = Q3_R + (1.5 * IQR_R)  
Lower_R = Q1_R - (1.5 * IQR_R)  
Rain = copydata[(copydata['rain'] < Lower_R) | (copydata['rain'] > Upper_R)] 
print(Rain.index.to_list())
print(len(Rain))
print("The outliers in rain are:", (Rain['rain'].tolist()))