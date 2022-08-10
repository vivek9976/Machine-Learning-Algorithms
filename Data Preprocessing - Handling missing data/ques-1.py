#question-1
from numpy.core.numeric import NaN
import pandas as pd
from matplotlib import pyplot as plt
 
data = pd.read_csv("landslide_data3_miss.csv")
df=data.isnull().sum()
#number of missing values in each attributes
print(df)
df=list(df)
tt=["dates","stationid","temperature","humidity","pressure","rain","lightavgw/o0","lightmax","moisture"]
fig = plt.figure(figsize = (10, 5))
plt.bar(tt, df, color ='maroon',
        width = 0.4)
plt.xlabel("attribute")
plt.ylabel("No. of missing value")
plt.show()


