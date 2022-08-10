#question-1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statistics as st

data = pd.read_csv("pima-indians-diabetes.csv")
pf=data.copy()
pf.drop('class', inplace=True, axis=1)

for col in pf.columns:
    q1 = np.percentile(pf[col], 25, interpolation='midpoint')  
    q3 = np.percentile(pf[col], 75, interpolation='midpoint')  
    iqr = q3 - q1  
    upperbound = q3 + (1.5 * iqr)  
    lowerbound = q1 - (1.5 * iqr)
    median = pf[col].median() 
    pf.loc[pf[col] < lowerbound, col] = median
    pf.loc[pf[col] > upperbound, col] = median
#part a
# min-max normalization
nordata=pf.copy()
before = {}
after = {}
for col in nordata.columns:
    newLowerBound=5
    newUpperBound=12
    mini = np.min(nordata[col])
    maxi = np.max(nordata[col])
    before[col] = [mini, maxi]
    ranges = maxi - mini
    newRange = newUpperBound - newLowerBound
    old = nordata[col].values.tolist()
    new = []
    
    for value in old:
        y=((value - mini) / ranges) * newRange + newLowerBound
        new.append(y)

    nordata[col] = nordata[col].replace(old, new)
    after[col] = [min(nordata[col]), max(nordata[col])]
print("The min and max value before normalization",before)
print()
print("The min and max value after normalization",after)
print()
#part b
#Standardization
beforemean = {}
beforestd = {}
aftermean = {}
afterstd = {}
for col in pf.columns:
    mean = pf[col].mean()
    std = pf[col].std()
    beforemean[col] = mean
    beforestd[col] = std
    older = pf[col].values.tolist()
    newer = []
    for value in older:
        y= (value - mean) / std
        newer.append(y)

    pf[col] = pf[col].replace(older, newer)
    aftermean[col] =round(pf[col].mean(), 3)
    afterstd[col] = round(pf[col].std(), 3)

print("The mean before standardization", beforemean)
print()
print("The standard deviation before standardization", beforestd)
print()
print("The mean after standardization", aftermean)
print()
print("The standard deviation after standardization", afterstd)
print()
#question2
meand = [0, 0]
cov = [[13, -3], [-3, 5]]
data = np.random.multivariate_normal(meand, cov, 1000)
ff= pd.DataFrame(data, columns=['x', 'y'])

# part a
plt.scatter(ff['x'], ff['y'])
plt.title("Scatter plot of data samples")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# part b
s, t = np.linalg.eig(cov)
print("The eigen values", s)
print("The eigen vectors", t)
# plotting
plt.scatter(ff['x'], ff['y'], marker='x')
plt.quiver(t[0][0], t[0][1], scale=1)
plt.quiver(t[1][0], t[1][1], scale=1)
plt.title(' Plotting eigen directions (with arrows/lines) onto the scatter plot of data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# part c
#direction of first eigen
unitvector1= [t[0][0] / ((t[0][0]) ** 2 + (t[0][1]) ** 2) ** (1 / 2),
         t[0][1] / ((t[0][0]) ** 2 + (t[0][1]) ** 2) ** (1 / 2)] 
sum1 = ff['x'] * unitvector1[0] + ff['y'] * unitvector1[1]
ff['sum1'] = sum1
# Projecting the data on first eigen direction
e1x = ff['sum1'] * unitvector1[0]
e1y = ff['sum1'] * unitvector1[1]
e1x = [round(num, 3) for num in e1x.tolist()]
e1y = [round(num, 3) for num in e1y.tolist()]
#scatter plots superimposed with eigen direction
plt.scatter(ff['x'], ff['y'], marker='x')
plt.quiver(t[0][0], t[0][1], scale=1)
plt.quiver(t[1][0], t[1][1], scale=1)
plt.scatter(e1x, e1y, marker='x')
plt.title('Projected values on first eigen direction')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# direction of second eigen
unitvector2 = [t[1][0] / ((t[1][0]) ** 2 + (t[1][1]) ** 2) ** (1 / 2),
         t[1][1] / ((t[1][0]) ** 2 + (t[1][1]) ** 2) ** (1 / 2)]  
sum2 = ff['x'] * unitvector2[0] + ff['y'] * unitvector2[1]
ff['sum2'] = sum2
# Projecting the data on to the second eigen direction
e2x = ff['sum2'] * unitvector2[0]
e2y = ff['sum2'] * unitvector2[1]
e2x = [round(num, 3) for num in e2x.tolist()]
e2y = [round(num, 3) for num in e2y.tolist()]
#scatter plots superimposed with eigen direction
plt.scatter(ff['x'], ff['y'], marker='x')
plt.quiver(t[0][0], t[0][1], scale=1)
plt.quiver(t[1][0], t[1][1], scale=1)
plt.scatter(e2x, e2y, marker='x')
plt.title('Projected values on second eigen direction')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# part d
# reconstruction of sample data using both eigen vector
ff = ff.drop(['sum2', 'sum1'], axis=1)
PCa = PCA(n_components=2)
gg= PCa.inverse_transform(PCa.fit_transform(ff))
mse = np.linalg.norm((gg - ff), None)
print("error between new and original matrix after reconstruction", mse)
# question 3

corrematrix = pf.corr()
p, s = np.linalg.eig(corrematrix.to_numpy())
eigen = {}
for i in range(len(p)):
    eigen[round(p[i], 3)] = [round(num, 3) for num in s[i]]
# sorting in descending order
sorteigen = sorted(eigen.items(), reverse=True)
eigenanaly = {}
for i in range(len(p)):
    eigenanaly[round(sorteigen[i][0], 3)] = [round(num, 3) for num in sorteigen[i][1]]
# part a
# reducing the multidimensional (d = 8) data into lower dimensions (l = 2)
PCa = PCA(n_components=2)
reduced = PCa.fit_transform(pf)
df= pd.DataFrame(reduced, columns=['y1', 'y2'])
print(df)
variance1 = st.variance(df['y1'].values.tolist())
variance2 = st.variance(df['y2'].values.tolist())
print("variances", round(variance1, 3), round(variance2, 3))
print("eigen values", round(sorteigen[0][0], 3), round(sorteigen[1][0], 3))
# plot of reduced dimensional data with l=2
plt.scatter(df['y1'], df['y2'], marker='x')
plt.title('plot of reduced dimensional data')
plt.xlabel('y1')
plt.ylabel('y2')
plt.show()
# part b
n = np.linspace(1, 8, 8)
plt.plot(n, eigenanaly.keys())
plt.title('eigen values in descending order')
plt.xlabel('position')
plt.ylabel('eigen values')
plt.show()
# part c
col = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
# calculating the  reconstruction error in terms of rmse considering the different values of l (=1, 2,3, ..., 8)
components = [1, 2, 3, 4, 5, 6, 7, 8]
RMSE = []
for n in components:
    PCa = PCA(n_components=n)
    reduced= PCa.fit_transform(pf)
    dfx= PCa.inverse_transform(PCa.fit_transform(pf))
    if n != 1:
        column = col[0:n]
        dfy= pd.DataFrame(data=reduced, columns=[column])
        covy = pd.DataFrame(dfy.cov().T.round(decimals=3))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("covariance matrix for value of l =", n, covy)

    rmse= np.linalg.norm((pf - dfx), None)
    RMSE.append(round(rmse, 3))

components = np.linspace(1, 8, 8)
plt.plot(components, RMSE)
plt.title('reconstruction error in RMSE vs l')
plt.xlabel('components')
plt.ylabel('rmse value')
plt.show()
# part d
# Compare the covariance matrix for the original data (8-dimensional) with that of the covariance 
#matrix for 8-dimensional representation obtained using PCA with l = 8.
covori= pd.DataFrame(pf.cov().T.round(decimals=3))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print("covariance matrix of ori-data", covori)
# covariance matrix of reconstructed data is already calculated in part c