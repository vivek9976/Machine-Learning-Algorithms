#lab7
#b20172 vivek jaiswal
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

#question 1
print("Question1")
#reading csv file 
df= pd.read_csv("Iris.csv")
data=df['Species']
#dropping the sprecies column from data
df = df.drop(['Species'], axis = 1)
print(df)
corrematrix = df.corr()
values, vector= np.linalg.eig(corrematrix.to_numpy())
print(values)
print(vector)

# reducing the multidimensional (d = 8) data into lower dimensions (l = 2)
PCa = PCA(n_components=2)
reduced = PCa.fit_transform(df)
df= pd.DataFrame(reduced, columns=['y1', 'y2'])
#data after using PCA reduction
print(df)

#plotting eigen values vs components
values=list(values)
components = [1,2,3,4]
plt.plot(components,values)
plt.title('eigen values vs components')
plt.xlabel('components')
plt.ylabel('eigen values')
plt.show()

#plotting data aftyer reduction
plt.scatter(df['y1'],df['y2'],color='b')
plt.show()

#question2
print("question2")
ktest = KMeans(n_clusters=3) #number of clusters to predict 3 

ktest.fit(df) #fitting the model to df 

y_pred = ktest.predict(df) #predicting labels (df) and saving to y_pred
print(y_pred)
#part a
print("part a")
#Plotting predicted labels as a scatter 
plt.scatter(df['y1'], df['y2'], c=y_pred, cmap=plt.cm.Paired) 
plt.scatter(ktest.cluster_centers_[:, 0], ktest.cluster_centers_[:, 1], c='red', marker='x')
plt.title('Data points and cluster centroids')
plt.show() 

#partb
print("part b")
print('distortion measure for k =3 is',ktest.inertia_)

#part c
print("part c")
clinit = []
for i in data:
    if i == 'Iris-setosa':
        clinit.append(0)
    elif i == 'Iris-virginica':
        clinit.append(2)
    else:
        clinit.append(1)

def purityscore(ytrue, ypred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(ytrue, ypred)  
    print(contingency_matrix)

    # Find optimal one-to-one mapping between cluster labels and true labels
    rowind, colind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[rowind, colind].sum() / np.sum(contingency_matrix)

print('purity score for k =3 is', purityscore(clinit, y_pred))

#question 3
print("Question 3")
K= [2, 3, 4, 5, 6, 7]
distortion = []
purity = []

for k in K:
    ktest = KMeans(n_clusters=k)
    ktest.fit(df)
    distortion.append(ktest.inertia_)
    purity.append(purityscore(clinit, ktest.predict(df)))

print('distortion measures are', distortion)
print('purity scores are', purity)

# plotting K vs distortion measure
plt.plot(K, distortion)
plt.title('distortion measure vs K')
plt.xlabel('K')
plt.ylabel('distortion measure')
plt.show()

#question 4
print("question 4")
K=3
# building gmm
gmm = GaussianMixture(n_components=3, random_state=42).fit(df)
gmmpred = gmm.predict(df)
df['gmmcluster'] = gmmpred
gmmcentres = gmm.means_
# part a
print("part a")
# plotting the scatter plot
plt.scatter(df[df.columns[0]], df[df.columns[1]], c=gmmpred, cmap='rainbow', s=15)
plt.scatter([gmmcentres[i][0] for i in range(K)], [gmmcentres[i][1] for i in range(K)], c='black', marker='o',
            label='cluster centres')
plt.legend()
plt.title('Data Points')
plt.show()

df = df.drop(['gmmcluster'], axis=1)
# part b
print("part b")
print('distortion measure for k =3 is', gmm.score(df) * len(df))

# part c
print('part c')
print('purity score for k =3 is', purityscore(clinit, gmmpred))

# question 5
print("question 5")
Ks= [2, 3, 4, 5, 6, 7]
totallog = []
purity = []
for k in Ks:
    gmm = GaussianMixture(n_components=k, random_state=42).fit(df)
    totallog.append(round((gmm.score(df) * len(df)), 3))
    purity.append(round(purityscore(clinit, gmm.predict(df)), 3))

print('The distortion measures are', totallog)
print('The purity scores are', purity)

# plotting K vs distortion measure
plt.plot(Ks, totallog)
plt.title('distortion measure vs K')
plt.xlabel('K')
plt.ylabel('distortion measure')
plt.show()

#question 6
print('question 6')
eps = [1, 1, 5, 5]
min_samples = [4, 10, 4, 10]
for i in range(4):
    dbscan_model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(df)
    DBSCAN_predictions = dbscan_model.labels_
    print(f'Purity score for eps={eps[i]} and min_samples={min_samples[i]} is',
          round(purityscore(clinit, DBSCAN_predictions), 3))
    plt.scatter(df[df.columns[0]], df[df.columns[1]], c=DBSCAN_predictions, cmap='flag', s=15)
    plt.title(f'Data Points for eps={eps[i]} and min_samples={min_samples[i]}')
    plt.show()
