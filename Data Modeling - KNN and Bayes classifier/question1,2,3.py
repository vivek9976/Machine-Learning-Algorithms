#question 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

data= pd.read_csv("SteelPlateFaults-2class.csv")

# storing column names
column = data.columns.to_numpy()
# splitting the dataset according to class
grouping= data.groupby('Class')

# converting the class data into numpy array
class0 = grouping.get_group(0).to_numpy()
class1 = grouping.get_group(1).to_numpy()

# splitting the classes dataset into training and testing sets
train0, test0 = train_test_split(class0, test_size=0.30, random_state=42, shuffle=True)
train1, test1 = train_test_split(class1, test_size=0.3, random_state=42, shuffle=True)

# combining the class wise splitted dataset
training = np.concatenate((train0, train1), axis=0)
testing = np.concatenate((test0, test1), axis=0)

# converting the splitted data into dataframe
train = pd.DataFrame(training, columns=column)
test = pd.DataFrame(testing, columns=column)

# saving the data of train and test into csv file
train.to_csv('SteelPlateFaults-train.csv')
test.to_csv('SteelPlateFaults-test.csv')

#applying KNN method to training and testing data
train_data= pd.read_csv("SteelPlateFaults-train.csv")
test_data=pd.read_csv("SteelPlateFaults-test.csv")
train_y=train_data['Class']
train_x=train_data.drop(['Class'], axis = 1)
test_data= pd.read_csv("SteelPlateFaults-test.csv")
test_y=test_data['Class']
test_x=test_data.drop(['Class'], axis = 1)

#part a and b of question 1
print("question 1 part a and b")
#for K=1
classifier1= KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2 )  
classifier1.fit(train_x, train_y)

#Predicting the test set result  
y1_pred= classifier1.predict(test_x)  
cm1= confusion_matrix(test_y, y1_pred)
as1= accuracy_score(test_y, y1_pred)
print("confusion matrix for k=1")
print(cm1)
print("accuracy for k=1",as1)

#for K=3
classifier3= KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2 )  
classifier3.fit(train_x, train_y)

#Predicting the test set result  
y3_pred= classifier3.predict(test_x)  
cm3= confusion_matrix(test_y, y3_pred)
as3= accuracy_score(test_y, y3_pred)
print()
print("confusion matrix for k=3")
print(cm3)
print("accuracy for k=3",as3)

#for K=5
classifier5= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 ) 
classifier5.fit(train_x, train_y)

#Predicting the test set result  
y5_pred= classifier5.predict(test_x)  
cm5= confusion_matrix(test_y, y5_pred)
as5= accuracy_score(test_y, y5_pred)
print()
print("confusion matrix for k=5")
print(cm5)
print("accuracy for k=5",as5)
print()
maxaccu=max(as1,as3,as5)
print("max accuracy",maxaccu)
print()
if(maxaccu==as1):
    print("K=1 has max accuracy")
elif(maxaccu==as3):
    print("K=3 has max accuracy")
elif(maxaccu==as5):
    print("K=5 has max accuracy")
    print()
#question 2
#min max normalization of traning data
#normalising train data
train_xx=train_x.copy()
for col in train_x.columns:
    newLowerBound=0
    newUpperBound=1
    mini = np.min(train_x[col])
    maxi = np.max(train_x[col])
    ranges = maxi - mini
    newRange = newUpperBound - newLowerBound
    old = train_x[col].values.tolist()
    new = []
    
    for value in old:
        y=((value - mini) / ranges) * newRange + newLowerBound
        new.append(y)

    train_x[col] = train_x[col].replace(old, new)

#normalising test data

for col in test_x.columns:
    newLowerBound=0
    newUpperBound=1
    mini = np.min(train_xx[col])
    maxi = np.max(train_xx[col])
    ranges = maxi - mini
    newRange = newUpperBound - newLowerBound
    old=test_x[col].values.tolist()
    new = []
    
    for value in old:
        y=((value - mini) / ranges) * newRange + newLowerBound
        new.append(y)

    test_x[col] = test_x[col].replace(old, new)

# saving the normalise data of train and test into csv file
train_x.to_csv('SteelPlateFaults-train-Normalised.csv')
test_x.to_csv('SteelPlateFaults-test-Normalised.csv')

#part a and b of question 2
print("question 2 part a and b")
#for K=1
classifier1= KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2 )  
classifier1.fit(train_x, train_y)

#Predicting the test set result  
y1_pred= classifier1.predict(test_x)  
cm1= confusion_matrix(test_y, y1_pred)
as1= accuracy_score(test_y, y1_pred)
print("confusion matrix for k=1")
print(cm1)
print("accuracy for k=1",as1)

#for K=3
classifier3= KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2 )  
classifier3.fit(train_x, train_y)

#Predicting the test set result  
y3_pred= classifier3.predict(test_x)  
cm3= confusion_matrix(test_y, y3_pred)
as3= accuracy_score(test_y, y3_pred)
print()
print("confusion matrix for k=3")
print(cm3)
print("accuracy for k=3",as3)

#for K=5
classifier5= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 ) 
classifier5.fit(train_x, train_y)

#Predicting the test set result  
y5_pred= classifier5.predict(test_x)  
cm5= confusion_matrix(test_y, y5_pred)
as5= accuracy_score(test_y, y5_pred)
print()
print("confusion matrix for k=5")
print(cm5)
print("accuracy for k=5",as5)
print()
maxaccu=max(as1,as3,as5)
print("max accuracy",maxaccu)
print()
if(maxaccu==as1):
    print("K=1 has max accuracy")
elif(maxaccu==as3):
    print("K=3 has max accuracy")
elif(maxaccu==as5):
    print("K=5 has max accuracy")
#question 3
print("question 3")
print()
# Building a Bayes Classifier
train = train.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
test= test.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
# splitting training dataset into its attributes and labels
x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, train.shape[1] - 1].values
# splitting testing dataset into its attributes and labels
x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, test.shape[1] - 1].values


# sample mean and covariance for class 0
train00=pd.DataFrame(train0, columns=column)
train00 = train00.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
x_train0 = train00.iloc[:, :-1].values
y_train0 = train00.iloc[:, train00.shape[1] - 1].values
mean0 = np.mean(x_train0, axis=0)
cov0 = np.cov(x_train0.T)
column_num = [x for x in range(1, 24)]
matrix0 = pd.DataFrame(x_train0, columns=column_num)
covari0 = pd.DataFrame(matrix0.cov().T.round(decimals=3))
covari0.to_csv('covariance_0.csv')
print("Mean of class 0:\n",[round(x, 3) for x in mean0])
# sample mean and covariance for class 1
train11=pd.DataFrame(train1, columns=column)
train11= train11.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
x_train1 = train11.iloc[:, :-1].values
y_train1 = train11.iloc[:, train11.shape[1] - 1].values
mean1 = np.mean(x_train1, axis=0)
cov1 = np.cov(x_train1.T)
column_num = [x for x in range(1, 24)]
matrix1 = pd.DataFrame(x_train1, columns=column_num)
covari1 = pd.DataFrame(matrix1.cov().T.round(decimals=3))
covari1.to_csv('covariance_1.csv')
print()

print("Mean of class 1:\n",[round(x, 3) for x in mean1])
# calculating prior probability for each class
prior0 = len(y_train0) / len(y_train)
prior1 = len(y_train1) / len(y_train)

#likelihood function
def likelihood(x, mean, cov):
    expo = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return (np.exp(expo)) / ((2 * np.pi) ** 11.5 * (np.linalg.det(cov)) ** 0.5)

# calculate the likelihood and predicting class 
yy_pred = []
for x in x_test:
    likeh0 = likelihood(x, mean0, cov0) * prior0
    likeh1 = likelihood(x, mean1, cov1) * prior1
    if likeh0 > likeh1:
        yy_pred.append(0)

    else:
        yy_pred.append(1)

print("The confusion matrix for Bayes model")
print(confusion_matrix(y_test, yy_pred))
print("The accuracy for Bayes model")
print(accuracy_score(y_test, yy_pred))
#question 4
print("question 4")
# Tabulating the best results of all three classifiers
compare = {"classifier": ["KNN", "KNN on normalised data", "Baye method"],
              "accuracy (in decimal)": [0.8961, 0.9556,0.9436]}
tt=pd.DataFrame(compare)
print("comparison b/w classifiers based upon classification accuracy")
print(tt)