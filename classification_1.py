# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:06:19 2018

@author: c_ymelpati

Classification Machine learning challange

"""

"""
Steps: 
    Importing the data
    Preprocessing the data
    If necessary cross validation
    applying the model
    testing the model
    evaluating the model
"""
########################## Importing the data #################################

import pandas as pd
import numpy as np

train = pd.read_csv("./Dataset/train.csv")
train.head()

# describe the train
train.describe()
train.info()

########################## Preprocessing the data #############################

# x and y 
y = train.iloc[:,2].values

# Unique values of y
y_unique = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']



# Encoding categorical data
# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# For one hot encode of damage_grade column of train

lben = LabelEncoder()
temp = lben.fit_transform(train.area_assesed)

ohe = OneHotEncoder()
temp = ohe.fit_transform(temp.reshape(-1,1)).toarray()

tempdf = pd.DataFrame(temp)
train = pd.concat([train, tempdf], axis = 1)
train.head()

# sucessfully one hot encoded the damage_grade column and removed it form train
train_removed_area_assesed = train.drop(['area_assesed', 'damage_grade'], axis = 1)

x = train_removed_area_assesed.values

# For y
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



########################## Cross validation## #################################
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y , test_size= 0.25,
                                                random_state= 5)

########################## Applying the models ################################

##### Model: 1 K nearst neighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 5
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X, y)

##### Model: 2 Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

xtrain_id = xtrain[:,0]
xtrain_wo_id = np.nan_to_num(np.delete(xtrain, 0,1))
xtest_id = xtest[:,0]
xtest_wo_id = np.nan_to_num(np.delete(xtest,0,1))
classifier.fit(xtrain_wo_id, ytrain)

# Predicting the Test set results
y_pred = classifier.predict(xtest)




######################




# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("./Dataset/train.csv")
# has_repair_started has nan values
dataset = dataset.fillna(dataset.mean())

X = dataset.drop(['damage_grade'], axis = 1).values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,2:] = sc.fit_transform(X_train[:,2:])
X_test[:,2:] = sc.transform(X_test[:,2:])

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




