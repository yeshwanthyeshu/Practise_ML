# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:13:47 2018

@author: c_ymelpati

Deep learning classification model
"""

import pandas as pd
import numpy as np

train = pd.read_csv("./Dataset/train.csv")
train.head()

test = pd.read_csv('./Dataset/test.csv')
test.head()

# describe the train
train.describe()
train.info()

# has_repair_started has null values:
train.isnull().any()
test.isnull().any()


train['has_repair_started'].fillna((train['has_repair_started'].mean(skipna = True)), inplace=True)
test['has_repair_started'].fillna((test['has_repair_started'].mean(skipna = True)), inplace=True)

# replacing mean with column mean
train.apply(lambda x: x.fillna(x.mean()),axis=0)

# Data preprossicing
## area_assesed should be converted into int or one hot encoding

# Encoding categorical data
# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# For one hot encode of damage_grade column of train

# for training
lben = LabelEncoder()
temp = lben.fit_transform(train.area_assesed)

ohe = OneHotEncoder()
temp = ohe.fit_transform(temp.reshape(-1,1)).toarray()

tempdf = pd.DataFrame(temp)
train = pd.concat([train, tempdf], axis = 1)
train.head()
# for test
lben = LabelEncoder()
temp = lben.fit_transform(test.area_assesed)

ohe = OneHotEncoder()
temp = ohe.fit_transform(temp.reshape(-1,1)).toarray()

tempdf = pd.DataFrame(temp)
test = pd.concat([test, tempdf], axis = 1)
test.head()





# as area_assesed is converted to one hot encoding drop the column
train = train.drop(['area_assesed'] , axis = 1)
train.head()
test = test.drop(['area_assesed'] , axis = 1)
test.head()




# label encoding the damage_grade columns
lben = LabelEncoder()
train.damage_grade = lben.fit_transform(train.damage_grade)


# To convert into numpy arrays
x = train.drop(['building_id', 'damage_grade'], axis = 1).values
y = train['damage_grade'].values

x = test.drop(['building_id'], axis = 1).values


# We have five stages for the target

############################### Applying the model ############################
# libraries
from keras.layers import Dense
from keras.models import Sequential

predictors = np.array(x)
target = np.array(y)
ncols = predictors.shape[1]

# Making model
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (ncols,) ) )
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))

# Compiling and fitting
model.compile(optimizer='adam', loss= 'mean_squared_error')
model.fit(predictors, target)

# Finding predictions
predictions = model.predict(predictors)
predictions = np.around(predictions)
predictions

# evaluation
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(predictors, target) 
mse



# Improving the model
from keras.utils import to_categorical
# One hot encoding
target = to_categorical(target)

from keras.callbacks import EarlyStopping
early_stopping_monitor =  EarlyStopping(patience = 2)
model = Sequential()
# Add the first layer
model.add(Dense(100, activation='relu', input_shape=(ncols,) ))
model.add(Dense(100, activation='relu' ))
# Add the output layer
model.add(Dense(5, activation='softmax'))
# Compile the model
model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model
model.fit(predictors, target ,epochs=5,validation_split= 0.25 ,callbacks = [early_stopping_monitor])
    
# Finding predictions
predictions = model.predict(predictors)
predictions




## still improving
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(ncols,) ))
model.add(Dense(100, activation='relu' ))
model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer='sgd',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(predictors, target ,epochs=5,validation_split= 0.25 ,
          callbacks = [early_stopping_monitor])
   
preds = model.predict(predictors)
preds[preds>=0.5] = int(1)
preds[preds<0.5] = int(0)

# Them mse : 0.3036057
out =[]
for i in preds:
    temp = list(i).index(1) +1
    temp = 'Grade '+str(temp)
    out.append(temp)

t = [[1,2],[2,1]]
for i in t:
    print(i.index(1))
[0,0,1].index(1)

t = [[1],[2],[0]]

for i in predictions:
    temp = 'Grade '+str(int(i[0]))
    out.append(temp)


outdf = pd.DataFrame(out)
bid = pd.DataFrame(test['building_id'])
output = pd.concat([bid, outdf], axis=1)
output.columns = ['building_id', 'damage_grade']

output.head()


output.to_csv("submission_1.csv", sep=',', index = False )

#########################Testing data##########################################

out = model.predict(x)
out
max(out)
min(out)

out = np.around(out)
out


