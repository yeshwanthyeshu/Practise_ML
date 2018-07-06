# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:18:57 2018

@author: c_ymelpati

Machine learning from scratch

"""

#libraries
import numpy as np
import pandas as pd
import matplotlib
import seaborn

###############################################################################
########################### Numpy #############################################
# generating random numbers
## float
r = np.random.random((2,3))
r
# returning floor
np.floor(r)
# rounding to nearest int
np.around(r)

# int (long)
r = np.random.randint(1,5,size=(2,3),dtype='l')
r

r[np.where(r == 4)]
i
list(r).index(1)

# mathematical operations
r.sum(axis = 1)
r.sum(axis = 0)
r.sum()

r.min(axis = 1)
r

# Normal distribution
r = np.random.normal(0.0, 1.0, size= 100)
np.mean(r)

r.shape
r.ndim

x = np.arange(9.).reshape(3, 3)
x
np.where( x > 5 )
np.where(x < 7)

# shape manipulation
a = np.random.randint(1,5,size=(2,3),dtype='l')
a.shape
a

# flatenning:
a.ravel()

a.shape= (3,2)
a
# transpose
a.transpose()

a.reshape(1,6) # does't have inplace operation
a
a.reshape(1,-1)
# stacking
a
b = [[1,2],[3,4],[4,5]]
b

np.vstack((a,b))
np.hstack((a,b))


np.hsplit(a,2)

# inverse of a array
a =np.random.randint(1,5, size= (2,2))
np.linalg.inv(a)

np.eye(3)
np.trace(a)

# solving a quadratic equations
a
c=np.array([1,2])
np.linalg.solve(a,c)

np.mat('1 2; 3 4')

np.linalg.eigvals(a)
np.linalg.eig(a) # eigen values and eigen vectors

# singular value decomposition: 
u, sigma, v = np.linalg.svd(a, full_matrices = False)
u
sigma
v
###############################################################################
########################### Pandas ############################################

pd.Series(np.array([1,2,3,4,5]))
# dict to series
mydic = {'name':['yeshu','sai','melpati'],'age':[24,34,15]}

pd.Series(mydic)
df = pd.DataFrame(mydic)

df.loc[0] # gives 0th observation
df['age']
df[['age','name']]

df.loc[0] # sees for the label of the index
df.iloc[0] # sees for the int of the index

df
df['new'] = [2,3,4]
df

df.drop('new', axis=1, inplace = True)
df

new = df.pop('new')
new
df

df.loc[0, 'name']
df

df.loc[[1,2],['age','name']]

# selection by condition:
df[df['age'] <24]
df.index.values

df.sort_index(axis=1, ascending= False)

# string operations
df['name'].str.extract('(\w+)', expand = False)
df['name'].str.upper()
df['name'].str.len()
df['name'].str.split('e')
df['name'].str.contains('e')

# one hot encoding:
df['gender'] =['male','female','male']
df


genders = pd.get_dummies(df)
genders # sparse matrix 

# applying functions
def log10(x):
    return np.log10(x)
log10(20)
log10(df['age'])
df['age'].apply(log10)

# lambda functions
df['age'].apply(lambda x:log10(x))

df['age'].apply(lambda x: 'elder'  if x > 20 else 'smaller')

# using map and lambda expression
items = [3,2,5,6,8]
def sqrt(x): return np.sqrt(x)

list(map(sqrt, items)) # map iterator

df['age'].map(sqrt)
df['age'].apply(sqrt)

list(map(lambda x: np.sqrt(x) , df['age']))
#################### Importing the data #######################################

# loading the CSV
pokemon = pd.read_csv('./pokemon/Pokemon.csv')
pokemon
# saving a csv file
df
df.to_csv("samplemydetails.csv", index = False)
# loading it
df = pd.read_csv("samplemydetails.csv")
df
 
# loading text 
txt = pd.read_table('./pokemon/data_semicolon.txt', sep =';')
txt


#loading the excel

ex = pd.read_excel('./pokemon/Pokemon.xlsx')
ex.head()

# reading data with .data  file format
dat = pd.read_table('./pokemon/Pokemon.data', sep=',')



