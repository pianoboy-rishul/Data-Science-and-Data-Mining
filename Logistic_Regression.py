# -*- coding: utf-8 -*-
"""
@author: GITAA
"""
# =============================================================================
# CLASSIFYING PERSONAL INCOME 
# =============================================================================
################################# Required packages ############################
# To work with dataframes
import pandas as pd 

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

###############################################################################
# =============================================================================
# Importing data
# =============================================================================
data_income = pd.read_csv('bank_test.csv')                                         #,na_values=[" ?"]) 
  
# Creating a copy of original data                                                                              # Additional strings (" ?") to recognize as NA
data = data_income.copy()

"""
#Exploratory data analysis:

#1.Getting to know the data
#2.Data preprocessing (Missing values)
#3.Cross tables and data visualization
"""
# =============================================================================
# Getting to know the data
# =============================================================================
#**** To check variables' data type
print(data.info())

#**** Check for missing values             
data.isnull()          
       
print('Data columns with null values:\n', data.isnull().sum())
#**** No missing values !

#**** Summary of numerical variables
summary_num = data.describe()
print(summary_num)            

#**** Summary of categorical variables
summary_cate = data.describe(include = "O")
print(summary_cate)

#**** Frequency of each categories
data['deposit'].value_counts()

#**** Checking for unique classes
print(np.unique(data['deposit'])) 

#**** There exists ' ?' instesd of nan

"""
Go back and read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
"""
data = pd.read_csv('bank_test.csv',na_values=[" ?"]) 

# =============================================================================
# Data pre-processing
# =============================================================================
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis=1 => to consider at least one column value is missing in a row

data2 = data.dropna(axis=0)
data3 = data2.copy()
data4 = data3.copy()
# Realtionship between independent variables
correlation = data2.corr()

# =============================================================================
# Cross tables & Data Visualization
# =============================================================================
# Extracting the column names
data2.columns   

# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

# Reindexing the salary status names to 0,1
data2['deposit']=data2['deposit'].map({'yes':0,'no':1})
print(data2['deposit'])

new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['deposit']))
print(features)

# Storing the output values in y
y=new_data['deposit'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

#Sensitivity
sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print("Sensitivity = ",sensitivity)

#Specificivity
specificivity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print("Specificivity = ",specificivity)

#Prevalence
prevalence = (confusion_matrix[1,0]+confusion_matrix[1,1])/(confusion_matrix[1,0]+confusion_matrix[1,1]+confusion_matrix[0,0]+confusion_matrix[0,1])
print("Prevalence = ",prevalence)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

# =============================================================================
# END OF SCRIPT
# =============================================================================
