# -*- coding: utf-8 -*-
"""
Analysis of UCI Breast Cancer Coimbra dataset using SVM

Author: Joe Skufca
Created:  15 June, 2019

"""
#%% Action

"""
For your homework, save a copy of this file under the filename

    FirstNameLastNameBCCHW.py
    
    
Edit as necessary to complet the requested work.

"""

#%% Libraries and functions

# Not sure which of these will get used

import pandas as pd
import numpy as np
# use seaborn plotting defaults
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


#%% Read data

# Download the data file from the UCI ML repository.  
# Place the file on you computer in a location that this command can read the file

bcc = pd.read_csv("dataR2.csv")

#%% Identify features

# We separate the date into the predictor variables and the response
X=bcc.loc[:,'Age':'MCP.1']
y=bcc.Classification



#%% Build SVM classifier on full dataset

# First, we will build the default svm using all the data and compute the accuracy

from sklearn import svm

mdl1=svm.SVC(gamma='auto')
mdl1.fit(X,y)

# prediction
y_pred=mdl1.predict(X)


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y, y_pred))

#%%  QUESTION

# The indicated accuracy is 100%, which would be great, but ... it is suspect.  
# What is the problem with what we have done?

# Your answer here

#%% Create a training and test set

from sklearn.model_selection import train_test_split

# specify random_state to allow reproducibility
X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.25,random_state=23) 

#%% QUESTION

# Explain what the previous command accomplishes?  

# Can you explain why I chose to specify the "random_state"  ?

# Your answer here

#%% Build default SVM on split set and evaluate accuracy

# ACTION REQUIRED - Modify this code cell so that it trains a default svm using 
# the training data and evaluates the accuracy on the test data

mdl2=svm.SVC()

mdl2.fit(X_train,y_train)

# prediction
y_pred=mdl2.predict(X_test)

# Insert code here
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#%% QUESTION

# Based on your reading of the book and tutorials, it is not reasonable to assume
# that the default SVM is the best.  
#
# What are the key things we should explore in looking for a "good" SVM

# Your answer here


#%% Build using linear kernel

# ACTION REQUIRED - Modify this code cell so that it trains an svm 
#  using a linear kernel but with default values for all other hyperparameter 
# the training data and evaluates the accuracy on the test data

# Insert code lines below
mdl3=svm.SVC(kernel="linear")

mdl3.fit(X_train,y_train)

# prediction
y_pred=mdl3.predict(X_test)

# Insert code here
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))








#%% Build using polynomial kernel

# You can also build with a polynomial kernel.  I think you will find it 
# significantly slower.  

# ACTION Uncomment the code below and run it.

# =============================================================================
mdl4=svm.SVC(kernel='poly')
mdl4.fit(X_train,y_train)

# prediction
y_pred=mdl4.predict(X_test)

print("Default Linear SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
# =============================================================================

#%%  QUESTION

# We are using the raw input values from the provided data.  The book notes that
# as part of feature engineering, we may consider scaling of the variables.

# Do a web search to see if it is important or not to scale variables when using SVM?

# COMMENT HERE as to whether you think we need to scale?


#%% Scaling variables
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler() # instance
scaler.fit(X_train) # fit the scaler to the test data

# scale both training and test data using the SAME SCALER
X_train_sc=scaler.transform(X_train)
X_test_sc=scaler.transform(X_test)

# train and test a linear SVM on scaled data (using default hyperparameters)
mdl5=svm.SVC(kernel='linear')
mdl5.fit(X_train_sc,y_train)

# prediction
y_pred=mdl5.predict(X_test_sc)

print("Scaled Linear SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%% QUESTION

# Did scaling make a difference?

# Your answer here


#%% Other metrics
print("Scaled Linear SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,pos_label=2))
print("Recall:",metrics.recall_score(y_test, y_pred,pos_label=2))
metrics.confusion_matrix(y_test,y_pred)

#%% QUESTION

# Why did I have to set  "pos_label=2" for the Precision and Recall?  
# I didn't have to do that in last week's exercise.

# Insert your answer here.

#%%  QUESTION

# Sometime `accuracy' might be too simple of a measure for us to evaluate 
# and tune performance

# Briefly describe the measures computed in the above cell.

# Insert your response here.


#%% Does parameter C make a difference

# QUESTION - What is the default value for C?

# Insert your answer here.

#%% Experiment with C

# Play around with the value for C and see if you can find a good value 
# that work for this data using linear kernel

mdl5=svm.SVC(kernel='linear',C=.04)  # manipulate this line

# Insert code lines here that will fit, then predict, then evaluate
# use the scaled data

#fit
mdl5.fit(X_train_sc,y_train)


# prediction
y_pred=mdl5.predict(X_test_sc)

# evaluate
print("Scaled Linear SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%% What is a good value for C  for  rbf

# Do the sam experimentation as above, but with the `rbf` kernel
mdl5=svm.SVC(kernel='rbf',C=9)

# Insert code lines here that will fit, then predict, then evaluate
# use the scaled data

#fit
mdl5.fit(X_train_sc,y_train)


# prediction
y_pred=mdl5.predict(X_test_sc)

# evaluate
print("Scaled rbf SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))


#%% What is a good value for C  for  poly 

# NOTE - because we are using the scaled data, you will find that poly is no longer slow

# Do the sam experimentation as above, but with the `rbf` kernel
mdl5=svm.SVC(kernel='poly',C=1.5)

# Insert code lines here that will fit, then predict, then evaluate
# use the scaled data

#fit
mdl5.fit(X_train_sc,y_train)


# prediction
y_pred=mdl5.predict(X_test_sc)

# evaluate
print("Scaled poly SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))


#%% Does parameter gamma make a difference for  rbf

# Experiment to find a good value for gamma.   
# Choose an appropriate value for C for this experiment

mdl5=svm.SVC(kernel='rbf',C=4, gamma=.25)

# Insert code lines here that will fit, then predict, then evaluate
# use the scaled data

#fit
mdl5.fit(X_train_sc,y_train)


# prediction
y_pred=mdl5.predict(X_test_sc)

# evaluate
print("Scaled rbf with C and gamma SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))



#%%  QUESTION - Feature Selection

# Here we will use a bit of a cheat.  
# Rather than us exploring to select a subset of features (as we have not yet
# really developed that methodology)
# Let's borrow from the paper reference on the UCI site.
# Go to the UCI page that talks about this dataset and follow the link to the
# paper written by Patricio
# https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3877-1

# What are the variables it suggests are 

# Insert your answer here



#%% QUESTION Can we do it with fewer predictors

X_train2=X_train_sc[:,[0,1,2,7]]
X_test2=X_test_sc[:,[0,1,2,7]]

# RESPOND - 

# Explain why the subsetting command is consistent with the suggestion from the paper

#%% Analysis

# Using the subsetted variables, experiment to find good values for
# kernel, C, and gamma

# EXPERIMENT
mdl5=svm.SVC(kernel='rbf',C=1, gamma=.1)

mdl5.fit(X_train2,y_train)

# prediction
y_pred=mdl5.predict(X_test2)

print("Scaled Linear SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,pos_label=2))
print("Recall:",metrics.recall_score(y_test, y_pred,pos_label=2))
metrics.confusion_matrix(y_test,y_pred)


#%% REFLECT

# Write a comment about what you have learned.  
