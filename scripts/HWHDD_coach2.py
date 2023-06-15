# -*- coding: utf-8 -*-
"""
Created on 2 July 2019

@author: Joe Skufca

This file provides a guided effort through the Heart Disease Homework assignment.

The goal of this assignment is to better understand how to tackle some
preprocessing steps, with particular focus on scaling.

Additionally, it intends to explore the use of cross-validation as a tool
for improved accuracy assessments and tuning of parameters.


"""
#%% Action

"""
For your homework, save a copy of this file under the filename

    FirstNameLastNameHDDHW.py
    
    
Edit as necessary to complet the requested work.

"""

#%% libraries and modules that we may need

# standards for data and computation
import pandas as pd
import numpy as np

# use seaborn plotting defaults
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# preprocessing from scikit learn
import sklearn.preprocessing as sklp

# Note: other commands may be loaded as neeeded.


#%% Task 1 - read as data frames
"""
Ensure you have downloaded the data from our course page.  
The data was taken from the UCI Machine Learning Library
Cleveland Heart Disease Data

""" 

dfCleve=pd.read_csv("cleve.txt", header=None,skiprows=20,sep='\s+')
dfProcessed=pd.read_csv("processed.cleveland.data", header=None)

#%%  Task 1a   
""" ACTION REQUIRED: Write code that looks at the first 5 rows of each dataset
to get some notion
that you have correctly loaded the data
""" 

# Insert code below



#%% Task 2a  Convert from dfCleve to dfProcessed

# I will build as individual fixes, with later consideration of a pipeline

# inspect the object before we start
dfCleve.dtypes

"""
ACTION REQUIRED:  Which fields are already numeric?

* Your answer here



"""

#%% Task 2b Buiding the numeric dataset from the cleve.txt
"""
NOTE -  Assignment statements in Python do not copy objects, 
they create bindings between a target and an object. 
A copy is sometimes needed so one can change one copy without changing the other.
"""
dfCP=dfCleve.copy() # create the new object

"""
LabelEncoder 

This object converts categorical to numeric and is a convenient way to convert
most variables to a numeric code.  However, it only assigns numbers in
alphabetical order, starting with 0.

To achieve the same encoding as dfProcessed, only a few of the variables satisfy.  
"""
le=sklp.LabelEncoder() #instantiate the object


dfCP[1]=le.fit_transform(dfCleve[1]) # number assigned is alphabetical
dfCP[5]=le.fit_transform(dfCleve[5])
dfCP[8]=le.fit_transform(dfCleve[8])


"""ACTION REQUIRED: Provide a brief description of what 

             fit_transform() 
             
achieves.

* YOUR ANSWER HERE             
"""


#%% Task 2c Buiding the numeric dataset 
"""
To match our encoding to that in the dfProcessed, we need a specific conversion
from category (text) to number.  

The pandas 'map' function may be the easiest way, converting IAW a 
dictionary which specifies the mapping.

"""

dfCP[2]=dfCleve[2].map({'angina':1,'abnang':2,'notang':3,'asympt':4})

""" ACTION REQUIRED: refer to the first 20 lines of "cleve.txt" to find the
desired mappings, then write code that achieves that mapping for columns

6,10,11,12,14

INSERT CODE BELOW.
"""

#%%  2c continued  

""" Note that dfProcessed does not include a 

"buff/sick" column, so we remove it from dfCP
"""

dfCP.drop(columns=13,inplace=True)

"""
ACTION REQUIRED:  Why do we need the option "inplace=True"?

* YOUR ANSWER HERE

"""

#%%  NO ACTION REQUIRED - INFORMATIONAL

"""
You should inspect to feel comfortable that you have properly encoded to 
achieve the same result in dfCP as dfProcessed.

NOTE - They will not match exactly.   The row ordering in the dataframes will
be slightly different.

"""

#%% Task 3 incorporating one-hot encoding

"""
As we have discussed - one-hot encoding is often an "obvious" need.  
Consequently, pandas includes a dataframe method to covert categrorical variables
to one-hot encoded dummy variables.
"""

dfOnehot=pd.get_dummies(dfCleve.dropna())

#%% Task 4 Scaling of numeric variables

# let's use MinMax to scale to unit output using 

scaler=sklp.MinMaxScaler()

scaler.fit(dfOnehot) # fit the scaler to the test data

# Not appropriate to use full scaled dataset
dfScaled=scaler.transform(dfOnehot)

"""
Above, I used the MinMax scaler.  

(a) Identify three other scalers availble in sklearn.prepocessing
(b) Briefly describe the difference between MinMaxScaler and StandardScaler.

** YOUR ANSWER HERE **

"""


#%% Task 5 comparing performance

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score

"""
What metric do you thing is most appropriate for this evaluation?  
Provide a short justification for your choice.

** YOUR ANSWER HERE **

"""


#%% Analysis with no manipulation of the data

""" For consistency - I will use dfCP as the starting point for these
analyses.

Additionally, I will construct a temporary dataframe, df1, to allow
resuse of some of my code lines.
"""

df1=dfCP.dropna()
X=df1.drop(columns=14)
y=(df1[14]>0)

"""
X  is my predictor variables
y  is the response variable - 
"""

mdl1=svm.SVC(C=1,gamma='auto')

scores = cross_val_score(mdl1, X, y, cv=10)
np.mean(scores)

"""
Provide a brief explanation of what is accomplished by the line

    scores = cross_val_score(mdl1, X, y, cv=10)

to include the meaning of the parameter choice  "cv=10"

** YOUR ANSWER HERE **

"""

#%% (Scenario) Evaluating performance with just one-hot  encoding

df1=dfOnehot.copy()
X=df1.loc[:,:'12_rev']
y=(df1['13_sick']>0)

"""
My command for creating predictor variable `X' and response variable 'y'
is somewhat different.

Justify/explain what I am doing.

** YOUR ANSWER HERE **

"""

#%% One-hot (continued)

mdl1=svm.SVC(C=1,gamma='auto')

scores = cross_val_score(mdl1, X, y, cv=10)

""" Add a line of code so that you can see the resultant mean score.
"""

# INSERT CODE HERE


#%% onehot and scaling
df1=scaler.transform(dfOnehot.copy()) # operate on a copy, not original data


# EDIT THE CODE BELOW
X=df1[:,????]
y=(df1[:,????]>0)

""" Correct the code above so that it selects the appropriate columns """



mdl1=svm.SVC(C=1,gamma='auto')

scores = cross_val_score(mdl1, X, y, cv=10)
np.mean(scores)


#%% (Scenario)  Just scaling, no one-hot

df1=dfCP.dropna()

#  EDIT CODE BELOW TO PROPERLY SELECT PREDICTOR AND RESPONSE
X=???
y=???

scaler2=sklp.MinMaxScaler()

scaler2.fit(X) # fit the scaler to the test data

Xs=scaler2.transform(X)

"""
Add code below to evaluate performace using the Xs data as predictor
"""

# INSERT CODE HERE


#%% Parameter tuning

"""
In the various code sections above, I wrote the code as if to use default
values of parameters.  

Explain why it would be appropriate to tune the parameter for each dataset
before making accuracy comparisons.

** YOUR ANSWER HERE.


"""

#%% onehot and scalinG AND full paramter tune

"""
You may have done some manual tuning of parameter, but let's explore
the built in capability to find good parameters.
"""

from sklearn.model_selection import GridSearchCV

df1=scaler.transform(dfOnehot.copy())
X=df1[:,0:30]
y=(df1[:,31]>0)

Cs = [0.001, 0.01, 0.1, 1, 10,100]
gammas = [0.001, 0.01,.05, 0.1, .15,  1]
param_grid = {'C': Cs, 'gamma' : gammas}

my_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=10)
my_search.fit(X, y)
print(my_search.best_params_)
print(my_search.best_score_)

"""
Modify Cs and gammas to identify what you think are good choices for these
parameters.

What was the best choice that you found?

** INSERT ANSWER HERE. **


"""
