import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import logging
logging.basicConfig(format='[%(asctime)s] %(message)s', filename='test_results.log', level=logging.INFO)
logging.info("####################################")
logging.info("START")
location = 'C:\\Users\\greg\\Desktop\\workspace\\dip\\oasis_cross-sectional.csv'
df=pd.read_csv(location)



# Cleaning the data

df = df.drop('Delay',1)  # delete a column
df = df.drop('Hand',1)
df = df[np.isfinite(df['SES'])]  # delete row in "SES" with NaN
df = df[np.isfinite(df['Educ'])]
df = df[np.isfinite(df['MMSE'])]
df = df[np.isfinite(df['CDR'])]

# Normalization

def calculate_intervals(col,n):
    mean_values = []
    intervals = []
    max_val = col.max()
    min_val = col.min()
    interval = (max_val - min_val)/ n
    for i in range(0,n):
        interval_min = min_val + i * interval # min of every interval
        interval_max = min_val+(i+1) * interval # max --//--
        mean_val = (interval_min + interval_max) / 2
        mean_values.append(mean_val)
        intervals.append((interval_min, interval_max))
    print(mean_values)
    return mean_values, intervals


def norma(col,intervals,mean_values):
    for i in col:
        for j,k in enumerate(intervals): # j is a counter, k is "intervals[j]
            if i >= k[0] and i < k[1]:
                col.replace(i, mean_values[j], inplace = True) # True -> replaces and saves new values
                break
    # print(col.head())


mean_values, intervals = calculate_intervals(df["ASF"], 10)
norma(df["ASF"], intervals, mean_values )
mean_values, intervals = calculate_intervals(df["nWBV"],10)
norma(df["nWBV"],intervals, mean_values)

# Change M/F to 0/1

df['M/F'].replace('M', 0, inplace = True)
df['M/F'].replace('F', 1, inplace = True)

df['CDR'].replace(0.0, 'a', inplace = True)
df['CDR'].replace(0.5, 'b', inplace = True)
df['CDR'].replace(1.0, 'c', inplace = True)
df['CDR'].replace(2.0, 'd', inplace = True)
print(df.head())

########################################

# shape (dimensions of data)
print(df.shape)

# descriptions
print(df.describe())

# class distribution
print(df.groupby('CDR').size())

# box and whisker plots
# df.plot(kind = 'box', subplots = True, layout = (2,5), sharex = False, sharey = False)
# plt.show()

#histograms
# df.hist()
# plt.show()

#scater plot matrix
# scatter_matrix(df)
# plt.show()

##################################

# Split-out validation dataset
array = df.values
X = array[:,[4,5,7]]
logging.info("col: 4,5,7")
Y = array[:,6]
# validation_size = 0.20
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
# # print(X)
# #Test options and evaluation metric
scoring = 'accuracy'
seed = 7

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))



#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)# KFold-> sklearn class.10fold val. test is a tenth of the dataset.
    cv_results = model_selection.cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    logging.info(msg)

logging.info("END")
############################

