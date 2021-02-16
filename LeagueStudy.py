#!/usr/bin/env python
# coding: utf-8

# In[33]:



import pandas as pd
import json
import numpy as np
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# from sklearn.externals import joblib
# from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from time import localtime
from datetime import datetime as dt
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from pathlib import Path

working_directory     = str(os.getcwd()) + "/"
# working_directoryPT = str(Path.home()) + "/"
# wd                  = "/Users/Jonat/Dropbox/CreateDawg/ImRight/input/"


# In[34]:


data = pd.read_csv('preproData.csv')
data.drop(['R_damageshare','B_damageshare'],axis=1,inplace=True)
# print(data['R_team'])
list(data.columns)


# In[35]:


from sklearn.model_selection import RepeatedStratifiedKFold
def hyperTune(model,X_Train,Y_Train,X_Test,Y_Test):
    #Hyperparameter find utilizing grid search cross validation 
    C = [1, 10, 100, 1000]
    mItr = [500]
    tol = [.001,.01]
#     class_weight = [{'Blue':0.5, 'Red':0.5}, {'Blue':0.4, 'Red':0.6}, {'Blue':0.6, 'Red':0.4}, {'Blue':0.7, 'Red':0.3},'none']
    solver = ['newton-cg', 'lbfgs','sag','saga']
    param_grid = dict(tol=tol,max_iter=mItr, C=C,  solver=solver)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='roc_auc',verbose=1,n_jobs=-1)
    grid_result = grid.fit(X_Train, Y_Train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#     print("========================================================")
#     print('Train Score: ', grid_result.best_score_)
#     print('Best Train Params: ', grid_result.best_params_)
#     # print('Validation score:',grid.score(X_Val,Y_Val))
#     print('Test score:',grid.score(X_Test,Y_Test))
# #     return grid_result.best_params_
    return grid_result.best_params_ , grid.score(X_Test, Y_Test)
def hyperTuneXGB(model,X_Train,Y_Train,X_Test,Y_Test):
    #Hyperparameter find utilizing grid search cross validation 
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'alpha': [0.0001, 0.05,0.1],
        'max_depth': [3, 4, 5]
    }
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(estimator=model,param_grid=params,cv=5,scoring='roc_auc',verbose=1,n_jobs=-1)
    grid_result = grid.fit(X_Train, Y_Train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_params_ , grid.score(X_Test, Y_Test)

def hyperTuneMLP(model,X_Train,Y_Train,X_Test,Y_Test):
    #Hyperparameter find utilizing grid search cross validation 
    parameter_space = {
    'hidden_layer_sizes': [(72,),(107,),(107,214,107),(107,50,20)],
    'activation': ['tanh', 'relu','logistic'],
    'solver': ['sgd', 'adam','lbfgs'],
    'alpha': [0.0001, 0.05,0.1],
    'learning_rate': ['constant','adaptive','invscaling'],
    }
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(estimator=model,param_grid=parameter_space,cv=5,scoring='roc_auc',verbose=1,n_jobs=-1)
    grid_result = grid.fit(X_Train, Y_Train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#     print("========================================================")
#     print('Train Score: ', grid_result.best_score_)
#     print('Best Train Params: ', grid_result.best_params_)
#     # print('Validation score:',grid.score(X_Val,Y_Val))
#     print('Test score:',grid.score(X_Test,Y_Test))
# #     return grid_result.best_params_
    return grid_result.best_params_ , grid.score(X_Test, Y_Test)
def gnb(X_Train,Y_Train,X_Test,Y_Test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    clf    = GaussianNB()
    clf.fit(X_Train,Y_Train)
    y_pred = clf.predict(X_Test)
    acc    = accuracy_score(Y_Test,y_pred)
    return clf, acc
def mlp(X_Train,Y_Train,X_Test,Y_Test):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
#     {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (72,), 'learning_rate': 'invscaling', 'solver': 'adam'}
#   {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (107, 214, 107), 'learning_rate': 'adaptive', 'solver': 'adam'}
    clf = MLPClassifier(max_iter=10000,activation= 'tanh', alpha= 0.0001,tol=.001, hidden_layer_sizes= (107, 214, 107), learning_rate= 'adaptive', solver= 'adam')
    clf.fit(X_Train,Y_Train)
    y_pred = clf.predict(X_Test)
    acc    = accuracy_score(Y_Test,y_pred)
    return clf, acc
#     clf = MLPClassifier(max_iter=1)

#     hyperparams, bestScore = hyperTuneMLP(clf,X_Train,Y_Train,X_Test,Y_Test)
#     return hyperparams, bestScore
    
def rforest(X_Train,Y_Train,X_Test,Y_Test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_Train,Y_Train)
    y_pred = clf.predict(X_Test)
    acc    = accuracy_score(Y_Test,y_pred)
    return clf, acc
def boosted(X_Train,Y_Train,X_Test,Y_Test):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
#     {'alpha': 0.05, 'colsample_bytree': 1.0, 'gamma': 5, 'max_depth': 3, 'min_child_weight': 10, 'subsample': 1.0}
    boost    = XGBClassifier(booster='gbtree',alpha='.05',colsample_bytree =1.0, gamma = 5, max_depth = 3, min_child_weight = 10, subsample= 1.0)
    boost.fit(X_Train,Y_Train)
    y_pred   = boost.predict(X_Test)
    accuracy = accuracy_score(Y_Test,y_pred)
    return boost, accuracy
#     boost = XGBClassifier()
#     hyperparams, bestScore = hyperTuneXGB(boost,X_Train,Y_Train,X_Test,Y_Test)
#     return hyperparams, bestScore
def logReg(X_Train,Y_Train,X_Test,Y_Test):
    logRegg = LogisticRegression()
#     {'C': 1000, 'max_iter': 500, 'solver': 'newton-cg', 'tol': 0.001}
#    {'C': 1000, 'max_iter': 500, 'solver': 'newton-cg', 'tol': 0.001}
#     logRegg = LogisticRegression(class_weight = 'none', max_iter= 10000, tol = 0.001, solver= 'newton-cg')
#     logRegg.fit(X_Train,Y_Train)
#     # model   = SelectFromModel(logRegg,prefit=True)
#     # x_new   = model.transform(X_Train)
#     # chopp   = model.get_support()
#     # for i in range(len(Headers)):
#     # if(chopp[i]==True):
#     # # print(Headers[i])
#     print("G: "+g+" Score: ",logRegg.score(X_Test,Y_Test))
#     return logRegg, logRegg.score(X_Test,Y_Test)
#     print("================================================================")

    # feature selection

    # get the hyperparameters
#     try:
#         hyperparams, bestScore = hyperTune(logRegg,X_Train,Y_Train,X_Test,Y_Test)
#         return hyperparams, bestScore
#     except ValueError:
#         print('lolwut')
    logRegg = LogisticRegression(C=1000, max_iter= 500, solver= 'newton-cg', tol= 0.001)
    logRegg.fit(X_Train,Y_Train)
    return logRegg, logRegg.score(X_Test,Y_Test)


# In[36]:


# from sklearn.feature_selection import SelectFromModel
# from numpy import sort
# from sklearn.metrics import accuracy_score

# rets = data['res']
# data.drop(['res'],axis=1,inplace=True)

# featSel = VarianceThreshold(.8 * (1 - .8))
# featSel.fit(data.copy())
# dataCom = data.copy()[data.columns[featSel.get_support(indices=True)]]
# dataCom['res'] = rets

# print(data.shape,dataCom.shape,"\n",list( set(list(data.columns)) - set(list(dataCom.columns))  ) )


# In[37]:


from numpy import sort
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train, test = train_test_split(data.copy(),test_size=.2)
# test, val   = train_test_split(test.copy(),test_size=.5)
y_train     = train['res']
y_test      = test['res']
# y_val       = val['res']
train.drop(['res'],axis=1,inplace=True)
test.drop(['res'],axis=1,inplace=True)
# val.drop(['res'],axis=1,inplace=True)

scalers = Normalizer()
train   = scalers.fit_transform(train)
test    = scalers.transform(test)
# val     = scalers.transform(val)
print(train.shape,y_train.shape,test.shape,y_test.shape)

print(logReg(train,y_train,test,y_test))

# print(mlp(train,y_train,test,y_test))
model = boosted(train,y_train,test,y_test)[0]
thresholds = sort(model.feature_importances_)
print(boosted(train,y_train,test,y_test)[1])
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(train)
#     # train model
#     selection_model = XGBClassifier(booster='gbtree',learning_rate=0.1,
#        max_delta_step=0, max_depth=3, 
#        objective='binary:logistic', silent=True)
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(test)
#     predictions = selection_model.predict(select_X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
#     print(rforest(train,y_train,test,y_test)[1])


# In[ ]:





# In[38]:


# %matplotlib inline


labels = list(data.columns[:len(list(data.columns))-1])
x_values = np.arange(1, len(labels) + 1, 1)
plt.bar([i for i in range(len(labels))],model.feature_importances_)
print(boosted(train,y_train,test,y_test)[1])
plt.xticks(x_values,labels)

# load data
# plot feature importance
plot_importance(model)
plt.show()

plt.rcParams["figure.figsize"] = (90,90)
plt.show()
 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




