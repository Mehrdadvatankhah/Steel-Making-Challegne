# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:04:58 2022

@author: m.vatankhah
"""

import pandas as pd
import numpy as np

X_ZOB_SLAB_inf_V = pd.read_csv("DataSet_DataStart-Contest/X_ZOB_SLAB_inf_V.csv")
Y_SLAB_DEF_UNDEF_ISG = pd.read_csv("DataSet_DataStart-Contest/Y_SLAB_DEF_UNDEF_ISG.csv")

X_3d = X_ZOB_SLAB_inf_V.copy()
X_3d = X_3d.drop(columns = 'Unnamed: 0')
X_3d = X_3d.drop(columns = ['SLAB_ID','ZOB_ID'])
X_3d.reset_index(level=0, inplace=True)
X_3d = X_3d.groupby(['CCM_TEC/ISG', 'index']).sum()

# Grouping on ISG - Y**
Y_3d = Y_SLAB_DEF_UNDEF_ISG.copy()
Y_3d = Y_3d.drop(columns = 'Unnamed: 0')
Y_3d = Y_3d.drop(columns = 'SLAB_ID')
Y_3d.reset_index(level=0, inplace=True)
Y_3d = Y_3d.groupby(['CCM_TEC/ISG', 'index']).sum()

X_3d_inf_V_arr = np.array(list(X_ZOB_SLAB_inf_V.groupby('CCM_TEC/ISG').apply(pd.DataFrame.to_numpy)))
Y_3d_arr = np.array(list(Y_3d.groupby('CCM_TEC/ISG').apply(pd.DataFrame.to_numpy)))

Y_3d_arr = np.array(list(Y_3d.groupby('CCM_TEC/ISG').apply(pd.DataFrame.to_numpy)))
pd.set_option("display.max_columns",None)

from numpy.ma.core import shape
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from numpy import arange
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler

result_acc = np.zeros((X_3d_inf_V_arr.shape[0],2,Y_3d_arr[1].shape[1]))
lassomodelFeatures = np.zeros((X_3d_inf_V_arr.shape[0],Y_3d_arr[1].shape[1],68))
for datasetsNo in range(X_3d_inf_V_arr.shape[0]):
    featuresAll = X_3d_inf_V_arr[datasetsNo].copy()
    labelsAll = Y_3d_arr[datasetsNo]
    #labelsAll.reset_index()

    scaler = MinMaxScaler()
    scaler.fit(featuresAll)

    labelsAll[labelsAll > 1] = 1
    count = 0

    #labelColumns = labelsAll.columns
    j =0
    for i in range(Y_3d_arr[datasetsNo].shape[1]):
        #print(labelsAll[i].unique())
        #cnt = labelsAll.loc[labelsAll[i] == 1, i].count()
        cnt = np.count_nonzero(labelsAll[:,i]==1)
        cnt2 = np.count_nonzero(labelsAll[:,i]==0)
        # print('cnt= ', cnt)
        percent = 10
        if labelsAll.shape[0] > 100 and cnt> 0 and cnt2 > 0 and cnt >= cnt2:
            percent = cnt /cnt2
        elif labelsAll.shape[0] > 100 and cnt> 0 and cnt2> 0 and cnt2 < cnt:
            percent = cnt2 /cnt
        if Y_3d_arr[datasetsNo].shape[0] == cnt or percent > 5 :
            #print(i, ' = ',labelsAll.loc[labelsAll[i] == 1, i].count())
            count +=1
            #labelsAll.drop([i], axis = 1, inplace = True)
        else:
            # if cnt < 100:
            #     # define oversampling strategy
            #     oversample = RandomOverSampler(sampling_strategy= 'minority')
            #     X_under, y_under = oversample.fit_resample(featuresAll,labelsAll[:,i])

                #undersample = RandomUnderSampler(sampling_strategy='majority')
                # fit and apply the transform
                #X_under, y_under = undersample.fit_resample(X_over,y_over)
                # summarize class distribution
                #print(i, '1000 = ', X_under.shape)
            #elif cnt < 5000:
                # define oversampling strategy
                #undersample = RandomUnderSampler(sampling_strategy='majority')
                # fit and apply the transform
                #X_under, y_under = undersample.fit_resample(featuresAll,labelsAll[i])

                #oversample = RandomOverSampler(sampling_strategy= 'minority')
                #X_under, y_under = oversample.fit_resample(X_under1, y_under1)

                # summarize class distribution
                #print(i, '5000 = ', X_under.shape)
            # else:
                # define oversampling strategy
            undersample = RandomUnderSampler(sampling_strategy='majority')
            # fit and apply the transform
            X_under, y_under = undersample.fit_resample(featuresAll,labelsAll[:,i])
                # summarize class distribution
                #X_under, _ , y_under , _ = train_test_split(X_under, y_under,test_size = 1-(5000/cnt), random_state=1, stratify= y_under)
                #print(i, 'all= ',X_under.shape)
            #if X_under.shape[0]<200000:
            # define model evaluation method
            cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=1)
            # define model
            lassomodel = LassoCV(alphas= np.logspace(-10, 1, 400), cv=cv, n_jobs=-1)
            #model.
            # fit model
            lassomodel.fit(X_under, y_under)
            # summarize chosen configuration
            #print('alpha: ',np.absolute(model.coef_) > 0)
            #fs = SelectKBest(score_func=f_regression, k=50)
            # apply feature selection
            #X_selected = fs.fit_transform(X_under, y_under)
            #X_selected = X_under.iloc[:,(np.absolute(lassomodel.coef_) > 0).tolist()]
            X_selected = X_under[:,(np.absolute(lassomodel.coef_) > 0)]
            no_of_features = X_selected.shape[1]

            x_train, x_test, y_train , y_test = train_test_split(X_selected, y_under,
                                                                 test_size = 0.2, random_state=1, stratify= y_under)

            #SVC_model = SVC(C=1.0, random_state=1, kernel='linear')

            # train
            #SVC_model.fit(x_train, y_train)

            # predict
            #predictions = SVC_model.predict(x_test)
            #predic = metrics.accuracy_score(y_test,predictions)

            model = MLPClassifier(learning_rate_init=0.005,random_state=1, max_iter=150, early_stopping = True,
                                  hidden_layer_sizes =(no_of_features,no_of_features,no_of_features,))
            model.fit(x_train, y_train)
            pred = model.predict_proba(x_test)
            acc = model.score(x_test, y_test)

            print('accuracy_score= ', acc)
            #result_acc[datasetsNo,i]= [X_selected.columns,no_of_features,acc]
            result_acc[datasetsNo,0,i]= acc
            result_acc[datasetsNo,1,i]= no_of_features
            lassomodelFeatures[datasetsNo,i,:] = lassomodel.coef_.copy()

    print(count) 

