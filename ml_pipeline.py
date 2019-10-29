#!/usr/bin/python
#-*- coding:utf-8 -*-#

import os
import random
from datetime import date, datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import sklearn
import xgboost as xgb
from feature_selector import FeatureSelector
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, MinMaxScaler,
                                   Normalizer, OneHotEncoder, StandardScaler)
from sklearn.svm import SVC
from sklearn_pandas import DataFrameMapper, cross_val_score
from xgboost import XGBClassifier

import pandas_profiling

np.random.seed(0)

#data load.
train = pd.read_csv('titanic3.csv')

#data eda.
profile = train.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file="report/data_eda_output.html")

#train test split.
train_labels = train['survived']
train = train.drop(columns = ['survived'])

X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.2)

#data preprocess.transform
maxunique = 1024
unique_stat = X_train.nunique()
numeric_cols = [i for i in X_train.columns if is_numeric_dtype(X_train[i])]
categorical_cols = [i for i in X_train.columns if i not in numeric_cols and unique_stat[i] < maxunique]

mapper = DataFrameMapper([
    (categorical_cols , [ SimpleImputer(strategy='constant', fill_value='missing'), OneHotEncoder(handle_unknown='ignore') ]),
    (numeric_cols     , [ SimpleImputer(strategy='median') , StandardScaler() ]),
 ] , df_out = True )

X_train_df = mapper.fit_transform(X_train.copy())
X_test_df = mapper.transform(X_test.copy())

#feature selection.
fs = FeatureSelector(data = X_train_df, labels = y_train)
fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'classification', 'eval_metric': 'auc', 
                                    'cumulative_importance': 0.99})
fs.feature_importances.head()

train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)
test_removed_all_once = X_test_df[train_removed_all_once.columns]

#model train pipeline.
xgb_param = {
             'eta':0.5
           , 'silent':0
           , 'objective':'binary:logistic'
           , 'booster' :'gbtree'
           , 'gamma':0.0001 
           , 'min_child_weight':20
           , 'subsample':0.8
           , 'colsample_bytree':0.8
           , 'eval_metric':'auc'
           , 'scale_pos_weight':1
           , 'eval_train':1
    }
clf = Pipeline(steps=[ 
                ('classifier', XGBClassifier(xgb_param) ) 
             ])

param_grid = {
    'classifier__max_depth': [4,5,6] 
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False , scoring='roc_auc')
grid_search.fit(train_removed_all_once, y_train)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(test_removed_all_once, y_test)))


#model persistant.
#from joblib import dump, load
#mapper_filename = 'tmp/mapper.joblib'
#dump(mapper, mapper_filename) 
#
#mapper_loaded = load(mapper_filename)
#
#clf_filename = 'tmp/clf.joblib'
#dump(grid_search , clf_filename)
#
#gcv_loaded = load(clf_filename)
