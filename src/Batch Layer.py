#!/usr/bin/env python

"""
Author: Dennis/JIAJIE LIANG
Created: 10/30/2015
FileName: Batch Layer.py
"""

# sklearn library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn import feature_extraction
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy

# feature_training: Retrieve Modeling per features
# Param@X_data: Training features
# Param@y_data: Label
# Param@feature_name: Feature Name
def feature_training(X_data, y_data, feature_name):

    # Retrieve the features
    X_Train, y_Train = X_data[feature_name].reshape(len(X_data[feature_name]), 1), y_data

    # Configuring the parameters
    parameters={'clf__gamma':(0.01, 0.02, 0.1, 0.3, 1), 'clf__C':(0.1, 0.3, 1, 3, 10, 30), }

    pipeline = Pipeline([('clf', SVC(kernel='linear', gamma=0.01, C=100, max_iter = 10))])

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1, scoring='accuracy')
    
    model = grid_search.fit(X_Train, y_Train)

    return model

# feature_distance: Retrieve distance per model, per feature
# Param@data: Training Sample
# Param@feature_name: model from feature_training
def feature_distance(data, feature_name, model):

    # Weight set for each weights
    result = 0
    data = data[feature_name]
    data = data.reshape(len(data), 1)
    result = model.decision_function(data).mean()

    return result

# one_click_transform: Encode feature
# Param@data: Traninkg Sample
def one_click_transform(data):

    enc = LabelEncoder()
    label_encoder = enc.fit(data[0:])
    float_class = label_encoder.transform(data[0:]).astype(float)
    
    print "[INFO] Transforming Success, Categories Generated "

    return float_class

# transform_features: Retrieve distance per model, per feature
# Param@X_data: Traninkg Sample
# Param@feature_name: model from feature_training
def transform_features(data):
    # Transform all features into numeric number representation
    for key in data.keys():
        print "[INFO] Transforming key: ", key
        data[key] = one_click_transform(data[key])

    print "[INFO] Transforming All Features Complete..."

    return data

# coef_sum: sum up the high plane coefficient
# Param@data: model coefficient
# return: sum of coefficient for each columns
def coef_sum(data):

    result = []

    for index in range(0, 9):
        print coef[index].sum()
        result.append(coef[index].sum())

    return result