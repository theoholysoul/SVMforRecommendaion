#!/usr/bin/env python

"""
Author: Dennis/JIAJIE LIANG
Created: 10/30/2015
FileName: Data Analytics Layer: LogisticRegression.py
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

# main function: proof of concepts on SVM training/testing models
# It takes in 3 given dataset (provided by Roy) and pre-processing those dataset
def LogisticRegression():

    # Data Pre-Processing: Join the username table and service log table
    df1 = pd.read_csv("NewForm1.csv")
    df2 = pd.read_csv("serviceExecutionLog_dataset2.csv")
    df3 = pd.merge(df1, df2, on = ['userName', 'executionStartTime'], how = 'left')

    # Uppercase transformation
    df3['model'] = df3['model'].map(str.upper)

    # Write out to csv file
    df3.to_csv("NewForm1WithExecutionTime.csv")

    # Data Pre-Processing: Join the Climate Dataset table to feature to train
    df4 = pd.read_csv("../storage/Climate_Datasets.csv")

    # Encoding: Grouping    
    df4['Dataset Group'] = df4['Dataset Group'].map(datasetgrouping)

    # Duplicate & Fillna
    df4['userName'] = df4['userName'].fillna('Unknown')
    df4['Users Group'] = df4['userName']
    df4['Users Group'] = df4['Users Group'].map(usergrouping)

    # Write out to FeaturesForTrain.csv
    df4.to_csv("FeaturesForTrain.csv")

    # Training/Testing Data and split  Preparation
    X, y = df4.astype(str).map(str.strip), df4['userName'].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # Pipeline building
    pipeline = Pipeline(['vect', TfidfVectorizer()), ('clf', LogisticRegression())])

    # Check the training data shape
    print X_train.shape

    # parameters setting
    parameters={'clf__gamma':(0.01, 0.02, 0.1, 0.3, 1), 'clf__C':(0.1, 0.3, 1, 3, 10, 30), }

    # training with grid_search: parameters fillin
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')

    # training with grid_search with X_train data
    grid_search.fit(X_train, y_train)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    
    # Predictions
    predictions = grid_search.predict(X_test)
    predictions_probability = grid_search.predict_proba(X_test)

    # Prediction Results 
    print 'Accuracy:', accuracy_score(y_test, predictions)
    print 'Confusion Matrix:', confusion_matrix(y_test, predictions)
    print 'Classification Report:', classification_report(y_test, predictions)