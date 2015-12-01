#!/usr/bin/env python

"""
Author: Dennis/JIAJIE LIANG
Created: 10/30/2015
FileName: Data Analytics_SVM.py
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

# SVM_Ranking_Model_Extraction_And_Encoding: this function is to extract the model and 
# pipeline into SVM Ranking
def SVM_Ranking_Model_Extraction_And_Encoding():

    # Pandas readin Training Samples
    df = pd.read_csv("FeatureToTrainWithoutTester.csv")
    df2 = df.copy()
    df2 = df2.drop(['Dataset Start Time', 'Dataset End Time', 'executionStartTime', 'Dataset Group', 'Users Group'], axis = 1)
    df2.head()

    # Feature Encoding
    transform_features(df2)
    df2.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)
    df2.head()
    
    # Encoded Features
    df = pd.read_csv("Transform_features.csv")

    # Training/Testing DataSet Split 
    df3 = df.copy()
    y = df3['userName']
    df3 = df3.drop(['userName'], axis = 1)
    X = df3
    X_train, X_test, y_train, y_test = X, X, y, y

    # SVM configuration
    parameters={'clf__gamma':(0.01, 0.02, 0.1, 0.3, 1), 'clf__C':(0.1, 0.3, 1, 3, 10, 30), }
    pipeline = Pipeline([('clf', SVC(kernel='rbf', gamma=0.01, C=100, max_iter = 100, probability = True))])
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1, scoring='accuracy')
    result2 = grid_search.fit(X_train, y_train)

    #coef = (result.best_estimator_.get_params()['clf'].coef_)
    #coef2 = coef_sum(coef)
    #coef2
    
    index = ['DatasetName', 'Agency', 'Instrument', 'Physical variable', 'var',
       'Units', 'Grid Dimension', 'Variable Name in Web Interface', 'model']


    # Model Estimation
    model = []

    for i in index:
        # Features' distance/relevant to category prediction
        model.append(feature_training(X_train, y_train, i))


    # Training data distance to single column PCA
    weight_set = numpy.zeros((len(X_train), len(index)))

    for j in range(0, len(X_train)):

        dict_index = 0

        for i in index:

            # Features' distance/relevant to category prediction
            model_extraction = model[dict_index]
            sample = X_train[j:j+1]
            weight = feature_distance(sample, i, model_extraction)
            weight_set[j, dict_index] = weight

            dict_index = dict_index + 1

            print "[INFO] Data Points: ", j, "Columns Iteration: ", dict_index
            print "[INFO] Weight : ", weight

        if j % 100 == 0:
            weight_set_file = pd.DataFrame(weight_set.copy())
            weight_set_file.to_csv("weight_set.csv")


    # Delivery: Training data with Label 
    Training_matrix = pd.DataFrame(weight_set.copy())
    Training_matrix['Label'] = y_train


    # SVM Ranking Formatting
    SVM_Rank_Formatted_Training_data  = Training_matrix.copy()

    for j in range(0, len(X_train)):
        for i in range(0, 9):
            SVM_Rank_Formatted_Training_data.ix[j, i] = str(i + 1) + ":" + str(SVM_Rank_Formatted_Training_data.ix[j, i])
            SVM_Rank_Formatted_Training_data.ix[j, 'Label'] = str(int(SVM_Rank_Formatted_Training_data.ix[j, 9]))

    # Columns Reorder
    Rank_format_columns = SVM_Rank_Formatted_Training_data.columns.tolist()
    Rank_format_columns = Rank_format_columns[-1:] + Rank_format_columns[:-1]
    SVM_Rank_Formatted_Training_data = SVM_Rank_Formatted_Training_data[Rank_format_columns]

    # Write to CSV format
    SVM_Rank_Formatted_Training_data.to_csv("SVM_Rank_Formatted_Training_data2.dat", index = False, sep = ' ', index_label = False, header = False)
    SVM_Rank_Formatted_Training_data.to_csv("SVM_Rank_Formatted_Training_data2.csv")

    predictions = grid_search.predict(X_test)

    # Prediction Results 
    print 'Accuracy:', accuracy_score(y_test, predictions)
    print 'Confusion Matrix:', confusion_matrix(y_test, predictions)
    print 'Classification Report:', classification_report(y_test, predictions)