#!/usr/bin/env python

"""
Author: Dennis/JIAJIE LIANG
Created: 10/30/2015
FileName: SVM_Code.py
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

# datasetgrouping: a preprocessing function for Dataset Name
# param@x: dataset.csv
def datasetgrouping(x):
    if x.startswith('ARGO') or x.startswith('AMSRE ') or x.startswith('ECMWF') or x.startswith('TOA'):
        return 1
    elif x.startswith('MODIS T') or x.startswith('TRMM P') or x.startswith('AMSR-E') or x.startswith('CERES'):
        return 2
    elif x.startswith('CMIP') or x.startswith('E2H') or x.startswith('ESM') or x.startswith('E2R'):
        return 3
    elif x.startswith('MODIS L') or x.startswith('AIR'):
        return 4
    elif x.startswith('GRACE'):
        return 5
    else:
        return x

# usergrouping: a preprocessing function for user group
def usergrouping(x):
    if x.startswith('abeatriz') or x.startswith('mroge') or x.startswith('cmartinezvi') or x.startswith('caquilinger') or x.startswith('jgristey') or x.startswith('gmarques'):
        return 1
    elif x.startswith('ksauter') or x.startswith('mclavner') or x.startswith('ochimborazo') or x.startswith('kzhang') or x.startswith('emaroon') or x.startswith('mlinz'):
        return 2
    elif x.startswith('fcannon') or x.startswith('dzermenodia') or x.startswith('jbrodie') or x.startswith('amerrifield') or x.startswith('fpolverari') or x.startswith('hwei'):
        return 3
    elif x.startswith('kwillmot') or x.startswith('nkille') or x.startswith('rbuchholz'):
        return 4
    elif x.startswith('htseng') or x.startswith('kneff') or x.startswith('jnanteza'):
        return 5
    else:
        return x

# user deleting: deleting admin/testing user
def deleteTester(x):
    if x == 'czhai' or x == 'lei' or x == 'admin' or x == 'btang' or x == 'jteixeira' or x == 'Unknown':
        return True
    return False

def TF_IDF():
	

# main function: proof of concepts on SVM training/testing models
# It takes in 3 given dataset (provided by Roy) and pre-processing those dataset
def main():

    # Data Pre-Processing: Join the username table and service log table
    df1 = pd.read_csv("NewForm1.csv")
    df2 = pd.read_csv("serviceExecutionLog_dataset2.csv")
    df3 = pd.merge(df1, df2, on = ['userName', 'executionStartTime'], how = 'left')

    # Uppercase transformation
    df3['model'] = df3['model'].map(str.upper)

    # Write out to csv file
    df3.to_csv("NewForm1WithExecutionTime.csv")

    # Data Pre-Processing: Join the Climate Dataset table to feature to train
    df4 = pd.read_csv("/Users/dennis/Documents/SVM-Tasks/Climate_Datasets.csv")

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
    pipeline = Pipeline([('vect', TfidfVectorizer(stop_words = 'english', lowercase = False)), ('clf', SVC(kernel=['rbf', 'linear'], gamma=0.01, C=100, max_iter = 100))])

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


def SVM_Model_Validation():
    # Recommendation for each users



if __name__ == '__main__':
    main()
    SVM_Ranking_Model_Extraction_And_Encoding()
    SVM_Model_Validation()