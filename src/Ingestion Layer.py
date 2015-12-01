#!/usr/bin/env python

"""
Author: Dennis/JIAJIE LIANG
Created: 10/30/2015
FileName: Ingestion Layer.py
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