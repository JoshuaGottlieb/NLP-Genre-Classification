import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from joblib import parallel_backend
import pickle

def run_model(X_train, X_test, y_train, y_test, model, model_params = None, grid_search = False,
              random_state = 42, scoring = make_scorer(accuracy_score), cv = 3,
              plot_confusion = False, display_report = False, pickle_ = False, pickle_dest = None):
    
    with parallel_backend('threading', n_jobs = -1):
        le = LabelEncoder()
        y_train_transformed = le.fit_transform(y_train)
        y_test_transformed = le.transform(y_test)
        
        if grid_search:
            classifier = GridSearchCV(estimator = model, param_grid = model_params,
                                 n_jobs = -1, scoring = scoring, cv = cv)
            if 'random_state' in classifier.get_params():
                classifier.set_params(**{'random_state': random_state})
        else:
            classifier = model
            if model_params is not None:
                classifier.set_params(**model_params)
            if 'random_state' in classifier.get_params():
                classifier.set_params(**{'random_state': random_state})
        
        classifier.fit(X_train, y_train_transformed)
                
        if any([plot_confusion, display_report]):
            y_pred = classifier.predict(X_test)
            y_score = classifier.score(X_test, y_test_transformed)
            print('Score: {}'.format(y_score))
        
        if plot_confusion:
            ConfusionMatrixDisplay(confusion_matrix(y_test_transformed, y_pred)).plot();
            
        if display_report:
            print(classification_report(y_test_transformed, y_pred))
            
        if pickle_:
            if pickle_dest is None:
                print('No pickle destination given, pickling skipped.')
            else:
                with open(pickle_dest, 'wb') as f:
                    pickle.dump(classifier, f)
        
    return classifier
    