import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from joblib import parallel_backend
import pickle

def run_model(X_train, X_test, y_train, y_test, model, model_params = None, grid_search = False,
              random_state = 42, scoring = make_scorer(accuracy_score), cv = 3,
              plot_confusion = False, display_report = False, pickle_ = False, pickle_dest = None):
    '''
    Fits a specified model on a train set, returns a fitted model.
    
    X_train: Train set data.
    X_test: Test set data.
    y_train: Train set labels/values.
    y_Test: Test set labels/values.
    model: Unfitted, baseline model with no parameters specified.
    model_params: Optional dictionary specifying the model parameters to set for the model. Default None.
                  If grid_search is True, dictionary represents the param_grid of the GridSearchCV object.
    grid_search: Bool specifying whether to perform a grid search. Default False.
    random_state: Integer setting the random state of the model, if needed. Default 42.
    scoring: String or scoring object. Scoring method to use when performing grid search. By default, use accuracy score.
    cv: Integer representing the number of cross-validation folds to use while grid searching. Default 3.
    plot_confusion: Denotes whether to plot the resultant confusion matrix on the test set. Default False.
    display_report: Denotes whether to print the classification report for the predictions on the test set. Default False.
    pickle_: Denotes whether to pickle the model. Default False.
    pickle_dest: String denoting the pickle destination. Required to pickle the model.
    '''
    
    with parallel_backend('threading', n_jobs = -1):
        # Some models require label encoded targets.
        le = LabelEncoder()
        y_train_transformed = le.fit_transform(y_train)
        y_test_transformed = le.transform(y_test)
        
        # If grid search, perform a grid search across the input param_grid.
        # Else, instantiate the model and set parameters. Add random state if needed.
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
        
        # Fit the model.
        classifier.fit(X_train, y_train_transformed)
        
        # If predictions or scoring are needed, create predictions and scores, print score.
        if any([plot_confusion, display_report]):
            y_pred = classifier.predict(X_test)
            y_score = classifier.score(X_test, y_test_transformed)
            print('Score: {}'.format(y_score))
        
        # If true, plot the confusion matrix using the predictions and labels for the test set.
        if plot_confusion:
            ConfusionMatrixDisplay(confusion_matrix(y_test_transformed, y_pred)).plot();
        
        # If true, print the classification report using the predictions and labels for the test set.
        if display_report:
            print(classification_report(y_test_transformed, y_pred))
        
        # If true, pickle the model.
        if pickle_:
            if pickle_dest is None:
                print('No pickle destination given, pickling skipped.')
            else:
                with open(pickle_dest, 'wb') as f:
                    pickle.dump(classifier, f)
        
    return classifier