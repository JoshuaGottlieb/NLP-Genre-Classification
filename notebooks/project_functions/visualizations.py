# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from joblib import parallel_backend
import pickle
import re
import dataframe_image as dfi

def encode_target(y):
    le = LabelEncoder()
    return le.fit_transform(y)

def get_scores_as_frame(X_test, y_test, model, model_label):
    score = model.score(X_test, y_test)
    df = pd.DataFrame([[model_label, score]], columns = ['model', 'accuracy'])
    
    return df

def get_all_scores(X_tests, y_test, models, model_labels):
    dfs = [get_scores_as_frame(X_tests[i], y_test, models[i], model_labels[i]) for i in range(len(models))]
    
    score_df = pd.concat(dfs, axis = 0)
    
    return score_df
    

# Configures axis and title labels
def configure_axislabels_and_title(xlabel, ylabel, title, ax):
    # Set fonts, padding, and fontsize for axis labels and title
    ax.set_xlabel(xlabel,
                  fontfamily = 'Arial',
                  fontsize = 24,
                  labelpad = 5)

    ax.set_ylabel(ylabel,
                  fontfamily = 'Arial',
                  fontsize = 24,
                  labelpad = 10)

    ax.set_title(title,
                 fontfamily = 'Arial',
                 fontsize = 32,
                 pad = 10)
    
    return



# Configures ticklabels and tick parameters
def configure_ticklabels_and_params(ax, labelsize = 16):
    # Set label sizes and tick lengths
    ax.tick_params(axis = 'both',
                   which = 'major',
                   labelsize = labelsize,
                   length = 8,
                   width = 1)

    # Set font for tick labels on both axes
    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")

    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        
    return

# Takes in a feature frame and returns a bar plot of the top 10 features by largest absolute coefficient
def plot_feature_importances(feature_frame):
    fig, ax = plt.subplots(figsize = (16, 8))

    plt.tight_layout()
    
    # Subselect the top 10 coefficients by magnitude
    top_10_features = feature_frame.nlargest(10, 'coefficient')

    # Create horizontal seaborn barplot
    sns.barplot(data = top_10_features, x = 'coefficient', y = 'feature',
                orient = 'h', ax = ax);
    
    return fig, ax

# Takes in X-, y-values and a fitted model
# Returns a fig, ax pair containing a seaborn heatmap of the confusion matrix
def plot_confusion_matrix_fancy(X_test, y_test, model):
    matrix = confusion_matrix(y_test, model.predict(X_test))
    
    fig, ax = plt.subplots(figsize = (10, 10))
    
    plt.tight_layout()
    
    # Plot seaborn heatmap, no decimals or scientific notation, no colorbar
    sns.heatmap(matrix, annot = True, fmt = 'g', cmap = 'viridis', linewidths = 1,
                cbar = False, annot_kws = {'fontsize': 24, 'fontfamily': 'Arial'}, ax = ax)
    
    return fig, ax