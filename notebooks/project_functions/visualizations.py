# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
# import dataframe_image as dfi

def encode_target(y):
    '''
    Helper function that applies a scikit-learn LabelEncoder to a target variable.
    
    y: Series containing values to encode.
    '''
    
    le = LabelEncoder()
    return le.fit_transform(y)

def get_scores_as_frame(X_test, y_test, model, model_label):
    '''
    Takes in a model and calculates the score using the test data. Returns a dataframe with results.
    
    X_test: Input variables to use for scoring. Must be in a format usable by model.score().
    y_test: Ground-truth labels to use for scoring. Must be in a format usable by model.score().
    model: Trained model to use for scoring.
    model_label: String representing model label to use for returning results.
    '''
    
    score = model.score(X_test, y_test)
    df = pd.DataFrame([[model_label, score]], columns = ['model', 'accuracy'])
    
    return df

def get_all_scores(X_tests, y_test, models, model_labels):
    '''
    Takes in a list of encoded test values and trained models, returns a dataframe tabulating scores for all models.
    
    X_tests: A list of objects to be used as X variables for model scoring. Must be in a format usable by model.score().
    y_test: Ground-truth labels to use for scoring. Must be in a format usable by model.score().
    models: A list of trained models to be used for scoring.
    model_labels: A list of strings representing model labels to be used for tabulating results.
    '''
    
    dfs = [get_scores_as_frame(X_tests[i], y_test, models[i], model_labels[i]) for i in range(len(models))]
    
    score_df = pd.concat(dfs, axis = 0)
    
    return score_df
    

def configure_axislabels_and_title(x_label, y_label, title, ax,
                                   axis_size = 24, title_size = 32,
                                   axis_pad = 10, title_pad = 10, font_name = 'Arial'):
    '''
    Configure the axis and title labels of a matplotlib axis object.
    
    x_label: String to use for labeling the x-axis.
    y_label: String to use for labeling the y-axis.
    title: String to use for the title of the ax object.
    ax: The matplotlib axis object to modify.
    axis_size: Float representing the font size of the axis labels. Default 24.0.
    title_size: Float representing the font size of the title. Default 32.0.
    axis_pad: Float representing the padding between graph and axis labels. Default 10.0.
    title_pad: Float representing the padding between graph and title. Default 10.0.
    font_name: String representing the name of the font to use for all labels. Default Arial.
    '''
    
    ax.set_xlabel(x_label, fontfamily = font_name, fontsize = axis_size, labelpad = axis_pad)
    ax.set_ylabel(y_label, fontfamily = font_name, fontsize = axis_size, labelpad = axis_pad)
    ax.set_title(title, fontfamily = font_name, fontsize = title_size, pad = title_pad)
    
    return



# Configures ticklabels and tick parameters
def configure_ticklabels_and_params(ax, label_size = 16, length = 8, width = 1, font_name = 'Arial',
                                    x_label_size = None, x_length = None, x_width = None,
                                    y_label_size = None, y_length = None, y_width = None,
                                    format_xticks = False, x_ticks_rounding = 1, x_ticks_rotation = 0,
                                    format_yticks = False, y_ticks_rounding = 1, y_ticks_rotation = 0):
    '''
    Configures the ticklabels and tick sizes of a matplotlib axis object.
    
    ax: Matplotlib axis object to format.
    label_size: Float, the font size of the major tick labels. Default 16.
    length: Float, the length of the major tick marks. Default 8.
    width: Float, the width of the major tick marks. Default 1.
    font_name: String, the font to use for tick labels. Default Arial.
    x_label_size: Float, the font size of the major tick labels on the x-axis. By default inherits from label_size.
    x_length: Float, the length of the major tick marks on the x-axis. By default inherits from length.
    x_width: Float, the width of the major tick marks on the x-axis. By default inherits from width.
    y_label_size: Float, the font size of the major tick labels on the y-axis. By default inherits from label_size.
    y_length: Float, the length of the major tick marks on the y-axis. By default inherits from length.
    y_width: Float, the width of the major tick marks on the y-axis. By default inherits from width.
    format_xticks: Bool, indicates whether to format numerical x-tick labels. Default False.
    x_ticks_rounding: Integer, the number by which to divide the x-tick labels (e.g. to round to 10s, set to 10). Default 1.
    x_ticks_rotation: Integer, the number in degrees by which to rotate the x-tick labels counter-clockwise. Default 0.
    format_yticks: Bool, indicates whether to format numerical y-tick labels. Default False.
    y_ticks_rounding: Integer, the number by which to divide the y-tick labels (e.g. to round to 10s, set to 10). Default 1.
    y_ticks_rotation: Integer, the number in degrees by which to rotate the y-tick labels counter-clockwise. Default 0.
    '''
    
    # Set individual axis attributes
    if x_label_size is None:
        x_label_size = label_size
    if y_label_size is None:
        y_label_size = label_size
    if x_length is None:
        x_length = length
    if y_length is None:
        y_length = length
    if x_width is None:
        x_width = width
    if y_width is None:
        y_width = width
    
    # Set label sizes and tick lengths
    ax.tick_params(axis = 'x', which = 'major', labelsize = x_label_size, length = x_length, width = x_width)
    ax.tick_params(axis = 'y', which = 'major', labelsize = y_label_size, length = y_length, width = y_width)

    # Set font for tick labels on both axes, format tick labels if numerical, rotate if needed
    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_rotation(x_ticks_rotation)
        # Rounds tick values and adds commas where appropriate
        if format_xticks:
            ax.get_yaxis().set_major_formatter(pltticker.FuncFormatter(lambda x, p: format(int(x / x_ticks_rounding),',')))

    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_rotation(y_ticks_rotation)
        # Rounds tick values and adds commas where appropriate
        if format_yticks:
            ax.get_yaxis().set_major_formatter(pltticker.FuncFormatter(lambda x, p: format(int(x / y_ticks_rounding),',')))
        
    return

def configure_legend(ax, labels = None, loc = 0, bbox_to_anchor = None, labelcolor = 'black', fontsize = 12,
                     frameon = False, facecolor = 'inherit', fancybox = False, framealpha = 0.8,
                     edgecolor = 'inherit', borderpad = 0.4, labelspacing = 0.5):
    '''
    Configures the legend of a matplotlib axis object.
    
    ax: Matplotlib axis object to format.
    labels: List of strings, to manually specify the labels to use for each artist. Default None uses automatic labeling.
    loc: Integer, string, or pair of floats, specifies the location of the legend according to matplotlib documentation.
         Default 0, equivalent to 'best'.
    bbox_to_anchor: 2-tuple or 4-tuple of floats, specifies the location of the bounding box of the legend. Default None.
    labelcolor: String or list of strings, specifies the color(s) of the labels. Default 'black'.
    fontsize: Integer, specifies the fontsize of the labels. Default 12.
    frameon: Bool, denotes whether the legend should be drawn on a patch (frame). Default False.
    facecolor: 'inherit' or color, denotes the background color of the legend. If 'inherit', match the color of the axis.
               Default 'inherit'.
    fancybox: Bool, denotes whether the legend's background should have rounded edges. Default False.
    framealpha: Float, denotes the alpha transparency of the legend's background. Default 0.8.
    edgecolor: 'inherit' or color, denotes the legend's background edge color. If 'inherit', match the color of the axis.
               Default 'inherit'.
    borderpad: Float, denotes the fractional whitespace inside the legend border, in font-size units. Default 0.4.
    labelspacing: Float, denotes the vertical space between the legend entries, in font-size units. Default 0.5.
    '''
    
    # Extract handles for artists
    handles = [i for i in ax.get_legend_handles_labels()][0]
    
    # If there is a bounding box, anchor to bounding box. Else draw legend using input parameters.
    if bbox_to_anchor is not None:
        ax.legend(handles = handles, labels = labels, loc = loc, bbox_to_anchor = None, labelcolor = labelcolor,
                  frameon = frameon, facecolor = facecolor, fontsize = fontsize, fancybox = fancybox,
                  framealpha = framealpha, borderpad = borderpad, labelspacing = labelspacing);
    else:
        ax.legend(handles = handles, labels = labels, loc = loc, labelcolor = labelcolor,
                  frameon = frameon, facecolor = facecolor, fontsize = fontsize, fancybox = fancybox,
                  framealpha = framealpha, borderpad = borderpad, labelspacing = labelspacing);
    
    return


def plot_confusion_matrix_fancy(X_test, y_test, model, figsize = (10,10)):
    '''
    Takes in test data and a fitted model, returns a matplotlib fig, ax object pair
    containing a seaborn heatmap of the confusion matrix results.
    
    X_test: Test data to use for predicting. Must be in a format usable by model.predict().
    y_test: Ground-truth labels for the test data.
    model: Fitted model to use for predictions.
    figsize: Tuple, size of the figure object to create in inches (horizontal, vertical). Default (10, 10).
    '''
    
    # Create the confusion matrix array
    matrix = confusion_matrix(y_test, model.predict(X_test))
    
    fig, ax = plt.subplots(figsize = figsize)
    
    plt.tight_layout()
    
    # Plot seaborn heatmap, no decimals or scientific notation, no colorbar
    sns.heatmap(matrix, annot = True, fmt = 'g', cmap = 'viridis', linewidths = 1,
                cbar = False, annot_kws = {'fontsize': 24, 'fontfamily': 'Arial'}, ax = ax)
    
    return fig, ax