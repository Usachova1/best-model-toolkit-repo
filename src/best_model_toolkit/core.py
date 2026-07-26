# -*- coding: utf-8 -*-
"""
Core functions for training, tuning and comparing scikit-learn / XGBoost
regression and classification models, with built-in plotting of
performance metrics (R^2, MAE, RMSE, F1, precision, recall, PR curves, etc).

This module was originally written as a Google Colab notebook and has been
packaged for reuse via pip.
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import *
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

"""
Functions for testing and evaluating regression models, as well as for plotting the performance metrics such as R-squared, MAE, and RMSE, using Matplotlib """


def test_model_reg(model, model_name, x_train, y_train, x_test, y_test):
    """
    Args:
        model: a machine learning model to be tested
        model_name:a string containing the name of the model
        x_train: a table of features for the training set
        y_train: a table of labels for the training set
        x_test: a table of features for the test set
        y_test: a table of labels for the test set

    Returns: dictionary of scores
    """
    # Train the model using the training data
    model.fit(x_train, y_train)

    # Use the trained model to make predictions on the testing and training data
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    # Compute R^2 scores for both training and testing data
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Compute mean absolute error (MAE) scores for both training and testing data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Compute root mean squared error (RMSE) scores for both training and testing data
    rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))

    # Store the scores in a dictionary
    scores = {
        'r2': {'train': r2_train, 'val': r2_test},
        'rmse': {'train': rmse_train, 'val': rmse_test},
        'mae': {'train': mae_train, 'val': mae_test}
    }

    # Plot the predicted vs. ground truth values for the testing data
    plt.scatter(y_test, y_test_pred, s=5)
    plt.xlabel('Original Data')
    plt.ylabel('Predicted')
    plt.title(model_name)
    plt.show()
    plt.close()

    return scores


def plot_r2(model_names, model_scores_t, model_scores_v):
    """
    Args:
        model_names: the names of the models being compared
        model_scores_t: the training scores for each model
        model_scores_v: the validation scores for each model

    Returns: The function does not have a Returns section, so it does not return anything
    """
    # Plot the R^2 scores for the training and validation data for each model
    plt.plot(model_names, model_scores_t, '*', label='training')
    plt.plot(model_names, model_scores_v, '*', label='validation')
    plt.ylabel(r'$R^2$')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-0.1, 1)
    plt.legend()
    plt.show()
    plt.close()


def plot_score_reg(model_names, model_scores_t, model_scores_v):
    """
    Args:
        model_names: names of the models being compared
        model_scores_t: the training scores for each model
        model_scores_v: the validation scores for each model

    Returns: The function does not have a Returns section, so it does not return anything
    """
    # Plot a generic score (e.g. R^2, accuracy) for the training and validation data for each model
    plt.plot(model_names, model_scores_t, '*', label='training')
    plt.plot(model_names, model_scores_v, '*', label='validation')
    plt.ylabel('score')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()
    plt.close()


def plot_mae(model_names, model_scores_t, model_scores_v):
    """
    Args:
        model_names: the names of the models being compared
        model_scores_t: the training scores for each model
        model_scores_v: the validation scores for each model

    Returns: The function does not have a Returns section, so it does not return anything
    """
    # Plot the mean absolute error (MAE) scores for the training and validation data for each model

    plt.plot(model_names, model_scores_t, '*', label='training')
    plt.plot(model_names, model_scores_v, '*', label='validation')
    plt.ylabel('MAE')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()
    plt.close()


def plot_rmse(model_names, model_scores_t, model_scores_v):
    """
    Args:
        model_names: the names of the models being compared
        model_scores_t: the training scores for each model
        model_scores_v: the validation scores for each model

    Returns: The function does not have a Returns section, so it does not return anything
    """
    # Plot the root mean squared error (RMSE) scores for the training and validation data for each model

    plt.plot(model_names, model_scores_t, '*', label='training')
    plt.plot(model_names, model_scores_v, '*', label='validation')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()
    plt.close()


def test_model_class(model, model_name, x_train, y_train, x_test, y_test):
    """
    Args:
        model: the machine learning model to be tested
        model_name: a string indicating the name of the model
        x_train:  the training data features
        y_train: the training data target variable
        x_test: the test data features
        y_test:  the test data target variable

    Returns: The function returns a dictionary containing various scores of the model, such as its accuracy, F1 score, precision, recall, and average precision score. If the model has a predict_proba method, the function also calculates the precision-recall curve.
    """

    model.fit(x_train, y_train)

    # Use the trained model to make predictions on the training and validation data
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    # Calculate the score of the model on the training and validation data
    score_train = model.score(x_train, y_train)
    score_test = model.score(x_test, y_test)

    # Calculate the F1 score of the model on the training and validation data
    f1_test = f1_score(y_test, y_test_pred)  # , average='micro')
    f1_train = f1_score(y_train, y_train_pred)  # , average='micro')

    # Calculate the precision of the model on the training and validation data
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)  # , average='micro')
    precision_train = precision_score(y_train, y_train_pred, zero_division=0)  # , average='micro')

    # Calculate the recall of the model on the training and validation data
    recall_test = recall_score(y_test, y_test_pred)  # , average='micro')
    recall_train = recall_score(y_train, y_train_pred)  # ,  average='micro')

    # Create a dictionary to store the scores of the model
    scores = {
        'Score': {'train': score_train, 'val': score_test},
        'F1': {'train': f1_train, 'val': f1_test},
        'Precision': {'train': precision_train, 'val': precision_test},
        'Recall': {'train': recall_train, 'val': recall_test},
        'AP': None,
        'PR_curve': None,
    }

    try:

        # If the model has a predict_proba method, use it to make predictions and calculate the average precision score
        y_test_pred_proba = model.predict_proba(x_test)
        y_train_pred_proba = model.predict_proba(x_train)

        # Convert the target values to one-hot encoding format
        ohe = OneHotEncoder(sparse_output=False)
        y_train_oh = ohe.fit_transform(y_train.reshape((-1, 1)))
        y_test_oh = ohe.fit_transform(y_test.reshape((-1, 1)))

        # Calculate the average precision score of the model on the training and validation data
        ap_test = average_precision_score(y_test_oh, y_test_pred_proba)
        ap_train = average_precision_score(y_train_oh, y_train_pred_proba)

        # Calculate the precision-recall curve of the model on the training and validation data
        pr_curve_test = precision_recall_curve(y_test, y_test_pred_proba[:, -1])
        pr_curve_train = precision_recall_curve(y_train, y_train_pred_proba[:, -1])

        # Add the average precision score and the precision-recall curve to the scores dictionary
        scores['AP'] = {'train': ap_train, 'val': ap_test}
        scores['PR_curve'] = {'train': pr_curve_train, 'val': pr_curve_test}
    except AttributeError as e:
        # If the model does not have a predict_proba method, print an error message
        print(e)

    # Display a confusion matrix for the model's predictions on the validation data
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, normalize='true')
    plt.title(model_name)
    plt.show()
    plt.close()

    return scores


def plot_score_generic(model_names, model_scores_t, model_scores_v, score_name):
    """
    Args:
       model_names: the names of the models being compared
       model_scores_t: the training scores for each model
       model_scores_v: the validation scores for each model

   Returns: The function does not have a Returns section, so it does not return anything
   """

    plt.plot(model_names, model_scores_t, '*', label='training')

    # Plot the validation scores
    plt.plot(model_names, model_scores_v, '*', label='validation')

    plt.ylabel(score_name)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()
    plt.close()


# A convenience function to plot scores with the default name "score"
def plot_score(model_names, model_scores_t, model_scores_v):
    """
    Args:
        model_names: the names of the models being compared
        model_scores_t: the training scores for each model
        model_scores_v: the validation scores for each model

    Returns: The function does not have a Returns section, so it does not return anything
    """

    plot_score_generic(model_names, model_scores_t, model_scores_v, score_name='score')


def plot_f1_score(model_names, model_scores_t, model_scores_v):
    """
    Args:
        model_names: the names of the models being compared
        model_scores_t: the training scores for each model
        model_scores_v: the validation scores for each model

    Returns: The function does not have a Returns section, so it does not return anything
    """

    plot_score_generic(model_names, model_scores_t, model_scores_v, score_name='f1')


# A convenience  function to plot precision scores
def plot_score_precision(model_names, model_scores_t, model_scores_v):
    plot_score_generic(model_names, model_scores_t, model_scores_v, score_name='precision')


# A convenience function to plot recall scores
def plot_score_recall(model_names, model_scores_t, model_scores_v):
    """
    Args:
        model_names: the names of the models being compared
        model_scores_t: the training scores for each model
        model_scores_v: the validation scores for each model

    Returns: The function does not have a Returns section, so it does not return anything
    """

    plot_score_generic(model_names, model_scores_t, model_scores_v, score_name='recall')


# A convenience function to plot average precision scores
def plot_score_ap(model_names, model_scores_t, model_scores_v):
    """
    Plots the recall scores for different models on both training and validation sets.

    Args:
        model_names (list): The names of the models being compared (e.g., list of strings).
        model_scores_t (list): The recall scores for each model on the training set (e.g., list of floats).
        model_scores_v (list): The recall scores for each model on the validation set (e.g., list of floats).

    Returns:
        None
    """

    plot_score_generic(model_names, model_scores_t, model_scores_v, score_name='AP')


# A generic function to plot precision-recall curves for different models
def plot_curve_generic(model_names, model_scores_t, model_scores_v, x_score_name, y_score_name):
    """
    Args:
        model_names: the names of the models being compared
        model_scores_t: the precision, recall, and thresholds for each model on the training dataset
        model_scores_v: the precision, recall, and thresholds for each model on the validation dataset
        x_score_name:  a string that represents the name of the score on the x-axis of the plot
        y_score_name:  string that represents the name of the score on the y-axis of the plo

    Returns:
    """
    # Plot the precision-recall curve for each model on the training set

    for mn, (precision, recall, thresholds) in zip(model_names, model_scores_t):
        plt.plot(recall, precision, '--', label=f'{mn} training')
        # Plot the precision-recall curve for each model on the validation set
    for mn, (precision, recall, thresholds) in zip(model_names, model_scores_v):
        plt.plot(recall, precision, '-', label=f'{mn} validation')

    plt.ylabel(y_score_name)
    plt.xlabel(x_score_name)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.close()




def grid_reg(model, params, features, target):
    """
    Performs grid search for hyperparameter tuning of the model.

    Args:
        model (object): The model object you want to tune.
        params (list or dict): A dictionary or list of dictionaries containing hyperparameters to be tested.
                              If params is a list, grid search is performed for each dictionary in the list.
        features (array-like): The input features used for training the model.
        target (array-like): The target variable used for training the model.

    Returns:
        model_grid (object): The tuned model object after performing grid search.
        grid_results (DataFrame): A pandas DataFrame containing the results of grid search.
        pairs (dict): A dictionary containing the mapping of parameter names to their string representations.
    """

    # Define scoring metrics
    global model_grid
    scoring = {'r2': make_scorer(r2_score),
               'rmse': make_scorer(mean_squared_error),
               'mae': make_scorer(mean_absolute_error)
               }
    
   # If the parameters are a dictionary, we can perform a single grid search
    model_grid = GridSearchCV(model, params, scoring=scoring, refit='r2')
    model_grid.fit(features, target)
    grid_results = pd.DataFrame(model_grid.cv_results_)

    # Add a column for the root mean squared error
    grid_results['root_mean_test_mse'] = list(map(lambda n: sqrt(n), grid_results['mean_test_rmse']))

    # Convert the parameter dictionaries to strings and create a list of parameter names
    params_list = grid_results['params']
    our_list = []
    features_list = list(set().union(*(d.keys() for d in params_list)))
    pairs = {}

    # Loop through each parameter dictionary and create a string representation
    for params_dict in params_list:
        our_str = ""
        for i, k in enumerate(features_list):
            if params_dict.get(k) is not None:
                our_str += chr(i + 65) + ':' + str(params_dict[k]) + " "
                pairs[k] = chr(i + 65)

        our_list.append(our_str.strip())

    # Replace the parameter dictionaries with their string representation
    grid_results['params'] = our_list
    our_list = grid_results['params']

    return model_grid, grid_results, pairs


def plot_cv_r2(model_name, model_params, model_r2, pairs):
    """
    Plots the cross-validation R2 scores of a model across different hyperparameters.

    Args:
        model_name (str): Name of the model for the plot title.
        model_params (list): List of parameter names used as x-axis ticks.
        model_r2 (array-like): Array-like object containing R2 scores for each combination of hyperparameters.
        pairs (dict): A dictionary containing the mapping of parameter names to their string representations.

    Returns:
        None
    """
    # Plots the cross-validation R2 scores of a model across different hyperparameters
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 10)
    plt.plot(model_params, model_r2, label='r2', color='blue')
    plt.xticks(rotation=90, ha='right')
    plt.title(model_name)

    # Create list of parameter labels from pairs dictionary
    parameter_labels = []
    for k, v in pairs.items():
        parameter_labels.append(f'{k}={v}')

    # Create text from parameter labels and add it to plot
    text = '\n'.join(parameter_labels)
    plt.legend()
    ax.text(1.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top')
    plt.show()
    plt.close()


def plot_mae_mse(model_name, model_params, model_mse, model_mae, pairs):
    """
        Plots the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of a model across different hyperparameters.

        Args:
            model_name (str): Name of the model for the plot title.
            model_params (list): List of parameter names used as x-axis ticks.
            model_mse (array-like): Array-like object containing MSE values for each combination of hyperparameters.
            model_mae (array-like): Array-like object containing MAE values for each combination of hyperparameters.
            pairs (dict): A dictionary containing the mapping of parameter names to their string representations.

        Returns:
            None
        """

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Set the figure size
    fig.set_size_inches(30, 10)

    # Plot the mean absolute error and root mean squared error
    #plt.plot(model_params, model_mae, label='mae')
    plt.plot(model_params, model_mse, label='rmse')

    # Rotate the x-axis tick labels by 90 degrees and align them to the right
    plt.xticks(rotation=90, ha='right')

    # Set the title of the plot to the model name
    plt.title(model_name)

    # Create a list of parameter labels based on the key-value pairs in the 'pairs' dictionary
    parameter_labels = []
    for k, v in pairs.items():
        parameter_labels.append(f'{k}={v}')

    # Join the parameter labels with newline characters and assign to the 'text' variable
    text = '\n'.join(parameter_labels)

    # Add a legend to the plot
    plt.legend()

    # Add the parameter labels to the plot as text in the upper-right corner
    ax.text(1.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top')

    plt.show()
    plt.close()


def grid_class(model, params, features, target):
    """
    Performs grid search for hyperparameter tuning of a classification model.

    Args:
        model (object): The classification model object you want to tune.
        params (list or dict): A dictionary or list of dictionaries containing hyperparameters to be tested.
                              If params is a list, grid search is performed for each dictionary in the list.
        features (array-like): The input features used for training the classification model.
        target (array-like): The target variable used for training the classification model.

    Returns:
        model_grid (object): The tuned classification model object after performing grid search.
        df_grid (DataFrame): A pandas DataFrame containing the results of the grid search.
        pairs (dict): A dictionary containing the mapping of feature names to their string representations.
    """

    # Define the scoring metrics for the grid search
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, zero_division=0, average='macro'),
               'recall': make_scorer(recall_score, average='macro'),
               'f1': make_scorer(f1_score, average='macro')}

    
    model_grid: GridSearchCV = GridSearchCV(model, params, scoring=scoring, refit='f1')
    model_grid.fit(features, target)
    df_grid = pd.DataFrame(model_grid.cv_results_)

    # Convert the parameter dictionaries to strings and replace them in the dataframe
    params_list = df_grid['params']
    our_list = []
    features_list = list(set().union(*(d.keys() for d in params_list)))
    pairs = {}

    for params_dict in params_list:

        our_str = ""
        for i, k in enumerate(features_list):
            if params_dict.get(k) is not None:
                our_str += chr(i + 65) + ':' + str(
                    params_dict[k]) + " "  # Convert the parameter dictionaries to strings
                pairs[k] = chr(i + 65)  # Store the corresponding character label for each feature

        our_list.append(our_str.strip())

    df_grid['params'] = our_list  # Replace the parameter dictionaries with the strings in the dataframe
    our_list = df_grid['params']

    return model_grid, df_grid, pairs


def plot_cv_metrics(model_name, model_params, model_f1, model_recall, model_precision, model_accuracy, pairs):
    """
    Plots various evaluation metrics for a model across different hyperparameters.

    Args:
        model_name (str): The name of the model for the plot title.
        model_params (list): List of parameter names used as x-axis ticks.
        model_f1 (array-like): Array-like object containing F1 scores for each combination of hyperparameters.
        model_recall (array-like): Array-like object containing recall scores for each combination of hyperparameters.
        model_precision (array-like): Array-like object containing precision scores for each combination of hyperparameters.
        model_accuracy (array-like): Array-like object containing accuracy scores for each combination of hyperparameters.
        pairs (dict): A dictionary containing the mapping of parameter names to their string representations.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 10)

    # Plot the F1 score, recall, precision, and accuracy
    plt.plot(model_params, model_f1, label='f1')
    plt.plot(model_params, model_recall, label='recall')
    plt.plot(model_params, model_precision, label='precision')
    plt.plot(model_params, model_accuracy, label='accuracy')

    # Rotate the x-axis tick labels by 90 degrees and align them to the right
    plt.xticks(rotation=90, ha='right')

    # Set the title of the plot to the model name
    plt.title(model_name)

    # Create a list of parameter labels based on the key-value pairs in the 'pairs' dictionary
    parameter_labels = []
    for k, v in pairs.items():
        parameter_labels.append(f'{k}={v}')

    # Join the parameter labels with newline characters and assign to the 'text' variable
    text = '\n'.join(parameter_labels)

    # Add a legend to the plot
    plt.legend()

    # Add the parameter labels to the plot as text in the upper-right corner
    ax.text(1.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top')

    plt.show()
    plt.close()


def corr_reg(x, y, **kws):
    """
    Calculate the Pearson correlation coefficient between x and y and add an annotation to the plot.

    Args:
        x (array-like): The x-axis data.
        y (array-like): The y-axis data.
        **kws: Additional keyword arguments.

    Returns:
        None
    """
    # Calculate the Pearson correlation coefficient between x and y
    r, _ = stats.pearsonr(x, y)

    # Get the current axes instance
    ax = plt.gca()

    # Add an annotation to the plot showing the correlation coefficient
    ax.annotate("r = {:.1f}".format(r),
                xy=(0.2, 0.95),
                xycoords=ax.transAxes, size=35)

def scatterplot(x, y, **kws):
    ax = plt.gca()
    sns.scatterplot(x=x, y=y, ax=ax, **kws)

def corrfunc(x, y, **kws):
    """
    Calculate the Pearson correlation coefficient between x and y and add an annotation to the plot.

    Args:
        x (array-like): The x-axis data.
        y (array-like): The y-axis data.
        **kws: Additional keyword arguments.

    Returns:
        None
    """

    # Calculate the Pearson correlation coefficient between x and y

    r, _ = stats.pearsonr(x, y)
    # Get the current axes instance
    ax = plt.gca()
    # Determine the number of annotations already on the plot
    n = len([c for c in ax.get_children() if
             isinstance(c, matplotlib.text.Annotation)])
    # Determine the position and color of the new annotation based on the hue category
    pos = (.1, .9 - .1 * n)
    color = sns.color_palette()[sns.color_palette().index(kws['color'])]

    # Add an annotation to the plot showing the correlation coefficient and hue category
    ax.annotate("{}: r = {:.2f}".format(kws['label'], r), xy=pos, xycoords=ax.transAxes, color=color,size = 30)


def plot_pair_grid_ref(df, hue):
    """
    Create a PairGrid with regression plots in the upper triangle,
    and kernel density estimates in the lower triangle.

    Args:
        df (DataFrame): The data to plot.
        hue (str): The column name in 'df' representing the hue category.

    Returns:
        None
    """

    # Create a PairGrid with regression plots in the upper triangle
    # and kernel density estimates in the lower triangle
    g = sns.PairGrid(data=df, hue=hue, height=4, aspect=1.5)
    g.map_upper(corr_reg)
    g.map_lower(scatterplot)
    g.map_diag(sns.histplot)
    


def plot_pair_grid_class(df, hue):
    """
        Create a PairGrid with regression plots in the upper triangle,
        and annotations of correlation coefficients in the lower triangle.

        Args:
            df (DataFrame): The data to plot.
            hue (str): The column name in 'df' representing the hue category.

        Returns:
            None
        """
    # Create a PairGrid with regression plots in the upper triangle
    # and annotations of correlation coefficients in the lower triangle
    g = sns.PairGrid(data=df, hue=hue, height=4, aspect=1.5)
    g.map_upper(corrfunc)
    g.map_lower(scatterplot)
    g.map_diag(sns.histplot)


def plot_corr_heatmap(df, figsize=(12, 10)):
    """
    Plot a correlation heatmap for all numeric columns of df.

    A compact alternative to a full PairGrid — instead of one subplot per
    pair of columns (which grows to n_columns^2 and becomes unreadable for
    wide dataframes), this shows every pairwise correlation in a single
    annotated heatmap.

    Args:
        df (DataFrame): The data to compute correlations for.
        figsize (tuple): Size of the matplotlib figure.

    Returns:
        None
    """
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_pair_grid_top_corr(df, target, top_n=5, hue=None):
    """
    Build a PairGrid using only the top_n features most correlated (by
    absolute Pearson correlation) with the target column, plus the target
    itself. Falls back to the full PairGrid if the target column cannot be
    identified inside df.

    Args:
        df (DataFrame): The data to plot. Must contain the target column.
        target: Either the target column name (str), or the target Series
            (e.g. y = df["yield"]) — its `.name` attribute is used to find
            the column in df.
        top_n (int): How many of the most-correlated features to include.
        hue (str or None): Column name in df to use as hue (for
            classification-style coloring). None for a plain regression
            style PairGrid.

    Returns:
        None
    """
    target_name = target if isinstance(target, str) else getattr(target, "name", None)

    if target_name is None or target_name not in df.columns:
        # Can't determine which column is the target — fall back to the
        # full pairplot rather than silently plotting nothing useful.
        if hue is not None:
            plot_pair_grid_class(df, hue)
        else:
            plot_pair_grid_ref(df, hue=hue)
        return

    corr = df.corr(numeric_only=True)[target_name].drop(labels=[target_name])
    top_features = list(corr.abs().sort_values(ascending=False).index[:top_n])
    cols = top_features + [target_name]
    if hue is not None and hue not in cols:
        cols = cols + [hue]

    g = sns.PairGrid(data=df[cols], hue=hue, height=4, aspect=1.5)
    g.map_upper(corrfunc if hue is not None else corr_reg)
    g.map_lower(scatterplot)
    g.map_diag(sns.histplot)


def _show_overview_plot(df, target, pairplot_type, top_n, hue):
    """
    Internal helper used by best_model to render the "overview" plot at
    the start of a run, according to pairplot_type:
        'heatmap'  -> plot_corr_heatmap(df)                     (default)
        'top_corr' -> plot_pair_grid_top_corr(df, target, top_n, hue)
        'full'     -> the original full PairGrid (all columns)
        'none'     -> skip the overview plot entirely
    """
    if pairplot_type == 'none':
        return
    elif pairplot_type == 'heatmap':
        plot_corr_heatmap(df)
    elif pairplot_type == 'top_corr':
        plot_pair_grid_top_corr(df, target, top_n=top_n, hue=hue)
    elif pairplot_type == 'full':
        if hue is not None:
            plot_pair_grid_class(df, hue)
        else:
            plot_pair_grid_ref(df, hue=hue)
    else:
        raise ValueError(
            f"Unknown pairplot_type: {pairplot_type!r}. "
            "Expected one of 'heatmap', 'top_corr', 'full', 'none'."
        )


# Function that performs a grid search with cross-validation to find the
# best hyperparameters for a set of regression or classification models,
# and returns a dictionary containing information on the models including
# the model object, best hyperparameters, and evaluation metrics.


def best_model(features, target, mode, grid, df, hue, models=None,
               pairplot_type='heatmap', top_n=5):
    """
    Perform model training and evaluation for both regression and classification problems.

    Args:
        features: Input features.
        target: Target values.
        mode: 'regression' or 'classification'.
        grid: 'Yes' or 'No', whether to perform grid search.
        df: Dataframe for plotting pair grids.
        hue: Hue variable for pair grids.
        models: Additional custom models with their parameters.
        pairplot_type: What overview plot to show at the start of the run:
            'heatmap'  - a correlation heatmap of all numeric columns in df
                         (default; compact and readable for wide dataframes).
            'top_corr' - a PairGrid limited to the top_n features most
                         correlated with the target, plus the target itself.
            'full'     - the original full PairGrid over every column in df
                         (grows to n_columns x n_columns subplots - can get
                         very large/unreadable for wide dataframes).
            'none'     - skip the overview plot entirely.
        top_n: Number of top-correlated features to include when
            pairplot_type='top_corr'. Ignored otherwise.

    Returns:
        models_fit_info: Dictionary containing model information, hyperparameters, and evaluation metrics.
    """

    # Splitting the dataset into train and test sets with a ratio of 0.2
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    # Initializing a variable to store the model fit information
    models_fit_info = None

    if mode == 'regression':
        # Show the chosen overview plot (correlation heatmap by default)
        _show_overview_plot(df, target, pairplot_type, top_n, hue=None)


        # Initializing a dictionary to store the models and their hyperparameters
        models_fit_info = {
            'LinearRegression': {'model': LinearRegression(), 'param': {}},
            'RandomForestRegressor': {
                'model': RandomForestRegressor(n_jobs=-1),
                'param': [
                    {'max_depth': [4, 16, 32, 42]},
                    {'n_estimators': [50, 100, 150, 200, 300, 400, 500]}
                ]
            },
            'DecisionTreeRegressor': {'model': DecisionTreeRegressor(),
                                      'param': {'max_features': [0.1, 0.2, 0.3], 'max_depth': [1, 2, 4, 8, 16],
                                                }},
            'Lasso': {'model': linear_model.Lasso(max_iter=50000),
                      'param': {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0]}},
            'Ridge': {'model': Ridge(), 'param': {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0, 2.3, 2.7, 3.0]}},
            'KNeighborsRegressor': {'model': KNeighborsRegressor(n_jobs=-1),
                                    'param': {'n_neighbors': [2, 3, 5, 10, 15, 17, 19, 21, 23, 25],
                                              'weights': ('uniform', 'distance')}},
            'GradientBoostingRegressor': {'model': GradientBoostingRegressor(),
                                          'param': {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                                    'n_estimators': [50, 100, 150, 200, 300, 400, 500]}},
            'AdaBoostRegressor': {'model': AdaBoostRegressor(),
                                  'param': {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                            'n_estimators': [50, 100, 150, 200, 300, 400, 500]}},
            'XGBRegressor': {'model': xgb.XGBRegressor(),
                             'param': {'objective': ['reg:squarederror'], 'max_depth': [1, 2, 4, 8, 16, 64, 128, 512],
                                       'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}}
        }
        if models != None:
            # Iterate through all models and store their info
            for model, model_params in models:
                models_fit_info[model.__class__.__name__] = {'model': model, 'param': model_params}

        # Iterate through all models, perform grid search and final training and evaluations
        for model_name, model_info in models_fit_info.items():
            if grid == 'Yes':
                # Perform grid search on the model
                model_grid, grid_results, pairs = grid_reg(model_info['model'], model_info['param'], features, target)

                # Uppdate model info with best parameters
                model_info['best_param'] = model_grid.best_params_
                model_class = model_info['model'].__class__

                # Create a new model instance with best parameters
                new_model_with_best_params = model_class(**model_info['best_param'])
                model_info['model'] = new_model_with_best_params

                # Plot metrics (r2, mae, mse) for cv model
                plot_cv_r2(model_name + 'CV', grid_results['params'], grid_results['mean_test_r2'], pairs)
                plot_mae_mse(model_name + 'CV', grid_results['params'], grid_results['mean_test_mae'],
                             grid_results['root_mean_test_mse'], pairs)

            # Train and evaluate the final model
            model = model_info['model']
            metrics_dict = test_model_reg(model, model_name, x_train, y_train, x_test, y_test)
            model_info['metrics'] = metrics_dict

        # Create a list of model names
        model_names = list(models_fit_info.keys())

        # Plot r2, rmse, and mae for train and validation sets
        for score_name, score_plot_fn in zip(['r2', 'rmse', 'mae'], [plot_r2, plot_rmse, plot_mae]):
            model_score_tra = [models_fit_info[mn]['metrics'][score_name]['train'] for mn in model_names]
            model_score_val = [models_fit_info[mn]['metrics'][score_name]['val'] for mn in model_names]

            score_plot_fn(model_names, model_score_tra, model_score_val)

        # Print a tidy summary table of every model's metrics
        print(summarize_results(models_fit_info, mode='regression').to_string(index=False))

        # Return the models_fit_info dictionary
        return models_fit_info

    if mode == 'classification':
        # Show the chosen overview plot (correlation heatmap by default)
        _show_overview_plot(df, target, pairplot_type, top_n, hue=hue)
        # Initializing a dictionary to store the models and their hyperparameters
        models_fit_info = {
            'DecisionTree': {'model': DecisionTreeClassifier(), 'param': {'max_depth': [1, 2, 4, 8, 16, 64, 128, 512],
                                                                          'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                                                           0.7, 1]}},
            # 'RandomForest': {'model': RandomForestClassifier(n_jobs = 1),'param' : {'max_depth':[1,2,4,6,10,328,16,64,128,256,512],'criterion':('gini', 'entropy')}},
            'RandomForest':
                {'model': RandomForestClassifier(n_jobs=-1),
                 'param': [{'max_depth': [1, 2, 4, 6, 10, 328, 16, 64, 128, 256, 512]},
                           {'criterion': ('gini', 'entropy')}]},
            'SVM': {'model': svm.SVC(), 'param': {'C': [0.1, 1, 10]}},
            'LogisticRegression': {'model': LogisticRegression(n_jobs=-1),
                                   'param': {'C': [0.5, 1.0, 2.0, 3.0, 10.0, 20.0]}},
            'KNeighborsClassifier': {'model': KNeighborsClassifier(n_jobs=-1),
                                     'param': {'n_neighbors': [3, 5, 10, 12, 15, 20, 25],
                                               'weights': ('uniform', 'distance')}},
            'GradientBoostingClassifier': {'model': GradientBoostingClassifier(),
                                           'param': {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                                     'n_estimators': [50, 100, 150, 200, 300, 400, 500]}},
            'AdaBoostClassifier': {'model': AdaBoostClassifier(),
                                   'param': {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                             'n_estimators': [50, 100, 150, 200, 300, 400, 500]}},
            'XGBClassifier': {'model': xgb.XGBClassifier(), 'param': {'max_depth': [1, 2, 4, 8, 16, 64, 128, 512],
                                                                      'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                                                        0.7]}},
            'GaussianNB': {'model': GaussianNB(), 'param': {}}
        }
        if models != None:
            # Iterate through all models and store their info
            for model, model_params in models:
                models_fit_info[model.__class__.__name__] = {'model': model, 'param': model_params}

        # Iterate through all models, perform grid search and final training and evaluations
        for model_name, model_info in models_fit_info.items():
            if grid == 'Yes':
                # Perform grid search on the model
                model_grid, df_grid, pairs = grid_class(model_info['model'], model_info['param'], features, target)
                # Uppdate model info with best parameters
                model_info['best_param'] = model_grid.best_params_
                model_class = model_info['model'].__class__
                new_model_with_best_params = model_class(**model_info['best_param'])
                # Create a new model instance with best parameters
                model_info['model'] = new_model_with_best_params
                # Plot metrics for cv_model
                plot_cv_metrics(model_name + 'CV', df_grid['params'], df_grid['mean_test_f1'],
                                df_grid['mean_test_recall'], df_grid['mean_test_precision'],
                                df_grid['mean_test_accuracy'], pairs)
            # Train and evaluate the final model
            model = model_info['model']
            metrics_dict = test_model_class(model, model_name, x_train, y_train, x_test, y_test)
            model_info['metrics'] = metrics_dict
        # Create a list of model names
        model_names = list(models_fit_info.keys())

        # for score_name, score_plot_fn in zip(['score', 'f1', 'precision', 'recall'], [plot_score, plot_f1_score, plot_score_precision,plot_score_recall]):
        # Plot model evaluation scores
    for score_name in ['Score', 'F1', 'Precision', 'Recall', 'AP']:
        selected_model_names = []
        model_score_tra = []
        model_score_val = []

        # Collect metrics for each model
        for mn in model_names:
            model_metrics = models_fit_info[mn]['metrics'][score_name]
            if model_metrics is None:
                continue

            selected_model_names.append(mn)
            model_score_tra.append(model_metrics["train"])
            model_score_val.append(model_metrics['val'])

        # Plot scores using a generic function
        plot_score_generic(selected_model_names, model_score_tra, model_score_val, score_name)

    # Plot PR curves
    for score_name, (x_score_name, y_score_name) in [('PR_curve', ('Recall', 'Precision'))]:
        selected_model_names = []
        model_score_tra = []
        model_score_val = []

        # Collect metrics for each model
        for mn in model_names:
            model_metrics = models_fit_info[mn]['metrics'][score_name]

            if model_metrics is None:
                continue

            selected_model_names.append(mn)
            model_score_tra.append(model_metrics['train'])
            model_score_val.append(model_metrics['val'])

        # Plot PR curves using a generic function
        plot_curve_generic(selected_model_names, model_score_tra, model_score_val, x_score_name=x_score_name,
                           y_score_name=y_score_name)

    # Print a tidy summary table of every model's metrics
    print(summarize_results(models_fit_info, mode='classification').to_string(index=False))

    # Return the model information dictionary
    return models_fit_info


def summarize_results(results, mode="regression", sort_by=None, ascending=False):
    """
    Turn the results dict returned by best_model() into a tidy,
    sorted pandas DataFrame -- one row per model, one column per metric --
    instead of a nested dict that's hard to read at a glance.

    Args:
        results: The dict returned by best_model().
        mode: 'regression' or 'classification'. Determines which metrics
            are pulled out of each model's `metrics` dict.
        sort_by: Column name to sort by. Defaults to the validation R2 for
            regression, and validation accuracy Score for classification.
        ascending: Sort order. Defaults to False (best model first).

    Returns:
        A pandas DataFrame with columns:
            regression:     model, best_param, r2_train, r2_val,
                             rmse_val, mae_val
            classification: model, best_param, score_val, f1_val,
                             precision_val, recall_val
    """
    rows = []

    if mode == "regression":
        for name, info in results.items():
            m = info.get("metrics", {})
            r2 = m.get("r2", {}) or {}
            rmse = m.get("rmse", {}) or {}
            mae = m.get("mae", {}) or {}
            rows.append({
                "model": name,
                "best_param": info.get("best_param"),
                "r2_train": r2.get("train"),
                "r2_val": r2.get("val"),
                "rmse_val": rmse.get("val"),
                "mae_val": mae.get("val"),
            })
        default_sort = "r2_val"

    elif mode == "classification":
        for name, info in results.items():
            m = info.get("metrics", {})
            score = m.get("Score", {}) or {}
            f1 = m.get("F1", {}) or {}
            precision = m.get("Precision", {}) or {}
            recall = m.get("Recall", {}) or {}
            rows.append({
                "model": name,
                "best_param": info.get("best_param"),
                "score_val": score.get("val"),
                "f1_val": f1.get("val"),
                "precision_val": precision.get("val"),
                "recall_val": recall.get("val"),
            })
        default_sort = "score_val"

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'regression' or 'classification'.")

    df = pd.DataFrame(rows)
    sort_col = sort_by or default_sort
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    return df

