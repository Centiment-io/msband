"""
Stress Detection
Machine Learning to process data from Affektive Band
Joseph Chu - Dec/11
"""

import requests
import logging
import time

# ML Imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score

# Globals
url = 'http://159.203.4.252/api/measurement?results_per_page=100'
page = '&page='


def plot_validation_curve(param_name, param_range, train_scores, test_scores):
    """Plot of Validation Curve

    """

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation Curve with SVM")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)

    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.savefig('vc_{}.png'.format(param_name))

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('learning_curve.png')

def data_visualize(df):
    """Data Visualization by plotting <HR,GSR> for each state's dataset

    Args:
        Panda DataFrame

    Returns:
        None
    """

    logging.info('Performing Data Visualization')

    normal = df[df['state'] == 'Normal']
    calm = df[df['state'] == 'Calm']
    stressed = df[df['state'] == 'Stressed']

    logging.info('The size of our dataset for the normal state {}'.format(normal.shape))
    logging.info('The size of our dataset for the calm state {}'.format(calm.shape))
    logging.info('The size of our dataset for the stressed state {}'.format(stressed.shape))

    ax = normal.plot(kind='scatter', xlim = (30,100), ylim = (0,15000), x='hr', y='gsr',
                                    color='DarkBlue', label='Normal')
    calm.plot(kind='scatter', xlim = (30,100), ylim = (0,15000),x='hr', y='gsr',
                                    color='DarkGreen', label='Calm', ax=ax)
    #stressed.plot(kind='scatter', xlim = (30,100), ylim = (0,15000),x='hr', y='gsr',
    #                                color='DarkRed', label='Stressed', ax=ax)
    plt.savefig('output.png')

def retrieve_data():
    """ Method to retrieve data from the Server using HTTP get requests

    Args: None

    Output: Panda df

    """
    logging.info('Retrieving Data')
    df = pd.DataFrame()
    r = requests.get(url)
    total_pages = r.json()['total_pages']

    for i in xrange(1,total_pages):
        new_url = url + page + str(i)
        new_r = requests.get(new_url)

        data_json = new_r.json()
        data_objects = data_json['objects']

        new_df = pd.DataFrame(data_objects)
        df = pd.concat([df, new_df])
    return df

def time_data(df):

    logging.info('Performing Data Visualization')

    normal = df[df['state'] == 'Normal']
    normal1 = normal[normal['id'] < 4506]
    normal2 = normal1[normal1['id'] > 3937]
    normal2['timestamp'] = pd.to_datetime(normal2['timestamp'])

    #calm = df[df['state'] == 'Calm']
    #stressed = df[df['state'] == 'Stressed']

    #logging.info('The size of our dataset for the normal state {}'.format(normal.shape))
    #logging.info('The size of our dataset for the calm state {}'.format(calm.shape))
    #logging.info('The size of our dataset for the stressed state {}'.format(stressed.shape))

    print normal1
    ax = normal2.plot(ylim = (30,100), x='timestamp', y='hr',
                                    color='DarkBlue', label='Normal')
    #calm.plot(kind='scatter', xlim = (30,100), ylim = (0,15000),x='hr', y='gsr',
    #                                color='DarkGreen', label='Calm', ax=ax)
    #stressed.plot(kind='scatter', xlim = (30,100), ylim = (0,15000),x='hr', y='gsr',
    #                                color='DarkRed', label='Stressed', ax=ax)
    plt.savefig('normalhrs.png')

    fig2 = plt.figure()
    ax1 = normal2.plot(x='timestamp', y='temp',
                                    color='DarkRed', label='Normal')
    plt.savefig('normaltemp.png')


def process_data(userid):
    """ Method to process data using ML algorithms from the Scikit-Learn Python
    Library.

    Training Data: X - N X 2 dimension numpy array (HR,GSR)
    Targets: T - N X 1 dimension numpy array

    Args: userid TODO:Currently only one user

    Output:

    """
    logging.info('Processing Data')

    # Retrive Data from db & Visualize
    df = retrieve_data()
    data_visualize(df)

    # State of 'None' is our unlabeled data set, so exclude
    df = df[df['state'] != 'None']

    # Data Preprocessing: Map string 'states' to numeric labels
    le = preprocessing.LabelEncoder()
    le.fit(df['state'])
    df['state'] = le.transform(df['state'])

    # Setting the dimensionality of our data set
    X_raw = df[['hr','gsr']].values
    T_raw = df[['state']].values
    T_raw = np.ravel(T_raw)

    # Further Preprocessing
    X_raw = StandardScaler().fit_transform(X_raw)
    # X = preprocessing.normalize(X)

    # Creating a training/test split 80-20
    X, X_test, T, y_test = train_test_split(X_raw, T_raw, test_size=0.20, random_state=42)

    # Set of Classifiers
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(C=15, gamma=15),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    #Iterate through Classifiers
    for name, clf in zip(names, classifiers):
        # Compute Cross-Validation Score using K-Folds
        start = time.clock()
        scores = cross_val_score(clf, X, T, cv=10, scoring='roc_auc')
        end = time.clock()
        logging.info('----Elapsed Time: {}---'.format(end-start))

         = clf.fit(X, T)

        roc_auc_score(
        train_scores = current.roc_auc_score(X, T)
        test_scores = current.roc_auc_score(X_test, y_test)

        # Log Results
        logging.info('Training ROC AUC Score of {0} = {1}'.format(name, np.mean(train_scores)))
        logging.info('Cross Validation ROC AUC Score of {0} = {1}'.format(name, np.mean(scores)))
        logging.info('Test ROC AUC Score of {0} = {1}'.format(name, np.mean(test_scores)))

    # Plot of hyperparameters and mean classification rate to determine B-V
    train_scores, validation_scores = validation_curve(SVC(), X, T, param_name="gamma", param_range=gammas,
                    cv=10, scoring="accuracy", n_jobs=1)
    plot_validation_curve('Gamma', gammas, train_scores, validation_scores)
    train_scores, validation_scores = validation_curve(SVC(), X, T, param_name="C", param_range=Cs,
                    cv=10, scoring="accuracy", n_jobs=1)
    plot_validation_curve('C', Cs, train_scores, validation_scores)


    return

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s     %(levelname)s:  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)
    process_data("affektive")

