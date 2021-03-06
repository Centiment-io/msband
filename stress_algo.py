"""
Stress Detection
Machine Learning to process data from Affektive Band
Joseph Chu - Dec/11
"""

import requests
import logging
import time
import csv

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
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

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

def plot_roc_curve(X, y, clf, name):

    print y
    y = label_binarize(y, classes=[0, 1, 2])
    print y
    n_classes = y.shape[1]

    n_samples, n_features = X.shape

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        start = time.clock()
        auc_val = cross_val_score(clf, X_train, y_train[:, i], cv=10, scoring='roc_auc').mean()
        end = time.clock()

        logging.info('----Elapsed Time: {}---'.format(end-start))
        logging.info('Cross Validation ROC AUC Score of Class {2} {0} = {1}'.format(name, auc_val, i))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)

    with open('{}_features.csv'.format(n_features), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter =",",quoting=csv.QUOTE_MINIMAL)

        cur_row = []
        cur_row.append(name)
        cur_row.append(str(end-start))
        for i in range(n_classes):
            cur_row.append('{:0.2f}'.format(roc_auc[i]))
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig('roc_curves_{}.png'.format(name))

        print ''.join(cur_row)
        writer.writerow(cur_row)

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

    logging.info('Plotting Time Series Data')
    #logging.info('The size of our dataset for the normal state {}'.format(normal.shape))
    #logging.info('The size of our dataset for the calm state {}'.format(calm.shape))
    #logging.info('The size of our dataset for the stressed state {}'.format(stressed.shape))

    # Retrieve Subset of entries for each state
    plot_time_data(df, 'Normal1', 3937, 4506)
    plot_time_data(df, 'Normal2', 389, 1542)
    plot_time_data(df, 'Normal3', 1543, 3678)
    plot_time_data(df, 'Calm1', 63, 388)
    plot_time_data(df, 'Calm2', 3679, 3936)
    plot_time_data(df, 'Stressed1', 4907, 4995)
    plot_time_data(df, 'Stressed2', 5256, 5277)

def plot_time_data(current, state, start, end):

    current1 = current[current['id'] < end]
    current2 = current1[current1['id'] > start]
    current2['timestamp'] = pd.to_datetime(current2['timestamp'])

    # Heart Rate Data
    plt.figure()
    current2.plot(x='timestamp', y='hr',
                        color='DarkRed', label='Heart Rate')
    plt.savefig('heart_rate_{}.png'.format(state))

    # GSR Data
    plt.figure()
    current2.plot(x='timestamp', y='gsr',
                        color='Blue', label='GSR')
    plt.savefig('GSR_{}.png'.format(state))

    # Skin Temperature Data
    plt.figure()
    current2.plot(x='timestamp', y='temp',
                        color='Orange', label='Skin Temperature')
    plt.savefig('skin_temp_{}.png'.format(state))

    #Acceleration Data
    plt.figure()
    ax = current2.plot(x='timestamp', y='accx',
                                    color='DarkBlue', label='accx')
    current2.plot(x='timestamp', y='accy',
                                    color='DarkGreen', label='accy', ax=ax)
    current2.plot(x='timestamp', y='accz',
                                    color='DarkRed', label='accz', ax=ax)
    plt.savefig('acceleration_{}.png'.format(state))

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
    #time_data(df)
    #data_visualize(df)

    # State of 'None' is our unlabeled data set, so exclude
    df = df[df['state'] != 'None']

    # Data Preprocessing: Map string 'states' to numeric labels
    le = preprocessing.LabelEncoder()
    le.fit(df['state'])
    df['state'] = le.transform(df['state'])

    # Setting the dimensionality of our data set
    #X_raw = df[['hr','gsr']].values
    X_raw = df[['hr','gsr','temp','accx','accy','accz']].values
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
        SVC(kernel="linear", C=0.025, probability=True),
        SVC(C=15, gamma=15, probability=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    #Iterate through Classifiers
    for name, clf in zip(names, classifiers):
        # Compute Cross-Validation Score using K-Folds
        logging.info('----Currently processing: {}---'.format(name))

        # Test
        plot_roc_curve(X_raw, T_raw, clf, name)

    return

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s     %(levelname)s:  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)
    process_data("affektive")

