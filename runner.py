import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import sys

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


def getTestingData():
    test_data = np.loadtxt("data/mnist_test.csv", delimiter=",")  # get the test data

    test_labels = np.asfarray(test_data[:, :1]).flatten()  # get all the test labels
    test_imgs = np.asfarray(test_data[:, 1:])  # get all the test images

    # standardize data
    test_imgs = scaler.transform(test_imgs)

    return test_imgs, test_labels

def getTrainingData():
    train_data = np.loadtxt("data/mnist_train.csv", delimiter=",")  # get the training data

    test_size = 0.1

    y = np.asfarray(train_data[:, :1]).flatten()  # get all the labels
    X = np.asfarray(train_data[:, 1:])  # get all the images

    # split training data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # standardize data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# used to see how the testAllModel function would work - quick version
def getSmallerTrainingData():
    digits = datasets.load_digits()
    n = len(digits.images)
    data = digits.images.reshape(n, -1)

    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=test_size)

    # standardize data
    scaler_temp = StandardScaler()
    X_train = scaler_temp.fit_transform(X_train)
    X_test = scaler_temp.transform(X_test)

    return X_train, X_test, y_train, y_test

def testSingleModel(hyperparameters):
    print("Getting Training Data...")
    X_train, X_test, y_train, y_test = getTrainingData()

    print("Training Model - this will take a few minutes ...")
    model = SVC(gamma=hyperparameters['gamma'], kernel=hyperparameters['kernel'],
                C=hyperparameters['C'], degree=hyperparameters['degree'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # model accuracy and confusion matrix
    print('\nSVM Trained Classifier Accuracy: ', model.score(X_test, y_test))
    print('\nAccuracy of Classifier on Validation Images: ', accuracy_score(y_test, y_pred))
    print('\nConfusion Matrix: \n', confusion_matrix(y_test, y_pred))


def buildFinalModel(hyperparameters):
    print("Getting Training Data...")
    X_train, X_test, y_train, y_test = getTrainingData()  # get training data
    print("Getting Testing Data...")
    test_imgs, test_labels = getTestingData()  # get testing data

    # creating an SVM model
    print("Training Model - this will take a few minutes ...")
    model = SVC(gamma=hyperparameters['gamma'], kernel=hyperparameters['kernel'],
                C=hyperparameters['C'], degree=hyperparameters['degree'])
    model.fit(X_train, y_train)  # train model
    y_pred = model.predict(test_imgs)  # final evaluation

    # model accuracy and confusion matrix
    print('\nSVM Trained Classifier Accuracy: ', model.score(test_imgs, test_labels))
    print('\nAccuracy of Classifier on Test Images: ', accuracy_score(test_labels, y_pred))
    print('\nConfusion Matrix: \n', confusion_matrix(test_labels, y_pred))


# function that carries out hyper parameter tuning using Grid Search and storing results in pickle file
def testAllModels():
    X_train, X_test, y_train, y_test = getTrainingData()  # get training data
    # get smaller training dataset (only for testing purposes)
    # X_train, X_test, y_train, y_test = getSmallerTrainingData()

    # setting hyperparameters for testing
    hyper_params = [
        {
            'kernel': ['poly'],
            'gamma': [0.01, 0.1, 1],
            'C': [0.01, 0.1, 1],
            'degree': [2, 3, 4]
        },
        {
            'kernel': ['linear'],
            'C': [0.001, 0.01, 0.1, 1]
        },
        {
            'kernel': ['rbf'],
            'gamma': [0.01, 0.1, 1],
            'C': [0.01, 0.1, 1],
        }
    ]

    model = SVC()  # create an SVM model
    folds = KFold(n_splits=3, shuffle=True, random_state=10)  # create a K-fold cross validator object
    model_cv = GridSearchCV(estimator=model, param_grid=hyper_params, scoring='accuracy',
                            cv=folds, verbose=2, return_train_score=True, n_jobs=-1)

    print("Training Model...")
    model_cv.fit(X_train, y_train)  # train model
    cv_results = pd.DataFrame(model_cv.cv_results_)  # convert results to data frame object

    # -- save results to a pickle file --
    # cv_results.to_pickle("./results.pkl")  # hyperparameters for test_size of 0.20

    # -- save results to a csv file --
    # cv_results.to_csv('results.csv') # main hyperparameters for test_size of 0.20

    # -- display results to console - optional --
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(cv_results)

    # get best result and hyperparams and display it
    best_result = model_cv.best_score_
    best_hyperparams = model_cv.best_params_
    print("The highest score was {0} having the hyperparameters {1}".format(best_result, best_hyperparams))

    return best_hyperparams


def runner():
    # test several different hyperparameters to find best model - warning this takes really long
    # hyperparameters = testAllModels()

    # the best hyperparameters found from experimentation - if above is commented
    hyperparameters = {'C': 0.01, 'degree': 3, 'gamma': 0.1, 'kernel': 'poly'}

    # testing a single model
    # testSingleModel(hyperparameters)

    # building the final model
    buildFinalModel(hyperparameters)


scaler = StandardScaler()  # scaler
runner()

