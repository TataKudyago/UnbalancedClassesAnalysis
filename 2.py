import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def read_data():
    data = pd.read_csv('creditcard.csv')
    y_train = data.iloc[:, -1].values
    x_train = data.drop(data.columns[len(data.columns)-1], axis=1).values
    return y_train, x_train


train_y, train_x = read_data()
testX, trainX, testY, trainY = train_test_split(train_x, train_y, test_size=0.5, random_state=0)
classifiers = (DecisionTreeClassifier(), KNeighborsClassifier())
grid_params = ({'criterion': ('gini', 'entropy')}, {'n_neighbors': np.arange(1, 7)})


for i in range(2):
    accuracy = GridSearchCV(estimator=classifiers[i], param_grid=grid_params[i], scoring='accuracy', n_jobs=-1)
    accuracy.fit(trainX, trainY)

    precision = GridSearchCV(estimator=classifiers[i], param_grid=grid_params[i], scoring='precision', n_jobs=-1)
    precision.fit(trainX, trainY)

    recall = GridSearchCV(estimator=classifiers[i], param_grid=grid_params[i], scoring='recall', n_jobs=-1)
    recall.fit(trainX, trainY)

    f = GridSearchCV(estimator=classifiers[i], param_grid=grid_params[i], scoring='f1', n_jobs=-1)
    f.fit(trainX, trainY)

    roc = GridSearchCV(estimator=classifiers[i], param_grid=grid_params[i], scoring='roc_auc', n_jobs=-1)
    roc.fit(trainX, trainY)

    print("accuracy: " + str(accuracy.score(testX, testY)))
    print("precision: " + str(precision.score(testX, testY)))
    print("recall: " + str(recall.score(testX, testY)))
    print("f: " + str(f.score(testX, testY)))
    print("roc: " + str(roc.score(testX, testY)))

