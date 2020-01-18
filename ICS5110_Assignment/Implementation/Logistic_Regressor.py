import pandas as pd
import numpy as np
from numpy import log
import matplotlib.pyplot as plt
import random


# Logistic Regressor Model
class LogisticRegressor:

    # Initialise hyperparameters
    def __init__(self, lr=0.01, max_iter=500, add_intercept=True):
        self.learning_rate = lr
        self.max_iter = max_iter
        self.add_intercept = add_intercept

    # Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Cost function for gradient descent
    def __cost_function(self, y, y_pred):
        return (-y * log(y_pred) - ((1 - y) * log(1 - y_pred))).mean()

    # Train the model
    def fit(self, X, y):

        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))

        # Initialise weights
        self.w = np.zeros(X.shape[1])

        # Keep a history of cost values
        self.cost_history = []

        # 1/m (m being the number of intances)
        OneOverM = 1 / X.shape[1]

        # Iterate
        for i in range(self.max_iter):
            # prediction
            y_pred = self.__sigmoid(np.dot(X, self.w.T))

            # cost
            cost = self.__cost_function(y, y_pred)
            self.cost_history.append(cost)

            # gradient vector
            gradient = np.dot(X.T, (y_pred - y)) * OneOverM
            self.w -= gradient * self.learning_rate

    # Predict an output
    def predict(self, X):
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        y_pred = self.__sigmoid(np.dot(X, self.w.T))
        return y_pred


# One Vs All Model
class LogisticRegressorOneVsAll:

    # Initialise hyperparameters
    def __init__(self, lr=0.01, max_iter=500, add_intercept=True):
        self.learning_rate = lr
        self.max_iter = max_iter
        self.add_intercept = add_intercept
        # self.y_1vsall = []
        # self.nb_classes = 3

    # Train the model
    def fit(self, X, y):
        self.regressors = []

        # # For each possible class
        # for c in range(self.nb_classes):
        #     y_one = np.where(y == c, 1, 0)
        #     self.y_1vsall.append(y_one)

        # For each "one vs. all" target set
        for y_one in y:
            # Build a model with target set
            model = LogisticRegressor(lr=self.learning_rate,
                                      max_iter=self.max_iter,
                                      add_intercept=self.add_intercept)
            # Train a model and add it to the list of regressors
            model.fit(X, y_one)
            self.regressors.append(model)

    # Predict an output
    def predict(self, X):
        final_pred = []
        y_pred_1vsall = []

        # For each regressor
        for model in self.regressors:
            y_pred = model.predict(X)  # was X_test
            y_pred_1vsall.append(y_pred)

        # For each instance
        for i in range(len(X)):
            best_pred = 0
            best_target = -1
            # Find the best prediction (model with highest probability)
            for j in range(len(self.regressors)):
                if y_pred_1vsall[j][i] > best_pred:
                    best_pred = y_pred_1vsall[j][i]
                    best_target = j
            # Add best prediction to the final result
            final_pred.append(best_target)

        # Return final prediction
        final_pred = np.asarray(final_pred)
        return final_pred