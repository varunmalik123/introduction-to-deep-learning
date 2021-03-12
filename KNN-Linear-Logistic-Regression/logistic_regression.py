"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5 # To threshold the sigmoid 
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1/(1 + np.exp(-z)) 
    
    def get_one_hot(targets, no):
        res = np.eye(no)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[no])    



    def hypothesis(self,theta, X):
        return 1 / (1 + np.exp(-(np.dot(theta, X.T)))) - 0.0000001

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        

        N, D = X_train.shape
        self.w = weights

        y_scaled = get_one_hot(y_train, 10)

        
        A = sigmoid(-1 * y_scaled * np.dot(self.w, np.transpose(X_train)))

        summation = np.dot(A*y_scaled, X_train)

        self.w = self.w + self.lr * (self.weight_decay + (1/N) *summation)

        return self.w
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        predicted = X_test.dot(self.w.T)
        return np.argmax(predicted,axis=1)
