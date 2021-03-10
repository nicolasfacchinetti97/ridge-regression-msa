import numpy as np

class RidgeRegression(object):
    def __init__(self, alfa=0.1):
        self.alfa = alfa

    def fit(self, X, y):
        n = X.shape[1]
        id = np.identity(n)
        X_T = np.transpose(X)

        C = np.dot(X_T, X) + self.alfa*id
        C = np.linalg.inv(C)
        self.w = np.linalg.multi_dot([C, X_T, y])
        return self

    def predict(self, X):
        return np.dot(X, self.w)