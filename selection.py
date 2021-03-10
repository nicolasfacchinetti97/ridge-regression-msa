import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class KFoldCV():
    def __init__(self, n_splits, shuffle = False):
        self.k = n_splits
        self.shuffle = shuffle
    
    def get_folds(self, data):
        'Given the data it divdie it in k folds'
        n = len(data)                                   # len data
        m = int(n/self.k)                               # num entry per folds
        folds = []
        i = 0
        for i in range(self.k-1):
            folds.append(data[i*m:(i+1)*m])
        folds.append(data[(i+1)*m:n])
        return folds
    
    def split(self, data):
        'Split the k folds in k train + test data'
        if self.shuffle:
            data.sample(frac=1)

        folds = self.get_folds(data)
        train_test = []
        for i in range(len(folds)):
            f = folds.copy()
            test = f.pop(i)
            train = pd.concat(f)

            train_test.append([train, test])
        return train_test

    def cross_validate(self, model, X, Y, loss_func):
        '''
        Take in input a model and the data and performe cross validation
        Input:
            - a ML model
            - X: dataset
            - Y: name of the column
            - loss function
        Return:
            - true and predicted elements for each fold
            - dictionary of metrics and statistics on each fold
        '''
        scaled_errors = []
        dict_metrics = {}
        for train, test in self.split(X):
            x_test = test.drop(Y, axis=1)
            x_train = train.drop(Y, axis=1)
            y_train = train[Y].copy()
            y_test = test[Y].copy()
            model.fit(x_train, y_train)                                             # fit the model
            y_predicted = model.predict(x_test)                                     # get prediction on test set
            
            experiments = list(zip(y_test, y_predicted))                            # buid a list of tuples composed by true and predicted element
            err_list = [loss_func(row[0], row[1]) for row in experiments]           # calculate the loss for each experiment according to a specific function
            scaled_error = self.k/len(X) * sum(err_list)                            # compute the scaled errors
            scaled_errors.append(scaled_error)
            
            print("Scaled error: " + str(scaled_error))
            print("R2: " + str(r2_score(y_test, y_predicted)))
            print("MAE: " + str(mean_absolute_error(y_test, y_predicted)))
            print("MSE: " + str(mean_squared_error(y_test, y_predicted)))
            print("\n")
        return scaled_errors


