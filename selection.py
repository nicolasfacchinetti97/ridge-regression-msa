import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class KFoldCV():
    def __init__(self, n_splits, shuffle = False, print = False):
        self.k = n_splits                               # number of splits
        self.shuffle = shuffle                          # shuffle or not the database
        self.n = 0                                      # number of records
        self.print = print                              # print data during computation
    
    def get_folds(self, data):
        'Given the data it divide it in k folds'
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
        self.n = len(data)                              # number of rows

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

    def compute_error(self, true_val, predict_val, loss_func):
        '''
        Compute the classification error according to a specific loss function
        '''
        experiments = list(zip(true_val, predict_val))                          # buid a list of tuples composed by true and predicted element
        err_list = [loss_func(row[0], row[1]) for row in experiments]           # calculate the loss for each experiment according to a specific function
        return self.k/self.n * sum(err_list)                                    # compute the scaled errors
        
    def get_train_test_splitted_data(self, train, test, column):
        '''
        Extract and split the data from a pandas dataframes to further evaluation
        '''
        x_test = test.drop(column, axis=1)
        x_train = train.drop(column, axis=1)
        y_train = train[column]
        y_test = test[column]
        return x_train, y_train, x_test, y_test

    def fit_and_evaluate(self, model, x_train, y_train, x_test, y_test, loss_function):
        '''
        Get a model, the train + test data and a loss function and fit the model according the data passed and the evaluate it
        '''
        model.fit(x_train, y_train)                                             # fit the model
        y_predicted = model.predict(x_test)                                     # get prediction on test set
        scaled_error = self.compute_error(y_test, y_predicted, loss_function)   # compute the loss according the loss function
    
        if self.print:
            print("Scaled error: " + str(scaled_error))
            # print("R2: " + str(r2_score(y_test, y_predicted)))
            # print("MAE: " + str(mean_absolute_error(y_test, y_predicted)))
            # print("MSE: " + str(mean_squared_error(y_test, y_predicted)))
            
        return scaled_error

    def cross_validate(self, model, X, Y, loss_func):
        '''
        Take in input a model and the data and performe cross validation
        Input:
            - a ML model
            - X: dataset
            - Y: name of the column
            - loss function
        Return:
            - mean of the error on each fold
        '''
        scaled_errors = []
        for train, test in self.split(X):
            x_train, y_train, x_test, y_test = self.get_train_test_splitted_data(train, test, Y)
            fold_error = self.fit_and_evaluate(model, x_train, y_train, x_test, y_test, loss_func)
            scaled_errors.append(fold_error)
        return 1/self.k * sum(scaled_errors)


class NestedCV():
    def __init__(self, outer, inner, shuffle = False):
        self.k_inner = inner
        self.k_outer = outer
        self.shuffle = shuffle
    
    def cross_validate(self, model, X, Y, loss_func, param_list):
        outerKFold = KFoldCV(self.k_outer)                                          
        innerKFold = KFoldCV(self.k_inner)

        outer_errors = []
        for i, (train_outer, test) in enumerate(outerKFold.split(X)):               # outer k fold cv
            print("External fold num: " + str(i+1))
            errors_by_parameter = {}
            for value in param_list:                                                # CV each algorithm on the interal fold
                m = model(value)
                error = innerKFold.cross_validate(m, train_outer, Y, loss_func)
                errors_by_parameter[value] = error
                print("Testing with value {} with error : {}".format(value, error))

            best_param = float(min(errors_by_parameter, key=errors_by_parameter.get))                             # get the value wich minimize the error
            x_train, y_train, x_test, y_test = outerKFold.get_train_test_splitted_data(train_outer, test, Y)
            best_model_on_fold = model(best_param)
            fold_error = outerKFold.fit_and_evaluate(best_model_on_fold, x_train, y_train, x_test, y_test, loss_func)
            outer_errors.append(fold_error)
            print("The best parameter on interal folds is {}, with error on external fold: {}\n".format(best_param, fold_error))
        return 1/self.k_outer * sum(outer_errors)

