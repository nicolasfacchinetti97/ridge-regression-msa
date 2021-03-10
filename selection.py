import pandas as pd

class KFoldCV():
    def __init__(self, n_splits, shuffle = False):
        self.k = n_splits
        self.shuffle = shuffle
    
    def get_folds(self, data):
        n = len(data)                                   # len data
        m = int(n/self.k)                               # num entry per folds
        folds = []
        i = 0
        for i in range(self.k-1):
            folds.append(data[i*m:(i+1)*m])
        folds.append(data[(i+1)*m:n])
        return folds
    
    def split(self, data):
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


