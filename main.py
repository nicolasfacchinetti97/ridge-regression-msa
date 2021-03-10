import pandas as pd
import numpy as np
from model import RidgeRegression

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler


def pca(x:np.ndarray, n_components=2)->np.ndarray:
    return PCA(n_components=n_components).fit_transform(x)

def encode_and_bind(original_dataframe, feature_to_encode):
    one_hot = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, one_hot], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res

def mean_absolute_error(real, predict):
    diff = np.abs(predict - real)
    return np.average(diff)

def plot_scatter_predicted(real, predict):
    # set up figure and ax
    fig, ax = plt.subplots()

    # plotting the points as scatter plot 
    ax.scatter(real, predict, color = "blue", s=1) 
  
    # plotting the identity function
    ax.plot([-0.50, 1], [-0.5, 1], color='red')
  
    # putting labels 
    plt.xlabel('real') 
    plt.ylabel('predicted') 
  
    plt.show()

def find_outliers(data, threshold):
    return np.where(np.abs(data) > threshold)

# def remove_outliers(data, threshold):
#     return  < 3).all(axis=1)]

# read data from csv file
data = pd.read_csv("cal-housing.csv")

# one-hot encode categorical values
data = encode_and_bind(data, "ocean_proximity")
# data = data.drop("ocean_proximity", axis = 1)

# remove row with NaN
data = data.dropna()

# data = (data - data.mean())/data.std()

y = data.median_house_value
x = data.drop("median_house_value", axis = 1)

# normalize features
x = (x - x.mean())/x.std()
# x = RobustScaler().fit_transform(x)


# scale data
y = (y - y.min())/(y.max() - y.min())
y = y - y.mean()

# search for outliers
threshold = 3
print(len(np.where(np.abs(x) > 3)[0]))
print(len(np.unique(np.where(np.abs(x) > 3)[0])))
print(find_outliers(x, 3))
outliers_index = np.unique(find_outliers(x, threshold)[0])
print(x.shape)
x = 
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

ridge = RidgeRegression(alfa=0.4)
#ridge = Ridge(alpha=0.001)
ridge.fit(x_train, y_train)

y_predicted = ridge.predict(x_test)

print("R2: " + str(r2_score(y_test, y_predicted)))

print("MAE: " + str(mean_absolute_error(y_test, y_predicted)))

print("MSE: " + str(mean_squared_error(y_test, y_predicted)))
# plot_scatter_predicted(y_test, y_predicted)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data_rescaled = scaler.fit_transform(x)

# plt.scatter(*pca(data_rescaled).T, s=1)
# plt.show()