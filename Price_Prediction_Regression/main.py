import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def plot_simple(ax, x, y, predictions, xlabel, ylabel):
    ax.scatter(x,y, c='green', edgecolor='white', label='data')
    ax.scatter(x, predictions, c='steelblue', label='predictions', s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    

def plot_residuals(ax, y_train, y_train_pred, y_test, y_test_pred):
    ax.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
    ax.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('Residuals (Error)')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=-3, xmax=3, color='black', lw=2)
    

def load_housing_data():
    url = 'http://jse.amstat.org/v19n3/decock/AmesHousing.txt'
    columns = ['Lot Area', 'Overall Qual','Overall Cond','Gr Liv Area','Total Bsmt SF','Neighborhood','House Style', 'Year Built','Utilities', 'SalePrice']
    df = pd.read_csv(url,sep='\t', usecols=columns)
    df = df.dropna(axis=1)
    # drop non numeric columns
    df = df.select_dtypes(include=[np.number])
    print('*** data shape: **\n', df.shape)
    print('\n****head****\n', df.head())
    return df


def getXY(df):
    targetCol = 'SalePrice'
    features = df.columns[df.columns != targetCol]
    X = df[features].values
    Y = df[targetCol].values
    return X, Y


df = load_housing_data()
X, Y =getXY(df)
stdScaller_x = StandardScaler()
stdScaller_y = StandardScaler()
Xstd = stdScaller_x.fit_transform(X)
Ystd = stdScaller_y.fit_transform(Y.reshape(-1,1))

xtrain, xtest, ytrain, ytest = train_test_split(Xstd, Ystd, test_size= 0.2, random_state=123)

#fit and predict Linear regression model
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)
lr_y_pred = lr_model.predict(xtest)

#fit ransac regression model
ransac_model = RANSACRegressor(LinearRegression(), max_trials = 100, min_samples =0.95, residual_threshold=None, random_state=123)
ransac_model.fit(xtrain, ytrain)
ransac_y_pred = ransac_model.predict(xtest)

print('\n**** r2 scores ******')
lr_r2_score = r2_score(ytest, lr_y_pred)
print(f'lr R2 score:{lr_r2_score:.3f}')
ransac_r2_score = r2_score(ytest, ransac_y_pred)
print(f'Ransac R2 score:{ransac_r2_score:.3f}')

#plot models predictions and residuals
fig, axes = plt.subplots(nrows=2, ncols= 2, figsize=(10,5))

plot_simple(axes[0,0], X[:,4], Y, stdScaller_y.inverse_transform(lr_model.predict(Xstd)), 'living area', 'SalePrice - lr')

plot_simple(axes[0,1], X[:,4], Y, stdScaller_y.inverse_transform(ransac_model.predict(Xstd)), 'living area', 'SalePrice - ransac')


plot_residuals(axes[1,0], ytrain, lr_model.predict(xtrain), ytest, lr_y_pred)
plot_residuals(axes[1,1], ytrain, ransac_model.predict(xtrain), ytest, ransac_y_pred)

plt.tight_layout()
plt.show()
    

