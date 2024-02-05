import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.tools.eval_measures import rmse
import seaborn as sns
import statsmodels.api as sm

data=pd.read_csv('CarPrice_Assignment_Simple.csv')

#Building data understanding 
print('shape of dataframe is:', data.shape)
data.info()
data['car_ID']=data['car_ID'].astype('object')

data.describe()





#build data understanding 
data['drivewheel'].value_counts()
#avg price for drivewheel
data.groupby('drivewheel')['price'].agg('mean').round(decimals=2)


#all numeric (float and int) variables in the dataset
data_numeric=data.select_dtypes(include=['Float64','int64'])
data_numeric.head()

#Correlation matrix
cor=data_numeric.corr()
cor.round(2)

plt.figure(figsize=(16,8))
sns.heatmap(cor.round(1),cmap='YlGnBu',annot=True)
plt.show()

#Data preparation
data.columns
X=data.loc[:,['drivewheel','carlength','carwidth','carheight','curbweight','horsepower']]
Y=data['price']

#Dummy variable creation

data_categorical=X.select_dtypes(include=['object'])
data_categorical.head()

#checking unique values in drivewheel column
data_categorical['drivewheel'].unique()
data_dummies=pd.get_dummies(data_categorical,drop_first=True)
data_dummies.head()

#Drop categorical variables
X=X.drop(list(data_categorical.columns),axis=1)
#concar dummy variables with X
X=pd.concat([X,data_dummies],axis=1)
X.head()


#Model building:splitting test and train data

#Adding constant
X=sm.add_constant(X)
X.head()
#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size = 0.8, test_size = 0.2,random_state=100)

X_train.shape
X_train.head()

X_test.shape
X_test.head()


#Model Building: Fitting linear model
#Performing regression 
model=sm.OLS(y_train,X_train).fit()
print(model.summary())

#Fit model which is already created

#generate predictions
ypred_train = model.predict(X_train)

#calc rmse
rmse_train=rmse(y_train,ypred_train)
rmse_train
y_train.head()
ypred_train.head()

#generate test prediction
ypred_test = model.predict(X_test)

#calc rmse
rmse_test=rmse(y_test,ypred_test)
rmse_test
y_test.head()
ypred_test.head()


#visualizing actual and predicted output
y_actual_vs_predicted=pd.concat([y_train,ypred_train],axis=1)
y_actual_vs_predicted.columns=['Actual price','predicted price']
y_actual_vs_predicted.head()