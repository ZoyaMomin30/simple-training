import pandas as pd 
df = pd.read_csv("delaney_solubility_with_descriptors.csv")

# print(df)

# Data preparation
# Data separation as X and y

y = df['logS']
print("printing logS")
# print(y)

x = df.drop('logS', axis=1)
print("printing F(x)")
# print(x)

#data splitting
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# print(x_test)
# print(x_train)

#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# we want to train the empty regression model on the following data set
lr.fit(x_train, y_train) 

# applying the model on training set to make a the prediction
y_lr_train_pred = lr.predict(x_train) # lr prediction on training set of x
y_lr_test_pred = lr.predict(x_test)

# print(y_lr_train_pred)
# print(y_lr_test_pred)

# MODEL PERFORMANCE

from sklearn.metrics import mean_squared_error, r2_score
# these 2 blocks are for training set 
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

#now we will do for testing set
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print("LR MSE (Train) :",  lr_train_mse)
# print("LR R2  (Train) :",lr_train_r2)

# print("LR MSE (Test) :",lr_test_mse)
# print("LR R2 (Test) :",lr_test_r2)

lr_results= pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns=["method", "Training MSE","Training r2", "testing MSE", "testing R2"]
# print(lr_results)


# RANDOM FOREST

# applying the model to make a prediction

#evaluate model perfeormance
#  quantitative values = regression models
# categorical values = classification models

# RAINFOREST

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# applying the model to make a prediction
y_rf_train_pred = rf.predict(x_train) # lr prediction on training set of x
y_rf_test_pred = rf.predict(x_test)

#model performance evaluation

# these 2 blocks are for training set 
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

#now we will do for testing set
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ["method", "Training MSE","Training r2", "testing MSE", "testing R2"]
print(rf_results)

df_models = pd.concat([lr_results, rf_results], axis=0)
# print(df_models.reset_index(drop=True))

# RIDGE REGRESSION
from sklearn.linear_model import Ridge

# Step1: training the ridge model with training data set
rd = Ridge(alpha=0.1)
rd.fit(x_train,y_train)

#Step2: applying the model on training set to make a the prediction
y_rd_train_pred = rd.predict(x_train)
y_rd_test_pred = rd.predict(x_test)

#Step3: Model Performance 
#for testing set 
rd_train_mse = mean_squared_error(y_train, y_rd_train_pred)
rd_train_r2 = r2_score(y_train, y_rd_train_pred)

# for training set
rd_test_mse = mean_squared_error(y_test, y_rd_test_pred)
rd_test_r2 = r2_score(y_test, y_rd_test_pred)

rd_results = pd.DataFrame(['Ridge Regression', rd_train_mse, rd_train_r2, rd_test_mse, rd_test_r2]).transpose()
rd_results.columns = ["method", "Training MSE","Training r2", "testing MSE", "testing R2"]
df_models = pd.concat([lr_results, rf_results, rd_results], axis=0)
print(df_models.reset_index(drop=True))


# Support Vector Machines
from sklearn.svm import SVR

svr = SVR(kernel='linear')
svr.fit(x_train, y_train)

#prediction
y_svr_train_pred = svr.predict(x_train)
y_svr_test_pred = svr.predict(x_test)

#model performance evaluation
svr_train_mse = mean_squared_error(y_train, y_svr_train_pred)
svr_train_r2 = r2_score(y_train, y_svr_train_pred)

svr_test_mse = mean_squared_error(y_test, y_svr_test_pred)
svr_test_r2 = r2_score(y_test, y_svr_test_pred)

svr_results = pd.DataFrame(['Support Vector Machines', svr_train_mse, svr_train_r2, svr_test_mse, svr_test_r2]).transpose()
svr_results.columns = ["method", "Training MSE","Training r2", "testing MSE", "testing R2"]
df_models = pd.concat([lr_results, rf_results, rd_results, svr_results], axis=0)
# print(df_models.reset_index(drop=True))

#Stochastic gradient descent
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
#training the model
sgd.fit(x_train, y_train)

#predicting the model 
y_sgd_train_pred = sgd.predict(x_train)
y_sgd_test_pred = sgd.predict(x_test)

#train data
y_sgd_train_mse = mean_squared_error(y_train, y_sgd_train_pred)
y_sgd_train_r2 = r2_score(y_train, y_sgd_train_pred)

#test data
y_sgd_test_mse = mean_squared_error(y_test, y_sgd_test_pred)
y_sgd_test_r2 = r2_score(y_test, y_sgd_test_pred)

sgd_results = pd.DataFrame(['Stochastic Gradient Descent', y_sgd_train_mse, y_sgd_train_r2, y_sgd_test_mse, y_sgd_test_r2]).transpose()
sgd_results.columns = ["method", "Training MSE","Training r2", "testing MSE", "testing R2"]
df_models = pd.concat([lr_results, rf_results, rd_results, svr_results, sgd_results], axis=0)
print(df_models.reset_index(drop=True))



# DATA VISUALISATION
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
graph = plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')

# plt.show()