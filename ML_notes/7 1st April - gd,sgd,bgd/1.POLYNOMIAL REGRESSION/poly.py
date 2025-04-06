import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:/Users/USER/Documents/Python/NareshIT/1 april/emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
# ---------------------------------------------------------------------------------
# linear model  -- linear algor ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# linear regression visualizaton 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('linear regression model (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred
# ---------------------------------------------------------------------------------
# polynomial model  ( bydefeaut degree - 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

# poly nomial visualization 

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('polymodel (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicton 

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

# ---------------------------------------------------------------------------------
# support vector regression model 
from sklearn.svm import SVR
svr_reg = SVR(kernel='poly', gamma ='scale', degree = 5,C = 1 )
svr_reg.fit(X,y)

# svr model prediction
svr_reg_pred = svr_reg.predict([[6.5]])
svr_reg_pred
# ---------------------------------------------------------------------------------

# knn regressor 
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=4, weights='distance')
knn_reg.fit(X,y)

# prediction 
knn_reg_pred = knn_reg.predict([[6.5]])
knn_reg_pred
# ---------------------------------------------------------------------------------
#decission tree algorithm
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='absolute_error', splitter='random')
dt_reg.fit(X,y)

dt_reg_pred = dt_reg.predict([[6.5]])
dt_reg_pred

# ---------------------------------------------------------------------------------
#Random forest
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(X,y)

rf_reg_pred = rf_reg.predict([[6.5]])
rf_reg_pred
# ---------------------------------------------------------------------------------





