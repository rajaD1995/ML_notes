import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:/Users/USER/Documents/Python/NareshIT/26 march/Investment.csv")


x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#eda transformer- to change 
x = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


m_slope = regressor.coef_
print(m_slope)

c_incept = regressor.intercept_
print(c_incept)

#x = np.append(arr= np.ones((50,1)).astype(int),values=x,axis=1) #become array

x = np.append(arr=np.full((50, 1), 42467).astype(int), values=x, axis=1)

x

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog = x_opt).fit()
regressor_OLS.summary()

#backword elemination
#highest p value is x4- remove it
x_opt = x[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog=y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,3]]
regressor_OLS = sm.OLS(endog=y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1]]
regressor_OLS = sm.OLS(endog=y, exog = x_opt).fit()
regressor_OLS.summary()
