
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Load the dataset
dataset = pd.read_csv(r'C:/Users/USER/Documents/Python/NareshIT/20 march SLR workshop/Salary_Data.csv')

x = dataset.iloc[:, :-1]   #Indepedent variables, dataframe

y = dataset.iloc[:, -1]  #Dependdent variable, series

# train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#x- df and y- series, so to make x_train & x_test array of float
x_train = x_train.values.reshape(-1, 1)

x_test = x_test.values.reshape(-1, 1)

#train to build a model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test) 

# Now copmare with test values with model.
plt.scatter(x_test, y_test, color = 'red')  
plt.plot(x_test, regressor.predict(x_test), color = 'blue', marker='o')  
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Uptill we just bulit a model. Yet to predict future for that c and m need to be calculated.
# Statistic calculation is yet to be calculated to check if it is good model.

#compare train values with model
plt.scatter(x_train, y_train, color = 'red')  
plt.plot(x_train, regressor.predict(x_train), color = 'blue', marker='o')  
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#--------------------Future prediction-----------------------------------------
#y=mx+c
#m(slope)
m=regressor.coef_
print(f"Coefficient: {m}")

#c- constant
c=regressor.intercept_
print("Intercept:",{c})

#comparision b/w predicted and actual test data.
comparision = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparision)

#comparision b/w predicted and actual train data.
comparision_train = pd.DataFrame({'Actual': y_train, 'Predicted': regressor.predict(x_train)})
print(comparision_train)

#prediction of salary of 12y experience.
salary_12y = m*12 + c
print(salary_12y)
# This is the the way you should predict the future.
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

#bias score
bias = regressor.score(x_train, y_train)
print(bias)

#variance score
variance = regressor.score(x_test, y_test)
print(variance)

#----------------------------------STATISTICS----------------------------------
#mean
dataset.mean()
dataset['YearsExperience'].mean()
dataset['Salary'].mean()

#median
dataset.median()
dataset['YearsExperience'].median()
dataset['Salary'].median()

#mode
dataset.mode()
dataset['YearsExperience'].mode()
dataset['Salary'].mode()

#VARIANCE: spread data towards mean.
dataset.var()
dataset['YearsExperience'].var()
dataset['Salary'].var()

#stadard deviation: compare to mean if it is less then the data points are close to mean.
#if std is high compare to mean then datapoints are more spread.
dataset.std()
dataset['YearsExperience'].std()
dataset['Salary'].std()

#Coefficient of Variation(CV)
#This will tell you how close the std is to mean. (std/mean)x100
from scipy.stats import variation

variation(dataset.values)
#std of YearsExperience is 52.5% of mean of YearsExperience.
#std of Salary is 35.4% of the mean of Salary.
variation(dataset['YearsExperience'].values) #0.5251
variation(dataset['Salary'].values)  #0.354


#How to know dependent variable has positive relation or negative with any dependent variable
# By correletion-- corr()
dataset.corr()
dataset['YearsExperience'].corr(dataset['Salary'])


#Skewness = 0: Symmetrical distribution.--Mean = Median = Mode

#Skewness > 0: Right skew (positive skew)--Mean > Median > Mode--Outliers on Right.
#   -Most values are concentrated on the left.
#   -Income distribution (few people earning very high salaries).
#   -A few extreme values (outliers) stretch the tail to the right.

#Skewness < 0: Left skew (negative skew)--Mean < Median < Mode--Outliers on Left.
#   -Exam scores where most students score high, but a few score very low.
#   -Most values are concentrated on the right.
#   -A few extreme values stretch the tail to the left.

#skewness
dataset.skew()
dataset['YearsExperience'].skew()
dataset['Salary'].skew()
#both attributes are positive skewness.
#   -Some employees have high salay
#   -Some employees have high years of experience too.
#   -most datapoints are concentreted on the left of the mean.


#Standard Error (SE)
#compare sample & population
#Standard Error (SE) measures the variability of a sample mean compared to the population mean.
#It tells how much the sample mean is likely to vary if we take multiple samples from the same population.

#Low SE:
# Sample mean is close to the population mean.
# Less variability between sample means.

❗# High SE:
# Sample mean may vary widely from the population mean.
# More variability between sample means.
dataset.sem()
dataset['YearsExperience'].sem()
dataset['Salary'].sem()


import scipy.stats as stats
#Z-score-- for scalling for ML model. scalling means bring down all numerical data in same range.
#feature scaling- (-3 to 3)
dataset.apply(stats.zscore)

#degree of freedom
a=dataset.shape[0]
b=dataset.shape[1]
degree_of_freedom = a-b
print(degree_of_freedom)


#sum of square regression(ssr)
y_mean = np.mean(y)
ssr = np.sum((y_pred-y_mean)**2)
print(ssr)


#SSE
y = y[0:6]
sse = np.sum((y-y_pred)**2)
print(sse)

#SST
mean_total = np.mean(dataset.values)
sst = np.sum((dataset.values-mean_total)**2)
print(sst)

#to check the model accuracy------------------------------------------------------

#r- square (0 to 1- for good model)
R_square = 1-(ssr/sst)
R_square
 
#bias score(Should be high 0.9 or 0.85 or like that)
bias = regressor.score(x_train, y_train)
#variance score(Should be high 0.9 or 0.85 or like that)
variance = regressor.score(x_test, y_test)

#Interpreting MSE
#✅ Low MSE:
    #Model predictions are close to actual values.
    #Better model performance.
    #MSE is much smaller than the variance of the target variable.
    #MSE is much smaller than the square of the mean.
    #🎒 Example 1: Predicting Student Marks
    #Actual Scores: [50, 60, 55, 70]
    #Predicted Scores: [52, 62, 54, 72]
    #MSE: 3.25
    #✅ Interpretation:
    #Since the scores range between 50 to 70, an MSE of 3.25 is low, indicating a good model.

#❗️ High MSE:
    #Model predictions deviate significantly from actual values.
    #Poor model performance.
    #MSE is close to or higher than the variance of the target variable.
    #MSE is comparable to or larger than the square of the mean.
    
    #💸 Example 2: Predicting Employee Salaries
    #Actual Salaries: [45,000, 50,000, 55,000, 60,000]
    #Predicted Salaries: [46,000, 49,000, 54,500, 61,000]
    #MSE: 625,000
    #❗️ Interpretation:
    #Since the salary values are in the range of 45,000 to 60,000, an MSE of 625,000 is relatively high, suggesting that the predictions have a higher error.
#Using RMSE for Better Interpretation
#✅ Why RMSE?
#MSE is in squared units, making it harder to interpret directly.
#Root Mean Squared Error (RMSE) is the square root of MSE and is in the same units as the target variable, making it easier to compare.

train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

print(regressor)