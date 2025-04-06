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
# poly_reg = PolynomialFeatures(degree=5)

poly_reg = PolynomialFeatures(
    degree=5,                      # The degree of the polynomial features. 
                                   # degree=2 means x, x², x1*x2 (interaction terms).
    
    interaction_only=False,      # If True, only interaction features are produced (e.g., x1*x2), no power terms (e.g., x1², x2²).
    
    include_bias=True,             # If True, includes a bias column (all ones) as the first feature. Set to False if your model already includes bias.
    
    order="C"                      # Order of output array: "C" (row-major, default) or "F" (column-major). Affects memory layout, not model logic.
)

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
# svr_reg = SVR(kernel='poly', gamma ='scale', degree = 5,C = 1 )
svr_reg = SVR(
    kernel="rbf",                   # Specifies the kernel type to be used: 
                                    # "linear", "poly", "rbf" (default), "sigmoid", or callable. Determines how input data is mapped.
    
    degree=3,                       # Degree of the polynomial kernel function ("poly" only). Ignored by other kernels.
    
    gamma="scale",                  # Kernel coefficient. 
                                    # "scale" = 1 / (n_features * X.var()), 
                                    # "auto" = 1 / n_features. Can also be a float.
    
    coef0=0.0,                      # Independent term in kernel function. Used in "poly" and "sigmoid".
    
    tol=1e-3,                       # Tolerance for stopping criterion. Smaller values lead to more precise solutions but slower training.
    
    C=1.0,                          # Regularization parameter. Higher C = less regularization = overfitting risk. Lower C = more regularization.
    
    epsilon=0.1,                    # Defines the epsilon-tube within which no penalty is associated in the training loss function. Controls sensitivity.
    
    shrinking=True,                # Whether to use shrinking heuristic (improves training time in some cases).
    
    cache_size=200,                # Size of the kernel cache (in MB). Increase if you have enough memory for faster computation.
    
    verbose=False,                 # Enable verbose output during training. Set to True to monitor progress.
    
    max_iter=-1                    # Hard limit on iterations. -1 = no limit. Set a positive value to limit training time.
)
svr_reg.fit(X,y)
# svr model prediction
svr_reg_pred = svr_reg.predict([[6.5]])
svr_reg_pred
# ---------------------------------------------------------------------------------

# knn regressor 
from sklearn.neighbors import KNeighborsRegressor
#knn_reg = KNeighborsRegressor(n_neighbors=4, weights='distance')


knn_reg = KNeighborsRegressor(
    n_neighbors=5,                   # Number of nearest neighbors to use. Lower = more sensitive to noise. Tune based on bias-variance balance.
    
    weights="uniform",              # How to weight the contribution of neighbors: 
                                    # "uniform" = all neighbors equal, 
                                    # "distance" = closer neighbors have more influence.
    
    algorithm="auto",               # Algorithm to compute nearest neighbors:
                                    # "auto" = selects best, 
                                    # "ball_tree", "kd_tree", or "brute".
    
    leaf_size=30,                   # Leaf size for BallTree or KDTree. Affects speed/memory. Tune when using those algorithms.
    
    p=2,                            # Power parameter for Minkowski distance: 
                                    # p=1 is Manhattan, p=2 is Euclidean (default). Affects distance calculation.
    
    metric="minkowski",             # Distance metric to use. Defaults to Minkowski; can be others like "euclidean", "manhattan", or custom.
    
    metric_params=None,             # Additional parameters for the metric function (if any). Usually None.
    
    n_jobs=None                     # Number of parallel jobs to run. -1 = all processors. Speeds up computation.
)
knn_reg.fit(X,y)

# prediction 
knn_reg_pred = knn_reg.predict([[6.5]])
knn_reg_pred
# ---------------------------------------------------------------------------------
#decission tree algorithm
from sklearn.tree import DecisionTreeRegressor
# dt_reg = DecisionTreeRegressor(criterion='absolute_error', splitter='random')
dt_reg = DecisionTreeRegressor(
    criterion="squared_error",      # Function to measure quality of a split: "squared_error" (default), "friedman_mse", "absolute_error", "poisson"
    
    splitter="best",                # Strategy used to choose the split at each node: "best" or "random". "random" can introduce randomness to avoid overfitting.
    
    max_depth=None,                       # The maximum depth of the tree. None means nodes are expanded until all leaves are pure. Controls overfitting.
    
    min_samples_split=2,                 # Minimum number of samples required to split an internal node. Higher value reduces overfitting but can underfit.
    
    min_samples_leaf=1,                  # Minimum number of samples required to be at a leaf node. Helps smooth the model and reduce overfitting.
    
    min_weight_fraction_leaf=0.0,        # Minimum weighted fraction of the input samples required to be at a leaf node. For datasets with sample weights.
    
    max_features=None,                   # Number of features to consider when looking for the best split. None means all features. Can reduce variance if limited.
    
    random_state=0,                   # Controls randomness for reproducibility. Use an integer (e.g., 42) for consistent results.
    
    max_leaf_nodes=None,                 # Limits the number of leaf nodes. Best-first strategy used if defined. Controls overfitting.
    
    min_impurity_decrease=0.0,          # A node will split if the split decreases impurity by at least this value. Useful for controlling tree growth.
    
    ccp_alpha=0.0,                       # Complexity parameter used for Minimal Cost-Complexity Pruning. Higher values prune more, reducing overfitting.
    
    monotonic_cst=None                   # Constraints for monotonic relationship between features and output. Use for monotonicity-aware modeling.
)
dt_reg.fit(X,y)

dt_reg_pred = dt_reg.predict([[6.5]])
dt_reg_pred

# ---------------------------------------------------------------------------------
#Random forest
from sklearn.ensemble import RandomForestRegressor
#rf_reg= RandomForestRegressor(random_state=0)
rf_reg = RandomForestRegressor(
    n_estimators=100,                     # Number of trees in the forest. More trees = better performance but slower. Common values: 100, 200, 500.
    
    criterion="squared_error",           # Function to measure quality of a split. "squared_error" is for regression tasks. Alternatives: "absolute_error", etc.
    
    max_depth=None,                      # Max depth of each tree. None means nodes expand until pure. Controls overfitting.
    
    min_samples_split=2,                # Minimum samples required to split a node. Higher value prevents overfitting but can underfit.
    
    min_samples_leaf=1,                 # Minimum samples required at a leaf node. Increases robustness by smoothing predictions.
    
    min_weight_fraction_leaf=0.0,       # Minimum weighted fraction of the input samples required to be at a leaf node. Used with sample weights.
    
    max_features=1.0,                   # Number of features to consider when looking for best split. Float = % of features (e.g., 0.5). Controls variance.
    
    max_leaf_nodes=None,                # Max number of leaf nodes per tree. Limits growth to avoid overfitting.
    
    min_impurity_decrease=0.0,         # A node splits if this threshold impurity decrease is met. Controls unnecessary splits.
    
    bootstrap=True,                     # Whether bootstrap samples are used when building trees. False = sampling without replacement.
    
    oob_score=False,                    # Whether to use out-of-bag samples to estimate generalization score. Set to True to validate without cross-validation.
    
    n_jobs=None,                        # Number of CPU cores to use. -1 = all cores. Speeds up training.
    
    random_state=None,                  # Controls randomness for reproducibility. Use an integer (e.g., 42) to get consistent results.
    
    verbose=0,                          # Controls verbosity. 0 = silent, >0 = prints training progress.
    
    warm_start=False,                   # When True, reuse previous solution and add more trees. Enables incremental learning.
    
    ccp_alpha=0.0,                      # Complexity parameter for Minimal Cost-Complexity Pruning. Higher = more pruning, less overfitting.
    
    max_samples=None,                   # Number or fraction of samples to draw from X to train each tree. Used only if bootstrap=True.
    
    monotonic_cst=None                  # Optional. Apply monotonicity constraints per feature. Useful in specific domain-aware modeling.
)
rf_reg.fit(X,y)

rf_reg_pred = rf_reg.predict([[6.5]])
rf_reg_pred
# ---------------------------------------------------------------------------------





