import numpy as np
import sklearn.datasets

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

num_splits = 5
alpha_vals = np.array([0.0, 0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 10.0])
error_vals = np.zeros_like(alpha_vals)
error_fold = np.zeros(num_splits)

X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)
X = PolynomialFeatures(degree=2, include_bias=True).fit_transform(X)

fold = KFold(n_splits=num_splits)
fold.get_n_splits(X)

for i in range(len(alpha_vals)):
    alpha = alpha_vals[i]
    model = Ridge(alpha=alpha, fit_intercept=False)

    for j, (train_idx, test_idx) in enumerate(fold.split(X)):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        model = model.fit(X_train, y_train)
        error = mean_squared_error(y_test, model.predict(X_test))
        error_fold[j] = error

    error_vals[i] = error_fold.mean()

np.savetxt('data/alpha-vs-error-kfold.csv', np.column_stack((alpha_vals, error_vals)), delimiter=',')
