import numpy as np
import sklearn.datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

polyf = PolynomialFeatures(degree=2, include_bias=True)
X_train = polyf.fit_transform(X_train)
X_test = polyf.fit_transform(X_test)

alpha_vals = np.array([0.0, 0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 10.0])
error_vals = np.zeros_like(alpha_vals)

for i in range(len(alpha_vals)):
    alpha = alpha_vals[i]
    model = Ridge(alpha=alpha, fit_intercept=False).fit(X_train, y_train)
    error = mean_squared_error(y_test, model.predict(X_test))
    error_vals[i] = error

np.savetxt('data/alpha-vs-error.csv', np.column_stack((alpha_vals, error_vals)), delimiter=',')
