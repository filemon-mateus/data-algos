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

model = Ridge(alpha=0.0, fit_intercept=False).fit(X_train, y_train)
error = mean_squared_error(y_test, model.predict(X_test))
print(error)
