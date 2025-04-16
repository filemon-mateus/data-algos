import numpy as np
import sklearn.datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)

model = LinearRegression().fit(X_train, y_train)
coefs = np.insert(model.coef_, 0, model.intercept_)
print(coefs)
