

class RegressionMixin:

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class RVRModel(RegressionMixin):
    def __init__(self):
        from skrvm import RVR
        self.model = RVR(
            verbose=False,
            kernel="rbf",
        )


class RidgeModel(RegressionMixin):
    def __init__(self):
        from sklearn.linear_model import Ridge
        self.model = Ridge(
        )


class LinearRegModel(RegressionMixin):
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()


