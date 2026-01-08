from models.base import Model

class LinearRegression(Model):
    def fit(self, X, y, optimizer):
        ...

    def predict(self, X):
        ...

    def score(self, X, y):
        ...