class Model:
    def fit(self, X, y, optimizer):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError