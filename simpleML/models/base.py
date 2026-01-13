from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, X, y, optimizer):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError
