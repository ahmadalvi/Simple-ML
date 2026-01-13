from .base import Model
from core.vector import Vector
from core.matrix import Matrix


class LinearRegression(Model):
    def __init__(self, complexity: int = 1):
        self.complexity = complexity
        self.theta = Vector([0] * (complexity + 1))

    def predict(self, X):
        return X.vector_mult(self.theta)

    def fit(self, X, y, optimizer):
        def loss_fn(theta):
            return mse_loss(theta, X, y)

        def grad_fn(theta):
            return mse_gradient(theta, X, y)

        self.theta = optimizer.optimize(self.theta, loss_fn, grad_fn)

        return self


def mse_loss(theta: Vector, X: Matrix, y: Vector) -> float:
    predictions = X.vector_mult(theta)
    return sum((yt - yp) ** 2 for yt, yp in zip(y.arr, predictions.arr)) / 2


def mse_gradient(theta: Vector, X: Matrix, y: Vector) -> Vector:
    predictions = X.vector_mult(theta)
    errors = Vector([(yp - yt) for yt, yp in zip(y.arr, predictions.arr)])
    return X.transpose().vector_mult(errors)
