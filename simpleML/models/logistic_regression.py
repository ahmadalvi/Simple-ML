from .base import Model
from core.vector import Vector
from core.matrix import Matrix
import math


class LogisticRegression(Model):
    def __init__(self, complexity: int = 1):
        self.complexity = complexity
        self.theta = Vector([0] * (complexity + 1))

    def sigmoid(self, z: Vector) -> Vector:
        return Vector([1 / (1 + math.exp(-zi)) for zi in z.arr])

    def predict(self, X: Matrix) -> Vector:
        probs = self.predict_proba(X)
        return Vector([1 if p >= 0.5 else 0 for p in probs.arr])

    def predict_proba(self, X: Matrix) -> Vector:
        return self.sigmoid(X.vector_mult(self.theta))

    def fit(self, X: Matrix, y: Vector, optimizer):
        def loss_fn(theta):
            return log_loss(theta, X, y)

        def grad_fn(theta):
            return log_gradient(theta, X, y)

        self.theta = optimizer.optimize(self.theta, loss_fn, grad_fn)

        return self

    def accuracy(self, X: Matrix, y: Vector) -> float:
        predictions = self.predict(X)
        correct = sum(1 for yp, yt in zip(predictions.arr, y.arr) if yp == yt)
        return correct / len(y.arr)


def log_loss(theta: Vector, X: Matrix, y: Vector) -> float:
    z = X.vector_mult(theta)
    probs = Vector([1 / (1 + math.exp(-zi)) for zi in z.arr])

    eps = 1e-15
    loss = 0.0

    for yi, pi in zip(y.arr, probs.arr):
        pi = max(eps, min(1 - eps, pi))
        loss += -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi))

    return loss


def log_gradient(theta: Vector, X: Matrix, y: Vector) -> Vector:
    return X.transpose().vector_mult(
        Vector(
            [
                (1 / (1 + math.exp(-xi)) - yi)
                for xi, yi in zip(X.vector_mult(theta).arr, y.arr)
            ]
        )
    )
