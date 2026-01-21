from .base import Model
from core.vector import Vector
from core.matrix import Matrix, diag
import math


class LogisticRegression(Model):
    def __init__(self, complexity: int = 1, l2: float = 0.0):
        self.complexity = complexity
        self.l2 = l2
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
            base_loss = log_loss(theta, X, y)
            reg_loss = (self.l2 / 2) * coef_sq(theta)
            return base_loss + reg_loss

        def grad_fn(theta):
            base_grad = log_gradient(theta, X, y)
            reg_grad = Vector([0.0] + [self.l2 * ti for ti in theta.arr[1:]])
            return base_grad + reg_grad

        def hess_fn(theta):
            p = self.sigmoid(X.vector_mult(theta))
            W = diag([pi * (1 - pi) for pi in p.arr])
            H = X.transpose().matrix_mult(W).matrix_mult(X)

            for i in range(1, H.rows):
                H.mat[i][i] += self.l2

            return H

        self.theta = optimizer.optimize(self.theta, loss_fn, grad_fn, hess_fn)

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


def coef_sq(theta):
    return sum(ti**2 for ti in theta.arr[1:])
