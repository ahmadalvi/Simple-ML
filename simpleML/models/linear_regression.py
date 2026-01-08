from .base import Model
from core.vector import Vector
from core.matrix import Matrix
from optim.base import Optimizer


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
        
        self.theta = Optimizer.optimize(
            theta = self.theta, 
            loss_fn = loss_fn,
            grad_fn = grad_fn
        )
    
        return self


def mse_loss(theta: Vector, X: Matrix, y: Vector) -> float:
    predictions = X.vector_mult(theta)
    return sum((yt - yp) ** 2 for yt, yp in zip(y, predictions)) / 2

def mse_gradient(theta: Vector, X: Matrix, y: Vector) -> Vector: 
    predictions = X.vector_mult(theta)
    errors = Vector([(yt - yp) for yt, yp in zip(y, predictions)])
    return X.transpose().vector_mult(errors)