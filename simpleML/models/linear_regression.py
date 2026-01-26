from .base import Model
from core.vector import Vector
from core.matrix import Matrix


class LinearRegression(Model):
    def __init__(self, complexity: int = 1, l2: float = 0.0):
        self.complexity = complexity
        self.l2 = l2
        self.theta = Vector([0] * (complexity + 1))

    def predict(self, X: Matrix) -> Vector:
        """Predict target values for given input data using the learned linear model.

        Args:
            X: A Matrix object where each row represents a data point and each column represents a feature.

        Returns:
            Returns a Vector object containing the predicted target values for each data point in X.
        """
        return X.vector_mult(self.theta)

    def fit(self, X, y, optimizer) -> "LinearRegression":
        """Fit the linear regression model to the provided training data.

        Args:
            X: A Matrix object where each row represents a data point and each column represents a feature.
            y: A Vector object containing the target values for each data point in X.
            optimizer: An optimizer object used to update the model parameters.

        Returns:
            Returns the fitted LinearRegression model instance.
        """

        def loss_fn(theta: Vector) -> float:
            """
            Compute the regularized mean squared error loss.

            Args:
                theta: A Vector object representing the model parameters.
            Returns:
                The computed loss as a float.
            """
            base_loss = mse_loss(theta, X, y)
            reg_loss = (self.l2 / 2) * coef_sq(theta)
            return base_loss + reg_loss

        def grad_fn(theta: Vector) -> Vector:
            """
            Compute the gradient of the regularized mean squared error loss.

            Args:
                theta: A Vector object representing the model parameters.
            Returns:
                A Vector object representing the gradient of the loss with respect to theta.
            """
            base_grad = mse_gradient(theta, X, y)
            reg_grad = [0.0] + [self.l2 * ti for ti in theta.arr[1:]]

            return base_grad + Vector(reg_grad)

        self.theta = optimizer.optimize(self.theta, loss_fn, grad_fn)

        return self


def mse_loss(theta: Vector, X: Matrix, y: Vector) -> float:
    predictions = X.vector_mult(theta)
    return sum((yt - yp) ** 2 for yt, yp in zip(y.arr, predictions.arr)) / 2


def mse_gradient(theta: Vector, X: Matrix, y: Vector) -> Vector:
    predictions = X.vector_mult(theta)
    errors = Vector([(yp - yt) for yt, yp in zip(y.arr, predictions.arr)])
    return X.transpose().vector_mult(errors)


def coef_sq(theta):
    return sum(ti**2 for ti in theta.arr[1:])
