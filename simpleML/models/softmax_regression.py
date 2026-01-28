from .base import Model
from core.vector import Vector
from core.matrix import Matrix, diag
import math


class SoftmaxRegression(Model):
    def __init__(self, complexity: int = 1, classes: int = 2, l2: float = 0.0):
        self.complexity = complexity
        self.classes = classes  
        self.l2 = l2
        self.W = Matrix([[0] * classes for _ in range(complexity + 1)])

    def softmax(self, z: Matrix) -> Matrix:
        """
        Apply the softmax function to a Vector.

        Args:
            z: A Matrix object.

        Returns:
            A Matrix object with the softmax function applied.
        """
        row_max = [max(row) for row in z.mat]
        exp_z = Matrix([[math.exp(zi - row_max[i]) for zi in row] for i, row in enumerate(z.mat)])
        sum_exp = [sum(row) for row in exp_z.mat]
        softmax_mat = Matrix(
            [
                [exp_z.mat[i][j] / sum_exp[i] for j in range(exp_z.cols)]
                for i in range(exp_z.rows)
            ]
        )
        return softmax_mat

    def predict(self, X: Matrix) -> Matrix:
        """
        Predict binary class labels for given input data using the learned logistic model.

        Args:
            X: A Matrix object where each row represents a data point and each column represents a feature.
        Returns:
            A Vector object containing the predicted class labels (0 or 1) for each data point in X.
        """

        probs = self.predict_proba(X)
        return Matrix([
            [row.index(max(row)) for row in probs.mat]
        ])

    def predict_proba(self, X: Matrix) -> Vector:
        """
        Predict probabilities for the positive class for given input data using the learned logistic model.
        Args:
            X: A Matrix object where each row represents a data point and each column represents a feature.
        Returns:
            A Vector object containing the predicted probabilities for the positive class for each data point in X.
        """
        return self.softmax(X.matrix_mult(self.W))

    def fit(self, X: Matrix, y: Vector, optimizer) -> "SoftmaxRegression":
        """
        Fit the softmax regression model to the provided training data.

        Args:
            X: A Matrix object where each row represents a data point and each column represents a feature.
            y: A Vector object containing the target binary class labels for each data point in X.
            optimizer: An optimizer object used to update the model parameters.
        Returns:
            Returns the fitted LogisticRegression model instance.
        """

        def loss_fn(W):
            base_loss = cross_entropy_loss(W, X, y)
            reg_loss = (self.l2 / 2) * coef_sq(W)
            return base_loss + reg_loss

        def grad_fn(W):
            base_grad = cross_entropy_gradient(W, X, y)
            reg_grad = Matrix([[0.0] + [self.l2 * ti for ti in W.mat[1:]]])
            return base_grad + reg_grad

        self.W = optimizer.optimize(self.W, loss_fn, grad_fn)
        return self

    def accuracy(self, X: Matrix, y: Vector) -> float:
        """Calculate the accuracy of the logistic regression model on the given data.
        Args:
            X: A Matrix object where each row represents a data point and each column represents a feature.
            y: A Vector object containing the true binary class labels for each data point in X.
        Returns:
            A float representing the accuracy of the model on the provided data.
        """
        predictions = self.predict(X)
        correct = sum(1 for yp, yt in zip(predictions.arr, y.arr) if yp == yt)
        return correct / len(y.arr)


def cross_entropy_loss(W: Matrix, X: Matrix, Y_one_hot: Matrix) -> float:
    return 1/X.rows * sum(
        -sum(Y_one_hot.mat[i][k] * math.log(X.mat[i].dot(W.get_column(k))) for k in range(W.cols))
        for i in range(X.rows)
    )

def cross_entropy_gradient(W: Matrix, X: Matrix, Y_one_hot: Matrix) -> Matrix:
    return 1/X.rows * X.transpose().matrix_mult(
        Matrix([
            [
                math.exp(X.mat[i].dot(W.get_column(k))) - Y_one_hot.mat[i][k] / sum(math.exp(X.mat[i].dot(W.get_column(k))) for k in range(W.cols))
                for k in range(W.cols)
            ]
            for i in range(X.rows)
        ])
    )


def coef_sq(theta):
    return sum(ti**2 for ti in theta.arr[1:])
