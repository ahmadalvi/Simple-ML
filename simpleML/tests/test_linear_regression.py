import unittest

from models.linear_regression import LinearRegression, mse_loss
from optim.gradient_descent import GradientDescent
from core.matrix import Matrix
from core.vector import Vector


class TestLinearRegression(unittest.TestCase):

    def test_simple_linear_fit(self):
        """
        Fit y = 2x + 1
        """
        X = Matrix(
            [
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
            ]
        )
        y = Vector([1, 3, 5, 7])

        model = LinearRegression(complexity=1)
        optimizer = GradientDescent(learning_rate=0.1, max_iter=5000, tol=1e-6)

        model.fit(X, y, optimizer)
        theta = model.theta.arr

        self.assertAlmostEqual(theta[0], 1.0, places=2)  # bias
        self.assertAlmostEqual(theta[1], 2.0, places=2)  # slope

    def test_predict(self):
        """
        Ensure predict works with learned theta
        """
        model = LinearRegression(complexity=1)
        model.theta = Vector([1.0, 2.0])  # y = 2x + 1

        X = Matrix(
            [
                [1, 4],
                [1, 5],
            ]
        )

        preds = model.predict(X)

        self.assertEqual(preds.arr[0], 9.0)
        self.assertEqual(preds.arr[1], 11.0)

    def test_loss_decreases_after_fit(self):
        """
        Loss should decrease after training
        """
        X = Matrix(
            [
                [1, 0],
                [1, 1],
                [1, 2],
            ]
        )
        y = Vector([1, 3, 5])

        model = LinearRegression(complexity=1)
        optimizer = GradientDescent(learning_rate=0.1, max_iter=2000)

        initial_loss = mse_loss(model.theta, X, y)
        model.fit(X, y, optimizer)
        final_loss = mse_loss(model.theta, X, y)

        self.assertLess(final_loss, initial_loss)

    def test_multidimensional_linear_fit(self):
        """
        Fit y = 3x1 + 2x2 + 1
        """
        X = Matrix(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        )
        y = Vector([1, 4, 3, 6])

        model = LinearRegression(complexity=2)
        optimizer = GradientDescent(learning_rate=0.1, max_iter=5000, tol=1e-6)

        model.fit(X, y, optimizer)
        theta = model.theta.arr

        self.assertAlmostEqual(theta[0], 1.0, places=2)
        self.assertAlmostEqual(theta[1], 3.0, places=2)
        self.assertAlmostEqual(theta[2], 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
