import unittest
from xml.parsers.expat import model

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

    def test_l2_does_not_regularize_bias(self):
        """
        Bias term theta[0] should not be affected by L2 regularization
        """
        X = Matrix(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]
        )
        y = Vector([3, 5, 7, 9])  # y = 2x + 1

        opt = GradientDescent(learning_rate=0.01, max_iter=1000)

        model_no_reg = LinearRegression(complexity=1, l2=0.0)
        model_no_reg.fit(X, y, opt)

        model_l2 = LinearRegression(complexity=1, l2=10.0)
        model_l2.fit(X, y, opt)
        # Bias should be approximately the same
        self.assertAlmostEqual(
            model_no_reg.theta.arr[0], model_l2.theta.arr[0], places=0
        )

    def test_l2_shrinks_weights(self):
        """
        L2 regularization should shrink non-bias weights
        """

        X = Matrix(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]
        )
        y = Vector([3, 5, 7, 9])  # y = 2x + 1

        opt = GradientDescent(learning_rate=0.01, max_iter=1000)

        model_no_reg = LinearRegression(complexity=1, l2=0.0)
        model_no_reg.fit(X, y, opt)

        model_l2 = LinearRegression(complexity=1, l2=10.0)
        model_l2.fit(X, y, opt)

        # Weight magnitude should be smaller with L2
        self.assertLess(abs(model_l2.theta.arr[1]), abs(model_no_reg.theta.arr[1]))

    def test_strong_l2_drives_weights_to_zero(self):
        """
        Very large L2 should heavily penalize weights
        """

        X = Matrix(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]
        )
        y = Vector([3, 5, 7, 9])

        opt = GradientDescent(learning_rate=0.01, max_iter=1000)

        model = LinearRegression(complexity=1, l2=100.0)
        model.fit(X, y, opt)

        # Weight should be near zero
        self.assertAlmostEqual(model.theta.arr[1], 0.0, places=1)

    def test_l2_increases_loss(self):
        """
        L2 regularization should increase loss for same parameters
        """

        X = Matrix(
            [
                [1, 1],
                [1, 2],
            ]
        )
        y = Vector([3, 5])

        theta = Vector([1.0, 2.0])
        l2 = 10.0

        def coef_sq(theta):
            return sum(ti**2 for ti in theta.arr[1:])

        base_loss = mse_loss(theta, X, y)
        reg_loss = base_loss + (l2 / 2) * coef_sq(theta)

        self.assertGreater(reg_loss, base_loss)


if __name__ == "__main__":
    unittest.main()
