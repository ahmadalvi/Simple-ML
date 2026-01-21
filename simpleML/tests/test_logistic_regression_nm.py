import unittest
from models.logistic_regression import (
    LogisticRegression,
    log_loss,
    log_gradient,
)
from core.vector import Vector
from core.matrix import Matrix
from optim.newton_raphson import NewtonOptimizer


class TestLogisticRegression(unittest.TestCase):

    def test_sigmoid_outputs_in_range(self):
        model = LogisticRegression(complexity=1)

        z = Vector([-10, 0, 10])
        probs = model.sigmoid(z)

        for p in probs.arr:
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)

    def test_log_loss_non_negative(self):
        X = Matrix(
            [
                [1, 0],
                [1, 1],
                [1, 2],
            ]
        )
        y = Vector([0, 0, 1])
        theta = Vector([0, 0])

        loss = log_loss(theta, X, y)
        self.assertGreaterEqual(loss, 0.0)

    def test_log_gradient_dimension(self):
        X = Matrix(
            [
                [1, 0],
                [1, 1],
                [1, 2],
            ]
        )
        y = Vector([0, 0, 1])
        theta = Vector([0, 0])

        grad = log_gradient(theta, X, y)
        self.assertEqual(grad.n, theta.n)

    def test_simple_separable_fit_newton(self):
        """
        Linearly separable dataset:
        y = 1 if x > 0, else 0
        """
        X = Matrix(
            [
                [1, -2],
                [1, -1],
                [1, 1],
                [1, 2],
            ]
        )
        y = Vector([0, 0, 1, 1])

        model = LogisticRegression(complexity=1)
        optimizer = NewtonOptimizer(
            max_iter=50,
            tol=1e-8,
        )

        model.fit(X, y, optimizer)
        preds = model.predict(X)

        self.assertEqual(preds.arr, y.arr)

    def test_predict_proba_matches_predict(self):
        X = Matrix(
            [
                [1, -1],
                [1, 1],
            ]
        )

        model = LogisticRegression(complexity=1)
        model.theta = Vector([0, 5])  # strong positive weight

        probs = model.predict_proba(X)
        preds = model.predict(X)

        for p, y in zip(probs.arr, preds.arr):
            self.assertEqual(y, 1 if p >= 0.5 else 0)

    def test_loss_decreases_after_fit(self):
        """
        Ensure training actually reduces log loss
        """
        X = Matrix(
            [
                [1, -2],
                [1, -1],
                [1, 1],
                [1, 2],
            ]
        )
        y = Vector([0, 0, 1, 1])

        model = LogisticRegression(complexity=1)
        optimizer = NewtonOptimizer(max_iter=20, tol=1e-8)

        initial_loss = log_loss(model.theta, X, y)
        model.fit(X, y, optimizer)
        final_loss = log_loss(model.theta, X, y)

        self.assertLess(final_loss, initial_loss)

    def test_newton_converges_quickly(self):
        """
        Newton should converge in very few iterations
        """
        X = Matrix(
            [
                [1, -3],
                [1, -2],
                [1, -1],
                [1, 1],
                [1, 2],
                [1, 3],
            ]
        )
        y = Vector([0, 0, 0, 1, 1, 1])

        model = LogisticRegression(complexity=1)
        optimizer = NewtonOptimizer(max_iter=10, tol=1e-8)

        model.fit(X, y, optimizer)
        preds = model.predict(X)

        accuracy = sum(1 for pi, yi in zip(preds.arr, y.arr) if pi == yi) / y.n

        self.assertGreaterEqual(accuracy, 0.9)


if __name__ == "__main__":
    unittest.main()
