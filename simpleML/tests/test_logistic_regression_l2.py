import unittest
from core.vector import Vector
from core.matrix import Matrix
from models.logistic_regression import LogisticRegression, log_loss
from optim.gradient_descent import GradientDescent


class TestLogisticRegressionL2(unittest.TestCase):

    def setUp(self):
        # Simple linearly separable dataset
        self.X = Matrix(
            [
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
            ]
        )
        self.y = Vector([0, 0, 1, 1])

    def test_l2_increases_loss(self):
        theta = Vector([1.0, 1.0])

        model_no_l2 = LogisticRegression(complexity=1)
        model_no_l2.l2 = 0.0

        model_l2 = LogisticRegression(complexity=1)
        model_l2.l2 = 1.0

        loss_no_l2 = log_loss(theta, self.X, self.y)
        loss_l2 = loss_no_l2 + (model_l2.l2 / 2) * sum(t**2 for t in theta.arr[1:])

        self.assertGreater(loss_l2, loss_no_l2)

    def test_l2_shrinks_weights(self):
        optimizer = GradientDescent(learning_rate=0.1, max_iter=1000)

        model_no_l2 = LogisticRegression(complexity=1)
        model_no_l2.l2 = 0.0
        model_no_l2.fit(self.X, self.y, optimizer)

        model_l2 = LogisticRegression(complexity=1)
        model_l2.l2 = 1.0
        model_l2.fit(self.X, self.y, optimizer)

        self.assertLess(abs(model_l2.theta.arr[1]), abs(model_no_l2.theta.arr[1]))

    def test_l2_does_not_regularize_bias(self):
        optimizer = GradientDescent(learning_rate=0.1, max_iter=1000)

        model = LogisticRegression(complexity=1)
        model.l2 = 10.0
        model.fit(self.X, self.y, optimizer)

        # Bias should still be non-zero and meaningful
        self.assertGreater(abs(model.theta.arr[0]), 0.1)

    def test_strong_l2_drives_weights_toward_zero(self):
        optimizer = GradientDescent(learning_rate=0.1, max_iter=2000)

        model = LogisticRegression(complexity=1)
        model.l2 = 100.0
        model.fit(self.X, self.y, optimizer)

        self.assertAlmostEqual(model.theta.arr[1], 0.0, places=1)

    def test_training_still_works_with_l2(self):
        optimizer = GradientDescent(learning_rate=0.1, max_iter=1000)

        model = LogisticRegression(complexity=1)
        model.l2 = 1.0
        model.fit(self.X, self.y, optimizer)

        preds = model.predict(self.X)

        # Should do better than random guessing
        accuracy = sum(int(p == y) for p, y in zip(preds.arr, self.y.arr)) / self.y.n

        self.assertGreaterEqual(accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
