import unittest
from core.vector import Vector
from optim.gradient_descent import GradientDescent
from optim.line_search import line_search


class TestGradientDescent(unittest.TestCase):

    def test_quadratic_convergence_1d(self):
        """
        Minimize f(x) = x^2
        """

        def loss(theta):
            return theta.arr[0] ** 2

        def grad(theta):
            return Vector([2 * theta.arr[0]])

        theta0 = Vector([10.0])
        optimizer = GradientDescent(learning_rate=0.1, tol=1e-6)

        theta = optimizer.optimize(theta0, loss, grad)

        self.assertAlmostEqual(theta.arr[0], 0.0, places=4)

    def test_quadratic_shifted(self):
        """
        Minimize f(x) = (x - 3)^2
        """

        def loss(theta):
            return (theta.arr[0] - 3) ** 2

        def grad(theta):
            return Vector([2 * (theta.arr[0] - 3)])

        theta0 = Vector([0.0])
        optimizer = GradientDescent(learning_rate=0.1)

        theta = optimizer.optimize(theta0, loss, grad)

        self.assertAlmostEqual(theta.arr[0], 3.0, places=4)

    def test_multidimensional_quadratic(self):
        """
        Minimize f(x, y) = x^2 + y^2
        """

        def loss(theta):
            return theta.arr[0] ** 2 + theta.arr[1] ** 2

        def grad(theta):
            return Vector([2 * theta.arr[0], 2 * theta.arr[1]])

        theta0 = Vector([5.0, -3.0])
        optimizer = GradientDescent(learning_rate=0.1, max_iter=10000, tol=1e-8, normalize_grad=False)

        theta = optimizer.optimize(theta0, loss, grad)

        self.assertAlmostEqual(theta.arr[0], 0.0, places=4)
        self.assertAlmostEqual(theta.arr[1], 0.0, places=4)

    def test_loss_monotonic_decrease(self):
        """
        Ensure GD moves downhill
        """
        losses = []

        def loss(theta):
            val = theta.arr[0] ** 2
            losses.append(val)
            return val

        def grad(theta):
            return Vector([2 * theta.arr[0]])

        theta0 = Vector([5.0])
        optimizer = GradientDescent(learning_rate=0.1, max_iter=5)

        optimizer.optimize(theta0, loss, grad)

        for i in range(1, len(losses)):
            self.assertLessEqual(losses[i], losses[i - 1])


if __name__ == "__main__":
    unittest.main()
