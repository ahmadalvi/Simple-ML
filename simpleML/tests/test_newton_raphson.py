import unittest
import math
from core.vector import Vector
from core.matrix import Matrix
from optim.newton_raphson import NewtonOptimizer


class TestNewtonRaphson(unittest.TestCase):

    def test_quadratic_minimization_1d(self):
        """
        Minimize f(x) = (x - 3)^2
        Global minimum at x = 3
        """

        def loss(theta):
            return (theta.arr[0] - 3) ** 2

        def grad(theta):
            return Vector([2 * (theta.arr[0] - 3)])

        def hess(theta):
            return Matrix([[2]])

        theta0 = Vector([0.0])
        optimizer = NewtonOptimizer(max_iter=20, tol=1e-8)

        theta_opt = optimizer.optimize(theta0, loss, grad, hess)

        self.assertAlmostEqual(theta_opt.arr[0], 3.0, places=6)

    def test_quadratic_minimization_2d(self):
        """
        Minimize f(x, y) = x^2 + y^2
        Minimum at (0, 0)
        """

        def loss(theta):
            return theta.arr[0] ** 2 + theta.arr[1] ** 2

        def grad(theta):
            return Vector(
                [
                    2 * theta.arr[0],
                    2 * theta.arr[1],
                ]
            )

        def hess(theta):
            return Matrix(
                [
                    [2, 0],
                    [0, 2],
                ]
            )

        theta0 = Vector([5.0, -3.0])
        optimizer = NewtonOptimizer(max_iter=10, tol=1e-8)

        theta_opt = optimizer.optimize(theta0, loss, grad, hess)

        self.assertAlmostEqual(theta_opt.arr[0], 0.0, places=6)
        self.assertAlmostEqual(theta_opt.arr[1], 0.0, places=6)

    def test_newton_converges_in_one_step_for_quadratic(self):
        """
        Newton should converge in ONE step for exact quadratic
        """

        def loss(theta):
            return (theta.arr[0] - 10) ** 2

        def grad(theta):
            return Vector([2 * (theta.arr[0] - 10)])

        def hess(theta):
            return Matrix([[2]])

        theta0 = Vector([0.0])
        optimizer = NewtonOptimizer(max_iter=5, tol=1e-12)

        theta_opt = optimizer.optimize(theta0, loss, grad, hess)

        self.assertAlmostEqual(theta_opt.arr[0], 10.0, places=8)

    def test_damping_prevents_singularity_crash(self):
        """
        Hessian nearly singular — damping should prevent crash
        """

        def loss(theta):
            return theta.arr[0] ** 4

        def grad(theta):
            return Vector([4 * theta.arr[0] ** 3])

        def hess(theta):
            return Matrix([[12 * theta.arr[0] ** 2]])

        theta0 = Vector([0.01])
        optimizer = NewtonOptimizer(max_iter=50, tol=1e-6, damping=1e-4)

        theta_opt = optimizer.optimize(theta0, loss, grad, hess)

        self.assertAlmostEqual(theta_opt.arr[0], 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
