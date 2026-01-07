import unittest
from optim.gradient_descent import gradient_descent


class TestGradientDescent(unittest.TestCase):

    def test_quadratic_convergence_1d(self):
        """
        Minimize f(x) = 2x^2 - 5x + 3
        True minimum at x = 1.25
        """

        def loss(theta):
            x = theta[0]
            return 2 * x * x - 5 * x + 3

        def grad(theta):
            x = theta[0]
            return [4 * x - 5]

        theta0 = [0.0]

        theta, converged, iters, final_val = gradient_descent(
            theta=theta0,
            rhoFn=loss,
            gradFn=grad,
            maxIter=200,
            toler=1e-4,
            lambdaStepSize=0.05,
        )

        self.assertTrue(converged)
        self.assertAlmostEqual(theta[0], 1.25, places=2)
        self.assertAlmostEqual(final_val, -0.125, places=2)
        self.assertGreater(iters, 0)

    def test_loss_decreases(self):
        """
        Ensure each step reduces the loss (catches sign errors)
        """

        def loss(theta):
            return (theta[0] - 3) ** 2

        def grad(theta):
            return [2 * (theta[0] - 3)]

        theta = [10.0]
        prev_loss = loss(theta)

        for _ in range(10):
            theta, _, _, _ = gradient_descent(
                theta=theta,
                rhoFn=loss,
                gradFn=grad,
                maxIter=1,  # single step
                lambdaStepSize=0.1,
            )
            curr_loss = loss(theta)
            self.assertLessEqual(curr_loss, prev_loss)
            prev_loss = curr_loss

    def test_zero_gradient_no_movement(self):
        """
        If gradient is zero, parameters should not change
        """

        def loss(theta):
            return 5.0

        def grad(theta):
            return [0.0]

        theta0 = [2.0]

        theta, converged, iters, final_val = gradient_descent(
            theta=theta0,
            rhoFn=loss,
            gradFn=grad,
            maxIter=10,
            lambdaStepSize=0.1,
        )

        self.assertEqual(theta, theta0)
        self.assertTrue(converged)

    def test_multidimensional_quadratic(self):
        """
        Minimize f(x, y) = (x - 1)^2 + (y + 2)^2
        """

        def loss(theta):
            x, y = theta[0], theta[1]
            return (x - 1) ** 2 + (y + 2) ** 2

        def grad(theta):
            x, y = theta[0], theta[1]
            return [2 * (x - 1), 2 * (y + 2)]

        theta0 = [10.0, -10.0]

        theta, converged, _, _ = gradient_descent(
            theta=theta0,
            rhoFn=loss,
            gradFn=grad,
            maxIter=300,
            lambdaStepSize=0.01,
        )

        self.assertTrue(converged)
        self.assertAlmostEqual(theta[0], 1.0, places=2)
        self.assertAlmostEqual(theta[1], -2.0, places=2)


if __name__ == "__main__":
    unittest.main()
