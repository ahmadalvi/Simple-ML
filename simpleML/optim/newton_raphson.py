from typing import Callable
from core.matrix import Matrix, identity
from core.vector import Vector
from .base import Optimizer


class NewtonOptimizer(Optimizer):
    def __init__(self, max_iter=100, tol=1e-6, damping=1e-4):
        super().__init__(max_iter=max_iter, tol=tol)
        self.damping = damping

    def optimize(
        self,
        theta: Vector,
        loss_fn,
        grad_fn: Callable[[Vector], Vector],
        hess_fn: Callable[[Vector], Matrix],
    ) -> Vector:
        for _ in range(self.max_iter):
            grad = grad_fn(theta)
            H = hess_fn(theta)

            H_damped = H.add(identity(H.rows).scalar_mult(self.damping))
            step = H_damped.inverse().vector_mult(grad)
            theta = theta - step

            if step.norm() < self.tol:
                break

        return theta
