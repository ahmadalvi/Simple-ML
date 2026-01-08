from core.vector import Vector
from .base import Optimizer

class GradientDescent(Optimizer):
    def __init__(
            self, 
            learning_rate = 0.1, 
            max_iter = 1000, 
            tol = 1e-6,
            normalize_grad = True
        ):
        super().__init__(max_iter = max_iter, tol = tol)
        self.learning_rate = learning_rate
        self.normalize_grad = normalize_grad

def optimize(self, theta: Vector, loss_fn, grad_fn) -> Vector:
    for _ in range(self.max_iter):
        grad = grad_fn(theta)

        # --- convergence check (skeleton) ---
            # if ||grad|| < self.tol:
            #     break

            # --- direction ---
            # d = -grad (optionally normalized)

            # --- update ---
            # theta = theta + step * d

        pass

    return theta
