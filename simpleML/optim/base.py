from abc import ABC, abstractmethod
from core.vector import Vector


class Optimizer(ABC):

    def __init__(self, max_iter=1000, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    @abstractmethod
    def optimize(self, theta: Vector, loss_fn, grad_fn):
        pass
