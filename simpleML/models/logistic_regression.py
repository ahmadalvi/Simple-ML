from .base import Model
from core.vector import Vector
from core.matrix import Matrix
import math


class LogisticRegression(Model):
    def __init__(self, complexity: int = 1):
        self.complexity = complexity
        self.theta = Vector([0] * (complexity + 1))

    def sigmoid(self, z: Vector) -> Vector:
        return Vector([1 / (1 + math.exp(-zi)) for zi in z.arr])
