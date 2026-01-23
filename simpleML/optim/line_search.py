from typing import Callable
from core.vector import Vector


def line_search(theta: Vector, d: Vector, loss_fn: Callable[[Vector], float]):
    lambdaStepSize = 0.1
    lambdaMax = 1.0

    lambdas = [0.0] + [
        k * lambdaStepSize for k in range(1, int(lambdaMax / lambdaStepSize) + 1)
    ]

    theta_candidates = [theta + d.scalarmult(l) for l in lambdas]
    values = [loss_fn(candidate) for candidate in theta_candidates]

    best_idx = values.index(min(values))

    return lambdas[best_idx]
