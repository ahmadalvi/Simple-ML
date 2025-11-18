import core.vector as vector
from core.vector import Vector
from core.matrix import Matrix
import core.matrix as matrix
from optim.gradient_descent import (
    gradient_descent as gd,
    testConvergence,
    gridLineSearch,
)


def main():
    def rho(theta):
        return 2 * theta[0] ** 2 - 5 * theta[0] + 3

    def grad(theta):
        return [4 * theta[0] - 5]

    t = gd(
        theta=[0],
        rhoFn=rho,
        gradFn=grad,
        lineSearchFn=gridLineSearch,
        testConvergenceFn=testConvergence,
        maxIter=10,
        toler=1e-6,
    )
    print(t)


if __name__ == "__main__":
    main()
