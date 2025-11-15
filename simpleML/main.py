import core.vector as vector
from core.vector import Vector
from core.matrix import Matrix
import core.matrix as matrix


def main():
    L = Matrix([
            [1, 0, 0],
            [2, 1, 0],
            [-1, 4, 1]
    ])
    b = Vector([1, 2, 3])

    y_expected = Vector([1, 0, 4])
    y = L.forward_substitution(b)
    print(y.arr)
    print(y_expected.arr)


if __name__ == "__main__":
    main()
