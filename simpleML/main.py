import core.vector as vector
from core.vector import Vector
from core.matrix import Matrix
import core.matrix as matrix

def main():
    m1 = Matrix(2, 2, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tr = matrix.transpose(m1)

    m1.disp()
    tr.disp()
if __name__ == "__main__":
    main()