import vector
from vector import Vector
from matrix import Matrix
import matrix

def main():
    m1 = Matrix(3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    tr = matrix.transpose(m1)

    m1.disp()
    tr.disp()
if __name__ == "__main__":
    main()