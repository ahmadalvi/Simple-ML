import vector
from vector import Vector
from matrix import Matrix
import matrix

def main():
    vector1 = Vector(4, [1, 2, 3, 4])
    vector2 = Vector.scalarmult(3, vector1)
    vector2.disp()
    
if __name__ == "__main__":
    main()