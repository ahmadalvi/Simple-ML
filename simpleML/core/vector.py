import math
from core.exceptions import DimensionError

class Vector():
    def __init__(self, arr):
        self.arr = arr
        self.n = len(arr)
    
    def __repr__(self):
        print("[ ", end="")
        for i in range(self.n):
            print(self.arr[i], end=" ")
        print("]")

    def __eq__(self, other):
        return isinstance(other, Vector) and self.arr == other.arr

    def dot(self, other) -> int:
        """ Calculate the dot product of 2 n-dimensional vectors

        Args:
            self: A vector object
            other: A vector object

        Returns:
            Returns the dot product as an real number

        Raises:
        DimensionError: If vectors are not the same size.
        
        """
        sum_ = 0
        
        if self.n != other.n:
            raise DimensionError("Error: Vectors must be the same size.")
        else:
            for i in range(self.n):
                sum_ += self.arr[i] * other.arr[i]

        return sum_


    def __add__(self, other) -> 'Vector':
        """ Add 2 n-dimensional vectors together

        Args:
            self: A vector object
            other: A vector object

        Returns:
            Returns the sum of the 2 vectors as a new vector

        Raises:
        DimensionError: If vectors are not the same size.
        
        """
        vec = Vector([])

        if self.n != other.n:
            raise DimensionError("Error: Vectors must be the same size.")
        else:
            for i in range(self.n):
                vec.arr.append(self.arr[i] + other.arr[i])
            
        return vec


    def __sub__(self, other) -> 'Vector':
        """ Subtract 2 n-dimensional vectors

        Args:
            self: A vector object
            other: A vector object

        Returns:
            Returns the difference as a new vector

        Raises:
        DimensionError: If vectors are not the same size.
        
        """
        vec = Vector([])

        if self.n != other.n:
            raise DimensionError("Error: Vectors must be the same size.")
        else:
            for i in range(self.n):
                vec.arr.append(self.arr[i] - other.arr[i])
        return vec


    def scalarmult(self, alpha: int) -> 'Vector':
        """ Multiply an n-dimensional vector with a scalar

        Args:
            self: A vector object
            alpha: A real number

        Returns:
            Returns the product as a new vector
        
        """
        vec = Vector([alpha * x for x in self.arr])
        
        return vec


    def norm(self, type: str = "euclidean") -> int:
        """ Calculate the norm of a vector

        Args:
            self: A vector object
            type: {euclidean, manhattan, inf}
                The order of magnitude to calculate

        Returns:
            Returns the magnitude of a vector as an integer
        
        """

        if self.n == 0:
            return 0

        if type == "manhattan":
            return sum([abs(x) for x in self.arr])
        elif type == "euclidean":
            return math.sqrt(sum([x**2 for x in self.arr]))
        elif type == "inf":
            m = max([abs(x) for x in self.arr])
            return m


    def cross(self, other) -> 'Vector':
        """ Calculates the cross products of 2 3-dimensional vectors

        Args:
            self: A vector object
            other: A vector object

        Returns:
            Returns the result of the cross product as a new vector

        Raises:
        DimensionError: If any vector is not 3-D.
        """

        if (other.n != 3 or other.n != 3):
            raise DimensionError("Error: Cross product is only defined for 3-dimensional vectors.")
        
        
        arr = [
            self.arr[1]*other.arr[2] - self.arr[2]*other.arr[1], 
            self.arr[2]*other.arr[0] - self.arr[0]*other.arr[2],
            self.arr[0]*other.arr[1] - self.arr[1]*other.arr[0]
        ]

        return Vector(arr)


    def proj(self, other) -> 'Vector': 
        """ Calculates the projection of v1 onto v2

        Args:
            self: A vector object
            other: A vector object

        Returns:
            Returns the result of the cross product as a new vector

        Raises:
        DimensionError: If v2 isn't non-zero
        """

        if (other.norm("manhattan") == 0):
            raise DimensionError("Error: v2 must be non-zero.")
        
        n = self.dot(other)
        d = other.dot(other)

        vec = other.scalarmult(n/d)

        return vec
