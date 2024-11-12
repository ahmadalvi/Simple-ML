import math

class Vector():
    def __init__(self, n, arr):
        self.n = n
        self.arr = arr
    
    def disp(self):
        max_len = len(str(max(self.arr))) + 2
        
        if max_len <= 3:
            for i in range(self.n):
                print("[{:^3}]".format(self.arr[i]))
        elif max_len <= 5:
            for i in range(self.n):
                print("[{:^5}]".format(self.arr[i]))
        elif max_len <= 7:
            for i in range(self.n):
                print("[{:^7}]".format(self.arr[i]))
        elif max_len <= 9:
            for i in range(self.n):
                print("[{:^9}]".format(self.arr[i]))
        elif max_len <= 11:
            for i in range(self.n):
                print("[{:^11}]".format(self.arr[i]))
        

def dot(v1: Vector, v2: Vector) -> int:
    """ Calculate the dot product of 2 n-dimensional vectors

    Args:
        v1: A vector object
        v2: A vector object

    Returns:
        Returns the dot product as an real number

    Raises:
      DimensionError: If vectors are not the same size.
    
    """
    sum_ = 0

    if v1.n != v2.n:
        return "Error: Vectors must be the same size."
    else:
        for i in range(v1.n):
            sum_ += v1.arr[i] * v2.arr[i]

    return sum_


def add(v1: Vector, v2: Vector) -> Vector:
    """ Add 2 n-dimensional vectors together

    Args:
        v1: A vector object
        v2: A vector object

    Returns:
        Returns the sum of the 2 vectors as a new vector

    Raises:
      DimensionError: If vectors are not the same size.
    
    """
    vec = Vector(v1.n, [])

    if v1.n != v2.n:
        return "Error: Vectors must be the same size."
    else:
        for i in range(v1.n):
            vec.arr.append(v1.arr[i] + v2.arr[i])
        
    return vec


def sub(v1: Vector, v2: Vector) -> Vector:
    """ Subtract 2 n-dimensional vectors

    Args:
        v1: A vector object
        v2: A vector object

    Returns:
        Returns the difference as a new vector

    Raises:
      DimensionError: If vectors are not the same size.
    
    """
    vec = Vector(v1.n, [])

    if v1.n != v2.n:
        return "Error: Vectors must be the same size."
    else:
        for i in range(v1.n):
            vec.arr.append(v1.arr[i] - v2.arr[i])

def scalarmult(alpha: int, v1: Vector) -> Vector:
    """ Multiply an n-dimensional vector with a scalar

    Args:
        alpha: A real number
        v1: A vector object

    Returns:
        Returns the product as a new vector
    
    """
    vec = Vector(v1.n, [alpha * x for x in v1.arr])
    
    return vec

def norm(v1: Vector, type: str = "euclidean") -> int:
    """ Calculate the norm of a vector

    Args:
        v1: A vector object
        type: {euclidean, manhattan, inf}
              The order of magnitude to calculate

    Returns:
        Returns the magnitude of a vector as an integer
    
    """

    if type == "manhattan":
        return sum([x for x in v1.arr])
    
    elif type == "euclidean":
        return math.sqrt(sum([x**2 for x in v1.arr]))
    
    elif type == "inf":
        m = max([abs(x) for x in v1.arr])
        return m


def cross(v1: Vector, v2: Vector) -> Vector:
    """ Calculates the cross products of 2 3-dimensional vectors

    Args:
        v1: A vector object
        v2: A vector object

    Returns:
        Returns the result of the cross product as a new vector

    Raises:
      DimensionError: If any vector is not 3-D.
    """

    vec = Vector(3, [])

    if (v1.n != 3 or v2.n != 3):
        return "Error: Cross product is only defined for 3-dimensional vectors."
    
    vec.arr[0] = v1.arr[1]*v2.arr[2] - v1.arr[2]*v2.arr[1]
    vec.arr[1] = v1.arr[2]*v2.arr[0] - v1.arr[0]*v2.arr[2]
    vec.arr[2] = v1.arr[0]*v2.arr[1] - v1.arr[1]*v2.arr[0]

    return vec

def proj(v1: Vector, v2: Vector) -> Vector: 
    """ Calculates the projection of v1 onto v2

    Args:
        v1: A vector object
        v2: A vector object

    Returns:
        Returns the result of the cross product as a new vector

    Raises:
      DimensionError: If v2 isn't non-zero
    """

    if (norm(v2, "manhattan") == 0):
        return "Error: v2 must be non-zero."
    
    n = Vector.dot(v1, v2)
    d = Vector.dot(v2, v2)

    vec = Vector.scalarmult(n/d, v2)

    return vec
