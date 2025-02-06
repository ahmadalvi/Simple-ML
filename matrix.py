class Matrix:
    def __init__(self, n: int, mat):
        self.n = n
        self.mat = mat

    def disp(self):
        rows = int(len(self.mat) / self.n)
        for i in range(rows):
            row = self.mat[i * self.n:(i + 1) * self.n]  # Slice the array into rows
            print(" ".join(f"{x:4}" for x in row))  # Format each row for alignment

class DimensionError(Exception):
    pass

def identity(n: int) -> Matrix:
    """ Create an nxn identity matrix

    Args:
        n: A positive integer
    
    Returns:
        Returns a Matrix object 
    
    """
    m = Matrix(n, [])
    length = n * n
    
    for i in range(length):
        if i % (n + 1) == 0:
            m.mat.append(1)
        else:
            m.mat.append(0)
    return m

def zeros(n: int) -> Matrix:
    """ Creates a matrix full of zeroes

    Args:
        n: A positive integer
    
    Returns:
        Returns a Matrix Object

    """
    m = Matrix(n, [])
    for i in range(n * n):
        m.mat.append(0)
    return m

def ones(n: int) -> Matrix:
    """ Creates a matrix full of ones

    Args:
        n: A positive integer
    
    Returns:
        Returns a Matrix Object
        
    """
    matrix = Matrix(n, [])
    for i in range(n * n):
        matrix.mat.append(1)

def trace(m: Matrix) -> int:
    """ Calculates the trace of a matrix

    Args:
        m: A Matrix object
    
    Returns:
        Returns an integer object, the sum of the diagonal elements of the matrix
    
    Raises:
        DimensionError: The matrix provided must be a square matrix
        
    """
    rows = int(len(m.mat) / m.n)

    if rows != m.n:
        raise DimensionError("Matrix must be square")

    _sum = 0
    for i in range(0, len(m.mat), m.n + 1):
        _sum += m.mat[i]
    return _sum


def add(m1: Matrix, m2: Matrix) -> Matrix:
    """ Add 2 matrices together
    
    Args:
        m1: A Matrix object
        m2: A Matrix object
    
    Returns:
        Returns the sum of the 2 matrices as a new matrix
    
    Raises:
        DimensionError: If the vecotrs are not the same size.
    
    """
    if (m1.n != m2.n):
        raise DimensionError("Matrices must have the same dimension")
    
    matrix = Matrix(m1.n, [])
    for i in range(len(m1.mat)):
        matrix.mat.append(m1.mat[i] + m2.mat[i])
    
    return matrix


def sub(m1: Matrix, m2: Matrix) -> Matrix:
    """ Subtract 2 matrices
    
    Args:
        m1: A Matrix object
        m2: A Matrix object
    
    Returns:
        Returns the difference of the 2 matrices as a new matrix
    
    Raises:
        DimensionError: If the vecotrs are not the same size.
    
    """
    if (m1.n != m2.n):
        raise ValueError("Matrices must have the same dimension")
    
    matrix = Matrix(m1.n, [])
    for i in range(len(m1.mat)):
        matrix.mat.append(m1.mat[i] - m2.mat[i])

    return matrix


def transpose(m: Matrix) -> Matrix:
    """ Transpose a matrix

    Args:
        m: A Matrix object
    
    Returns:
        Returns the transpose of a matrix
    """

    rows = int(len(m.mat) / m.n)
    cols = m.n

    transposed = [
        m.mat[row + col * rows]  # Transpose index calculation
        for col in range(cols)
        for row in range(rows)
    ]

    t = Matrix(rows, transposed)
    return t