class DimensionError(Exception):
    """Exception raised for errors in the dimension of vectors."""
    
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"DimensionError: {self.message}"