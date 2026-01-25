def one_hot_encode(labels, num_classes) -> list[list[int]]:
    """
    Convert a list of integer labels to one-hot encoded format.

    Args:
        labels (list of int): List of integer labels.
        num_classes (int): Total number of classes.
    Returns:
        list of list of int: One-hot encoded representation of the labels.
    """
