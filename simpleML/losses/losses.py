def mse(y_true, y_pred):
    """
    Mean Squared Error Loss

    Parameters:
    y_true (list or array-like): True labels
    y_pred (list or array-like): Predicted labels

    Returns:
    float: Mean Squared Error
    """
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def mae(y_true, y_pred):
    """
    Mean Absolute Error Loss

    Parameters:
    y_true (list or array-like): True labels
    y_pred (list or array-like): Predicted labels

    Returns:
    float: Mean Absolute Error
    """
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def cross_entropy(y_true, y_pred, type="binary"):
    """
    Cross Entropy Loss

    Parameters:
    y_true (list or array-like): True labels
    y_pred (list or array-like): Predicted probabilities
    type (str): Type of cross entropy ('binary' or 'categorical')

    Returns:
    float: Cross Entropy Loss
    """
    import math

    n = len(y_true)
    if type == "binary":
        return (
            -sum(
                yt * math.log(yp) + (1 - yt) * math.log(1 - yp)
                for yt, yp in zip(y_true, y_pred)
            )
            / n
        )
    elif type == "categorical":
        return (
            -sum(
                sum(yt[i] * math.log(yp[i]) for i in range(len(yt)))
                for yt, yp in zip(y_true, y_pred)
            )
            / n
        )
    else:
        raise ValueError("Type must be 'binary' or 'categorical'")
