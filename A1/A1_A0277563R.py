import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0277563R(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray

    """

    # your code goes here
    X_transpose = np.transpose(X)
    XTX = np.dot(X_transpose, X)
    InvXTX = np.linalg.inv(XTX)

    w = np.dot(np.dot(InvXTX, X_transpose), y)

    # return in this order
    return InvXTX, w
