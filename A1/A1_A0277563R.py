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


X = np.array([[2, 4], [-4, 3], [5, -7], [6, 3], [0, -8]])
y = np.array([[5], [-3], [4], [9], [2]])

XTX_inv, w = A1_A0277563R(X, y)
print(XTX_inv, w)
