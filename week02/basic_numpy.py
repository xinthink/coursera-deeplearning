# Exercise: Implement the sigmoid function using numpy.
# Instructions: x could now be either a real number, a vector, or a matrix.
# The data structures we use in numpy to represent these shapes (vectors, matrices...)
# are called numpy arrays. You don't need to know more for now.
import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


x = np.array([1, 2, 3])
s = sigmoid(x)
ds = sigmoid_derivative(x)
print("sigmoid({}) = {}".format(x, s))
print("sigmoid_derivative({}) = {}".format(x, ds))
