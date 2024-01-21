from random import random

import numpy as np
# Define the size of the layers
from main import *
# Initialize the weights and biases


# one array for the weights inputs to hidden, array from hidden layer -> output, array from hidden node biases, single variable
# def create_weights_and_biases(num_hidden_nodes):
#     # create weight matrix with random values
#     weight_matrix = np.random.randn(5, num_hidden_nodes)
#
#     # create bias matrix with zeros
#     bias_matrix = np.random.randn(1, num_hidden_nodes)
#
#     return [weight_matrix, bias_matrix]
#
# print(create_weights_and_biases(5))
def sigmoid(z):
    """
    Compute the sigmoid function for the input z.

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))

    return s

import numpy as np

def create_matrices():
    w1 = np.random.uniform(low=-0.4, high=0.4, size=(5, 5))
    w2 = np.random.uniform(low=-0.4, high=0.4, size=(5, 1))
    return w1, w2

w1, w2 = create_matrices()

x1, y1 = split_data()

# print(x1)
# print(y1)


def init_params():
    w1 = np.random.rand(5,5)
    b1 = np.random.rand(5, 1)
    w2 = np.random.rand(5, 1)
    b2 = np.random.rand(1,1)


    return w1, w2, b1, b2

init_params()

def ReLU(Z):
    return np.maximum(0, Z)



def forward_pass(w1, w2, b1, b2, X):

    Z1 = w1.dot(X.T) + b1
    A1 = ReLU(Z1)
    Z2 = w2.T.dot(A1) + b2
    A2 = ReLU(Z2)
    return Z1, A1, Z2, A2

def deriv_ReLU(Z):
    return Z > 0

def backward_pass(z1, a1, z2, a2, w2, x1, y1):
    m = 580
    one_hot_Y = y1.T # y1 is the real y output values
    dZ2 = a2 - one_hot_Y #a2 is the output values from the hidden to output one hot is just it transposed
    dW2 = 1 / m * dZ2.dot(a1.T) # m is the number of rows, a1 is the values from the input to the hidden layer. dZ2 is the difference
    db2 = 1 / m * np.sum(dZ2) # m is the number of columns.
    print(dZ2[0][0])
    dZ1 = w2.T.dot(dZ2) * deriv_ReLU(z1) #Z1 is the predicted ioutput before it is sigmoided
    dW1 = 1 / m * dZ1.T.dot(x1.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def train(x1, y1, alpha, iterations):
    w1, w2, b1, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_pass(w1, w2, b1, b2, x1)
        dW1, db1, dW2, db2 = backward_pass(z1, a2, z2, a2, w1,x1, y1)
        w1, b1, w2, b2 = update_params(w1, b2, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("iterations : ", i)
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y1))
    return w1, b1, w2, b2

w1, b1, w2, b2 = train(x1, y1, 0.10, 500)


# def backward_pass(x1, y1, a2, a1, w1, w2):
#     # calculate error at output layer
#
#     y1 = y1[:, None]
#     print("the subtraction of y2 and y1")
#     print(a2-y1)
#     delta2 = (a2 - y1) * a2 * (1 - a2)
#
#     # calculate error at hidden layer
#     # print(w2.T.shape, " is w2.T")
#     # print(a1.shape, " is a1")
#     # print(delta2.shape)
#     # print("delta2 : ", delta2,"\nw2 : ", w2.T,"\na1 : ",  a1)
#     delta1 = np.dot(w2.T, delta2).T * a1 * (1 - a1)
#     # calculate gradients
#     grad_w2 = np.outer(delta2, a1)
#     grad_w1 = np.outer(delta1, x1)
#     # print("delta1 is ", delta1)
#     # print("delta2 is ", delta2)
#
#     # update weights
#     w2 -= 0.1 * grad_w2
#     w1 -= 0.1 * grad_w1
#
#     # return updated weights
#     return w1, w2



# print(x1)
# train(x1, y1)
# train(x1, y1)
# train(x1, y1, w1, w2)

# In this function, we use the numpy library's randn function to create the bias matrix, which has dimensions of (1, num_hidden_nodes) since we have num_hidden_nodes hidden nodes. We initialize the values in the bias matrix using the randn function, which generates random values from a normal distribution.
# Note that we changed the shape of the bias matrix from (num_hidden_nodes,) to (1, num_hidden_nodes), so that it has the same shape as the weight matrix. This will make it easier to perform matrix multiplication between the input matrix and the weight matrix, and to add the bias matrix to the result.
