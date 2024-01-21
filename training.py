import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from main import split_data
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def create_weight_bias():
    np.random.seed(3129) #3129 #88442
    # Initialize the weights and biases for the neural network
    # W1 has a shape of (5, 5), which means there are 5 neurons in the first hidden layer and 5 features in the input data
    # b1 has a shape of (5, 1), which means there is one bias term for each neuron in the first hidden layer
    # W2 has a shape of (5, 1), which means there are 5 neurons in the output layer
    # b2 has a shape of (1, 1), which means there is one bias term for the output layer
    W1 = np.random.rand(5, 5) # 5, 2
    b1 = np.random.rand(5, 1) # 2, 1
    W2 = np.random.rand(5, 1) # 2,1
    b2 = np.random.rand(1, 1) #1,1
    return W1, b1, W2, b2

def tanh(Z):
    return np.tanh(Z)

def tanh_deriv(Z):
    t = np.tanh(Z)
    return 1 - t**2


def ReLU(Z):
    # The function takes an input vector Z and applies the ReLU
    # function element-wise to each element of the vector
    return np.maximum(Z, 0)

def forward_prop(W1, b1, W2, b2, x1):
    # Inp_Hidden is the input to the first hidden layer
    Inp_Hidden = W1.T.dot(x1.T) + b1
    # Activ_1 is the output of the first hidden layer
    # to Inp_Hidden.
    Activ_1 = sigmoid(Inp_Hidden)
    # Hidden_Out is the input to the second hidden layer, which is computed by taking the transpose of W2 and
    # multiplying it by the output of the first hidden layer Activ_1, and then adding the bias term b2.

    Hidden_Out = W2.T.dot(Activ_1) + b2
    # Activ_2 is the output of the second hidden layer to Hidden_Out.

    Activ_2 = sigmoid(Hidden_Out)

    return Inp_Hidden, Activ_1, Hidden_Out, Activ_2

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_deriv(Z):
    s = sigmoid(Z)
    return s * (1 - s)


def ReLU_deriv(Z):
    # This function computes the derivative of the ReLU activation function
    return Z > 0

# Used to store the history of the change in input values which is used in momentum.
dW1_list = []

def linear_regression(X, Y):
    n = X.shape[1] + 1
    m = X.shape[0]
    X_new = np.concatenate((np.ones((m, 1)), X), axis=1)
    Y = Y.reshape(m, 1)
    theta = np.zeros((n, 1))
    alpha = 0.01
    iterations = 1000
    for i in range(iterations):
        hypothesis = np.dot(X_new, theta)
        loss = hypothesis - Y
        gradient = np.dot(X_new.T, loss) / m
        theta = theta - alpha * gradient
    return theta

v_dW1 = 0

def backward_prop_with_momentum(Z1, A1, Z2, A2, W1, W2, X, Y, learning_rate, momentum, dW1_list):
    # This function performs the backpropagation step of a neural network with momentum.
    # The function takes as inputs the input data X, the output data Y, and the weights and biases of the network.

    m,n = Y.shape

    # dZ2 is the error term for the output layer
    dZ2 = A2.T - Y

    # dW2 is the gradient of the cost function with respect to the weights W2 in the output layer.
    dW2 = 1 / m * (A1.dot(dZ2))

    # db2 is the gradient of the cost function with respect to the bias b2 in the output layer.
    db2 = 1 / m * np.sum(dZ2)

    # dZ1 is the error term for the first hidden layer, which is computed as the product of the weights W2
    dZ1 = W2.dot(dZ2.T) * sigmoid_deriv(Z1)

    # dW1 is the gradient of the cost function with respect to the weights W1 in the first hidden layer.
    dW1 = 1 / m * X.T.dot(dZ1.T)

    # Check if it's the first iteration to initialize change_x
    if len(dW1_list) == 0:
        # First iteration, initialize change_x
        change_x = np.zeros_like(dW1)
    else:
        # Use previous update direction
        change_x = dW1_list[-1]

    # Apply momentum to the update
    change_x = learning_rate * dW1 + momentum * change_x

    # Update weights and save dW1 for next iteration
    dW1_list.append(change_x)

    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2, change_x


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Get dimensions of output data

    m,n = Y.shape

    # Get dimensions of output data
    dZ2 = A2.T - Y

    # Compute the gradient of the cost function with respect to the weights W2 in the output layer
    dW2 = 1 / m * (A1.dot(dZ2))

    # Compute the gradient of the cost function with respect to the bias b2 in the output layer
    db2 = 1 / m * np.sum(dZ2)

    # Calculate the error term for the first hidden layer
    dZ1 = W2.dot(dZ2.T) * ReLU_deriv(Z1)

    # Compute the gradient of the cost function with respect to the weights W1 in the first hidden layer
    dW1 = 1 / m * X.T.dot(dZ1.T)

    # Compute the gradient of the cost function with respect to the bias b1 in the first hidden layer
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def momentum_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, change_x):

    # Update weights of the first hidden layer considering momentum
    W1 = W1 - change_x

    # Update bias of the first hidden layer
    b1 = b1 - alpha * db1

    # Update weights of the output layer
    W2 = W2 - alpha * dW2

    # Update bias of the output layer
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# takes alpha of the previous dW1, dW2 take delta 1 and delta 2 which are the previous dW1 for 1 and 2 and multiply it by 0.9 and
# dw1 = dw1 + alpha * previous dW1.
# second variable initialise and after momentum update it to the current dW1. During the next iteration update after
# pass through
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    change = 0.0
    # Update weights of the first hidden layer using learning rate
    W1 = W1 - alpha * dW1

    # Update bias of the first hidden layer using learning rate
    b1 = b1 - alpha * db1

    # Update weights of the output layer using learning rate
    W2 = W2 - alpha * dW2

    # Update bias of the output layer using learning rate
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# dW1, db1, dW2, db2, change_x = backward_prop_with_momentum(Z1, A1, Z2, A2, W1, W2, X, Y, 0.1, 0.9, dW1_list)
# W1, b1, W2, b2 = momentum_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, change_x)

# this saves the mean squared error in a list
mse_list = []

# saves the iterations in alist
iterations_list = []

def gradient_descent(X, Y, alpha, iterations):

    # Create initial weight and bias values
    W1, b1, W2, b2 = create_weight_bias()

    # Perform gradient descent for the specified number of iterations
    for i in range(iterations):

        # Calculate the forward propagation step
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        # W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Calculate gradients and update weights with momentum
        dW1, db1, dW2, db2,change_x = backward_prop_with_momentum(Z1, A1, Z2, A2, W1, W2, X, Y, alpha, 0.9, dW1_list)
        W1, b1, W2, b2 = momentum_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, change_x)

        # print(A2)

        # Append mean squared error and iteration index to their respective lists
        mse_list.append(find_mse(Y, A2))
        iterations_list.append(i)

    # Print the final mean squared error
    print(mse_list[-1])

    return W1, b1, W2, b2, mse_list, iterations_list

def find_mse(Y, A2):
    # Calculate the mean squared error between Y and A2
    mse = np.mean((Y - A2.T) ** 2)
    return mse

# Returns the formatted data which can be used for training.
x1, y1, x1_val, y1_val = split_data()

#Execute the gradient descent optimization for the neural network
W1, b1, W2, b2, mse_list, iterations_list = gradient_descent(x1, y1, 0.1, 2000)

mse_list = mse_list[1:]
iterations_list = iterations_list[1:]
# Plot the mean squared error (MSE) as a function of the number of iterations (epochs)
# using matplotlib's pyplot.
y=mse_list
x=iterations_list

# Create a line plot of the mean squared error values versus the iteration indices
plt.plot(x,y)

#Label the x-axis as 'Epochs'
plt.xlabel('Epochs')

#Label the y-axis as 'MSE'
plt.ylabel('MSE')

# Set the plot title to 'Gradient Descent using sigmoid activation'
plt.title("Gradient Descent using sigmoid activation")

#Display the plot
plt.show()

