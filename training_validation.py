import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from main import split_data
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt




def create_weight_bias():
    W1 = np.random.rand(5, 5)
    b1 = np.random.rand(5, 1)
    W2 = np.random.rand(5, 1)
    b2 = np.random.rand(1, 1)
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)



def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, x1):
    Inp_Hidden = W1.T.dot(x1.T) + b1
    Activ_1 = ReLU(Inp_Hidden)
    Hidden_Out = W2.T.dot(Activ_1) + b2
    Activ_2 = ReLU(Hidden_Out)
    # print("This is A2", A2.shape, A2, )
    # print(A2)
    return Inp_Hidden, Activ_1, Hidden_Out, Activ_2


def ReLU_deriv(Z):
    return Z > 0

dW1_list = []


# def backward_prop_momentum(Z1, A1, Z2, A2, W1, W2, X, Y):
#     m, n = Y.shape
#     # print("This si A2 above",A2)
#     # change_x(t) = step_size * f'(x(t-1)) + momentum * change_x(t-1)
#
#     dZ2 = A2.T - Y
#     dW2 = 1 / m * (A1.dot(dZ2))
#     db2 = 1 / m * np.sum(dZ2)
#     dZ1 = W2.dot(dZ2.T) * ReLU_deriv(Z1)
#     # change_x(t) = step_size * f'(x(t-1)) + momentum * change_x(t-1)
#
#     dW1 = 0.1 * dW1_list[0] + 0.9 *
#     dW1 = 1 / m * X.T.dot(dZ1.T)
#     dW1_list[0] = dW1
#     print("This is the previous dW1", dW1_list)
#     db1 = 1 / m * np.sum(dZ1)
#     return dW1, db1, dW2, db2
v_dW1 = 0
def backward_prop_with_momentum(Z1, A1, Z2, A2, W1, W2, X, Y, learning_rate, momentum, dW1_list):

    m,n = Y.shape

    # print("This si A2 above",A2)
    dZ2 = A2.T - Y
    dW2 = 1 / m * (A1.dot(dZ2))
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.dot(dZ2.T) * ReLU_deriv(Z1)

    dW1 = 1 / m * X.T.dot(dZ1.T)
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
    m,n = Y.shape

    dZ2 = A2.T - Y
    dW2 = 1 / m * (A1.dot(dZ2))
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.dot(dZ2.T) * ReLU_deriv(Z1)
    dW1 = 1 / m * X.T.dot(dZ1.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def momentum_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, change_x):

    # gradient = dW1
    # # calculate update
    # new_change = alpha * gradient + 0.9 * dW1_list[0]
    # # take a step
    # # save the change
    # new_change2 = alpha * dW2 + 0.9 * dW2_list[0]
    W1 = W1 - change_x

    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# takes alpha of the previous dW1, dW2 take delta 1 and delta 2 which are the previous dW1 for 1 and 2 and multiply it by 0.9 and
# dw1 = dw1 + alpha * previous dW1.
# second variable initialise and after momentum update it to the current dW1. During the next iteration update after
# pass through

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # change_x = step_size * f'(x)
    # x = x – change_x
    # change_x(t) = step_size * f'(x(t-1)) + momentum * change_x(t-1)
    # x(t) = x(t-1) – change_x(t)

    change = 0.0

    # calculate gradient


    W1 = W1 - alpha * dW1

    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    # print(len(A2))
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    # print("\nPredictions : ", predictions,"\nY values : ",  Y)
    # print(len(predictions))
    return np.sum(predictions == Y) / Y.size


# dW1, db1, dW2, db2, change_x = backward_prop_with_momentum(Z1, A1, Z2, A2, W1, W2, X, Y, 0.1, 0.9, dW1_list)
# W1, b1, W2, b2 = momentum_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, change_x)


mse_list = []
iterations_list = []
def gradient_descent(X, Y, alpha, iterations, x1_val, y1_val):
    W1, b1, W2, b2 = create_weight_bias()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2,x1_val)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, x1_val, y1_val)

        # dW1, db1, dW2, db2, change_x = backward_prop_with_momentum(Z1, A1, Z2, A2, W1, W2, X, Y, 0.1, 0.9, dW1_list)
        # W1, b1, W2, b2 = momentum_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, change_x)

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)


        mse_list.append(find_mse(y1_val, A2))
        # print(A2)
        iterations_list.append(i)
        #
        # if i % 10 == 0:
        #     # print("Iteration: ", i)
        #     # print(A2)
        #     predictions = get_predictions(A2)
        #     # print("The accuracy: ", get_accuracy(predictions, Y))
        #     print(Y.shape)
        #     print(A2.shape)

    return W1, b1, W2, b2, mse_list, iterations_list, A2
#
# def find_mse(Y, A2):
#     print(Y.shape, A2.shape)
#     mse = mean_squared_error(Y, A2.T)
#     return mse

def find_mse(Y, A2):
    # Calculate the mean squared error between Y and A2
    mse = np.mean((Y - A2.T) ** 2)
    return mse

x1, y1, x1_val, y1_val = split_data()

W1, b1, W2, b2, mse_list, iterations_list, A2 = gradient_descent(x1, y1, 0.10, 2000, x1_val, y1_val)

print(A2[0])
print(y1_val)

# mse_list = mse_list[1:]
# iterations_list = iterations_list[1:]
# y=mse_list
# x=iterations_list
# plt.plot(x,y)
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.title("Gradient Descent with Momentum")
# plt.show()


def compare_datasets(predicted, actual):
    plt.scatter(predicted, actual, label="Predicted vs Actual", marker="o", alpha=0.5)

    # Add a reference line for perfect predictions
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--',
             label='Perfect Prediction Line')

    plt.xlabel('Predicted values (A2)')
    plt.ylabel('Actual values (y)')
    plt.title('Comparison of Predicted Output and Actual y Values')
    plt.legend()
    plt.grid()
    plt.show()


# Example usage:
# A2 = list of predicted output from backpropagation algorithm
# y = list of actual y values
print(A2.shape, y1_val.shape)
compare_datasets(A2, y1_val)




