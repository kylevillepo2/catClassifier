import numpy as np
import h5py
#from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
#from public_tests import *

import copy

np.random.seed(1)

def sigmoid(Z):
    """
    sigmoid activation function
    Arguments:
    Z -- pre-activation parameter
    
    Returns:
    A -- the activation value
    activation_cache -- cache that contains Z needed for backpropagation
    """
    
    A = 1 / (1 + np.exp(-Z))
    activation_cache = Z
    
    return A, activation_cache

def relu(Z):
    """
    relu activation function
    
    Arguments:
    Z -- pre-activation parameter
    
    Returns:
    A -- the activation value
    activation_cache -- cache that contains Z needed for backpropagation
  
    """
    
    A = np.maximum(0, Z)
    activation_cache = Z
    
    return A, activation_cache

def initialize_parameters(n_x, n_h, n_y):
    """
    initialize_parameters for a 2-layer neural network
    Arguments:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    
    Returns:
    parameters -- a python dictionary containing the parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    #use np.random.randn because it generates random numbers from the standard normal distribution while np.random.rand generates random numbers from a uniform distribution [0, 1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    #we multiply by 0.01 because we want the values to be closer to 0 to allow for the gradient to be greater during backpropagation
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters


def initialize_parameters_deep(layer_dims):
    """
    initialization for an L-layer Neural Network
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in the network
                  ex: layer_dims[0] = size of the input layer
    Returns:
    parameters -- python dictionary containing the parameters 
                     Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                     bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # number of layers in the network
    
    for l in range(1, L):
        # using xavier initialization to initialize parameters
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1. / layer_dims[l - 1])


        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

def linear_forward(A, W, b):
    """
    linear part of forward propagation
    
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, m)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of current layer, 1)
    
    returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W", and "b"; stored for backpropagation
    """
    
    Z = np.dot(W, A) + b
    
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    
    Arguments:
    A_prev -- activations from the previous layer (or input data): (size of previous layer, m)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector: numpy array of shape (size of current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: 
                  "sigmoid" or "relu"
    
    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for backpropagation
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache

def L_model_forward(X, parameters):
    """
    implement forward propagation for [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    
    Arguments:
    X -- data, numpy array of shape (input size, m)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output layer (last layer)
    caches -- list of caches containing ever cache of linear_activation_forward()
              (there are L of them, which are indexed from 0 to L-1)
    """
    
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network, divide by two because 
                             # there is a weight and bias for each layer
    
    #implement [LINEAR -> RELU] L-1 times. Add cache to the caches list
    # the for loop starts at 1 because layer 0 is the input layer
    for l in range(1, L):
        A_prev = A
        
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
        
    
    #Implement LINEAR -> SIGMOID (output layer). Add cache to the caches list
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    
    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function using cross-entropy cost.
    
    Arguments:
    AL -- probability vector corresponding to my predictions, shape: (1, m)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, m)
    
    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1] # number of examples
    
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    
    cost = np.squeeze(cost) # to make sure the cost's shape is what we expect 
                            # (e.g.turns [[1]] into 1)
    
    return cost


def linear_backward(dZ, cache):
    """
    linear portion of back propagation for a single layer (layer l)
    
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from forward propagation in current layer
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1] # number of training samples
    
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def sigmoid_backward(dA, activation_cache):
    """
    implements the backward propagation for SIGMOID unit. 
    
    Arguments: 
    dA -- gradient of the cost with respect to the activation
    activation_cache -- cache that contains Z
    
    Returns:
    dZ -- the gradient of the cost with respect to the linear output (of curr layer l)
    """
    Z = activation_cache
    
    # Sigmoid derivative
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    
    return dZ
    
def relu_backward(dA, activation_cache):
    """
    implements the backward propagation for RELU unit.
    
    Arguments:
    dA -- gradient of the cost with respect to the activation
    activation_cache -- cache that contains Z
    
    Returns:
    dZ -- the gradient of the cost with respect to the linear output
    """
    Z = activation_cache
    
    # Relu derivative
    
    dZ = np.array(dA, copy=True)
    
    # When Z <= 0, we should set dZ to 0 as well
    dZ[Z <= 0] = 0
    
    return dZ
    
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
                
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing my parameters
    grads -- python dictionary containing my gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    
    return parameters

def predict(X, Y, parameters):
    """
    Arguments:
    X -- input data of shape (number of features, number of examples)
    Y -- true labels of shape (1, number of examples)
    parameters -- a dictionary containing model parameters (weights and biases)
    
    Returns:
    predictions -- predicted labels of shape (1, number of examples)
    accuracy -- accuracy of the predictions compared to the true labels
    """
    
    
    
    # compute linear_activations forward
    AL, caches = L_model_forward(X, parameters)
    
    
    # Convert probabilities to 0 or 1 based on a threshold of 0.5
    predictions = (AL > 0.5).astype(int)
    
    # Calculate accuracy (percentage of correct predictions)
    accuracy = np.mean(predictions == Y)
    
    print(f"Accuracy: {accuracy}")
    
    return predictions, accuracy


                        
def compute_cost_L2(AL, Y, lambda_):
    """
    Implement the cost function using cross-entropy cost.
    
    Arguments:
    AL -- probability vector corresponding to my predictions, shape: (1, m)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, m)
    
    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1] # number of examples
    
    cross_entropy_cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    
    L2_regularization_cost = 0
    
    L = len(parameters) // 2  # number of layers (divide by 2 since params contains W and b)
    
    for l in range(1, L + 1):
        L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
    
    L2_regularization_cost = (lambda_ / (2 * m)) * L2_regularization_cost
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    cost = np.squeeze(cost) # to make sure the cost's shape is what we expect 
                            # (e.g.turns [[1]] into 1)
    
    return cost
    