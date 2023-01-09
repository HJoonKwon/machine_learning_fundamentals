import os

import h5py
import numpy as np


def update_parameters(parameters: dict, grads: dict, learning_rate: float):

    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


def load_data(data_dir: str):
    train_dataset = h5py.File(os.path.join(data_dir, 'train_catvnoncat.h5'),
                              "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(os.path.join(data_dir, 'test_catvnoncat.h5'), "r")
    test_set_x_orig = np.array(
        test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(layer_dims: list) -> dict:
    np.random.seed(1)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A_prev: np.ndarray, W: np.ndarray,
                   b: np.ndarray) -> tuple[np.ndarray, tuple]:
    # A_prev: [prev_layer_size, m]
    # W: [current_layer_size, prev_layer_size]
    # b: [current_layer_size, 1]
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation='relu'):
    # A_prev: [prev_layer_size, m]
    # W: [current_layer_size, prev_layer_size]
    # b: [current_layer_size, 1]
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'relu':
        A, activation_cache = relu_forward(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid_forward(Z)
    else:
        print(f"activation {activation} is not supported")
    return A, (linear_cache, activation_cache)


def deep_linear_activation_forward(
        X: np.ndarray, parameters: dict) -> tuple[np.ndarray, np.ndarray]:

    # num_layer
    L = len(parameters) // 2
    caches = []

    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters["W" + str(l)],
                                             parameters["b" + str(l)],
                                             activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A,
                                          parameters["W" + str(L)],
                                          parameters["b" + str(L)],
                                          activation='sigmoid')
    caches.append(cache)
    return AL, caches


def linear_backward(dZ: np.ndarray,
                    cache: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    A_prev, W, b = cache
    m = dZ.shape[1]
    # dJ/dA_prev = dJ/dZ * dZ/dA_prev
    dA_prev = W.T @ dZ
    # dJ/dW = dJ/dZ * dZ/dW
    dW = 1 / m * dZ @ A_prev.T
    # dJ/db = dJ/dZ * dZ/db
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA: np.ndarray,
                               cache: tuple,
                               activation='relu'):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        print(f"activation {activation} is not supported")

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def deep_linear_activation_backward(AL, Y, caches):

    Y = Y.reshape(AL.shape)

    # number of examples
    m = AL.shape[1]

    # number of layers
    L = len(caches)

    # gradient of cross entroy loss
    dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

    grads = {}

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward(dAL,
                                                    caches[L - 1],
                                                    activation='sigmoid')

    for l in reversed(range(L - 1)):
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 1)],
                                                     caches[l],
                                                     activation='relu')
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    return grads


def cross_entropy(AL, Y):
    m = Y.shape[1]
    loss = (-1 / m) * (Y @ np.log(AL).T + (1 - Y) @ (np.log(1 - AL).T))
    return np.squeeze(loss)


def sigmoid_forward(z: np.ndarray) -> tuple[np.ndarray, ...]:
    a = 1 / (1 + np.exp(-z))
    cache = z
    return a, cache


def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    # dA : (n, m)
    # dZ : (n, m)
    # dJ/dz = dJ/da * da/dz
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_forward(z: np.ndarray) -> tuple[np.ndarray, ...]:
    a = np.maximum(0, z)
    cache = z
    return a, cache


def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
