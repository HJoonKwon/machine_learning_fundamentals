import os

import h5py
import numpy as np


def load_data(data_dir: str) -> tuple[np.ndarray, ...]:
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


# Define blocks to define MLP


def linear_foward(A_prev: np.ndarray, W: np.ndarray,
                  b: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    Z = W @ A_prev + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(
    A_prev: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    activation='relu'
) -> tuple[np.ndarray, tuple[np.ndarray, tuple[np.ndarray, ...]]]:
    Z, linear_cache = linear_foward(A_prev, W, b)
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    else:
        assert False, f"Not supported activation function:{activation}"
    return A, (linear_cache, activation_cache)


def linear_backward(dZ: np.ndarray, cache: tuple[np.ndarray, ...]):
    A_prev, W, b = cache
    m = dZ.shape[1]
    dW = 1 / m * dZ @ A_prev.T
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    return (dW, db)


def linear_activation_backward(dA: np.ndarray,
                               cache: tuple[np.ndarray, tuple[np.ndarray,
                                                              ...]],
                               activation: str = 'relu'):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        assert False, f"Not supported activation function:{activation}"

    dW, db = linear_backward(dZ, linear_cache)

    return (dW, db)


def cross_entropy(AL: np.ndarray, Y: np.ndarray) -> float:
    loss = -(Y @ np.log(AL).T + (1 - Y) @ np.log(1 - AL).T)
    return np.squeeze(loss)


def sigmoid(Z: np.ndarray) -> tuple[np.ndarray, ...]:
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dAL: np.ndarray, cache: np.ndarray) -> np.ndarray:
    Z = cache
    AL, _ = sigmoid(Z)
    dZ = (1 - AL) * AL * dAL
    return dZ


def relu(Z: np.ndarray) -> tuple[np.ndarray, ...]:
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


class MultiLayerPerceptron():
    def __init__(self):
        pass

    def initialize_parameters(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def update_paramters(self):
        pass
