from cases_neural_network import *

from ml_algorithms.neural_netowork import *


def test_linear_forward():
    A, W, b = linear_forward_test_case()
    Z, linear_cache = linear_forward(A, W, b)
    Z_gt = np.array([[3.26295337, -1.23429987]])
    assert np.allclose(Z, Z_gt)


def test_linear_activation_forward():
    A_prev, W, b = linear_activation_forward_test_case()
    A, linear_activation_cache = linear_activation_forward(
        A_prev, W, b, activation="sigmoid")
    A_gt = np.array([[0.96890023, 0.11013289]])
    assert np.allclose(A, A_gt)

    A, linear_activation_cache = linear_activation_forward(A_prev,
                                                           W,
                                                           b,
                                                           activation="relu")
    A_gt = np.array([[3.43896131, 0.]])
    assert np.allclose(A, A_gt)


def test_L_model_forward_2hidden():
    X, parameters = L_model_forward_test_case_2hidden()
    model = MultiLayerPerceptron(layer_dims=[1]*4)
    model.parameters = parameters
    AL = model.forward(X)
    AL_gt = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    assert np.allclose(AL, AL_gt)


def test_compute_cost():
    Y, AL = compute_cost_test_case()
    cost = cross_entropy(AL, Y)
    cost_gt = 0.2797765635793422
    assert np.allclose(cost, cost_gt)


def test_linear_backward():
    dZ, linear_cache = linear_backward_test_case()
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    dA_prev_gt = np.array([[-1.15171336, 0.06718465, -0.3204696, 2.09812712],
                           [0.60345879, -3.72508701, 5.81700741, -3.84326836],
                           [-0.4319552, -1.30987417, 1.72354705, 0.05070578],
                           [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
                           [-2.52214926, 2.67882552, -0.67947465, 1.48119548]])
    dW_gt = np.array(
        [[0.07313866, -0.0976715, -0.87585828, 0.73763362, 0.00785716],
         [0.85508818, 0.37530413, -0.59912655, 0.71278189, -0.58931808],
         [0.97913304, -0.24376494, -0.08839671, 0.55151192, -0.10290907]])
    db_gt = np.array([[-0.14713786], [-0.11313155], [-0.13209101]])
    assert np.allclose(dA_prev, dA_prev_gt)
    assert np.allclose(dW, dW_gt)
    assert np.allclose(db, db_gt)


def test_linear_activation_backward():
    dAL, linear_activation_cache = linear_activation_backward_test_case()
    dA_prev, dW, db = linear_activation_backward(dAL,
                                                 linear_activation_cache,
                                                 activation='sigmoid')
    dA_prev_gt = np.array([[0.11017994, 0.01105339], [0.09466817, 0.00949723],
                           [-0.05743092, -0.00576154]])
    dW_gt = np.array([[0.10266786, 0.09778551, -0.01968084]])
    db_gt = np.array([[-0.05729622]])
    assert np.allclose(dA_prev, dA_prev_gt)
    assert np.allclose(dW, dW_gt)
    assert np.allclose(db, db_gt)
    dA_prev, dW, db = linear_activation_backward(dAL,
                                                 linear_activation_cache,
                                                 activation='relu')
    dA_prev_gt = np.array([[0.44090989, -0.], [0.37883606, -0.],
                           [-0.2298228, 0.]])
    dW_gt = np.array([[0.44513824, 0.37371418, -0.10478989]])
    db_gt = np.array([[-0.20837892]])
    assert np.allclose(dA_prev, dA_prev_gt)
    assert np.allclose(dW, dW_gt)
    assert np.allclose(db, db_gt)


# def test_deep_linear_activation_backward():
#     AL, Y_assess, caches = L_model_backward_test_case()
#     grads = deep_linear_activation_backward(AL, Y_assess, caches)
#     dW1_gt = np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
#                        [
#                            0.,
#                            0.,
#                            0.,
#                            0.,
#                        ], [0.05283652, 0.01005865, 0.01777766, 0.0135308]])
#     db1_gt = np.array([[-0.22007063], [0.], [-0.02835349]])
#     dA1_gt = np.array([[0.12913162, -0.44014127], [-0.14175655, 0.48317296],
#                        [0.01663708, -0.05670698]])
#     assert np.allclose(grads['dW1'], dW1_gt)
#     assert np.allclose(grads['db1'], db1_gt)
#     assert np.allclose(grads['dA1'], dA1_gt)


# def test_update_parameters():
#     parameters, grads = update_parameters_test_case()
#     parameters = update_parameters(parameters, grads, 0.1)
#     W1_gt = np.array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
#                       [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
#                       [-1.0535704, -0.86128581, 0.68284052, 2.20374577]])
#     b1_gt = np.array([[-0.04659241], [-1.28888275], [0.53405496]])
#     W2_gt = np.array([[-0.55569196, 0.0354055, 1.32964895]])
#     b2_gt = np.array([[-0.84610769]])
#     assert np.allclose(parameters['W1'], W1_gt)
#     assert np.allclose(parameters['b1'], b1_gt)
#     assert np.allclose(parameters['W2'], W2_gt)
#     assert np.allclose(parameters['b2'], b2_gt)


# if __name__ == '__main__':
#     test_linear_forward()
#     test_linear_activation_forward()
#     test_L_model_forward_2hidden()
#     test_linear_backward()
#     test_linear_activation_backward()
#     test_deep_linear_activation_backward()
#     test_update_parameters()
#     test_compute_cost()
