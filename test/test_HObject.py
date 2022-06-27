import math

from datacontrolreach.LipschitzApproximator import LipschitzApproximator, f_approximate
from datacontrolreach.interval import Interval
import time
from jax.tree_util import register_pytree_node_class
import jax
import datacontrolreach.jumpy as jp
from datacontrolreach.HObject import HObject, inverse_contraction_B, inverse_contraction_C, hc4revise_lin_eq, contract_row_wise


def test_contract_row_wise(seed):

    # Generate the data for test
    x = 3.0
    u = jp.array([1.0, 2.0, 3.0])
    g = Interval(jp.array([0.0, 0.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(jp.array([0.0, 0.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '1. row wise contraction fails : {} , {}\n'.format(answer, new_approx)

    # Generate the data for test
    x = 3.0
    u = jp.array([1.0, 0.0, 3.0])
    g = Interval(jp.array([0.0, 0.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(jp.array([0.0, 0.0, 2/3.0]), jp.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '2. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = 3.0
    u = jp.array([0.0, 0.0, 3.0])
    g = Interval(jp.array([0.0, 0.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(jp.array([0.0, 0.0, 1.0]), jp.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = 3.0
    u = jp.array([1.0, 2.0, 0.0])
    g = Interval(jp.array([0.0, 0.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(jp.array([1.0, 1.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '4. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '4. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = -1.0
    u = jp.array([1.0, 2.0, 3.0])
    g = Interval(jp.array([-1.0, -1.0, -1.0]), jp.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(jp.array([-1.0, -1.0, -1.0]), jp.array([1.0, 1.0, 2/3.0]))
    assert (answer == new_approx).all(), '5. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '5. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = -1.0
    u = jp.array([1.0,])
    g = Interval(jp.array([-1.0]), jp.array([1.0]))

    new_approx = contract_row_wise(x, g, u)
    answer = Interval(jp.array([-1.0]), jp.array([-1.0]))
    assert (answer == new_approx).all(), '6. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '6. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)


def test_contract_C(seed):

    # Generate the data for test
    A = jp.array([1,2,3]) # 3x1
    B = jp.ones((3,2))          # 3x2
    C = Interval(jp.array([-10.0, -10.0]), jp.array([10.0, 10.0])) # 2x1

    new_c = inverse_contraction_C(A, B, C)
    print(new_c)
    # assert (answer == new_approx).all(), '1. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
# test_contract_C(1)

def test_contract_B(seed):

    # Generate the data for test
    A = jp.array([[1],[2],[3]]) # 3x1
    B = Interval(jp.array([[-10.0, -10.0], [-10.0, -10.0], [-10.0, -10.0]]),
                jp.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]))           # 3x2
    C = jp.ones((2,1))# 2x1

    new_b = inverse_contraction_B(A, B, C)
    print(new_b)
    # assert (answer == new_approx).all(), '1. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
test_contract_B(1)


def test_H_object(seed):
    # Generate the data for test
    # This is the test case where H = F(X) + G(X)*U
    shape_x = (3,)  # = N
    shape_u = (2,)  # = M
    known_functions = []
    H = lambda x, u, known, unknown: unknown[0](x) + jp.matmul(unknown[1](x), u)
    unknown_approximators = [
        LipschitzApproximator(shape_x, shape_x, jp.array([10.0, 5.0, 3.0])), # This is F(X), which outputs a Nx1 array
        LipschitzApproximator(shape_x, (shape_x[0], shape_u[0]),
                              jp.array([[10.0, 5.0],  # This is G(X), which outputs a NxM array
                                        [10.0, 5.0],
                                        [10.0, 5.0]]))

    ]
    contractions = [  # recall Xdot = H = F(X) + G(X) * U
        lambda x, u, xdot, known, unknown: xdot - jp.matmul(unknown[1](x), u),  # F(X) = xdot - G(X) * U
        lambda x, u, xdot, known, unknown: inverse_contraction_B(xdot - unknown[0](x), unknown[1](x), u),
                                                # G(X) = (xdot - F(X)) U^-1
                                                # Recall we cannot use pseudo-inverse and must use inverse_contraction_B instead
    ]

    h_obj = HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = unknown_approximators[0](x) + jp.matmul(unknown_approximators[1](x), u)
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj.contract(x,u, x_dot)
    expected_x_dot = Interval(x_dot)
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([10.0 - math.sqrt(3.0) * 10, 30 - math.sqrt(3.0) * 5, 45 - math.sqrt(3) * 3]),
                              jp.array([10.0 + math.sqrt(3.0) * 10, 30 + math.sqrt(3.0) * 5, 45 + math.sqrt(3) * 3]))
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '3.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

def test_H_object2(seed):
    # Generate the data for test
    # This is the test case where H = (F(X) + Fknown(x)) + (G(X) + Gknown(x))*U
    shape_x = (3,)  # = N
    shape_u = (2,)  # = M
    known_functions = [lambda x: jp.array([0.0, 0.0, 0.0]),
                       lambda x: jp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
                       ]
    H = lambda x, u, known, unknown: (unknown[0](x) + known[0](x)) + jp.matmul((unknown[1](x) + known[1](x)), u)
    unknown_approximators = [
        LipschitzApproximator(shape_x, shape_x, jp.array([10.0, 5.0, 3.0])), # This is F(X), which outputs a Nx1 array
        LipschitzApproximator(shape_x, (shape_x[0], shape_u[0]),
                              jp.array([[10.0, 5.0],  # This is G(X), which outputs a NxM array
                                        [10.0, 5.0],
                                        [10.0, 5.0]]))

    ]
    contractions = [  # recall Xdot = H = (F(X) + Fknown(x)) + (G(X) + Gknown(x))*U
        lambda x, u, xdot, known, unknown: xdot - known[0](x) - jp.matmul((unknown[1](x) + known[1](x)), u),  # F(X) = xdot - G(X) * U
        lambda x, u, xdot, known, unknown: inverse_contraction_B(xdot - unknown[0](x) - known[0](x), unknown[1](x) + known[1](x), u) - known[1](x),
                                                # G(X) = (xdot - Funknown(X) - fknown(x)) U^-1
                                                # Recall we cannot use pseudo-inverse and must use inverse_contraction_B instead
                                                # Inverse contraction does the operation X = G*U, where the first arg = x, second = estimate for G, third = u
    ]

    h_obj = HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = unknown_approximators[0](x) + jp.matmul(unknown_approximators[1](x), u)
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj.contract(x,u, x_dot)
    expected_x_dot = Interval(x_dot)
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([10.0 - math.sqrt(3.0) * 10, 30 - math.sqrt(3.0) * 5, 45 - math.sqrt(3) * 3]),
                              jp.array([10.0 + math.sqrt(3.0) * 10, 30 + math.sqrt(3.0) * 5, 45 + math.sqrt(3) * 3]))
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '3.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

def test_H_object3(seed):
    # Generate the data for test
    # This is the test case where H = (F(X) + Fknown(x)) + (G(X) + Gknown(x))*U
    shape_x = (3,)  # = N
    shape_u = (2,)  # = M
    known_functions = [lambda x: jp.array([1.0, 1.0, 1.0]),
                       lambda x: jp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
                       ]
    H = lambda x, u, known, unknown: (unknown[0](x) + known[0](x)) + jp.matmul((unknown[1](x) + known[1](x)), u)
    unknown_approximators = [
        LipschitzApproximator(shape_x, shape_x, jp.array([10.0, 5.0, 3.0])), # This is F(X), which outputs a Nx1 array
        LipschitzApproximator(shape_x, (shape_x[0], shape_u[0]),
                              jp.array([[10.0, 5.0],  # This is G(X), which outputs a NxM array
                                        [10.0, 5.0],
                                        [10.0, 5.0]]))

    ]
    contractions = [  # recall Xdot = H = (F(X) + Fknown(x)) + (G(X) + Gknown(x))*U
        lambda x, u, xdot, known, unknown: xdot - known[0](x) - jp.matmul((unknown[1](x) + known[1](x)), u),  # F(X) = xdot - G(X) * U
        lambda x, u, xdot, known, unknown: inverse_contraction_B(xdot - unknown[0](x) - known[0](x), unknown[1](x) + known[1](x), u) - known[1](x),
                                                # G(X) = (xdot - Funknown(X) - fknown(x)) U^-1
                                                # Recall we cannot use pseudo-inverse and must use inverse_contraction_B instead
                                                # Inverse contraction does the operation X = G*U, where the first arg = x, second = estimate for G, third = u
    ]

    h_obj = HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = (unknown_approximators[0](x) + known_functions[0](x)) + jp.matmul((unknown_approximators[1](x) + known_functions[1](x)), u)
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj.contract(x,u, x_dot)
    expected_x_dot = Interval(x_dot)
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([10.0 - math.sqrt(3.0) * 10, 30 - math.sqrt(3.0) * 5, 45 - math.sqrt(3) * 3]),
                              jp.array([10.0 + math.sqrt(3.0) * 10, 30 + math.sqrt(3.0) * 5, 45 + math.sqrt(3) * 3]))
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '3.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)



def test_H_object4(seed):
    # This is the test case where H = Known1(x,u) + known2(x,u) * unknown(x)
    shape_x = (3,)  # = N
    shape_u = (2,)  # = M
    k = 4 # k unknown terms

    known_functions = [lambda x, u: jp.ones(shape_x),
                       lambda x, u: jp.ones((shape_x[0], k)),
                       ]
    H = lambda x, u, known, unknown: known[0](x,u) + jp.matmul(known[1](x,u), unknown[0](x))

    unknown_approximators = [
        LipschitzApproximator(shape_x, (k, shape_x[0]), jp.ones((k, shape_x[0]))), # lipschitz is all 1's
    ]
    contractions = [
        lambda x, u, xdot, known, unknown: xdot - known[0](x,u) - jp.matmul((unknown[1](x) + known[1](x)), u),  # F(X) = xdot - G(X) * U
    ]  # TODO

    h_obj = HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = known_functions[0](x,u) + jp.matmul(known_functions[1](x,u), unknown_approximators[0](x))
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj.contract(x,u, x_dot)
    expected_x_dot = Interval(x_dot)
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([10.0 - math.sqrt(3.0) * 10, 30 - math.sqrt(3.0) * 5, 45 - math.sqrt(3) * 3]),
                              jp.array([10.0 + math.sqrt(3.0) * 10, 30 + math.sqrt(3.0) * 5, 45 + math.sqrt(3) * 3]))
    result = h_obj.get_x_dot(x, u)
    assert (expected_x_dot == result).all(), '3.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)



