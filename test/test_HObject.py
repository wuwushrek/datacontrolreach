import math

from datacontrolreach.LipschitzApproximator import LipschitzApproximator, f_approximate, init_LipschitzApproximator, approximate
from datacontrolreach.interval import Interval
import time
from jax.tree_util import register_pytree_node_class
import jax
import datacontrolreach.jumpy as jp
from datacontrolreach.HObject import HObject, inverse_contraction_B, inverse_contraction_C, hc4revise_lin_eq, contract_row_wise, init_HObject, get_x_dot, contract


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

    x = Interval(3.0, 5.0)
    u = jp.array([1.0, 2.0, 0.0])
    g = Interval(jp.array([0.0, 0.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(jp.array([1.0, 1.0, 0.0]), jp.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '7. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '7. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)


def test_contract_C(seed):

    # Generate the data for test
    A = jp.array([1,2,3]) # 3x1
    B = jp.ones((3,2))          # 3x2
    C = Interval(jp.array([-10.0, -10.0]), jp.array([10.0, 10.0])) # 2x1
    new_c = inverse_contraction_C(A, B, C)
    expected = Interval(jp.array([[-7.0, -7.0]]), jp.array([[10.0, 10.0]]))
    assert (new_c == expected).all(), '1. C contraction fails : {} , {}\n'.format(new_c, expected)

    A = jp.array([1,2,3]) # 3x1
    B = jp.zeros((3,2))          # 3x2
    C = Interval(jp.array([-10.0, -10.0]), jp.array([10.0, 10.0])) # 2x1
    new_c = inverse_contraction_C(A, B, C)
    expected = Interval(jp.array([-10.0, -10.0]), jp.array([10.0, 10.0]))
    assert (new_c == expected).all(), '2. C contraction fails : {} , {}\n'.format(new_c, expected)

    A = jp.array([1]) # 1x1
    B = jp.zeros((1,2))          #1x2
    C = Interval(jp.array([-10.0, -10.0]), jp.array([10.0, 10.0])) # 2x1
    new_c = inverse_contraction_C(A, B, C)
    expected = Interval(jp.array([-10.0, -10.0]), jp.array([10.0, 10.0]))
    assert (new_c == expected).all(), '2. C contraction fails : {} , {}\n'.format(new_c, expected)

    A = jp.array([1]) # 1x1
    B = jp.ones((1,2))          #1x2
    C = Interval(jp.array([-10.0, -10.0]), jp.array([10.0, 10.0])) # 2x1
    new_c = inverse_contraction_C(A, B, C)
    expected = Interval(jp.array([-9.0, -9.0]), jp.array([10.0, 10.0]))
    assert (new_c == expected).all(), '2. C contraction fails : {} , {}\n'.format(new_c, expected)

def test_contract_B(seed):

    # Generate the data for test
    A = jp.array([1,2,3]) # 3x
    B = Interval(jp.array([[-10.0, -10.0], [-10.0, -10.0], [-10.0, -10.0]]),
                jp.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]))           # 3x2
    C = jp.ones((2,))# 2x
    new_b = inverse_contraction_B(A, B, C)
    expected = Interval(jp.array([[-9.0, -9.0],[-8.0, -8.0],[-7.0, -7.0] ]),
                        jp.array([[10.0, 10.0],[10.0, 10.0],[10.0, 10.0] ]))
    assert (new_b == expected).all(), '1. B contraction fails : {} , {}\n'.format(new_b, expected)

    A = jp.array([1,2,3]) # 3x1
    B = Interval(jp.array([[-10.0, -10.0], [-10.0, -10.0], [-10.0, -10.0]]),
                jp.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]))           # 3x2
    C = jp.zeros((2,))# 2x1
    new_b = inverse_contraction_B(A, B, C)
    expected = Interval(jp.array([[-10.0, -10.0],[-10.0, -10.0],[-10.0, -10.0] ]),
                        jp.array([[10.0, 10.0],[10.0, 10.0],[10.0, 10.0] ]))
    assert (new_b == expected).all(), '2. B contraction fails : {} , {}\n'.format(new_b, expected)

    A = jp.array([1]) # 1x
    B = Interval(jp.array([[-10.0, -10.0]]),
                jp.array([[10.0, 10.0]]))           # 1x2
    C = jp.zeros((2,))# 2x
    new_b = inverse_contraction_B(A, B, C)
    expected = Interval(jp.array([[-10.0, -10.0]]), jp.array([[10.0, 10.0]]))
    assert (new_b == expected).all(), '2. B contraction fails : {} , {}\n'.format(new_b, expected)

    A = jp.array([1]) # 1x1
    B = Interval(jp.array([[-10.0, -10.0]]),
                jp.array([[10.0, 10.0]]))           # 1x2
    C = jp.ones((2,))# 2x1
    new_b = inverse_contraction_B(A, B, C)
    expected = Interval(jp.array([[-9.0, -9.0]]), jp.array([[10.0, 10.0]]))
    assert (new_b == expected).all(), '2. B contraction fails : {} , {}\n'.format(new_b, expected)

def test_H_object(seed):
    # Generate the data for test
    # This is the test case where H = F(X) + G(X)*U
    shape_x = (3,)  # = N
    shape_u = (2,)  # = M
    known_functions = []
    H = lambda x, u, known, unknown: approximate(unknown[0], x) + jp.matmul(approximate(unknown[1], x), u)
    unknown_approximators = [
        init_LipschitzApproximator(shape_x, shape_x, jp.array([10.0, 5.0, 3.0]), 10), # This is F(X), which outputs a Nx1 array
        init_LipschitzApproximator(shape_x, (shape_x[0], shape_u[0]),
                              jp.array([[10.0, 5.0],  # This is G(X), which outputs a NxM array
                                        [10.0, 5.0],
                                        [10.0, 5.0]]),
                                   10)

    ]
    contractions = [  # recall Xdot = H = F(X) + G(X) * U
        lambda x, u, xdot, known, unknown: xdot - jp.matmul(unknown[1](x), u),  # F(X) = xdot - G(X) * U
        lambda x, u, xdot, known, unknown: inverse_contraction_B(xdot - unknown[0](x), unknown[1](x), u),
                                                # G(X) = (xdot - F(X)) U^-1
                                                # Recall we cannot use pseudo-inverse and must use inverse_contraction_B instead
    ]

    h_obj = init_HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = approximate(unknown_approximators[0], x) + jp.matmul(approximate(unknown_approximators[1], x), u)
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj = contract(h_obj, x, u, x_dot)
    expected_x_dot = Interval(x_dot)
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([10.0 - math.sqrt(3.0) * 10, 30 - math.sqrt(3.0) * 5, 45 - math.sqrt(3) * 3]),
                              jp.array([10.0 + math.sqrt(3.0) * 10, 30 + math.sqrt(3.0) * 5, 45 + math.sqrt(3) * 3]))
    result = get_x_dot(h_obj, x, u)
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
        init_LipschitzApproximator(shape_x, shape_x, jp.array([10.0, 5.0, 3.0]), 10), # This is F(X), which outputs a Nx1 array
        init_LipschitzApproximator(shape_x, (shape_x[0], shape_u[0]),
                              jp.array([[10.0, 5.0],  # This is G(X), which outputs a NxM array
                                        [10.0, 5.0],
                                        [10.0, 5.0]]),
                                   10)

    ]
    contractions = [  # recall Xdot = H = (F(X) + Fknown(x)) + (G(X) + Gknown(x))*U
        lambda x, u, xdot, known, unknown: xdot - known[0](x) - jp.matmul((unknown[1](x) + known[1](x)), u),  # F(X) = xdot - G(X) * U
        lambda x, u, xdot, known, unknown: inverse_contraction_B(xdot - unknown[0](x) - known[0](x), unknown[1](x) + known[1](x), u) - known[1](x),
                                                # G(X) = (xdot - Funknown(X) - fknown(x)) U^-1
                                                # Recall we cannot use pseudo-inverse and must use inverse_contraction_B instead
                                                # Inverse contraction does the operation X = G*U, where the first arg = x, second = estimate for G, third = u
    ]

    h_obj = init_HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = unknown_approximators[0](x) + jp.matmul(unknown_approximators[1](x), u)
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj = contract(h_obj, x,u, x_dot)
    expected_x_dot = Interval(x_dot)
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([10.0 - math.sqrt(3.0) * 10, 30 - math.sqrt(3.0) * 5, 45 - math.sqrt(3) * 3]),
                              jp.array([10.0 + math.sqrt(3.0) * 10, 30 + math.sqrt(3.0) * 5, 45 + math.sqrt(3) * 3]))
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '3.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

test_H_object2(1)

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
        init_LipschitzApproximator(shape_x, shape_x, jp.array([10.0, 5.0, 3.0]), 10), # This is F(X), which outputs a Nx1 array
        init_LipschitzApproximator(shape_x, (shape_x[0], shape_u[0]),
                              jp.array([[10.0, 5.0],  # This is G(X), which outputs a NxM array
                                        [10.0, 5.0],
                                        [10.0, 5.0]]),
                                   10)

    ]
    contractions = [  # recall Xdot = H = (F(X) + Fknown(x)) + (G(X) + Gknown(x))*U
        lambda x, u, xdot, known, unknown: xdot - known[0](x) - jp.matmul((unknown[1](x) + known[1](x)), u),  # F(X) = xdot - G(X) * U
        lambda x, u, xdot, known, unknown: inverse_contraction_B(xdot - unknown[0](x) - known[0](x), unknown[1](x) + known[1](x), u) - known[1](x),
                                                # G(X) = (xdot - Funknown(X) - fknown(x)) U^-1
                                                # Recall we cannot use pseudo-inverse and must use inverse_contraction_B instead
                                                # Inverse contraction does the operation X = G*U, where the first arg = x, second = estimate for G, third = u
    ]

    h_obj = init_HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = (unknown_approximators[0](x) + known_functions[0](x)) + jp.matmul((unknown_approximators[1](x) + known_functions[1](x)), u)
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj = contract(h_obj, x, u, x_dot)
    expected_x_dot = Interval(x_dot)
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([10.0 - math.sqrt(3.0) * 10, 30 - math.sqrt(3.0) * 5, 45 - math.sqrt(3) * 3]),
                              jp.array([10.0 + math.sqrt(3.0) * 10, 30 + math.sqrt(3.0) * 5, 45 + math.sqrt(3) * 3]))
    result = get_x_dot(h_obj, x, u)
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
        init_LipschitzApproximator(shape_x, (k,), jp.ones((k,)), 10), # lipschitz is all 1's
    ]
    contractions = [
        lambda x, u, xdot, known, unknown: inverse_contraction_C(xdot - known[0](x,u), known[1](x, u), unknown[0](x))

    ]

    h_obj = init_HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = known_functions[0](x,u) + jp.matmul(known_functions[1](x,u), unknown_approximators[0](x))
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj = contract(h_obj, x,u, x_dot)
    expected_x_dot = Interval(jp.array([-3999999.0000000,-3999999.0000000,-3999999.0000000,]), jp.array([4000001.0000000, 4000001.0000000, 4000001.0000000]))
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    # fails to contract because our unknown terms were all nonzero, so we cannot contract any one individually
    x = jp.array([2.0, 3.0, 4.0])
    expected_x_dot = Interval(jp.array([-3999999.0000000,-3999999.0000000,-3999999.0000000,]), jp.array([4000001.0000000, 4000001.0000000, 4000001.0000000]))
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '3.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)


def test_H_object5(seed):
    # This is the test case where H = Known1(x,u) + known2(x,u) * unknown(x)
    shape_x = (3,)  # = N
    shape_u = (2,)  # = M
    k = shape_x[0] + shape_u[0] # k unknown terms

    known_functions = [lambda x, u: jp.zeros(shape_x),
                       lambda x, u: jp.vstack(( jp.concatenate((x, u)),
                                               jp.concatenate((x, u)),
                                               jp.concatenate((x, u)))),
                       ]

    H = lambda x, u, known, unknown: known[0](x,u) + jp.matmul(known[1](x,u), unknown[0](x))  # given that known[0] is zeros,
                                                                                              # And known[1] is x x x u u
                                                                                                              # x x x u u
                                                                                                              # x x x u u
                                                                                            # This form is equilvalent to F + GU
                                                                                            # where f = x x x...
                                                                                            # and G = u u
    unknown_approximators = [
        init_LipschitzApproximator(shape_x, (k,), jp.ones((k,)), 10, # lipschitz is all 1's
                                boundsOnFunctionValues = Interval(jp.ones((k,)) * -100.0, jp.ones((k,)) * 100.0, )), # BOUND IS -100 TO 100
    ]
    contractions = [
        lambda x, u, xdot, known, unknown: inverse_contraction_C(xdot - known[0](x,u), known[1](x, u), unknown[0](x))

    ]

    h_obj = init_HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

    x = jp.array([1.0, 2.0, 3.0])
    u = jp.array([0.0, 0.0])
    expected_x_dot = known_functions[0](x,u) + jp.matmul(known_functions[1](x,u), unknown_approximators[0](x))
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

    x_dot = jp.array([10.0, 30.0, 45.0]) # lets say we sample this x dot at the above x,u
    h_obj = contract(h_obj, x, u, x_dot)
    expected_x_dot = Interval(jp.array([-555.0000000,-555.0000000,-555.0000000,]), jp.array([600.0000000, 600.0000000, 600.0000000]))
    result = get_x_dot(h_obj, x, u)
    assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

