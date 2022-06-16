import math

from datacontrolreach.LipschitzApproximator import LipschitzApproximator, f_approximate
from datacontrolreach.interval import Interval
import time
from jax.tree_util import register_pytree_node_class
import jax
import datacontrolreach.jumpy as jp
from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent
from datacontrolreach.HObject import HObject, inverse_contraction

# Generate the data for test
# This is the test case where H = F(X) + G(X)*U
shape_x = (3,)  # = N
shape_u = (2,)  # = M
known_functions = []
H = lambda x, u, known, unknown: unknown[0](x) + jp.matmul(unknown[1](x), u)
unknown_approximators = [
    LipschitzApproximator(shape_x, shape_x, jp.array([10.0, 5.0, 3.0])),  # This is F(X), which outputs a Nx1 array
    LipschitzApproximator(shape_x, (shape_x[0], shape_u[0]),
                          jp.array([[10.0, 5.0],  # This is G(X), which outputs a NxM array
                                    [10.0, 5.0],
                                    [10.0, 5.0]]))

]
contractions = [  # recall Xdot = H = F(X) + G(X) * U
    lambda x, u, xdot, known, unknown: xdot - jp.matmul(unknown[1](x), u),  # F(X) = xdot - G(X) * U
    lambda x, u, xdot, known, unknown: inverse_contraction(xdot - unknown[0](x), unknown[1](x), u),
    # G(X) = (xdot - F(X)) U^-1
    # Recall we cannot use pseudo-inverse and must use inverse_contraction instead
]

h_obj = HObject(shape_x, shape_u, known_functions, unknown_approximators, H, contractions)

x = jp.array([1.0, 2.0, 3.0])
u = jp.array([0.0, 0.0])
expected_x_dot = unknown_approximators[0](x) + jp.matmul(unknown_approximators[1](x), u)
result = h_obj.get_x_dot(x, u)
assert (expected_x_dot == result).all(), '1.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)

x_dot = jp.array([10.0, 30.0, 45.0])  # lets say we sample this x dot at the above x,u
h_obj.contract(x, u, x_dot)
expected_x_dot = Interval(x_dot)
result = h_obj.get_x_dot(x, u)
assert (expected_x_dot == result).all(), '2.H object prediction error. Expected {} , got {}\n'.format(expected_x_dot, result)



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
        lambda x, u, xdot, known, unknown: inverse_contraction(xdot - unknown[0](x), unknown[1](x), u),
                                                # G(X) = (xdot - F(X)) U^-1
                                                # Recall we cannot use pseudo-inverse and must use inverse_contraction instead
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
