import math

from datacontrolreach.LipschitzApproximator import LipschitzApproximator, f_approximate, init_LipschitzApproximator, approximate, add_data
from datacontrolreach.interval import Interval
import numpy as np
import time
from jax.tree_util import register_pytree_node_class
import jax
import datacontrolreach.jumpy as jp

def test_approximate(seed):
    np.random.seed(seed)

    # Generate the data for test
    input_shape = (3,)
    output_shape = (2,)
    lipschitz = np.array([10.0, 5.0])
    f_value_bounds = Interval(np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    lipAprox = init_LipschitzApproximator(input_shape, output_shape, lipschitz, 10, f_value_bounds)

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([2.0, 3.0, 4.0])
    assert (approximate(lipAprox ,x1) == f_value_bounds).all(), '1. No data approximation fails : {} , {}\n'.format(approximate(lipAprox, x1), f_value_bounds)
    assert (approximate(lipAprox, x2) == f_value_bounds).all(), '2. No data approximation fails : {} , {}\n'.format(approximate(lipAprox, x2), f_value_bounds)

    lipAprox = add_data(lipAprox, x1, Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    assert (approximate(lipAprox, x1) == Interval(np.array([0, 0]), np.array([0.0, 0.0]))).all(), '3. Approximation fails : {} , {}\n'.format(approximate(lipAprox, x1), Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    answer = Interval(np.array([math.sqrt(3.0) * -10.0, math.sqrt(3.0) * -5.0]), np.array([math.sqrt(3.0) * 10.0, math.sqrt(3.0) * 5.0]))
    assert (approximate(lipAprox, x2) == answer).all(), '4. Approximation fails : {} , {}\n'.format(approximate(lipAprox, x2), answer)

def test_approximate_2d(seed):
    np.random.seed(seed)

    # Generate the data for test
    input_shape = (3,)
    output_shape = (2, 2)
    lipschitz = np.array([[10.0, 5.0], [3.0,2.0]])
    f_value_bounds = Interval(np.array([[-100.0, -100.0],[-100.0, -100.0]]), np.array([[100.0, 100.0],[100.0, 100.0]]))

    lipAprox = init_LipschitzApproximator(input_shape, output_shape, lipschitz, 100, f_value_bounds)
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([2.0, 3.0, 4.0])
    assert (approximate(lipAprox, x1) == f_value_bounds).all(), '1. No data approximation fails : {} , {}\n'.format(approximate(lipAprox, x1), f_value_bounds)
    assert (approximate(lipAprox, x2) == f_value_bounds).all(), '2. No data approximation fails : {} , {}\n'.format(approximate(lipAprox, x2), f_value_bounds)

    lipAprox = add_data(lipAprox, x1, Interval(np.array([[0.0, 0.0],[0.0, 0.0]])))
    assert (approximate(lipAprox, x1) == Interval(np.array([[0.0, 0.0],[0.0, 0.0]]))).all(), '3. Approximation fails : {} , {}\n'.format(approximate(lipAprox, x1), Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    answer = Interval(np.array([[math.sqrt(3.0) * -10.0, math.sqrt(3.0) * -5.0], [math.sqrt(3.0) * -3.0, math.sqrt(3.0) * -2.0]]), np.array([[math.sqrt(3.0) * 10.0, math.sqrt(3.0) * 5.0], [math.sqrt(3.0) * 3.0, math.sqrt(3.0) * 2.0]]))
    assert (approximate(lipAprox, x2) == answer).all(), '4. Approximation fails : {} , {}\n'.format(approximate(lipAprox, x2), answer)


def test_approximate_weights(seed):
    np.random.seed(seed)

    # Generate the data for test
    input_shape = (3,)
    output_shape = (2,)
    lipschitz = np.array([10.0, 5.0])
    f_value_bounds = Interval(np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
    importance_weights = np.array([[1.0,1.0,0.0], [1.0,1.0,1.0]]) # this says the first output does not depend on the last input at all

    lipAprox = init_LipschitzApproximator(input_shape, output_shape, lipschitz, 100, f_value_bounds, importanceWeights=importance_weights)
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([1.0, 2.0, 4.0])
    assert (approximate(lipAprox, x1) == f_value_bounds).all(), '1. No data approximation fails : {} , {}\n'.format(approximate(lipAprox, x1), f_value_bounds)
    assert (approximate(lipAprox, x2) == f_value_bounds).all(), '2. No data approximation fails : {} , {}\n'.format(approximate(lipAprox, x2), f_value_bounds)

    lipAprox = add_data(lipAprox, x1, Interval(np.array([0.0, 0.0])))
    assert (approximate(lipAprox, x1) == Interval(np.array([0, 0]), np.array([0.0, 0.0]))).all(), '3. Approximation fails : {} , {}\n'.format(approximate(lipAprox, x1), Interval(np.array([0.0, 0.0])))

    # note the first output should be 0 because it does not care about the 3rd input being different
    # note normed 2 distance is only 1 instead of 3 this time
    answer = Interval(np.array([0.0, -math.sqrt(1.0) * 5.0]), np.array([0.0, math.sqrt(1.0) * 5.0]))
    assert (approximate(lipAprox, x2) == answer).all(), '4. Approximation fails : {} , {}\n'.format(approximate(lipAprox, x2), answer)

    # note the first output should be 0 because it does not care about the 3rd input being different
    # note normed 2 distance is only 1 instead of 3 this time
    x2 = np.array([1.0, 2.0, 10.0])
    answer = Interval(np.array([0.0, -math.sqrt(7.0**2) * 5.0]), np.array([0.0, math.sqrt(7.0**2) * 5.0]))
    assert (approximate(lipAprox, x2) == answer).all(), '5. Approximation fails : {} , {}\n'.format(approximate(lipAprox, x2), answer)

# [TODO]: Change Jacobian Test according to the change in f_approximation jvp
def test_jacobian_approximate(seed):
    np.random.seed(seed)

    # Generate the data for test
    # Generate the data for test
    input_shape = (3,)
    output_shape = (2,)
    lipschitz = jp.array([10.0, 5.0])
    f_value_bounds = Interval(jp.array([-100.0, -100.0]), jp.array([100.0, 100.0]))

    lipAprox = init_LipschitzApproximator(input_shape, output_shape, lipschitz, 100, f_value_bounds)
    x1 = jp.array([1.0, 2.0, 3.0])
    lipAprox = add_data(lipAprox, x1, Interval(jp.array([0.0, 0.0])))

    # test interval + interval
    value, derivative = jax.jvp(f_approximate,
                                (lipAprox.boundsOnFunctionValues, lipAprox.x_data, lipAprox.f_x_data, lipAprox.importanceWeights, lipAprox.lipschitzConstants, x1),
                                (lipAprox.boundsOnFunctionValues, lipAprox.x_data, lipAprox.f_x_data, lipAprox.importanceWeights, lipAprox.lipschitzConstants, x1))
    assert (derivative == jp.matmul(jp.ones(output_shape + input_shape), x1) * lipschitz * Interval(-1.0, 1.0)).all(), 'Jacobian for Subtraction between intervals fails : {} , {}\n'.format(derivative, Interval(-1.0, 1.0) * lipschitz).all()

def test_jacobian_approximate_2d(seed):
    np.random.seed(seed)

    # Generate the data for test
    # Generate the data for test
    input_shape = (3,)
    output_shape = (2,2)
    lipschitz = jp.array([[10.0, 5.0], [10.0, 5.0]])
    f_value_bounds = Interval(jp.array([[-100.0, -100.0],[-100.0, -100.0]]), jp.array([[100.0, 100.0],[100.0, 100.0]]))

    lipAprox = init_LipschitzApproximator(input_shape, output_shape, lipschitz, 100, f_value_bounds)
    x1 = jp.array([1.0, 2.0, 3.0])
    add_data(lipAprox, x1, Interval(np.array([[0.0, 0.0],[0.0, 0.0]])))

    # test interval + interval
    value, derivative = jax.jvp(f_approximate,
                                (lipAprox.boundsOnFunctionValues, lipAprox.x_data, lipAprox.f_x_data, lipAprox.importanceWeights, lipAprox.lipschitzConstants, x1),
                                (lipAprox.boundsOnFunctionValues, lipAprox.x_data, lipAprox.f_x_data, lipAprox.importanceWeights, lipAprox.lipschitzConstants, x1))
    assert (derivative == jp.matmul(jp.ones(output_shape + input_shape), x1) * lipschitz * Interval(-1.0, 1.0)).all(), 'Jacobian for Subtraction between intervals fails : {} , {}\n'.format(derivative, Interval(-1.0, 1.0) *  lipschitz).all()

