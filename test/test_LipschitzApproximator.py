import math

from datacontrolreach.LipschitzApproximator import LipschitzApproximator
from datacontrolreach.interval import Interval
import numpy as np
import time
from jax.tree_util import register_pytree_node_class


# Generate the data for test
#input_shape = (3,)
#output_shape = (2,2)
#lipschitz = np.array([[10, 5],[10, 5]])
#f_value_bounds = Interval(np.array([[-100, -100],[-100, -100]]), np.array([[100, 100],[100, 100]]))
#lipAprox = LipschitzApproximator(input_shape, output_shape, lipschitz, f_value_bounds)

#lipAprox.add_data([0,0,0], Interval(np.array([[0.0, 0.0],[0.0, 0.0]])))
#print(lipAprox.approximate([0,0,0]))

def test_approximate(seed):
    np.random.seed(seed)

    # Generate the data for test
    input_shape = (3,)
    output_shape = (2,)
    lipschitz = np.array([10, 5])
    f_value_bounds = Interval(np.array([-100, -100]), np.array([100, 100]))

    lipAprox = LipschitzApproximator(input_shape, output_shape, lipschitz, f_value_bounds)
    x1 = [1, 2, 3]
    x2 = [2, 3, 4]
    assert (lipAprox.approximate(x1) == f_value_bounds).all(), '1. No data approximation fails : {} , {}\n'.format(lipAprox.approximate(x1), f_value_bounds)
    assert (lipAprox.approximate(x2) == f_value_bounds).all(), '2. No data approximation fails : {} , {}\n'.format(lipAprox.approximate(x2), f_value_bounds)

    lipAprox.add_data(x1, Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    assert (lipAprox.approximate(x1) == Interval(np.array([0, 0]), np.array([0.0, 0.0]))).all(), '3. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x1), Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    answer = Interval(np.array([math.sqrt(3.0) * -10.0, math.sqrt(3.0) * -5.0]), np.array([math.sqrt(3.0) * 10.0, math.sqrt(3.0) * 5.0]))
    assert (lipAprox.approximate(x2) == answer).all(), '4. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x2), answer)


def test_approximate_2d(seed):
    np.random.seed(seed)

    # Generate the data for test
    input_shape = (3,)
    output_shape = (2, 2)
    lipschitz = np.array([[10, 5], [3,2]])
    f_value_bounds = Interval(np.array([[-100, -100],[-100, -100]]), np.array([[100, 100],[100, 100]]))

    lipAprox = LipschitzApproximator(input_shape, output_shape, lipschitz, f_value_bounds)
    x1 = [1, 2, 3]
    x2 = [2, 3, 4]
    assert (lipAprox.approximate(x1) == f_value_bounds).all(), '1. No data approximation fails : {} , {}\n'.format(lipAprox.approximate(x1), f_value_bounds)
    assert (lipAprox.approximate(x2) == f_value_bounds).all(), '2. No data approximation fails : {} , {}\n'.format(lipAprox.approximate(x2), f_value_bounds)

    lipAprox.add_data(x1, Interval(np.array([[0.0, 0.0],[0.0, 0.0]])))
    assert (lipAprox.approximate(x1) == Interval(np.array([[0.0, 0.0],[0.0, 0.0]]))).all(), '3. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x1), Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    answer = Interval(np.array([[math.sqrt(3.0) * -10.0, math.sqrt(3.0) * -5.0], [math.sqrt(3.0) * -3.0, math.sqrt(3.0) * -2.0]]), np.array([[math.sqrt(3.0) * 10.0, math.sqrt(3.0) * 5.0], [math.sqrt(3.0) * 3.0, math.sqrt(3.0) * 2.0]]))
    assert (lipAprox.approximate(x2) == answer).all(), '4. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x2), answer)

def test_approximate_weights(seed):
    np.random.seed(seed)

    # Generate the data for test
    input_shape = (3,)
    output_shape = (2,)
    lipschitz = np.array([10, 5])
    f_value_bounds = Interval(np.array([-100, -100]), np.array([100, 100]))
    importance_weights = np.array([[1,1,0], [1,1,1]]) # this says the first output does not depend on the last input at all

    lipAprox = LipschitzApproximator(input_shape, output_shape, lipschitz, f_value_bounds, importanceWeights=importance_weights)
    x1 = [1, 2, 3]
    x2 = [1, 2, 4]
    assert (lipAprox.approximate(x1) == f_value_bounds).all(), '1. No data approximation fails : {} , {}\n'.format(lipAprox.approximate(x1), f_value_bounds)
    assert (lipAprox.approximate(x2) == f_value_bounds).all(), '2. No data approximation fails : {} , {}\n'.format(lipAprox.approximate(x2), f_value_bounds)

    lipAprox.add_data(x1, Interval(np.array([0.0, 0.0])))
    assert (lipAprox.approximate(x1) == Interval(np.array([0, 0]), np.array([0.0, 0.0]))).all(), '3. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x1), Interval(np.array([0.0, 0.0])))

    # note the first output should be 0 because it does not care about the 3rd input being different
    # note normed 2 distance is only 1 instead of 3 this time
    answer = Interval(np.array([0.0, -math.sqrt(1.0) * 5.0]), np.array([0.0, math.sqrt(1.0) * 5.0]))
    assert (lipAprox.approximate(x2) == answer).all(), '4. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x2), answer)

    # note the first output should be 0 because it does not care about the 3rd input being different
    # note normed 2 distance is only 1 instead of 3 this time
    x2 = [1, 2, 10]
    answer = Interval(np.array([0.0, -math.sqrt(7.0**2) * 5.0]), np.array([0.0, math.sqrt(7.0**2) * 5.0]))
    assert (lipAprox.approximate(x2) == answer).all(), '5. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x2), answer)

def test_approximate_norm1(seed):
    np.random.seed(seed)

    # Generate the data for test
    input_shape = (3,)
    output_shape = (2,)
    lipschitz = np.array([10, 5])
    f_value_bounds = Interval(np.array([-100, -100]), np.array([100, 100]))

    lipAprox = LipschitzApproximator(input_shape, output_shape, lipschitz, f_value_bounds, norm_order=1)
    x1 = [1, 2, 3]
    x2 = [2, 3, 4]
    assert (lipAprox.approximate(x1) == f_value_bounds).all(), '1. No data approximation fails : {} , {}\n'.format(
        lipAprox.approximate(x1), f_value_bounds)
    assert (lipAprox.approximate(x2) == f_value_bounds).all(), '2. No data approximation fails : {} , {}\n'.format(
         lipAprox.approximate(x2), f_value_bounds)

    lipAprox.add_data(x1, Interval(np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    assert (lipAprox.approximate(x1) == Interval(np.array([0, 0]), np.array(
         [0.0, 0.0]))).all(), '3. Approximation fails : {} , {}\n'.format(lipAprox.approximate(x1),
                                                                         Interval(np.array([0.0, 0.0]),
                                                                                  np.array([0.0, 0.0])))
    answer = Interval(np.array([3.0 * -10.0, 3.0 * -5.0]),
                      np.array([3.0 * 10.0, 3.0 * 5.0]))
    assert (lipAprox.approximate(x2) == answer).all(), '4. Approximation fails : {} , {}\n'.format(
        lipAprox.approximate(x2), answer)

