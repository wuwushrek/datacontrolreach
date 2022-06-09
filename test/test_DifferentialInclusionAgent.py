import math

from datacontrolreach.LipschitzApproximator import LipschitzApproximator, f_approximate
from datacontrolreach.interval import Interval
import numpy as np
import time
from jax.tree_util import register_pytree_node_class
import jax
import datacontrolreach.jumpy as jp
from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent
from datacontrolreach.DifferentialInclusionAgent import contract_row_wise
from datacontrolreach.dynsideinfo import hc4revise_lin_eq

def test_contract_row_wise(seed):
    np.random.seed(seed)

    # Generate the data for test
    x = 3.0
    u = np.array([1.0, 2.0, 3.0])
    g = Interval(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '1. row wise contraction fails : {} , {}\n'.format(answer, new_approx)

    # Generate the data for test
    x = 3.0
    u = np.array([1.0, 0.0, 3.0])
    g = Interval(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(np.array([0.0, 0.0, 2/3.0]), np.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '2. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = 3.0
    u = np.array([0.0, 0.0, 3.0])
    g = Interval(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = 3.0
    u = np.array([1.0, 2.0, 0.0])
    g = Interval(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert (answer == new_approx).all(), '4. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = -1.0
    u = np.array([1.0, 2.0, 3.0])
    g = Interval(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 2/3.0]))
    assert (answer == new_approx).all(), '5. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)

    # Generate the data for test
    x = -1.0
    u = np.array([1.0,])
    g = Interval(np.array([-1.0]), np.array([1.0]))
    new_approx = contract_row_wise(x, g, u)
    answer = Interval(np.array([-1.0]), np.array([-1.0]))
    assert (answer == new_approx).all(), '6. row wise contraction fails : {} , {}\n'.format(answer, new_approx)
    assert (hc4revise_lin_eq(x, u, g) == new_approx).all(), '3. row wise contraction fails : {} , {}\n'.format(hc4revise_lin_eq(x, u, g), new_approx)