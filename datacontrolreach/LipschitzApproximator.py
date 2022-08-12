 # has to go in this order because of circular dependencies....
import copy

import numpy as np
import datacontrolreach.jumpy as jp
from datacontrolreach import interval
from datacontrolreach.interval import Interval
import jax
from functools import partial
from typing import NamedTuple

# A class representing a function approximator which uses Lipschitz constants to bound a function output
class LipschitzApproximator(NamedTuple):
    shapeOfInputs:tuple                 # Tuple of the input shape
    shapeOfOutputs:tuple                # tuple of the output shape

    lipschitzConstants:jp.array         # Lipschitz constants with respect to the outputs.
    boundsOnFunctionValues:Interval     # Bounds the possible outputs of the function with respect to each output
    importanceWeights:jp.array          # How important each input is with respect to outputs. Must of shape outputs x inputs

    x_data:jp.array                     # Data we have collected on the function. This is input records
    f_x_data:Interval                   # Data we have collected on the function. This is output estimates via intervals

    max_data_size:int                   # Max data we can collect. This must be limited for jax to work
    current_index:int                   # Record keeping for updating new data.


def init_LipschitzApproximator(shapeOfInputs, shapeOfOutputs, lipschitzConstants, max_data_size:int, boundsOnFunctionValues:Interval = None, importanceWeights = None):
    # check shapes to make sure everything is compatible

    assert shapeOfOutputs == lipschitzConstants.shape, 'Shape of outputs {} does not match shape of Lipschitz constants {}\n'.format(shapeOfOutputs, jp.shape(lipschitzConstants))
    assert boundsOnFunctionValues is None or shapeOfOutputs == boundsOnFunctionValues.shape, 'Bounds on function values must match the output shape. Got {} shape for bounds, {} for outputs'.format(boundsOnFunctionValues.shape, shapeOfOutputs)
    assert importanceWeights is None or (shapeOfOutputs + shapeOfInputs) == importanceWeights.shape, 'Importance weights shape must be MxN, where is M is the output shape and N is the input shape. Got {} for shape, expected {}'.format(importanceWeights.shape, (shapeOfOutputs + shapeOfInputs))

    # Assume some large function bounds if not provided. Do not assume infinity, even though its mathematically better, because it causes NaN
    boundsOnFunctionValues = boundsOnFunctionValues if boundsOnFunctionValues is not None else Interval(jp.full(shapeOfOutputs, -1e6), jp.full(shapeOfOutputs, 1e6))
    importanceWeights = importanceWeights if importanceWeights is not None else jp.ones(shapeOfOutputs + shapeOfInputs)

    # Initially we are empty of data. So we make it all 0's for inputs, outputs will be max values so we dont affect predictions
    x_data = jp.zeros((max_data_size,) + shapeOfInputs)
    f_x_data = Interval(jp.zeros((0, ) + shapeOfOutputs ))
    for i in range(max_data_size):
        f_x_data =   jp.vstack( (f_x_data,  jp.reshape(boundsOnFunctionValues, (1,) + shapeOfOutputs)))

    assert f_x_data.shape == (max_data_size, ) + shapeOfOutputs

    return LipschitzApproximator(shapeOfInputs, shapeOfOutputs, lipschitzConstants, boundsOnFunctionValues, importanceWeights, x_data, f_x_data, max_data_size, 0)


def approximate(LipschitzApproximator:LipschitzApproximator, x_to_predict):
    # assert LipschitzApproximator.shapeOfInputs == x_to_predict.shape, 'x_to_predict size wrong. Expected {} , got {}\n'.format(LipschitzApproximator.shapeOfInputs, x_to_predict.shape)
    return f_approximate(LipschitzApproximator.boundsOnFunctionValues,
                         LipschitzApproximator.x_data,
                         LipschitzApproximator.f_x_data,
                         LipschitzApproximator.importanceWeights,
                         LipschitzApproximator.lipschitzConstants,
                         x_to_predict)

def add_data(la:LipschitzApproximator, x, f_x: Interval):
    assert x.shape == la.shapeOfInputs,      'x size wrong. Expected {} , got {}\n'.format(la.shapeOfInputs, x.shape)
    assert f_x.shape == la.shapeOfOutputs,   'f_x size wrong. Expected {} , got {}\n'.format(la.shapeOfOutputs, f_x.shape)

    index = la.current_index
    x_data = la.x_data.at[index, :].set(x)
    f_x_data = Interval(la.f_x_data.lb.at[index, :].set(f_x.lb),
                        la.f_x_data.ub.at[index, :].set(f_x.ub))


    return LipschitzApproximator(la.shapeOfInputs, la.shapeOfOutputs,
                                 la.lipschitzConstants,
                                 la.boundsOnFunctionValues,
                                 la.importanceWeights,
                                 x_data, f_x_data,
                                 la.max_data_size, index + 1)

# Note I have to pull this function out of the class because Jax cannot jit or differentiate class methods
# So we pull the function out, and call it from the class
@partial(jax.custom_jvp, nondiff_argnums=(0,1,2,3,4))
def f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, x_to_predict):
    # initial bound is -inf, inf
    # Deep copy
    f_x_bounds = Interval(jp.array(boundsOnFunctionValues.lb), jp.array(boundsOnFunctionValues.ub))

    # for each data point, find bound based on lipschitz constants. Then find intersection
    for index in range(x_data.shape[0]):
        x = x_data[index]
        f_x = f_x_data[index]

        difference_x = x - x_to_predict

        # this process calculates the norm of the difference in X of the data and the X we are trying to predict
        # first we (elementwise) square the distance (norm 2)
        normed_difference_x = difference_x ** 2

        # then we multiply by the importance weights of every output with respect to every input. This outputs a distance with respect to each output
        weighted_distances = jp.matmul(importanceWeights, normed_difference_x)

        # then we have to undo the (elementwise) power operation from above, sqrt in the case of norm 2
        normed_weighted_distances = jp.sqrt(weighted_distances)

        # elementwise multiply the distance by the lipschitz constant to get a bound on how much the function value could possibly change. It is shaped like a cone.
        # possible_change_output = jp.multiply(lipschitzConstants, normed_weighted_distances)
        possible_change_output = lipschitzConstants * normed_weighted_distances
        # cone = jp.multiply(Interval(-1.0, 1.0), possible_change_output)
        cone = Interval(-1.0, 1.0) * possible_change_output

        bound = f_x + cone  # add the prediction at this point + the cone generated by X distance * lipschitz to get a bound
        f_x_bounds = bound & f_x_bounds  # finds the intersection of current bound and bound generated by this datapoint

    # after intersecting all bounds, return
    return f_x_bounds


# We cannot find the derivative at x. However, we do know the Lipschitz constant, so the derivative is between -Lipschitz and +Lipschitz
# so we can return that
@f_approximate.defjvp
def jvp_f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, primal, tangents):
    x_to_predict, = primal
    x_to_predict_dot, = tangents
    value = f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, x_to_predict)
    derivative = jp.matmul(importanceWeights, x_to_predict_dot) * lipschitzConstants * Interval(-1.0, 1.0)
    return value, derivative