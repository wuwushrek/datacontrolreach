import jax
import jax.numpy as jnp

import datacontrolreach.jumpy as jp
from datacontrolreach.jumpy import Interval

from functools import partial

from typing import NamedTuple

from jax.experimental.host_callback import id_print

# Define a maximum value for function bounds
MAX_FUNCTION_BOUNDS = 1e6

class LipschitzApproximator(NamedTuple):
    """ A class representing a function approximator which uses Lipschitz
        constants and data to bound a function output
        Attributes:
        lipschitzConstants      : Lipschitz constants with respect to the outputs.
        boundsOnFunctionValues  : Bounds the possible outputs of the function with
                                    respect to each output
        importanceWeights       : How important each input is with respect to outputs.
                                    Must be of the same shape as the state of the system
        xData                   : Data we have collected on the function.
                                    This is input records
        fxData                  : Data we have collected on the function.
                                    This is output estimates via intervals
        currentIndex            : Where to add new data. Record keeping for updating new data.
    """
    lipschitzConstants: jnp.ndarray
    boundsOnFunctionValues: Interval
    importanceWeights: jnp.ndarray
    xData: jnp.ndarray
    fxData: Interval
    currentIndex: jnp.ndarray


def init_LipschitzApproximator(lipschitzConstants, maxDataSize:int, numStates:int,
                                boundsOnFunctionValues:Interval = None,
                                importanceWeights = None):
    """A wrapper for initializing a LipschitzApproximator instance

    Args:
        lipschitzConstants (TYPE): Lipschitz constants with respect to the outputs.
        maxDataSize (int): The maximum size of the data buffer
        numStates (int): Description
        boundsOnFunctionValues (Interval, optional): Bounds the possible outputs of the functions
        importanceWeights (None, optional): How important each input is with respect to outputs.

    Returns:
        TYPE: LipschitzApproximator
    """
    # Define the number of unknown terms
    lipschitzConstants = jp.array(lipschitzConstants)
    shapeOfOutputs = lipschitzConstants.shape

    # Assume some large function bounds if not provided.
    # Do not assume infinity, even though its mathematically better, because it causes NaN
    if boundsOnFunctionValues is None:
        boundsOnFunctionValues = Interval(lb = jnp.full(shapeOfOutputs, -MAX_FUNCTION_BOUNDS),
                                          ub = jnp.full(shapeOfOutputs, MAX_FUNCTION_BOUNDS)
                                        )
    else:
        flb = jnp.array([bo[0] for bo in boundsOnFunctionValues ])
        fub = jnp.array([bo[1] for bo in boundsOnFunctionValues ])
        boundsOnFunctionValues = Interval(flb, fub)

    # If the importance weights is not given, assume the unknown functions
    # depend on every state variables of the system
    if importanceWeights is None:
        importanceWeights = jnp.ones((shapeOfOutputs[0], numStates))
    else:
        importanceWeights = jnp.array(importanceWeights)

    # Initially we are empty of data.
    # So we make it all 0's for inputs,
    # outputs will be max values so we dont affect predictions
    xData = jp.zeros((maxDataSize, numStates))
    fxData = jp.block_array([boundsOnFunctionValues for _ in range(maxDataSize)])

    return LipschitzApproximator(lipschitzConstants = lipschitzConstants,
                                boundsOnFunctionValues = boundsOnFunctionValues,
                                importanceWeights = importanceWeights,
                                xData = xData, fxData = fxData,
                                currentIndex = jnp.array(0))


def add_data(la:LipschitzApproximator, x, fx: Interval):
    """Summary

    Args:
        la (LipschitzApproximator): A Lipschitz-Based approximator
        x (TYPE): A state value
        fx (Interval): An estimate of the unknown function at that state

    Returns:
        TYPE: An update LipschitzApproximator
    """

    index = la.currentIndex
    xData = la.xData.at[index, :].set(x)
    fxData = jp.index_update(la.fxData, index, fx)
    newIndex = (index + 1) % la.xData.shape[0]
    return la._replace(xData=xData, fxData=fxData, currentIndex=newIndex)



# @partial(jax.custom_jvp, nondiff_argnums=(0,))
@jax.custom_jvp
def approximate(la: LipschitzApproximator, xToPredict):
    """Given a LipschitzApproximator object, this function provides an estimate
        of the vector field at the given state xToPredict

    Args:
        la (LipschitzApproximator): A Lipschitz-Based approximator
        xToPredict (TYPE): The state for which we want to predict function values

    Returns:
        TYPE: Over-approximation of the function values at the given state
    """
    # initial bound is -inf, inf or whatever the user sets
    fxBounds = Interval(la.boundsOnFunctionValues)

    # for each data point, find bound based on lipschitz constants.
    # Then find intersection
    def calculate_new_bound_given_past_data(fxBound, xs):
        x, fx = xs

        differenceX = x - xToPredict

        # This process computes the norm of the difference in X of the data
        # and the X we are trying to predict

        # First elementwise square the difference to the x to predict
        normedDifferenceX = differenceX ** 2

        # Then we multiply by the importance weights of every output with respect to every input.
        # This outputs a distance with respect to each output
        weightedDistances = jp.matmul(la.importanceWeights, normedDifferenceX)

        # Then we have to undo the (elementwise) power operation from above, sqrt in the case of norm 2
        normedWeightedDistances = jp.sqrt(weightedDistances)

        # Elementwise multiply the distance by the lipschitz constant to get a bound
        # on how much the function value could possibly change. It is shaped like a cone.
        possibleChangeOutput = la.lipschitzConstants * normedWeightedDistances

        # cone = jp.multiply(Interval(-1.0, 1.0), possible_change_output)
        cone = Interval(-1.0, 1.0) * possibleChangeOutput

        # Add the prediction at this point + the cone generated by X distance * lipschitz to get a bound
        bound = fx + cone

        # finds the intersection of current bound and bound generated by this datapoint
        fxBound = bound & fxBound

        return fxBound, None

    # Proceed to intersecting all bounds and return the resul
    fxBounds, _ = jax.lax.scan(calculate_new_bound_given_past_data, fxBounds, (la.xData, la.fxData))
    return fxBounds

@approximate.defjvp
def jvp_approximate(primal, tangents):
    """Compute the gradient of approximate. We cannot find the derivative at x.
       However, we do know the Lipschitz constant so the derivative is between
       -Lipschitz and +Lipschitz and we return that

    Args:
        la (LipschitzApproximator): A Lipschitz-Based approximator
        primal (TYPE): The primal variale for jvp computation
        tangents (TYPE): The tangent variable for jbp computation

    Returns:
        TYPE: Tuple (value, and grad * tangent)
    """
    # Extract the x used for computing approximate
    la, xToPredict = primal
    # Extract the tangent variable
    _, xToPredictDot = tangents
    # Approximate the unknown functions at xToPredict
    value = approximate(la, xToPredict)
    # Now the jvp is simply defined as described in the function definition
    derivative = jp.matmul(la.importanceWeights, xToPredictDot) \
                    * la.lipschitzConstants * Interval(-1.0, 1.0)
    return value, derivative

# # @approximate.defjvp
# def jvp_approximate(la: LipschitzApproximator, primal, tangents):
#     """Compute the gradient of approximate. We cannot find the derivative at x.
#        However, we do know the Lipschitz constant so the derivative is between
#        -Lipschitz and +Lipschitz and we return that

#     Args:
#         la (LipschitzApproximator): A Lipschitz-Based approximator
#         primal (TYPE): The primal variale for jvp computation
#         tangents (TYPE): The tangent variable for jbp computation

#     Returns:
#         TYPE: Tuple (value, and grad * tangent)
#     """
#     # Extract the x used for computing approximate
#     xToPredict, = primal
#     # Extract the tangent variable
#     xToPredictDot, = tangents
#     # Approximate the unknown functions at xToPredict
#     value = approximate(la, xToPredict)
#     # Now the jvp is simply defined as described in the function definition
#     derivative = jp.matmul(la.importanceWeights, xToPredictDot) \
#                     * la.lipschitzConstants * Interval(-1.0, 1.0)
#     return value, derivative
