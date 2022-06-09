 # has to go in this order because of circular dependencies....
import numpy as np
import datacontrolreach.jumpy as jp
from datacontrolreach import interval
from datacontrolreach.interval import Interval
import jax
from functools import partial

# @jax.tree_util.register_pytree_node_class
class LipschitzApproximator:
    """ A class to approximate a function given Lipschitz constants for each dimension and data in the form x, Interval of F(x)
    """
    def __init__(self, shapeOfInputs, shapeOfOutputs, lipschitzConstants, boundsOnFunctionValues:Interval = None, importanceWeights = None):
        # check shapes to make sure everything is compatible
        assert shapeOfOutputs == jp.shape(lipschitzConstants)
        assert boundsOnFunctionValues is None or shapeOfOutputs == jp.shape(boundsOnFunctionValues)
        assert importanceWeights is None or (shapeOfOutputs + shapeOfInputs) == jp.shape(importanceWeights)

        # check data type. Int does not work!!!! (idk why, jax throws errors)
        assert lipschitzConstants.dtype == np.dtype('float64')
        assert boundsOnFunctionValues is None or boundsOnFunctionValues.dtype == np.dtype('float64')
        assert importanceWeights is None or importanceWeights.dtype == np.dtype('float64')

        self.shapeOfInputs = shapeOfInputs
        self.shapeOfOutputs = shapeOfOutputs
        self.lipschitzConstants = lipschitzConstants
        self.boundsOnFunctionValues = boundsOnFunctionValues if boundsOnFunctionValues is not None else Interval(jp.full(shapeOfOutputs, float('-inf')), jp.full(shapeOfOutputs, float('inf')))
        self.importanceWeights = importanceWeights if importanceWeights is not None else jp.ones(shapeOfOutputs + shapeOfInputs)

        # The first index is the data index. Initially we have no data so it is 0. We will add data to these arrays.
        self.x_data = jp.zeros(((0,) + shapeOfInputs))
        self.f_x_data = Interval(jp.zeros(((0, ) + shapeOfOutputs )))

    def approximate(self, x_to_predict):
        assert self.shapeOfInputs == jp.shape(x_to_predict), 'x_to_predict size wrong. Expected {} , got {}\n'.format(self.shapeOfInputs, jp.shape(x_to_predict))
        assert x_to_predict.dtype == np.dtype('float64')
        return f_approximate(self.boundsOnFunctionValues, self.x_data, self.f_x_data, self.importanceWeights, self.lipschitzConstants, x_to_predict)

    def add_data(self, x, f_x: Interval):
        assert jp.shape(x) == self.shapeOfInputs, 'x size wrong. Expected {} , got {}\n'.format(self.shapeOfInputs, jp.shape(x))
        assert jp.shape(f_x) == self.shapeOfOutputs, 'f_x size wrong. Expected {} , got {}\n'.format(self.shapeOfOutputs, jp.shape(f_x))
        self.x_data =     jp.vstack(( self.x_data,     jp.reshape(x,   (1,) + self.shapeOfInputs)))
        self.f_x_data =   jp.vstack(( self.f_x_data,   jp.reshape(f_x, (1,) + self.shapeOfOutputs)))

    # needed to make it a pytree node for jitable, differentiable
    #def tree_flatten(self):
    #    children = (self.shapeOfInputs, self.shapeOfOutputs, self.lipschitzConstants, self.boundsOnFunctionValues, self.datapoints,)  # arrays / dynamic values
    #    aux_data = None
    #    return children, aux_data

    #@classmethod
    #def tree_unflatten(cls, aux_data, children):
    #    return cls(*children, **aux_data)


# Note I have to pull this function out of the class because Jax cannot jit or differentiate class methods
# So we pull the function out, and call it from the class
@jax.custom_jvp
def f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, x_to_predict):
    # initial bound is -inf, inf
    f_x_bounds = boundsOnFunctionValues

    # for each data point, find bound based on lipschitz constants. Then find intersection
    for index in range(jp.shape(x_data)[0]):
        x = x_data[index]
        f_x = f_x_data[index]

        difference_x = jp.subtract(x, x_to_predict)

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

        bound = jp.add(f_x, cone)  # add the prediction at this point + the cone generated by X distance * lipschitz to get a bound
        f_x_bounds = bound & f_x_bounds  # finds the intersection of current bound and bound generated by this datapoint

    # after intersecting all bounds, return
    return f_x_bounds

# We cannot find the derivate at x. However, we do know the Lipschitz constant, so the derivative is between -Lipschitz and +Lipschitz
# so we can return that
@f_approximate.defjvp
def jvp_f_approximate( primal, tangents):
    """ Addition between two intervals
    """
    boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, x_to_predict = primal
    boundsOnFunctionValues_dot,  x_data_dot, f_x_data_dot, importanceWeights_dot, lipschitzConstants_dot, x_to_predict_dot = tangents
    value = f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, x_to_predict)
    derivative = Interval(-1.0, 1.0) *lipschitzConstants
    return value, derivative