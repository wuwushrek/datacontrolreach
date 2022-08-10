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

        assert shapeOfOutputs == lipschitzConstants.shape, 'Shape of outputs {} does not match shape of Lipschitz constants {}\n'.format(shapeOfOutputs, jp.shape(lipschitzConstants))
        assert boundsOnFunctionValues is None or shapeOfOutputs == boundsOnFunctionValues.shape, 'Bounds on function values must match the output shape. Got {} shape for bounds, {} for outputs'.format(boundsOnFunctionValues.shape, shapeOfOutputs)
        assert importanceWeights is None or (shapeOfOutputs + shapeOfInputs) == importanceWeights.shape, 'Importance weights shape must be MxN, where is M is the output shape and N is the input shape. Got {} for shape, expected {}'.format(importanceWeights.shape, (shapeOfOutputs + shapeOfInputs))

        # # check data type. Int does not work!!!! (idk why, jax throws errors)
        # # [NOTE] float32 or float64 ARE float, so will just check if it is a float instead of differentiationg between 32 or 64
        # assert lipschitzConstants.dtype == float
        # assert boundsOnFunctionValues is None or boundsOnFunctionValues.dtype == float
        # assert importanceWeights is None or importanceWeights.dtype == float

        self.shapeOfInputs = shapeOfInputs
        self.shapeOfOutputs = shapeOfOutputs
        self.lipschitzConstants = lipschitzConstants
        # Assume some large function bounds if not provided. Do not assume infinity, even though its mathematically better, because it causes NaN
        self.boundsOnFunctionValues = boundsOnFunctionValues if boundsOnFunctionValues is not None else Interval(jp.full(shapeOfOutputs, -1e6), jp.full(shapeOfOutputs, 1e6))
        self.importanceWeights = importanceWeights if importanceWeights is not None else jp.ones(shapeOfOutputs + shapeOfInputs)

        # The first index is the data index. Initially we have no data so it is 0. We will add data to these arrays.
        self.x_data = jp.zeros(((0,) + shapeOfInputs))
        self.f_x_data = Interval(jp.zeros(((0, ) + shapeOfOutputs )))

    def __call__(self, x_to_predict):
        return self.approximate(x_to_predict)

    def approximate(self, x_to_predict):
        assert self.shapeOfInputs == x_to_predict.shape, 'x_to_predict size wrong. Expected {} , got {}\n'.format(self.shapeOfInputs, x_to_predict.shape)
        # assert x_to_predict.dtype == float
        return f_approximate(self.boundsOnFunctionValues, self.x_data, self.f_x_data, self.importanceWeights, self.lipschitzConstants, x_to_predict)

    def add_data(self, x, f_x: Interval):
        assert x.shape == self.shapeOfInputs, 'x size wrong. Expected {} , got {}\n'.format(self.shapeOfInputs, x.shape)
        assert f_x.shape == self.shapeOfOutputs, 'f_x size wrong. Expected {} , got {}\n'.format(self.shapeOfOutputs, f_x.shape)
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


#[TODO] Fix This as it should be non-differentiable with respect to the first 4 parameters
# Note I have to pull this function out of the class because Jax cannot jit or differentiate class methods
# So we pull the function out, and call it from the class
@partial(jax.custom_jvp, nondiff_argnums=(0,1,2,3,4))
# @jax.custom_jvp
def f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, x_to_predict):
    # initial bound is -inf, inf
    f_x_bounds = boundsOnFunctionValues

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


# We cannot find the derivate at x. However, we do know the Lipschitz constant, so the derivative is between -Lipschitz and +Lipschitz
# so we can return that
@f_approximate.defjvp
def jvp_f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, primal, tangents):
    x_to_predict, = primal
    x_to_predict_dot, = tangents
    value = f_approximate(boundsOnFunctionValues, x_data, f_x_data, importanceWeights, lipschitzConstants, x_to_predict)
    derivative = jp.matmul(importanceWeights, x_to_predict_dot) * lipschitzConstants * Interval(-1.0, 1.0)
    return value, derivative