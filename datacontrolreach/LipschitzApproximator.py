 # has to go in this order because of circular dependencies....
import numpy as np
import datacontrolreach.jumpy as jp
from datacontrolreach import interval
from datacontrolreach.interval import Interval
import jax


@jax.tree_util.register_pytree_node_class
class LipschitzApproximator:
    """ A class to approximate a function given Lipschitz constants for each dimension and data in the form x, Interval of F(x)
    """
    def __init__(self, shapeOfInputs, shapeOfOutputs, lipschitzConstants, boundsOnFunctionValues:Interval = None, importanceWeights = None, norm_order = 2):
        assert shapeOfOutputs == np.shape(lipschitzConstants)
        assert boundsOnFunctionValues is None or shapeOfOutputs == np.shape(boundsOnFunctionValues)
        assert importanceWeights is None or (shapeOfOutputs + shapeOfInputs) == np.shape(importanceWeights)

        self.shapeOfInputs = shapeOfInputs
        self.shapeOfOutputs = shapeOfOutputs
        self.lipschitzConstants = lipschitzConstants
        self.boundsOnFunctionValues = boundsOnFunctionValues if boundsOnFunctionValues is not None else Interval(jp.full(shapeOfOutputs, float('-inf')), jp.full(shapeOfOutputs, float('inf')))
        self.importanceWeights = importanceWeights if importanceWeights is not None else np.ones(shapeOfOutputs + shapeOfInputs)
        self.norm_order = norm_order
        self.datapoints = []

    def approximate(self, x_to_predict):
        assert self.shapeOfInputs == np.shape(x_to_predict)
        # initial bound is -inf, inf
        f_x_bounds = self.boundsOnFunctionValues

        # for each data point, find bound based on lipschitz constants. Then find intersection
        for x, f_x in self.datapoints:
            difference_x = jp.abs(jp.subtract(x, x_to_predict))

            # this process calculates the norm of the difference in X of the data and the X we are trying to predict
            # first we (elementwise) square the distance (in case of norm 2)
            normed_difference_x = difference_x ** self.norm_order

            # then we multiply by the importance weights of every output with respect to every input. This outputs a distance with respect to each output
            weighted_distances = np.matmul(self.importanceWeights, normed_difference_x)

            # then we have to undo the (elementwise) power operation from above, typically sqrt in the case of norm 2
            normed_weighted_distances = weighted_distances ** (1.0/self.norm_order)

            # elementwise multiply the distance by the lipschitz constant to get a bound on how much the function value could possibly change. It is shaped like a cone.
            possible_change_output = jp.multiply(self.lipschitzConstants, normed_weighted_distances)
            cone = jp.multiply(possible_change_output, Interval(-1.0, 1.0))

            bound = jp.add(f_x, cone) # add the prediction at this point + the cone generated by X distance * lipschitz to get a bound
            f_x_bounds = bound & f_x_bounds   # finds the intersection of current bound and bound generated by this datapoint

        # after intersecting all bounds, return
        return f_x_bounds

    def add_data(self, x, f_x: Interval):
        assert jp.shape(x) == self.shapeOfInputs
        assert jp.shape(f_x) == self.shapeOfOutputs
        self.datapoints.append((x, f_x))

    # needed to make it a pytree node for jitable, differentiable
    def tree_flatten(self):
        children = (self.shapeOfInputs, self.shapeOfOutputs, self.lipschitzConstants, self.boundsOnFunctionValues, self.datapoints,)  # arrays / dynamic values
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


