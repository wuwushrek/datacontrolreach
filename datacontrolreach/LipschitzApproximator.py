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
    def __init__(self, shapeOfInputs, shapeOfOutputs, lipschitzConstants, boundsOnFunctionValues:Interval = None):
        """ Initialize an interval. The lower bound must always be greater that the upper bound
            :param lb : The lower bound of the interval (vector or matrix)
            :param ub : The upper bound of the interval (vector or matrix)
        """
        assert shapeOfOutputs == np.shape(lipschitzConstants)
        assert boundsOnFunctionValues is None or shapeOfOutputs == np.shape(boundsOnFunctionValues)

        self.shapeOfInputs = shapeOfInputs
        self.shapeOfOutputs = shapeOfOutputs
        self.lipschitzConstants = lipschitzConstants
        self.boundsOnFunctionValues = boundsOnFunctionValues if boundsOnFunctionValues is not None else Interval(np.full(shapeOfOutputs, float('-inf')), np.full(shapeOfOutputs, float('inf')))
        self.datapoints = []

    def approximate(self, x_to_predict):
        assert self.shapeOfInputs == np.shape(x_to_predict)
        # initial bound is -inf, inf
        f_x_bounds = self.boundsOnFunctionValues

        # for each data point, find bound based on lipschitz constants. Then find intersection
        for x, f_x in self.datapoints:
            bound = f_x + self.lipschitzConstants * np.linalg.norm(np.subtract(x, x_to_predict), 2) * Interval(-1.0, 1.0)
            f_x_bounds = bound & f_x_bounds   # finds the intersection

        # after intersecting all bounds, return
        return f_x_bounds

    def add_data(self, x, f_x: Interval):
        assert np.shape(x) == self.shapeOfInputs
        assert np.shape(f_x) == self.shapeOfOutputs
        self.datapoints.append((x, f_x))

    # needed to make it a pytree node
    def tree_flatten(self):
        children = (self.shapeOfInputs, self.shapeOfOutputs, self.lipschitzConstants, self.boundsOnFunctionValues, self.datapoints,)  # arrays / dynamic values
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


