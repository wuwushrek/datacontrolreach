from functools import partial
import jax
from collections import deque
import datacontrolreach.jumpy as jp
import numpy as np


@jax.custom_jvp
def sum(list):
    total = 0
    for value in list[0][0]:
        total += value
    return total

@sum.defjvp
def jvp_f_approximate(primal, tangents):
    """ Addition between two intervals
    """
    list = primal,
    list_dot = tangents,
    return sum(list), 1.0

# list = jp.zeros((0, ))


#list = jp.concatenate((list, jp.add(jp.zeros((1, )), 1.0)))
#value, derivative = jax.jvp(sum, (list,), (list,))
#print(value, derivative)

#list = jp.concatenate((list, jp.add(jp.zeros((1, )), 10.0)))
#value, derivative = jax.jvp(sum, (list,), (list,))
#print(value, derivative)

#list = jp.concatenate((list, jp.add(jp.zeros((1, )), 53.0)))
#value, derivative = jax.jvp(sum, (list,), (list,))
#print(value, derivative)

init = np.zeros((0, 3, 3))
arr1 = np.zeros((1, 3,3))
init = np.vstack((init, arr1))
init = np.vstack((init, arr1))
init = np.vstack((init, arr1))
init = np.vstack((init, arr1))



x = np.zeros((1,3))
print(x, np.shape(x), type(x))
print(x[0], np.shape(x[0]), type(x[0]))
