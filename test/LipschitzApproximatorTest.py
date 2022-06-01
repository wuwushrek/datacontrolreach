from datacontrolreach.LipschitzApproximator import LipschitzApproximator
from datacontrolreach.interval import Interval
import numpy as np
import time
from jax.tree_util import register_pytree_node_class

input_shape = (3,)
output_shape = (2,)
lipschitz = np.array([10, 5])
f_value_bounds = Interval(np.array([-100, -100]), np.array([100, 100]))

lipAprox = LipschitzApproximator(input_shape, output_shape, lipschitz, f_value_bounds)
x1 = [1,2,3]
x2 = [2,3,4]
print("initial approximation at ", x1, " is", lipAprox.approximate(x1))
print("initial approximation at ", x2, " is", lipAprox.approximate(x2))

lipAprox.add_data(x1, Interval(np.array([0, 0]), np.array([0.0, 0.0])))

print("final approximation at ", x1, " is", lipAprox.approximate(x1))
print("final approximation at ", x2, " is", lipAprox.approximate(x2))

show_example(lipAprox)

value, derivative = jax.jvp(LipschitzApproximator.approximate, (lipAprox,), (lipAprox,))
