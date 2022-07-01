# Modified from Brax github: https://github.com/google/brax/blob/main/brax/jumpy.py

# pylint:disable=redefined-builtin
"""Numpy backend for JAX that is called for jax arrays or intervals or any abstract sets."""

from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
import numpy as onp

F = TypeVar('F', bound=Callable)

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

import datacontrolreach.interval as itvl

ndarray = Union[jnp.ndarray, itvl.Interval]  # pylint:disable=invalid-name
pi = jnp.pi
inf = jnp.inf
float32 = jnp.float32
int32 = jnp.int32
float64 = jnp.float64
int64 = jnp.int64


def _which_np(*args):
  """Returns np or jnp depending on args."""
  for a in args:
    if isinstance(a, itvl.Interval):
      return itvl
  return jnp

def _is_array(args):
  return hasattr(args, 'shape')


def any(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether any array element along a given axis evaluates to True."""
  return _which_np(a).any(a, axis=axis)


def all(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether all array elements along a given axis evaluate to True."""
  return _which_np(a).all(a, axis=axis)


def take(tree: Any, i: Union[ndarray, Sequence[int]], axis: int = 0) -> Any:
  """Returns tree sliced by i."""
  np = _which_np(i)
  if isinstance(i, list) or isinstance(i, tuple):
    i = np.array(i, dtype=int)
  return jax.tree_map(lambda x: np.take(x, i, axis=axis, mode='clip'), tree)


def norm(x: ndarray,
         axis: Optional[Union[Tuple[int, ...], int]] = None) -> ndarray:
  """Returns the array norm."""
  np = _which_np(x, axis)
  if np is itvl:
    return np.norm(x, axis=axis)
  return np.linalg.norm(x, axis=axis)


def index_update(x: ndarray, idx: ndarray, y: ndarray) -> ndarray:
  """Pure equivalent of x[idx] = y."""
  np = _which_np(x)
  if np is itvl:
    return np.index_update(x, idx, y)
  return x.at[idx].set(y)

def mean(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Compute the arithmetic mean along the specified axis."""
  return _which_np(a).mean(a, axis=axis)


def arange(start: int, stop: int) -> ndarray:
  """Return evenly spaced values within a given interval."""
  return _which_np().arange(start, stop)


def dot(x: ndarray, y: ndarray) -> ndarray:
  """Returns dot product of two arrays."""
  return _which_np(x, y).dot(x, y)


def matmul(x1: ndarray, x2: ndarray) -> ndarray:
  """Matrix product of two arrays."""
  return _which_np(x1, x2).matmul(x1, x2)


def inv(a: ndarray) -> ndarray:
  """Compute the (multiplicative) inverse of a matrix."""
  return _which_np(a).linalg.inv(a)


def square(x: ndarray) -> ndarray:
  """Return the element-wise square of the input."""
  return _which_np(x).square(x)


def tile(x: ndarray, reps: Union[Tuple[int, ...], int]) -> ndarray:
  """Construct an array by repeating A the number of times given by reps."""
  return _which_np(x).tile(x, reps)


def repeat(a: ndarray, repeats: Union[int, ndarray]) -> ndarray:
  """Repeat elements of an array."""
  return _which_np(a, repeats).repeat(a, repeats=repeats)


def sin(angle: ndarray) -> ndarray:
  """Returns trigonometric sine, element-wise."""
  return _which_np(angle).sin(angle)


def cos(angle: ndarray) -> ndarray:
  """Returns trigonometric cosine, element-wise."""
  return _which_np(angle).cos(angle)


def logical_not(x: ndarray) -> ndarray:
  """Returns the truth value of NOT x element-wise."""
  return _which_np(x).logical_not(x)


def logical_or(x: ndarray, y: ndarray) -> ndarray:
  """Returns the truth value of x OR y element-wise."""
  return _which_np(x, y).logical_or(x, y)


def logical_and(x: ndarray, y: ndarray) -> ndarray:
  """Returns the truth value of x AND y element-wise."""
  return _which_np(x,y).logical_and(x, y)


def multiply(x1: ndarray, x2: ndarray) -> ndarray:
  """Multiply arguments element-wise."""
  return _which_np(x1, x2).multiply(x1, x2)


def exp(x: ndarray) -> ndarray:
  """Returns the exponential of all elements in the input array."""
  return _which_np(x).exp(x)


def sum(a: ndarray, axis: Optional[int] = None):
  """Returns sum of array elements over a given axis."""
  return _which_np(a).sum(a, axis=axis)


def concatenate(x: Sequence[ndarray], axis=0) -> ndarray:
  """Join a sequence of arrays along an existing axis."""
  return _which_np(*x).concatenate(x, axis=axis)

def vstack(x: Sequence[ndarray]) -> ndarray:
  """Join a sequence of arrays along an existing axis."""
  return _which_np(*x).vstack(x)


def sqrt(x: ndarray) -> ndarray:
  """Returns the non-negative square-root of an array, element-wise."""
  return _which_np(x).sqrt(x)


def where(condition: ndarray, x: ndarray, y: ndarray) -> ndarray:
  """Return elements chosen from `x` or `y` depending on `condition`."""
  assert (isinstance(x, itvl.Interval) and isinstance(y, itvl.Interval)) or (not isinstance(y, itvl.Interval) and not isinstance(x, itvl.Interval)) 
  return _which_np(x, y).where(condition, x, y)


def zeros(shape, dtype=float) -> ndarray:
  """Return a new array of given shape and type, filled with zeros."""
  if isinstance(dtype, itvl.Interval):
    return itvl.zeros(shape)
  return jnp.zeros(shape, dtype=dtype)


def zeros_like(a: ndarray) -> ndarray:
  """Return an array of zeros with the same shape and type as a given array."""
  return _which_np(a).zeros_like(a)


def ones(shape, dtype=float) -> ndarray:
  """Return a new array of given shape and type, filled with ones."""
  if isinstance(dtype, itvl.Interval):
    return itvl.ones(shape)
  return jnp.ones(shape, dtype=dtype)


def ones_like(a: ndarray) -> ndarray:
  """Return an array of ones with the same shape and type as a given array."""
  return _which_np(a).ones_like(a)

def full(shape, fill_value : ndarray, dtype=None):
  return _which_np(fill_value).full(shape, fill_value=fill_value, dtype=dtype)


def tranpose(x1: ndarray, axes = None) -> ndarray:
  """Multiply arguments element-wise."""
  return _which_np(x1).transpose(x1, axes)


def reshape(a: ndarray, newshape: Union[Tuple[int, ...], int]) -> ndarray:
  """Gives a new shape to an array without changing its data."""
  return _which_np(a).reshape(a, newshape)


def flatten_list(x):
  return [a for i in x for a in flatten_list(i)] if isinstance(x, list) else [x]

def list_iter(x):
  return _which_np(*flatten_list(x))


def block_array(_object: Any, dtype=None) -> ndarray:
  """Creates an array given a list."""
  assert isinstance(_object, list), "The object {} should be a list!".format(_object)
  if list_iter(_object) is jnp:
    return jnp.array(_object) # Will be improved
  else:
    return itvl.block_array(_object, dtype)


def array(_object: Any, dtype=None) -> ndarray:
  """Creates an array given a list."""
  return jnp.array(_object) # Will be improved


def abs(a: ndarray) -> ndarray:
  """Calculate the absolute value element-wise."""
  return _which_np(a).abs(a)

def minimum(x1: ndarray, x2: ndarray) -> ndarray:
  """Element-wise minimum of array elements."""
  return _which_np(x1, x2).minimum(x1, x2)


def maximum(x1: ndarray, x2: ndarray) -> ndarray:
  """Element-wise minimum of array elements."""
  return _which_np(x1, x2).maximum(x1, x2)


def amin(x: ndarray) -> ndarray:
  """Returns the minimum along a given axis."""
  return _which_np(x).amin(x)

def amax(x: ndarray) -> ndarray:
  """Returns the maximum along a given axis."""
  return _which_np(x).amax(x)

def argmin(x: ndarray) -> ndarray:
  """Returns the indexes of maximum along a given axis."""
  return _which_np(x).argmin(x)