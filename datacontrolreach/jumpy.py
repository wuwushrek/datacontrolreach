# Modified from Brax github: https://github.com/google/brax/blob/main/brax/jumpy.py

# pylint:disable=redefined-builtin
"""Numpy backend for JAX that is called for non-jit/non-jax arrays or intervals / abstract sets."""

from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import jax
from jax import core
from jax import numpy as jnp
import numpy as onp

F = TypeVar('F', bound=Callable)

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

import datacontrolreach.interval as itvl

ndarray = Union[onp.ndarray, jnp.ndarray, itvl.Interval]  # pylint:disable=invalid-name
tree_map = jax.tree_map  # works great with jax or numpy as-is
tree_util = jax.tree_util
pi = onp.pi
inf = onp.inf
float32 = onp.float32
int32 = onp.int32


def _in_jit() -> bool:
  """Returns true if currently inside a jax.jit call."""
  return core.cur_sublevel().level > 0


def _which_np(*args):
  """Returns np or jnp depending on args."""
  for a in args:
    if isinstance(a, itvl.Interval):
      return itvl
    if (isinstance(a, jnp.ndarray) and not isinstance(a, onp.ndarray)):
      return jnp
  if _in_jit():
    return jnp
  return onp

def _is_array(args):
  return isinstance(args, jnp.ndarray) or isinstance(args, onp.ndarray)


def vmap(fun: F, include: Optional[Sequence[bool]] = None) -> F:
  """Creates a function which maps ``fun`` over argument axes."""
  if _in_jit():
    in_axes = 0
    if include:
      in_axes = [0 if inc else None for inc in include]
    return jax.vmap(fun, in_axes=in_axes)

  def _batched(*args):
    args_flat, args_treedef = jax.tree_flatten(args)
    vargs, vargs_idx = [], []
    rets = []
    if include:
      for i, (inc, arg) in enumerate(zip(include, args_flat)):
        if inc:
          vargs.append(arg)
          vargs_idx.append(i)
    else:
      vargs, vargs_idx = list(args_flat), list(range(len(args_flat)))
    for zvargs in zip(*vargs):
      for varg, idx in zip(zvargs, vargs_idx):
        args_flat[idx] = varg
      args_unflat = jax.tree_unflatten(args_treedef, args_flat)
      rets.append(fun(*args_unflat))
    return jax.tree_map(lambda *x: onp.stack(x), *rets)
  return _batched


def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1) -> Tuple[Carry, Y]:
  """Scan a function over leading array axes while carrying along state."""
  if _in_jit():
    return jax.lax.scan(f, init, xs, length, reverse, unroll)
  else:
    xs_flat, xs_tree = jax.tree_flatten(xs)
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [x[i] for x in xs_flat]
      carry, y = f(carry, jax.tree_unflatten(xs_tree, xs_slice))
      ys.append(y)
    stacked_y = jax.tree_map(lambda *y: onp.vstack(y), *maybe_reversed(ys))
    return carry, stacked_y


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
  if np is jnp:
    return x.at[idx].set(y)
  if np is itvl:
    return np.index_update(x, idx, y)
  x = onp.copy(x)
  x[idx] = y
  return x


def safe_norm(x: ndarray,
              axis: Optional[Union[Tuple[int, ...], int]] = None) -> ndarray:
  """Calculates a linalg.norm(x) that's safe for gradients at x=0.
  Avoids a poorly defined gradient for jnp.linal.norm(0) see
  https://github.com/google/jax/issues/3058 for details
  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm
  Returns:
    Norm of the array x.
  """
  np = _which_np(x)
  if np is jnp:
    is_zero = jnp.allclose(x, 0.)
     # temporarily swap x with ones if is_zero, then swap back
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis)
    n = jnp.where(is_zero, 0., n)
    return n
  elif np is onp:
    n = onp.linalg.norm(x, axis=axis)
    return n
  raise NotImplementedError


def any(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether any array element along a given axis evaluates to True."""
  return _which_np(a).any(a, axis=axis)


def all(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether all array elements along a given axis evaluate to True."""
  return _which_np(a).all(a, axis=axis)


def mean(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Compute the arithmetic mean along the specified axis."""
  return _which_np(a).mean(a, axis=axis)


def arange(start: int, stop: int) -> ndarray:
  """Return evenly spaced values within a given interval."""
  return _which_np().arange(start, stop)


def dot(x: ndarray, y: ndarray) -> ndarray:
  """Returns dot product of two arrays."""
  return _which_np(x, y).dot(x, y)


def outer(a: ndarray, b: ndarray) -> ndarray:
  """Compute the outer product of two vectors."""
  return _which_np(a, b).outer(a, b)


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


def floor(x: ndarray) -> ndarray:
  """Returns the floor of the input, element-wise.."""
  return _which_np(x).floor(x)


def cross(x: ndarray, y: ndarray) -> ndarray:
  """Returns cross product of two arrays."""
  return _which_np(x, y).cross(x, y)


def sin(angle: ndarray) -> ndarray:
  """Returns trigonometric sine, element-wise."""
  return _which_np(angle).sin(angle)


def cos(angle: ndarray) -> ndarray:
  """Returns trigonometric cosine, element-wise."""
  return _which_np(angle).cos(angle)


def arctan2(x1: ndarray, x2: ndarray) -> ndarray:
  """Returns element-wise arc tangent of x1/x2 choosing the quadrant correctly."""
  return _which_np(x1, x2).arctan2(x1, x2)


def arccos(x: ndarray) -> ndarray:
  """Trigonometric inverse cosine, element-wise."""
  return _which_np(x).arccos(x)


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


def exp(x: ndarray) -> ndarray:
  """Returns the exponential of all elements in the input array."""
  return _which_np(x).exp(x)


def sign(x: ndarray) -> ndarray:
  """Returns an element-wise indication of the sign of a number."""
  return _which_np(x).sign(x)


def sum(a: ndarray, axis: Optional[int] = None):
  """Returns sum of array elements over a given axis."""
  return _which_np(a).sum(a, axis=axis)

def cumsum(a: ndarray, axis: Optional[int] = None):
  """Returns sum of array elements over a given axis."""
  return _which_np(a).cumsum(a, axis=axis)


def random_prngkey(seed: int) -> ndarray:
  """Returns a PRNG key given a seed."""
  if _which_np() is jnp:
    return jax.random.PRNGKey(seed)
  else:
    rng = onp.random.default_rng(seed)
    return rng.integers(low=0, high=2**32, dtype='uint32', size=2)


def random_uniform(rng: ndarray,
                   shape: Tuple[int, ...] = (),
                   low: Optional[float] = 0.0,
                   high: Optional[float] = 1.0) -> ndarray:
  """Sample uniform random values in [low, high) with given shape/dtype."""
  if _which_np(rng) is jnp:
    return jax.random.uniform(rng, shape=shape, minval=low, maxval=high)
  else:
    return onp.random.default_rng(rng).uniform(size=shape, low=low, high=high)


def random_split(rng: ndarray, num: int = 2) -> ndarray:
  """Splits a PRNG key into num new keys by adding a leading axis."""
  if _which_np(rng) is jnp:
    return jax.random.split(rng, num=num)
  else:
    rng = onp.random.default_rng(rng)
    return rng.integers(low=0, high=2**32, dtype='uint32', size=(num, 2))


def segment_sum(data: ndarray,
                segment_ids: ndarray,
                num_segments: Optional[int] = None) -> ndarray:
  """Computes the sum within segments of an array."""
  if _which_np(data, segment_ids) is jnp:
    s = jax.ops.segment_sum(data, segment_ids, num_segments)
  else:
    if num_segments is None:
      num_segments = onp.amax(segment_ids) + 1
    s = onp.zeros((num_segments,) + data.shape[1:])
    onp.add.at(s, segment_ids, data)
  return s


def top_k(operand: ndarray, k: int) -> ndarray:
  """Returns top k values and their indices along the last axis of operand."""
  if _which_np(operand) is jnp:
    return jax.lax.top_k(operand, k)
  else:
    ind = onp.argpartition(operand, -k)[-k:]
    return operand[ind], ind


def stack(x: List[ndarray], axis=0) -> ndarray:
  """Join a sequence of arrays along a new axis."""
  return _which_np(*x).stack(x, axis=axis)


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


def diag(v: ndarray, k: int = 0) -> ndarray:
  """Extract a diagonal or construct a diagonal array."""
  return _which_np(v).diag(v, k)


def clip(a: ndarray, a_min: ndarray, a_max: ndarray) -> ndarray:
  """Clip (limit) the values in an array."""
  return _which_np(a, a_min, a_max).clip(a, a_min, a_max)


def eye(n: int) -> ndarray:
  """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
  return _which_np().eye(n)


def zeros(shape, dtype=float) -> ndarray:
  """Return a new array of given shape and type, filled with zeros."""
  if isinstance(dtype, itvl.Interval):
    return itvl.zeros(shape)
  return _which_np().zeros(shape, dtype=dtype)


def zeros_like(a: ndarray) -> ndarray:
  """Return an array of zeros with the same shape and type as a given array."""
  return type(a)(0.) if isinstance(a, float) or isinstance(a, int) else _which_np(a).zeros_like(a)


def ones(shape, dtype=float) -> ndarray:
  """Return a new array of given shape and type, filled with ones."""
  if isinstance(dtype, itvl.Interval):
    return itvl.ones(shape)
  return _which_np().ones(shape, dtype=dtype)


def ones_like(a: ndarray) -> ndarray:
  """Return an array of ones with the same shape and type as a given array."""
  return type(a)(1.) if isinstance(a, float) or isinstance(a, int) else _which_np(a).ones_like(a)

def full(shape, fill_value : ndarray, dtype=None):
  return _which_np(fill_value).full(shape, fill_value=fill_value, dtype=dtype)

# def full_like(shape, fill_value : ndarray, dtype=None):
#   return _which_np(fill_value).full(shape, fill_value=fill_value, dtype=dtype)

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


def array(_object: Any, dtype=None) -> ndarray:
  """Creates an array given a list."""
  if isinstance(_object, jnp.ndarray) or isinstance(_object, onp.ndarray) or isinstance(_object,itvl.Interval):
    return _object

  if type(_object) is object:
    return _object

  # Otherwise check if the object is iterable
  try:
    np = list_iter(_object)
  except TypeError:
    np = _which_np(_object)  # object is not iterable (e.g. primitive type)
    
  return np.array(_object, dtype)

def asarray(_object : Any, dtype=None):
  assert not isinstance(_object, itvl.Interval)
  np = _which_np(_object)
  if _in_jit() and np == onp:
    np = jnp
  return np.asarray(_object, dtype=dtype)


def abs(a: ndarray) -> ndarray:
  """Calculate the absolute value element-wise."""
  return _which_np(a).abs(a)