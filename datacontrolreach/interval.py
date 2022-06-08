import jax
import math
import numpy as np
import datacontrolreach.jumpy as jp
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union
from functools import partial
import copy
CHECK_VALID_INTERVAL = False

# jax_jit = jax.jit

@jax.tree_util.register_pytree_node_class
class Interval:
    """ A wrapper for treating interval quantities as pytrees for jax jitable and
        classical numeric function evaluations
    """
    def __init__(self, lb, ub=None):
        """ Initialize an interval. The lower bound must always be greater that the upper bound
            :param lb : The lower bound of the interval (vector or matrix)
            :param ub : The upper bound of the interval (vector or matrix)
        """
        # If lb is already an interval
        if isinstance(lb, Interval):
            self._lb = lb.lb
            self._ub = lb.ub
        else:
            # TODO : Do not copy ndarray here, use the correct value
            self._lb = jp.array(lb)
            self._ub = jp.array(ub) if ub is not None else copy.deepcopy(jp.array(lb)) # this must be a deep copy

        # Check if the interval makes sense --> Lower bound less than upper bound
        if CHECK_VALID_INTERVAL:
            no_lb_less_ub = self._lb >= self._ub
            self._lb = jp.where(no_lb_less_ub, -jp.inf, self._lb)
            self._ub = jp.where(no_lb_less_ub, jp.inf, self._ub)

    #__numpy_ufunc__ = None
    #__array_ufunc__ = None

    # Arithmetic
    def __add__(self, other):
        """ Addition between two intervals
        """
        return iv_add(self, other)

    # Addition between a non interval and an interval
    __radd__ = __add__

    def __sub__(self, other):
        """ Substraction between two intervals
        """
        return iv_sub(self, other)

    def __rsub__(self, other):
        """ Substraction between a noninterval and an interval
        """
        return iv_rsub(self, other)

    def __mul__(self, other):
        """ Multiplication (elementwise) between two intervals
        """
        return iv_mult(self, other)

    # Multiplication between a non interval and an interval
    __rmul__ = __mul__

    def __div__(self, other):
        """ Division (elemtwise) between two intervals. [-inf, Inf] is given for denominator 
            containing the value 0 
        """
        return iv_div(self, other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        """ Division between a non interval and an interval quantity
        """
        return iv_rdiv(self, other)

    __rtruediv__ = __rdiv__

    def __pow__(self, other):
        """ Computes the power between an interval and a scalar positive integer 
        """
        # Only allow a power as an integer or float -> Interval power not allowed

        return iv_pow(self, other)

    def __rpow__(self, other):
        """ Properly defined only for adequate power value
        """
        return iv_rpow(self, other)

    # Matrix vector  or Tensor matrix or Matrix matrix multiplication
    def __matmul__(self, other):
        """ Perform multiplication betweem two interval matrices, an interval matrice and a vector
            an interval tensor and a vector
        """
        # A function to squeeze a vector of interval by summing each component
        sum_vec = lambda x : Interval(lb=jp.sum(x.lb), ub=jp.sum(x.ub))

        # Temporary function to do prod sum of two interval vector
        dot_interval = lambda v1, v2 : sum_vec(v1 * v2)
        
        m_shape = len(self.shape)
        o_shape = len(other.shape)

        if  m_shape == 1 and o_shape == 1:
            return dot_interval(self, other)
        
        if m_shape == 1 and o_shape == 2:
            assert self.shape[0] == other.shape[0]
            return vmap_interval(dot_interval, in_axes=(None, 1))(self, other)

        if m_shape == 2 and o_shape == 1:
            assert self.shape[1] == other.shape[0]
            return vmap_interval(dot_interval, in_axes=(0, None))(self, other)

        if m_shape == 2 and o_shape == 2:
            assert self.shape[1] == other.shape[0]
            row_comp = vmap_interval(dot_interval, in_axes=(None, 1))
            return vmap_interval(row_comp, in_axes=(0, None))(self, other)

        if m_shape == 3 and o_shape == 1:
            assert self.shape[1] == other.shape[0]
            mid_comp = vmap_interval(dot_interval, in_axes=(1, None))
            return vmap_interval(mid_comp, in_axes=(0,None))(self, other)

        raise NotImplementedError

    matmul = __matmul__
    dot = __matmul__

    # Unary operators
    def __neg__(self):
        """ Negation of an interval
        """
        return Interval(-self._ub, -self._lb)

    def __pos__(self):
        """ Identity function
        """
        return Interval(lb=self._lb, ub=self._ub)

    def __abs__(self):
        """ Absolute value of an interval
        """
        return iv_abs(self)

    def squeeze(self):
        return Interval(lb=self._lb.squeeze(), ub=self._ub.squeeze())

    def reshape(self, newshape):
        return Interval(lb=self._lb.reshape(newshape), ub=self._ub.reshape(newshape))

    def transpose(self, axes=None):
        return Interval(lb=self._lb.transpose(axes), ub=self._ub.reshape(axes))

    @property
    def mid(self):
        return (self._lb + self._ub) * 0.5
    
    @property
    def width(self):
        """ Width of an interval
        """
        return self._ub - self._lb

    def intersect_intvect(self):
        """ Squeezing an interval vector/matrix into a single interval
        """
        return Interval(lb=jp.amax(self._lb), ub = jp.amin(self._ub))

    def contains(self, other, tol=1e-8):
        """ Check if the current interval contains the given interval
        """
        lb_val = other.lb if isinstance(other, Interval) else other
        ub_val = other.ub if isinstance(other, Interval) else other
        return jp.logical_and(self._lb <= lb_val+tol, self._ub >= ub_val-tol)

    # Numpy methods
    def mean(self, axis=None):
        """ Mean value of an interval array """
        return Interval(lb=jp.mean(self.lb, axis=axis), ub=jp.mean(self.ub, axis=axis))

    def sum(self, axis=None):
        return Interval(lb=jp.sum(self.lb, axis=axis), ub=jp.sum(self.ub, axis=axis))

    def sqrt (self):
        """ COmpute the square root of an interval
        """
        return Interval(lb=jp.sqrt(self._lb), ub= jp.sqrt(self.ub))

    def norm(self, axis=0):
        """ Compute the norm of an interval
        """
        if self.ndim == 0:
            return self.__abs__()
        pow2_int = self.__pow__(2)
        return Interval(lb=jp.sum(pow2_int.lb, axis=axis), ub=jp.sum(pow2_int.ub, axis=axis)).sqrt()

    def cos(self):
        """ Cosinus of an interval
        """
        return iv_cos(self)

    def sin(self):
        """ Sinus of an interval
        """
        return (self-jp.pi/2).cos()

    def tan(self):
        """ Tangent of an interval
        """
        return iv_tan(self)

    def log(self):
        """ Log of an interval
        """
        # The function log is monotonic
        return Interval(lb=jax.numpy.log(self._lb), ub=jax.numpy.log(self._ub))


    def exp(self):
        """ Exp of an interval
        """
        # The function exp is monotonic
        return Interval(lb=jp.exp(self._lb), ub=jp.exp(self._ub))

    def arctan(self):
        """ Exp of an interval
        """
        # The function exp is monotonic
        return Interval(lb=jp.arctan(self._lb), ub=jp.arctan(self._ub))


    # Comparisons
    def __lt__(self, other):
        lb_val = other.lb if isinstance(other, Interval) else other
        return jp.where(self._ub < lb_val, True, False)

    def __le__(self, other):
        lb_val = other.lb if isinstance(other, Interval) else other
        return jp.where(self._ub <= lb_val, True, False)

    def __eq__(self, other):
        lb_val = other.lb if isinstance(other, Interval) else other
        ub_val = other.ub if isinstance(other, Interval) else other
        return jp.logical_and(self._lb == lb_val, self._ub == ub_val)

    def __ge__(self, other):
        ub_val = other.ub if isinstance(other, Interval) else other
        return jp.where(self._lb >= ub_val, True, False)

    def __gt__(self, other):
        ub_val = other.ub if isinstance(other, Interval) else other
        return jp.where(self._lb > ub_val, True, False)

    def __ne__(self, other):
        return jp.logical_not( self == other)

    # Bitwise operations -> Intersection and Union of intervals
    def __and__(self, other):
        if isinstance(other, Interval):
            return Interval(lb=jp.maximum(self._lb, other.lb), ub=jp.minimum(self._ub, other.ub))
        else:
            return Interval(lb=jp.maximum(self._lb, other), ub=jp.minimum(self._ub, other))

    def __or__(self, other):
        if isinstance(other, Interval):
            return Interval(lb=jp.minimum(self._lb, other.lb), ub=jp.maximum(self._ub, other.ub))
        else:
            return Interval(lb=jp.minimum(self._lb, other), ub=jp.maximum(self._ub, other))

    def __getitem__(self, indx):
        return Interval(lb=self._lb[indx], ub=self._ub[indx])

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    def __repr__(self):
        if len(self.shape) == 0:
            return 'iv[{:.6f},{:.6f}]'.format(self._lb, self._ub)
        if len(self.shape) == 1:
            return '[{}]'.format(','.join('iv[{:.7f},{:.7f}]'.format(lb, ub) for lb, ub in zip(self._lb, self._ub) ))
        if len(self.shape) == 2:
            return '[{}]'.format('\n'.join(' '.join(['iv[{:.6f},{:.6f}]'.format(__lb,__ub) for __lb,__ub in zip(_lb,_ub)]) \
                                    for _lb, _ub in zip(self._lb, self._ub)))
        else:
            return 'Interval[{}]'.format(','.join('({:.6f},{:.6f})'.format(lb, ub) for lb, ub in zip(self._lb.ravel(), self._ub.ravel()) ))

    def tree_flatten(self):
        return ((self._lb, self._ub), None)

    @classmethod
    def tree_unflatten(cls, _, args):
        return cls(*args)

    @property
    def size(self):
        return self._lb.size

    def __len__(self):
        return self._lb.shape[0]

    @property
    def shape(self):
        return self._lb.shape

    @property
    def ndim(self):
        return self._lb.ndim

    @property
    def dtype(self):
        return self._lb.dtype


def vmap_interval(fun, in_axes= None):
    """Creates a function which maps ``fun`` over argument axes."""
    if jp._in_jit():
        return jax.vmap(fun, in_axes=in_axes if in_axes is not None else 0)

    def _batched(*args):
        # Convert the argument using tree unflatten
        args_flat, args_treedef = zip(*[jax.tree_flatten(ag) for ag in args])
        args_flat, args_treedef = list(args_flat), list(args_treedef)
        # List of axes for mapping
        vargs, vargs_idx, ax_v = [], [], []
        rets = []
        if in_axes:
            for i, (inc, arg) in enumerate(zip(in_axes, args_flat)):
                if inc is not None:
                    vargs.append(arg)
                    vargs_idx.append(i)
                    ax_v.append(inc)
        else:
            vargs, vargs_idx, ax_v = list(args_flat), list(range(len(args_flat))), [0] * len(args_flat)
        for ax in range(vargs[0][0].shape[ax_v[0]]):
            for zvargs, idx, ax_arr in zip(vargs, vargs_idx, ax_v):
                args_flat[idx] = [zv.take(ax, ax_arr) for zv in zvargs]
            args_unflat = [jax.tree_unflatten(at, af) for at, af in zip(args_treedef, args_flat)]
            rets.append(fun(*args_unflat))
        return jax.tree_map(lambda *x: jp.stack(x), *rets)
    return _batched


def scan_interval(f, init, xs, length=None, reverse=False, unroll: int = 1):
    """Scan a function over leading array axes while carrying along state."""
    if jp._in_jit():
        return jax.lax.scan(f, init, xs, length, reverse, unroll)
    else:
        xs_flat, xs_tree = jax.tree_flatten(xs)
        carry = init
        ys = []
        maybe_reversed = reversed if reverse else lambda x: x
        for i in maybe_reversed(range(xs_flat[0].shape[0])):
            xs_slice = [x[i] for x in xs_flat]
            carry, y = f(carry, jax.tree_unflatten(xs_tree, xs_slice))
            ys.append(y)
        stacked_y = jax.tree_map(lambda *y: jp.stack(y), *maybe_reversed(ys))
        return carry, stacked_y

def while_interval(cond_fun, body_fun, init_val):
    """Scan a function over leading array axes while carrying along state."""
    if jp._in_jit():
        return jax.lax.while_loop(cond_fun, body_fun, init_val)
    else:
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

def concatenate(x: Sequence[Interval], axis=0) -> Interval:
    """Join a sequence of arrays along an existing axis."""
    return Interval(lb=jp.concatenate([(_x.lb if _x.ndim>0 else _x.lb.reshape((1,))) for _x in x], axis), 
                    ub=jp.concatenate([(_x.ub if _x.ndim>0 else _x.ub.reshape((1,))) for _x in x], axis))

def vstack(x: Sequence[Interval]) -> Interval:
    """Join a sequence of arrays along an existing axis."""
    return Interval(lb=jp.vstack([(_x.lb if _x.ndim>0 else _x.lb.reshape((1,))) for _x in x]), 
                    ub=jp.vstack([(_x.ub if _x.ndim>0 else _x.lb.reshape((1,))) for _x in x]))

def where(cond, x, y):
    """ WHere-like operations between intervals
    """
    true_fn = lambda _x : _x[0]
    false_fn = lambda _x : _x[1]
    if isinstance(cond, bool) or cond.ndim == 0:
        return jax.lax.cond(cond, true_fn, false_fn, (x,y))
    return vmap_interval(jax.lax.cond, in_axes=(0,None,None,0))(cond, true_fn, false_fn, (x, y))
    # return jax.lax.cond(cond, lambda *_x : x, lambda *_x : y, x)
    # res_lb = jp.where(cond, x.lb, y.lb)
    # res_ub = jp.where(cond, x.ub, y.ub)
    # return Interval(lb=res_lb, ub=res_ub)

def block_array(mlist, dtype=None):
    """ Build an interval block matrix from the given set of intervals
    """
    assert isinstance(mlist, list)
    if isinstance(mlist[0], list):
        res_lb = [ [ o_i.lb for o_i in o_l] for o_l in mlist]
        res_ub = [ [ o_i.ub for o_i in o_l] for o_l in mlist]
        return Interval(res_lb, res_ub)
    else:
        res_lb = [ o_l.lb for o_l in mlist]
        res_ub = [ o_l.ub for o_l in mlist]
        return Interval(res_lb, res_ub)

######### Some NUmpy functions #########
norm = lambda x, axis=None : x.norm(axis)
index_update = lambda x, idx, y : Interval(lb=jp.index_update(x.lb, idx, y.lb if isinstance(y,Interval) else y), \
                                            ub=jp.index_update(x.ub, idx, y.ub if isinstance(y,Interval) else y))
mean = lambda a, axis : a.mean(axis)
dot = lambda x, y : x @ y
matmul = lambda x1, x2 : x1 @ x2
square = lambda x : x**2
sin = lambda x : x.sin()
cos = lambda x : x.cos()
tan = lambda x : x.tan()
exp = lambda x : x.exp()
log = lambda x : x.log()
multiply = lambda x1, x2 : x1 * x2
sum = lambda x, axis=None : x.sum(axis)
sqrt = lambda x : x.sqrt()
zeros_like = lambda a : Interval(lb=jp.zeros_like(a.lb))
ones_like = lambda a : Interval(lb=jp.ones_like(a.lb))
ones = lambda shape, dtype=None : Interval(jp.ones(shape, dtype))
zeros = lambda shape, dtype=None : Interval(jp.zeros(shape, dtype))
transpose = lambda x1, axes=None : x1.transpose(axes)
reshape = lambda a, newshape : a.reshape(newshape)
array = lambda a, dtype=None : block_array(a, dtype)
full = lambda shape, fill_value, dtype=None : Interval(lb=jp.full(shape, fill_value.lb), ub=jp.full(shape, fill_value.ub))

# Define the JVP rules for the interval arithmetic
@partial(jax.custom_jvp)
def iv_add(x, y):
    """ Addition between two intervals
    """
    if isinstance(x, Interval) and isinstance(y, Interval):
        return Interval(lb=x.lb+y.lb, ub=x.ub+y.ub)
    elif isinstance(x, Interval):
        return Interval(lb=x.lb+y, ub=x.ub+y)
    else:  # y is instance of interval
        return Interval(lb=y.lb+x, ub=y.ub+x)

@iv_add.defjvp
def jvp_iv_add(primal, tangents):
    """ Addition between two intervals
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_add(x, y), iv_add(xdot, ydot)


@jax.custom_jvp
def iv_sub(x, y):
    """ Addition between two intervals
    """
    if isinstance(x, Interval) and isinstance(y, Interval):
        return Interval(lb=x.lb-y.ub, ub=x.ub-y.lb)
    elif isinstance(x, Interval):
        return Interval(lb=x.lb - y, ub=x.ub - y)
    else:  # y is instance of interval
        return Interval(lb=x-y.ub, ub=x-y.lb)

@iv_sub.defjvp
def jvp_iv_sub(primal, tangents):
    """ Addition between two intervals
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_sub(x, y), iv_sub(xdot, ydot)

@jax.custom_jvp
def iv_rsub(x, y):
    """ Addition between two intervals
    """
    return Interval(lb=y-x.ub, ub=y-x.lb)

@iv_rsub.defjvp
def jvp_iv_rsub(primal, tangents):
    """ Addition between two intervals
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_rsub(x, y), iv_rsub(xdot, ydot)

@jax.custom_jvp
def iv_mult(x, other):
    """ Multiplication (elementwise) between two intervals
    """
    if isinstance(x, Interval) and isinstance(other, Interval):
        val_1, val_2, val_3, val_4 = x.lb * other.lb, x.lb * other.ub, x.ub * other.lb, x.ub * other.ub
        n_lb = jp.minimum(val_1, jp.minimum(val_2, jp.minimum(val_3,val_4)))
        n_ub = jp.maximum(val_1, jp.maximum(val_2, jp.maximum(val_3,val_4)))
        return Interval(lb= n_lb, ub=n_ub)
    elif isinstance(x, Interval):
        temp_1, temp_2 = x.ub * other, x.lb * other
        return Interval(lb=jp.minimum(temp_1, temp_2), ub=jp.maximum(temp_1, temp_2))
    else:  # other is interval, x is not
        temp_1, temp_2 = other.ub*x, other.lb * x
        return Interval(lb=jp.minimum(temp_1, temp_2), ub=jp.maximum(temp_1, temp_2))

@iv_mult.defjvp
def jvp_iv_mult(primal, tangents):
    """ Addition between two intervals
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_mult(x,y), y*xdot+x*ydot

@jax.custom_jvp
def iv_div(x, other):
    """ Division (elemtwise) between two intervals. [-inf, Inf] is given for denominator 
        containing the value 0 
    """
    if isinstance(other, Interval):
        if jp.logical_and(other.lb <= 0,  other.ub >= 0).any():
            raise ZeroDivisionError("Possible divide by 0 error. Range {} contains 0.".format(other))
        # no longer need the coeff because -inf, inf is undesirable anyway
        #coeff_val = jp.where(jp.logical_or(other.lb > 0,  other.ub < 0), 1., 0.)
        #ret = x * Interval(lb=1.0/(other.ub*coeff_val), ub=1.0/(other.lb*coeff_val))

        return x * Interval(lb=1.0/other.ub, ub=1.0/other.lb)
    else:
        #if other == 0.0:
        #    raise ZeroDivisionError()
        return x*(1.0 / other)

@iv_div.defjvp
def jvp_iv_div(primal, tangents):
    """ Addition between two intervals
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_div(x,y), iv_div(xdot, y)-iv_div(x*ydot, y**2)

@jax.custom_jvp
def iv_rdiv(x, other):
    """ Division between a non interval and an interval quantity
    """
    coeff_val = jp.where(jp.logical_or(x.lb > 0, x.ub < 0), 1., 0.)
    return Interval(lb=1.0/(x.ub*coeff_val), ub=1.0/(x.lb*coeff_val)) * other

@iv_rdiv.defjvp
def jvp_iv_rdiv(primal, tangents):
    """ Division between a non interval and an interval quantity
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_rdiv(x,y), iv_div(ydot, x)-iv_div(xdot*y, x**2)


@partial(jax.custom_jvp)#, nondiff_argnums=(1, ))
def iv_pow(x, other):
    """ Computes the power between an interval and a scalar positive integer 
    """
    # Only allow a power as an integer or float -> Interval power not allowed
    if isinstance(other, Interval):
        raise NotImplementedError
    elif (isinstance(other, int) or isinstance(other, float)) and other >= 0:
        ub_power, lb_power = x.ub**other, x.lb**other
        if other == 0:
            return Interval(lb=jp.ones_like(x.lb), ub=jp.ones_like(x.ub))
        elif other == 1:
            return Interval(lb=x.lb, ub=x.ub)
        elif other % 2 == 0:
            res_lb = jp.where(jp.logical_and(x.lb < 0, x.ub > 0), 0.0, jp.minimum(ub_power, lb_power))
            res_ub = jp.maximum(ub_power, lb_power)
        else:
            res_lb = lb_power
            res_ub = ub_power
        return Interval(lb=res_lb, ub=res_ub)
    else:
        raise NotImplementedError

@iv_pow.defjvp
def jvp_iv_pow(primal, tangents):
    """ Division between a non interval and an interval quantity
    """
    x, pow = primal
    xdot, powdot = tangents
    dx = jp.zeros_like(x) if pow == 0 else (xdot if pow == 1 else pow*iv_pow(x, pow-1)*xdot)
    dy = powdot * x**pow * x.log()
    return iv_pow(x, pow), dx + dy

@partial(jax.custom_jvp, nondiff_argnums=(0,))
def iv_rpow(x, other):
    """ Properly defined only for adequate power value. Calculates other ^ x
    """
    if isinstance(x, Interval):
        raise NotImplementedError    # In case the right hand side is a number and the power is an interval (not allowed)
    return iv_pow(other, x)
    #ub_power, lb_power = other**x, other**x
    #return Interval(lb=jp.minimum(lb_power, ub_power), ub=jp.maximum(lb_power, ub_power))

@iv_rpow.defjvp
def jvp_iv_rpow(primal, tangents):
    """ Division between a non interval and an interval quantity
    """
    x, pow = primal
    xdot, powdot = tangents
    res = iv_rpow(pow, x)
    return res, jp.log(pow) * res

@jax.custom_jvp
def iv_cos(x):
    """ Cosinus of an interval
    """
    scaleMin = x.lb % (2*jp.pi)
    scaleMax = x.ub % (2*jp.pi)
    cos1 = jp.cos(scaleMin)
    cos2 = jp.cos(scaleMax)
    minMaxInv = scaleMin > scaleMax
    min_less_pi = scaleMin < jp.pi
    max_great_pi = scaleMax > jp.pi
    pi_contain = jp.logical_or(jp.logical_and(minMaxInv, jp.logical_or(min_less_pi, max_great_pi)), 
                    jp.logical_and(jp.logical_not(minMaxInv), jp.logical_and(min_less_pi, max_great_pi)))
    lbval = jp.where(pi_contain, -1.0, jp.minimum(cos1, cos2))
    ubval = jp.where(minMaxInv, 1.0, jp.maximum(cos1, cos2))
    return Interval(lb=lbval, ub=ubval)

@iv_cos.defjvp
def jvp_iv_cos(primal, tangents):
    """ Division between a non interval and an interval quantity
    """
    x,  = primal
    xdot, = tangents
    return iv_cos(x), -iv_cos(x-jp.pi/2)*xdot

@jax.custom_jvp
def iv_tan(x):
    """ Cosinus of an interval
    """
    return iv_cos(x-jp.pi/2) / iv_cos(x)

@iv_tan.defjvp
def jvp_iv_tan(primal, tangents):
    """ Cosinus of an interval
    """
    x,  = primal
    xdot, = tangents
    sin_v = iv_cos(x-jp.pi/2)
    cos_v = iv_cos(x)
    return sin_v/cos_v, xdot/(cos_v**2)

@jax.custom_jvp
def iv_abs(x):
    """ Cosinus of an interval
    """
    ub_abs, lb_abs = jp.abs(x.ub), jp.abs(x.lb)
    res_lb = jp.where(jp.logical_and(x.lb < 0, x.ub > 0), 0.0, jp.minimum(ub_abs, lb_abs))
    res_ub = jp.maximum(lb_abs, ub_abs)
    return Interval(lb=res_lb, ub=res_ub)

@iv_abs.defjvp
def jvp_iv_abs(primal, tangents):
    """ Cosinus of an interval
    """
    x,  = primal
    xdot, = tangents
    return iv_abs(x), jp.where(x > 0, xdot, jp.where(x < 0, -xdot, xdot * Interval(-1., 1.)))
