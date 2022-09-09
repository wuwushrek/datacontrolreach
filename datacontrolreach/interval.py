import jax
import jax.numpy as jnp

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
    TOL_ITVL = 1e-4

    def __init__(self, lb, ub=None):
        """ Initialize an interval. The lower bound must always be greater that the upper bound
            :param lb : The lower bound of the interval (vector or matrix)
            :param ub : The upper bound of the interval (vector or matrix)
        """
        # If lb is already an interval
        if isinstance(lb, Interval):
            self._lb = jnp.array(lb.lb)
            self._ub = jnp.array(lb.ub)
        else:
            # TODO : Do not copy ndarray here, use the correct value
            # lb = lb if hasattr(lb, 'shape') else jnp.array(lb)
            # ub = ub if (ub is None or hasattr(ub, 'shape')) else jnp.array(ub)
            # self._lb = lb
            # self._ub = ub if ub is not None else lb
            self._lb = jnp.array(lb, dtype=float)
            self._ub = jnp.array(ub, dtype=float) if ub is not None else jnp.array(lb, dtype=float) # No need for deep copy as it is done by jnp and the constructor is pure

        assert jnp.shape(self._ub) == jnp.shape(self._lb) # make sure dims are the same!

        # Check if the interval makes sense --> Lower bound less than upper bound
        if CHECK_VALID_INTERVAL:
            no_lb_less_ub = self._lb >= self._ub
            self._lb = jnp.where(no_lb_less_ub, -jnp.inf, self._lb)
            self._ub = jnp.where(no_lb_less_ub, jnp.inf, self._ub)

    __numpy_ufunc__ = None # these are needed to make numpy behave when we multiply numpy array element wise * interval array
    __array_ufunc__ = None # if you remove these, numpy handles the elementwise multiplication and returns an array of intervals, instead of an interval

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
        sum_vec = lambda x : Interval(lb=jnp.sum(x.lb), ub=jnp.sum(x.ub))

        # Temporary function to do prod sum of two interval vector
        dot_interval = lambda v1, v2 : sum_vec(v1 * v2)
        
        m_shape = len(self.shape)
        o_shape = len(other.shape)

        if  m_shape == 1 and o_shape == 1:
            return dot_interval(self, other)
        
        if m_shape == 1 and o_shape == 2:
            assert self.shape[0] == other.shape[0]
            return jax.vmap(dot_interval, in_axes=(None, 1))(self, other)

        if m_shape == 2 and o_shape == 1:
            assert self.shape[1] == other.shape[0]
            return jax.vmap(dot_interval, in_axes=(0, None))(self, other)

        if m_shape == 2 and o_shape == 2:
            assert self.shape[1] == other.shape[0]
            row_comp = lambda v1, v2: jax.vmap(dot_interval, in_axes=(None, 1))(v1, v2)
            return jax.vmap(row_comp, in_axes=(0, None))(self, other)

        if m_shape == 3 and o_shape == 1:
            assert self.shape[1] == other.shape[0]
            mid_comp = jax.vmap(dot_interval, in_axes=(1, None))
            return jax.vmap(mid_comp, in_axes=(0,None))(self, other)

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
        return Interval(lb=jnp.amax(self._lb), ub = jnp.amin(self._ub))

    def contains(self, other, tol=1e-6):
        """ Check if the current interval contains the given interval
        """
        lb_val = other.lb if isinstance(other, Interval) else other
        ub_val = other.ub if isinstance(other, Interval) else other
        return jnp.logical_and(self._lb <= lb_val+tol, self._ub >= ub_val-tol)

    # Numpy methods
    def mean(self, axis=None):
        """ Mean value of an interval array """
        return Interval(lb=jnp.mean(self.lb, axis=axis), ub=jnp.mean(self.ub, axis=axis))

    def sum(self, axis=None):
        return Interval(lb=jnp.sum(self.lb, axis=axis), ub=jnp.sum(self.ub, axis=axis))

    def sqrt (self):
        """ COmpute the square root of an interval
        """
        return Interval(lb=jnp.sqrt(self._lb), ub= jnp.sqrt(self.ub))

    def norm(self, axis=0):
        """ Compute the norm of an interval
        """
        if self.ndim == 0:
            return self.__abs__()
        pow2_int = self.__pow__(2)
        return Interval(lb=jnp.sum(pow2_int.lb, axis=axis), ub=jnp.sum(pow2_int.ub, axis=axis)).sqrt()

    def cos(self):
        """ Cosinus of an interval
        """
        return iv_cos(self)

    def sin(self):
        """ Sinus of an interval
        """
        return (self-jnp.pi/2).cos()

    def tan(self):
        """ Tangent of an interval
        """
        return iv_tan(self)

    def log(self):
        """ Log of an interval
        """
        # The function log is monotonic
        return Interval(lb=jnp.log(self._lb), ub=jnp.log(self._ub))


    def exp(self):
        """ Exp of an interval
        """
        # The function exp is monotonic
        return Interval(lb=jnp.exp(self._lb), ub=jnp.exp(self._ub))

    def arctan(self):
        """ Exp of an interval
        """
        # The function exp is monotonic
        return Interval(lb=jnp.arctan(self._lb), ub=jnp.arctan(self._ub))


    # Comparisons
    def __lt__(self, other):
        lb_val = other.lb if isinstance(other, Interval) else other
        return jnp.where(self._ub < lb_val, True, False)

    def __le__(self, other):
        lb_val = other.lb if isinstance(other, Interval) else other
        return jnp.where(self._ub <= lb_val, True, False)

    def __eq__(self, other):
        lb_val = other.lb if isinstance(other, Interval) else other
        ub_val = other.ub if isinstance(other, Interval) else other
        return jnp.logical_and(jnp.abs(self._lb - lb_val) < self.TOL_ITVL, jnp.abs(self._ub - ub_val) < self.TOL_ITVL)

    def __ge__(self, other):
        ub_val = other.ub if isinstance(other, Interval) else other
        return jnp.where(self._lb >= ub_val, True, False)

    def __gt__(self, other):
        ub_val = other.ub if isinstance(other, Interval) else other
        return jnp.where(self._lb > ub_val, True, False)

    def __ne__(self, other):
        return jnp.logical_not( self == other)

    # Bitwise operations -> Intersection and Union of intervals
    def __and__(self, other):
        if isinstance(other, Interval):
            return Interval(lb=jnp.maximum(self._lb, other.lb), ub=jnp.minimum(self._ub, other.ub))
        else:
            return Interval(lb=jnp.maximum(self._lb, other), ub=jnp.minimum(self._ub, other))

    def __or__(self, other):
        if isinstance(other, Interval):
            return Interval(lb=jnp.minimum(self._lb, other.lb), ub=jnp.maximum(self._ub, other.ub))
        else:
            return Interval(lb=jnp.minimum(self._lb, other), ub=jnp.maximum(self._ub, other))

    def __getitem__(self, indx):
        return Interval(lb=self._lb[indx], ub=self._ub[indx])

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    def __repr__(self):
        base_str = 'iv[{},{}]' if isinstance(self._lb, jax.core.Tracer) else 'iv[{:.5f},{:.5f}]'
        if len(self.shape) == 0:
            return base_str.format(self._lb, self._ub)
        if len(self.shape) == 1:
            return '[{}]'.format(','.join('iv[{},{}]'.format(lb, ub) for lb, ub in zip(self._lb, self._ub) ))
        if len(self.shape) == 2:
            s = [[base_str.format(__lb,__ub) for __lb,__ub in zip(_lb,_ub)] \
                    for (_lb, _ub) in zip(self._lb, self._ub)]
            lens = [max(map(len, col)) for col in zip(*s)]
            fmt = ', '.join('{{:{}}}'.format(x) for x in lens) # \t
            table = ['[{}]'.format(fmt.format(*row)) for row in s]
            return '[{}]'.format('\n'.join(table))
            # return '[{}]'.format('\n'.join(' '.join(['iv[{:.6f},{:.6f}]'.format(__lb,__ub) for __lb,__ub in zip(_lb,_ub)]) \
            #                         for _lb, _ub in zip(self._lb, self._ub)))
        else:
            return 'Interval[{}]'.format(','.join('({},{})'.format(lb, ub) for lb, ub in zip(self._lb.ravel(), self._ub.ravel()) ))

    @classmethod
    def restore(cls, x, y):
        obj = object.__new__(cls)
        obj._lb = x
        obj._ub = y
        return obj

    def tree_flatten(self):
        return ((self._lb, self._ub), None)

    @classmethod
    def tree_unflatten(cls, _, args):
        return Interval.restore(*args)

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

def concatenate(x: Sequence[Interval], axis=0) -> Interval:
    """Join a sequence of arrays along an existing axis."""
    return Interval(lb=jnp.concatenate([(_x.lb if _x.ndim>0 else _x.lb.reshape((1,))) for _x in x], axis), 
                    ub=jnp.concatenate([(_x.ub if _x.ndim>0 else _x.ub.reshape((1,))) for _x in x], axis))

def vstack(x: Sequence[Interval]) -> Interval:
    """Join a sequence of arrays along an existing axis."""

    return Interval(lb=jnp.vstack([(_x.lb if _x.ndim>0 else _x.lb.reshape((1,))) for _x in x]), 
                    ub=jnp.vstack([(_x.ub if _x.ndim>0 else _x.lb.reshape((1,))) for _x in x]))

def where(cond, x, y):
    """ Where-like operations between intervals
    """
    assert isinstance(x, Interval) and isinstance(y, Interval), "x={} and y={} should be interval".format(x, y)
    true_fn = lambda _x : _x[0]
    false_fn = lambda _x : _x[1]
    if isinstance(cond, bool) or cond.ndim == 0:
        return jax.lax.cond(cond, true_fn, false_fn, (x,y))
    return jax.vmap(jax.lax.cond, in_axes=(0,None,None,0))(cond, true_fn, false_fn, (x, y))

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
index_update = lambda x, idx, y : Interval(lb=x.lb.at[idx].set(y.lb if isinstance(y,Interval) else y), \
                                            ub=x.ub.at[idx].set(y.ub if isinstance(y,Interval) else y))
mean = lambda a, axis : a.mean(axis)
dot = lambda x, y : x @ y
matmul = lambda x1, x2 : Interval(x1) @ Interval(x2) # Maybe turn back --> I hate converting all time
square = lambda x : x**2
sin = lambda x : x.sin()
cos = lambda x : x.cos()
tan = lambda x : x.tan()
exp = lambda x : x.exp()
log = lambda x : x.log()
multiply = lambda x1, x2 : x1 * x2
sum = lambda x, axis=None : x.sum(axis)
sqrt = lambda x : x.sqrt()
zeros_like = lambda a : Interval(lb=jnp.zeros_like(a.lb))
ones_like = lambda a : Interval(lb=jnp.ones_like(a.lb))
ones = lambda shape, dtype=None : Interval(jnp.ones(shape, dtype))
zeros = lambda shape, dtype=None : Interval(jnp.zeros(shape, dtype))
transpose = lambda x1, axes=None : x1.transpose(axes)
reshape = lambda a, newshape : a.reshape(newshape)

# array = lambda a, dtype=None : block_array(a, dtype)
full = lambda shape, fill_value, dtype=None : Interval(lb=jnp.full(shape, fill_value.lb), ub=jnp.full(shape, fill_value.ub))
subtract = lambda x,y: iv_sub(x,y)

# Define the JVP rules for the interval arithmetic
@partial(jax.custom_jvp)
def iv_add(x, y):
    """ Addition between two intervals
    """
    if isinstance(x, Interval) and isinstance(y, Interval):
        return Interval(lb=x.lb+y.lb, ub=x.ub+y.ub)
    if isinstance(x, Interval):
        return Interval(lb=x.lb+y, ub=x.ub+y)
    else:
        return Interval(lb=x+y.lb, ub=x+y.ub)

@iv_add.defjvp
def jvp_iv_add(primal, tangents):
    """ Addition between two intervals
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_add(x, y), iv_add(xdot, ydot)


@jax.custom_jvp
def iv_sub(x, y):
    """ Substraction between two intervals
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
        n_lb = jnp.minimum(val_1, jnp.minimum(val_2, jnp.minimum(val_3,val_4)))
        n_ub = jnp.maximum(val_1, jnp.maximum(val_2, jnp.maximum(val_3,val_4)))
        return Interval(lb= n_lb, ub=n_ub)
    elif isinstance(x, Interval):
        temp_1, temp_2 = x.ub*other, x.lb * other
        return Interval(lb=jnp.minimum(temp_1, temp_2), ub=jnp.maximum(temp_1, temp_2))
    else: # other is interval, x is not
        temp_1, temp_2 = x*other.ub, x * other.lb
        return Interval(lb=jnp.minimum(temp_1, temp_2), ub=jnp.maximum(temp_1, temp_2))

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

        # Let's keep -inf and +inf for now --> Not sure if we could jit it with raise
        coeff_val = jnp.where(jnp.logical_or(other.lb > 0,  other.ub < 0), 1., 0.)
        return x * Interval(lb=1.0/(other.ub*coeff_val), ub=1.0/(other.lb*coeff_val))
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
    coeff_val = jnp.where(jnp.logical_or(x.lb > 0, x.ub < 0), 1., 0.)
    return Interval(lb=1.0/(x.ub*coeff_val), ub=1.0/(x.lb*coeff_val)) * other

@iv_rdiv.defjvp
def jvp_iv_rdiv(primal, tangents):
    """ Division between a non interval and an interval quantity
    """
    x, y = primal
    xdot, ydot = tangents
    return iv_rdiv(x,y), iv_div(ydot, x)-iv_div(xdot*y, x**2)


# Let's keep the power nondifferentiable for now wit respect to the second variable
# In particular if the second variable are int variables, YOU CAN'T DIFFERENTIATE THROUGH
# If the user want to do x^y he can use exp(log)
# This function allows only x**other, where other is fixed by default and nonnegative
# It will mess up the condition on other > 0 -> Will fix it later
@partial(jax.custom_jvp, nondiff_argnums=(1,))
def iv_pow(x, other):
    """ Computes the power between an interval and a scalar positive integer 
    """
    # Only allow a power as an integer or float -> Interval power not allowed
    if isinstance(other, Interval):
        raise NotImplementedError
    elif (isinstance(other, int) or isinstance(other, float)) and other >= 0:
        ub_power, lb_power = x.ub**other, x.lb**other
        if other == 0:
            return Interval(lb=jnp.ones_like(x.lb), ub=jnp.ones_like(x.ub))
        elif other == 1:
            return Interval(lb=x.lb, ub=x.ub)
        elif other % 2 == 0:
            res_lb = jnp.where(jnp.logical_and(x.lb < 0, x.ub > 0), 0.0, jnp.minimum(ub_power, lb_power))
            res_ub = jnp.maximum(ub_power, lb_power)
        else:
            res_lb = lb_power
            res_ub = ub_power
        return Interval(lb=res_lb, ub=res_ub)
    else:
        raise NotImplementedError

@iv_pow.defjvp
def jvp_iv_pow(pow_val, primal, tangents):
    """ Division between a non interval and an interval quantity
    """
    x,  = primal
    xdot, = tangents
    return iv_pow(x,pow_val), jnp.zeros_like(x) if pow_val == 0 else (xdot if pow_val == 1 else pow_val*iv_pow(x, pow_val-1)*xdot)

@partial(jax.custom_jvp, nondiff_argnums=(0,))
def iv_rpow(x, other):
    """ Properly defined only for adequate power value. Calculates other ^ x
    """
    # In case the right hand side is a number and the power is an interval (not allowed)
    ub_power, lb_power = other**self._ub, other**self._lb
    return Interval(lb=jnp.minimum(lb_power, ub_power), ub=jnp.maximum(lb_power, ub_power))

@iv_rpow.defjvp
def jvp_iv_rpow(primal, tangents):
    """ Division between a non interval and an interval quantity
    """

    x, pow_val = primal
    xdot, = tangents
    res = iv_rpow(pow_val,x)
    return res, jnp.log(pow_val) * res

@jax.custom_jvp
def iv_cos(x):
    """ Cosinus of an interval
    """
    scaleMin = x.lb % (2*jnp.pi)
    scaleMax = x.ub % (2*jnp.pi)
    cos1 = jnp.cos(scaleMin)
    cos2 = jnp.cos(scaleMax)
    minMaxInv = scaleMin > scaleMax
    min_less_pi = scaleMin < jnp.pi
    max_great_pi = scaleMax > jnp.pi
    pi_contain = jnp.logical_or(jnp.logical_and(minMaxInv, jnp.logical_or(min_less_pi, max_great_pi)), 
                    jnp.logical_and(jnp.logical_not(minMaxInv), jnp.logical_and(min_less_pi, max_great_pi)))
    lbval = jnp.where(pi_contain, -1.0, jnp.minimum(cos1, cos2))
    ubval = jnp.where(minMaxInv, 1.0, jnp.maximum(cos1, cos2))
    return Interval(lb=lbval, ub=ubval)

@iv_cos.defjvp
def jvp_iv_cos(primal, tangents):
    """ JVP of cosinus function
    """
    x,  = primal
    xdot, = tangents
    return iv_cos(x), -iv_cos(x-jnp.pi/2)*xdot

@jax.custom_jvp
def iv_tan(x):
    """ Tan of an interval
    """
    return iv_cos(x-jnp.pi/2) / iv_cos(x)

@iv_tan.defjvp
def jvp_iv_tan(primal, tangents):
    """ JVP of Tan of an interval
    """
    x,  = primal
    xdot, = tangents
    sin_v = iv_cos(x-jnp.pi/2)
    cos_v = iv_cos(x)
    return sin_v/cos_v, xdot/(cos_v**2)

@jax.custom_jvp
def iv_abs(x):
    """ Absolute value of an interval
    """
    ub_abs, lb_abs = jnp.abs(x.ub), jnp.abs(x.lb)
    res_lb = jnp.where(jnp.logical_and(x.lb < 0, x.ub > 0), 0.0, jnp.minimum(ub_abs, lb_abs))
    res_ub = jnp.maximum(lb_abs, ub_abs)
    return Interval(lb=res_lb, ub=res_ub)

@iv_abs.defjvp
def jvp_iv_abs(primal, tangents):
    """ Cosinus of an interval
    """
    x,  = primal
    xdot, = tangents
    return iv_abs(x), where(x > 0, xdot, where(x < 0, -xdot, xdot * Interval(-1.,1.)))
