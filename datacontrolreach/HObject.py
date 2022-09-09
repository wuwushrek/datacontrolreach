import jax
import jax.numpy as jnp

import datacontrolreach.jumpy as jp
from datacontrolreach.jumpy import Interval

from datacontrolreach.LipschitzApproximator import LipschitzApproximator
from datacontrolreach.LipschitzApproximator import add_data
from datacontrolreach.LipschitzApproximator import init_LipschitzApproximator

from jax.experimental.host_callback import id_print


def init_HObject(H_fun=lambda x,u,unk:x , params_unknown=[], contractions=[]):
    """ Create a set of LipschitzApproximator corresponding to the unknown terms
        in the dynamics as well as a set of utilities function to append data to
        obtain the differential inclusion and to improve the approximation quality

    Args:
        H_fun (TYPE, optional): Side information-based estimator of the vetcor field
        params_unknown (list, optional): Parameters to construct the LipschitzApproximator
        contractions (list, optional): A set of contractor functions to update the approximation

    Returns:
        TYPE: approx_params : List[LipschitzApproximator], get_x_dot, contract
    """
    # Some check
    assert len(params_unknown) == len(contractions),\
        'Params and contractions lists should match. Got {} and {}'.format(len(params_unknown),len(contractions))

    # Construct the LipschitzApproximator
    approx_params = [init_LipschitzApproximator(**_params) for _params in params_unknown]

    # Define the function to obtain xdot
    get_xdot = lambda params, x, u : H_fun(x, u, params)

    # Define a function to update the Lipschitz approximators
    def contract(params, x, u, xdot):
        return [ add_data(_param, x, _contract(x, u, xdot, params)) \
                    for (_param, _contract) in zip(params, contractions)]

    return approx_params, get_xdot, contract


def inverse_contraction_B(A: jp.ndarray, B_approx: jp.Interval, C: jp.ndarray):
    """Assume we have A = B @ C where we want to estimate a tighter value on B given
        some initial overapproximation B_approx, and the values of A and C.

    Args:
        A (jp.ndarray): A vector or an Interval vector
        B_approx (jp.Interval): An interval Matrix, value to contract
        C (jp.ndarray): A vector or an Interval vector

    Returns:
        TYPE: Description
    """
    assert A.ndim == 1, 'A must be a vector, IE of size (N,). Got {}'.format(A.ndim)
    assert B_approx.ndim == 2, 'B must be a Matrix, IE of size (N,M). Got {}'.format(B_approx.ndim)
    assert C.ndim == 1, 'C must be a vector, IE of size (M,). Got {}'.format(C.ndim)
    assert B_approx.shape[1] == C.shape[0], \
        "B_approx @ C must be doable, got B={} and C={}".format(B_approx.shape, C.shape)

    def row_wise(x):
        a, bapprx = x
        y = contract_row_wise(a, bapprx, C)
        return y
    ys = jax.vmap(row_wise)((A, B_approx))
    return ys


def inverse_contraction_C(A: jp.ndarray, B: jp.ndarray, C_approx: jp.Interval):
    """Assume we have A = B @ C where we want to estimate a tighter value on C given
        some initial overapproximation C_approx, and the values of A and B.

    Args:
        A (jp.ndarray): A vector or an Interval vector
        B (jp.ndarray): A matrix or an interval Matrix
        C_approx (jp.Interval): A vector of Interval values

    Returns:
        TYPE: Contracted values of C
    """
    assert A.ndim == 1, 'A must be a vector, IE of size (N,). Got {}'.format(A.ndim)
    assert B.ndim == 2, 'B must be a Matrix, IE of size (N,M). Got {}'.format(B.ndim)
    assert C_approx.ndim == 1, 'C must be a vector, IE of size (M,). Got {}'.format(C_approx.ndim)
    assert C_approx.shape[0] == B.shape[1], \
        "B @ C_approx must be doable, got B={} and C={}".format(C_approx.shape, B.shape)

    def row_wise(carry, x):
        a, b = x
        y = contract_row_wise(a, C_approx, b)
        new_carry = carry & y
        return new_carry, None
    carry, _ = jax.lax.scan(row_wise, C_approx, (A, B))
    return carry



def contract_row_wise(a: jp.ndarray, x: jp.Interval, b: jp.ndarray):
    """This function is designed to find contractions for a single row
        of the type a = x @ b, where x is a vector of unknown variables with
        known initial over-approximation. a is a scalar or interval scalar

    Args:
        a (jp.ndarray): a scalar/interval scalar quantity
        x (jp.Interval): An Interval vector quantity
        b (jp.ndarray): a vector/ Interval vector quantity

    Returns:
        TYPE: Contracted value of x
    """
    assert a.ndim == 0, 'a should be a 0D-array. Got {} instead.'.format(a.shape)
    assert x.shape == b.shape, \
        'x and b must have the same size. Got {} and {} respectively'.format(x.size, b.size)
    return hc4revise_lin_eq(a, b, x)



def hc4revise_lin_eq(rhs : jp.ndarray, coeffs : jp.ndarray, unk : jp.Interval):
    """Apply HC4Revise for the constraint coeffs @ unk = rhs and return the contracted vars
    This function assumes that coeffs are not zeros

    Args:
        rhs (jp.ndarray): The right hand side of the hc4revise
        coeffs (jp.ndarray): The coefficient of the constraints
        unk (jp.Interval): Initial overapproximation of the range of the
                        variables in the constraints

    Returns:
        TYPE: Contracted version of unk
    """
    # Save temporary higher order sums
    _S = unk * coeffs

    _S_lb, _S_ub = jnp.cumsum(_S.lb[::-1]), jnp.cumsum(_S.ub[::-1])
    _S = Interval(lb=_S_lb, ub=_S_ub)
    Csum = _S[-1] & rhs # The sum and the rhs are equals

    # Perform contraction along a given variable and propagate the contracted sum
    def op(carry, extra):
        _unk, _c, _csum = extra
        # Trick to avoid division by 0
        zero_notin_c = jp.logical_or(_c > 0, _c < 0)

        _ct = jp.where(zero_notin_c, _c, 1. if not isinstance(coeffs, Interval) else Interval(1.))
        Cunk = jp.where(zero_notin_c, ((carry - _csum) & (_c * _unk))/_ct, _unk)
        # Use the newly Cunk to provide a contraction of the remaining sum
        carry = (carry - (Cunk * _c)) & _csum
        return carry, Cunk

    _, cunk = jax.lax.scan(op, Csum, (unk, coeffs, jp.concatenate((_S[::-1], Interval(0.)))[1:]) )
    return cunk
