 # has to go in this order because of circular dependencies....
import copy
import datacontrolreach.jumpy as jp
from datacontrolreach import interval
from datacontrolreach.interval import Interval
from datacontrolreach.LipschitzApproximator import LipschitzApproximator, add_data, approximate
import jax
import jax.numpy as jnp
import random
from functools import partial
from jax import jit
from typing import NamedTuple


class HObject(NamedTuple):
    shape_x:tuple
    shape_u:tuple
    known_functions:tuple
    unknown_approximations:tuple
    H:callable
    contractions:tuple

def init_HObject(shape_x, shape_u, known_functions, unknown_approximations, H, contractions):
    assert len(contractions) == len(unknown_approximations), 'There must be 1 contraction per unknown approximation. Got {} unknown contractions and {} approximations'.format(len(unknown_approximations), len(contractions))
    #self.shape_x = shape_x
    #self.shape_u = shape_u
    #self.known_functions = known_functions # must be a list of functions of X, U
    #self.unknown_approximations = unknown_approximations # must be a list of LipschitzApproximators which take in X
    #self.H = H # the function specifying how to calculate x dot
               # Takes in inputs X,U and uses the known and unknown functions of X
               # Its signature must be H(x,u,known,unknown) -> x_dot
    #self.contractions = contractions # must be a list of functions taking in x,u,xdot,known, unknown, and returning
                                     # a contraction of the output of unknown approximation[i] at value x
    return HObject(shape_x, shape_u, tuple(known_functions), tuple(unknown_approximations), H, tuple(contractions))

# this functions needs to be overwritten by the end user. It needs to return an X dot from X and U by using the known functions
# and unknown approximations
def get_x_dot(hobject:HObject, x, u):
    assert x.shape == hobject.shape_x
    assert u.shape == hobject.shape_u
    return hobject.H(x,u, hobject.known_functions, hobject.unknown_approximations)


def contract(hobject:HObject, x, u, x_dot):
    new_approximations = []
    for index in range(len(hobject.contractions)):
        contracted_value = hobject.contractions[index](x,u,x_dot, hobject.known_functions, hobject.unknown_approximations)
        new_approximations.append(add_data(hobject.unknown_approximations[index], x, contracted_value))
    new_approximations = tuple(new_approximations)
    return HObject(         hobject.shape_x, hobject.shape_u,
                            hobject.known_functions,
                            new_approximations,
                            hobject.H,
                            hobject.contractions
                            )






# # Assume we have A = B * C
# # We are trying to calculate A * C^-1 = B, which is a contraction for B
# # Normal pseudo-inverse operation is not sufficient. The current estimate of B allows us to be more precise
# # than a pseudo-inverse.
# # Given values for A, C and an estimate for B, we can contract the approximation for B
# A clean scan -> No need to count the index or access by index (that's the whole point of scan)
def inverse_contraction_B(A, B_approx:Interval, C):
    assert A.ndim == 1, 'A must be a vector, IE of size (N,). Got {}'.format(A.ndim)
    assert B_approx.ndim == 2, 'B must be a Matrix, IE of size (N,M). Got {}'.format(B_approx.ndim)
    assert C.ndim == 1, 'C must be a vector, IE of size (M,). Got {}'.format(C.ndim)

    def row_wise(x):
        a, bapprx = x
        y = contract_row_wise(a, bapprx, C)
        return y
    ys = jax.vmap(row_wise)((A, B_approx))
    return ys

# Assume we have A = B * C
# We are trying to calculate B^-1 * A = C, which is a contraction for C
# Normal pseudo-inverse operation is not sufficient. The current estimate of C allows us to be more precise
# than a pseudo-inverse.
# Given values for A, B and an estimate for C, we can contract the approximation for C
def inverse_contraction_C(A, B, C_approx:Interval):
    assert A.ndim == 1, 'A must be a vector, IE of size (N,)'
    assert B.ndim == 2, 'B must be a Matrix, IE of size (N,M)'
    assert C_approx.ndim == 1, 'C must be a vector, IE of size (M,)'
    def row_wise(carry, x):
        a, b = x
        y = contract_row_wise(a, C_approx, b)
        new_carry = carry & y
        return new_carry, y
    carry, _ = jax.lax.scan(row_wise, C_approx, (A, B))
    return carry


# this function is designed to find contractions for a single row of either B or C
# A (scalar) = Sum(B(vector) * C(vector))
# Since B and C are both vectors, which one we contract only changes the order. This allows us to solve either B or C
# simply swap the order of the arguments
# Returns a contracted vector of intervals
def contract_row_wise(dot_product, vector1: Interval, vector2):
    assert vector1.ndim == 1, 'vector1 must be a vector, IE of size (M,). Got {}'.format(vector1.ndim)
    assert vector2.ndim == 1, 'vector2 must be a vector, IE of size (M,). Got {}'.format(vector2.ndim)
    return hc4revise_lin_eq(dot_product, vector2, vector1)



def hc4revise_lin_eq(rhs : jp.ndarray, coeffs : jp.ndarray, unk : Interval):
    """ Apply HC4Revise for the constraint coeffs @ vars = rhs and return the contracted vars
        This function assumes that coeffs are not zeros
        :param rhs          : The right hand side of the hc4revise
        :param coeffs       : The coefficient of the constraints
        :param vars         : Initial overapproximation of the range of the variables in the constraints
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


