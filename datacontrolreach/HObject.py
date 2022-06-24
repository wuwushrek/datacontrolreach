 # has to go in this order because of circular dependencies....
import copy
import datacontrolreach.jumpy as jp
from datacontrolreach import interval
from datacontrolreach.interval import Interval
from datacontrolreach.LipschitzApproximator import LipschitzApproximator
import jax
import random
from functools import partial
from jax import jit


class HObject:
    """ A class to approximate a function given Lipschitz constants for each dimension and data in the form x, Interval of F(x)
    """
    def __init__(self, shape_x, shape_u, known_functions, unknown_approximations, H, contractions):
        assert len(contractions) == len(unknown_approximations)

        self.shape_x = shape_x
        self.shape_u = shape_u
        self.known_functions = known_functions # must be a list of functions of X, U
        self.unknown_approximations = unknown_approximations # must be a list of LipschitzApproximators which take in X
        self.H = H # the function specifying how to calculate x dot
                   # Takes in inputs X,U and uses the known and unknown functions of X
                   # Its signature must be H(x,u,known,unknown) -> x_dot
        self.contractions = contractions # must be a list of functions taking in x,u,xdot,known, unknown, and returning
                                         # a contraction of the output of unknown approximation[i] at value x
        self.data_collected = 0

    # this functions needs to be overwritten by the end user. It needs to return an X dot from X and U by using the known functions
    # and unknown approximations
    def get_x_dot(self, x, u):
        assert jp.shape(x) == self.shape_x
        assert jp.shape(u) == self.shape_u
        return self.H(x,u, self.known_functions, self.unknown_approximations)


    def contract(self, x, u, x_dot):
        for index in range(len(self.contractions)):
            contracted_value = self.contractions[index](x,u,x_dot, self.known_functions, self.unknown_approximations)
            self.unknown_approximations[index].add_data(x, contracted_value)

        # keep track of how many data points we have found
        self.data_collected += 1




# Assume we have A = B * C
# We are trying to calculate A * C^-1 = B, which is a contraction for B
# Normal pseudo-inverse operation is not sufficient. The current estimate of B allows us to be more precise
# than a pseudo-inverse.
# Given values for A, C and an estimate for B, we can contract the approximation for B
def inverse_contraction_B(A, B_approx:Interval, C):
    new_b_interval = Interval(jp.zeros((jp.shape(B_approx))))
    carry = (A, C, 0) # third one is index
    xs = new_b_interval
    def row_wise(carry, x):
        y = contract_row_wise(carry[0][carry[2]], x, carry[1])
        return (carry[0], carry[1], carry[2] + 1), y
    carry, ys = jax.lax.scan(row_wise, carry, xs)
    return ys

# Assume we have A = B * C
# We are trying to calculate B^-1 * A = C, which is a contraction for C
# Normal pseudo-inverse operation is not sufficient. The current estimate of C allows us to be more precise
# than a pseudo-inverse.
# Given values for A, B and an estimate for C, we can contract the approximation for C
def inverse_contraction_C(A, B, C_approx:Interval):
    new_c_interval = Interval(jp.zeros((jp.shape(C_approx))))
    carry = (A, B, 0) # third one is index
    xs = new_c_interval
    def row_wise(carry, x):
        y = contract_row_wise(carry[0][carry[2]], x, carry[1])
        return (carry[0], carry[1], carry[2] + 1), y
    carry, ys = jax.lax.scan(row_wise, carry, xs)
    return ys


# this function is designed to find contractions for a single row of either B or C
# A (scalar) = Sum(B(vector) * C(vector))
# Since B and C are both vectors, which one we contract only changes the order. This allows us to solve either B or C
# simply swap the order of the arguments
# Returns a contracted vector of intervals
def contract_row_wise(A, B_approx: Interval, C):
    return hc4revise_lin_eq(A, C, B_approx)


def hc4revise_lin_eq(rhs : jp.ndarray, coeffs : jp.ndarray, unk : Interval):
    """ Apply HC4Revise for the constraint coeffs @ vars = rhs and return the contracted vars
        This function assumes that coeffs are not zeros
        :param rhs          : The right hand side of the hc4revise
        :param coeffs       : The coefficient of the constraints
        :param vars         : Initial overapproximation of the range of the variables in the constraints
    """
    # Save temporary higher order sums
    _S = unk * coeffs

    _S_lb, _S_ub = jp.cumsum(_S.lb[::-1]), jp.cumsum(_S.ub[::-1])
    _S = Interval(lb=_S_lb, ub=_S_ub)
    Csum = _S[-1] & rhs # The sum and the rhs are equals

    # Perform contraction along a given variable and propagate the contracted sum
    def op(carry, extra):
        _unk, _c, _csum = extra
        # Trick to avoid division by 0
        zero_notin_c = jp.logical_or(_c > 0, _c < 0)
        _ct = jp.where(zero_notin_c, _c, 1.)
        Cunk = jp.where(zero_notin_c, ((carry - _csum) & (_c * _unk))/_ct, _unk) ################# Here
        # Use the newly Cunk to provide a contraction of the remaining sum
        carry = (carry - (Cunk * _c)) & _csum
        return carry, Cunk

    unk_last, cunk = interval.scan_interval(op, Csum, (unk[:-1], coeffs[:-1], _S[::-1][1:]))
    # Trick to avoid division by zero
    _coeff_nzeros = jp.logical_or(coeffs[-1] > 0, coeffs[-1] < 0)
    _coeff1 = jp.where(_coeff_nzeros, coeffs[-1], 1.)
    unk_last = jp.where(_coeff_nzeros, unk_last /_coeff1,  unk[-1])
    return interval.concatenate((cunk, unk_last))




