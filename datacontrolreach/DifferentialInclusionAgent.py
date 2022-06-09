 # has to go in this order because of circular dependencies....
import copy
import numpy as np
import datacontrolreach.jumpy as jp
from datacontrolreach import interval
from datacontrolreach.interval import Interval
from datacontrolreach.LipschitzApproximator import LipschitzApproximator
import jax
import random


class DifferentialInclusionAgent:
    """ A class to approximate a function given Lipschitz constants for each dimension and data in the form x, Interval of F(x)
    """
    def __init__(self, state_space, action_space, lipschitz_f, lipschitz_g, bound_f=None, bound_g=None, known_f=None, known_g=None):
        self.number_states = state_space.shape[0]
        self.number_actions = action_space.shape[0]
        self.state_space = state_space   # maybe dont need this
        self.action_space = action_space  # only used during excitation
        if bound_f is None:
            bound_f = Interval(np.full((self.number_states,), -1000000.0), np.full((self.number_states, ), 1000000.0))
        if bound_g is None:
            bound_g = Interval(np.full((self.number_states, self.number_actions), -1000000.0), np.full((self.number_states, self.number_actions), 1000000.0))
        self.f_approximation = LipschitzApproximator((self.number_states, ), (self.number_states, ),                   lipschitz_f, bound_f)
        self.g_approximation = LipschitzApproximator((self.number_states, ), (self.number_states, self.number_actions), lipschitz_g, bound_g)
        self.known_f = known_f if known_f is not None else lambda x: jp.zeros((self.number_states,))
        self.known_g = known_g if known_g is not None else lambda x: jp.zeros((self.number_states, self.number_actions))
        self.data_collected = 0

    # recall the equation is X_dot = (F_known(X) + F_unknown(X)) + (G_known(X) + G_unknown(X)) * U
    # therefore, F_unknown(X) = (X_dot - (G_known(X) + G_unknown(X)) * U) - F_known(X)
    # therefore, G_unknown(X) = (   (X_dot - (F_known(X) + F_unknown(X)))    * inverse(u)      ) - G_known(X)
    def update(self, state, action, state_dot):
        # find current approximations
        f_approx = self.f_approximation.approximate(state)
        g_approx = self.g_approximation.approximate(state)

        # Contract F
        self.contract_f( state, state_dot, action, f_approx, g_approx)

        # Contract G
        self.contract_g( state, state_dot, action, f_approx, g_approx)

        # keep track of how many data points we have found
        self.data_collected += 1

    # the main action method. Chooses between exploration (which is done via excitation)
    # or exploitation. This decision is made based on the amount of data we have already collected.
    def act(self, state):
        # exploration if data collected is small
        if self.data_collected <= self.number_actions:
            return self.excitation(state)
        # exploitation otherwise
        else:
            return self.control_theory(state)

    # This method returns an action based on excitation. The action is 0 in all dimensions except for 1, in which it is random
    # this is to explore how the action space affects the environment
    # note the first action we ever take is all 0s
    def excitation(self, state):
        ret = np.zeros((self.number_actions, ))
        if self.data_collected == 0:
            return ret
        elif 0 < self.data_collected <= self.number_actions:
            index_to_change = self.data_collected-1
            ret[index_to_change] = self.action_space.sample()[index_to_change]
            return ret
        else:
            index_to_change = random.randint(0, self.number_actions)
            ret[index_to_change] = self.action_space.sample()[index_to_change]
            return ret

    def control_theory(self, state):
        raise NotImplementedError

    def predict_next_state(self, state, action):
        # find x dot
        f_approx = self.f_approximation(state)
        g_approx = self.g_approximation(state)
        state_dot = f_approx + g_approx * action

        # convert x dot to next state
        # TODO

    def contract_f(self, state, state_dot, action, f_approx, g_approx):
        # calculate new bounds for F at the current state
        g_times_u = jp.matmul(g_approx +self.known_g(state), action)

        # Take x_dot - GU - Known f
        new_f_interval = state_dot - g_times_u - self.known_f(state)

        # intersect with current approximation
        new_f_interval = new_f_interval & f_approx

        # update the F approximator
        self.f_approximation.add_data(state, new_f_interval)


    def contract_g(self, state, state_dot, action, f_approx, g_approx):
        # verify action matrix is not all 0. If it is all 0, we cannot update G
        # if even a single element is non-zero, we can update G
        non_zero = np.any(action)
        if not non_zero:
            return

        # calculate new bounds for G at the current state
        X = state_dot - (f_approx + self.known_f(state))

        new_g_interval = {}
        for row in range(len(state)):
            print("state_row", state[row])
            print("action", action)
            print("g_approx[row]", g_approx[row], type(g_approx[row]), "\n\n")
            g_row = contract_row_wise(state[row], action, g_approx[row])
            new_g_interval.add(g_row)
        #print("new_g_interval = ", new_g_interval)

        # update the G approximator
        self.g_approximation.add_data(state, new_g_interval)



# this function is designed to find contractions for a single row.
# Starting at the function X_dot = F(X) + G(X)*U
# We can find X_DOT - F(X) = G(X)*U
# To simplify notation, assume X_DOT - F(X) = X and G(X)*U = G*U
# thus we have X = G*U
# Given values for X, U and an estimate for G, we can contract the approximation for G
# For jit purposes, we want to be able to do this rowwise. Row wise, we get the following for each row:
# X_i = Sum for m = 1 to the length(U) of (G_approximate_i_m * U_m )
# Solving for each value of G, we get
# new_G_approx_i_a = (X_i - Sum for m = 1 to the length(U) except m=a of (G_approximate_i_m * U_m )) / U_a
# This function expects X to be a scalar, G_approx to be a vector of intervals, and U to be a vector of scalars
# Returns a contract vector of intervals representing G
def contract_row_wise(X, G_approx: Interval, U):
    return hc4revise_lin_eq(X, U, G_approx)


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
        Cunk = jp.where(zero_notin_c, ((carry - _csum) & (_c * _unk))/_ct, _unk)
        # Use the newly Cunk to provide a contraction of the remaining sum
        carry = (carry - (Cunk * _c)) & _csum
        return carry, Cunk

    print("\n\nunk", unk)
    print("coeffs", coeffs)
    print("_S", _S)

    print("\nunk[:-1]", unk[:-1])
    print("coeffs[:-1]", coeffs[:-1])
    print("_S[::-1][1:]", _S[::-1][1:])

    unk_last, cunk = interval.scan_interval(op, Csum, (unk[:-1], coeffs[:-1], _S[::-1][1:]))
    # Trick to avoid division by zero
    _coeff_nzeros = jp.logical_or(coeffs[-1] > 0, coeffs[-1] < 0)
    _coeff1 = jp.where(_coeff_nzeros, coeffs[-1], 1.)
    unk_last = jp.where(_coeff_nzeros, unk_last /_coeff1,  unk[-1])
    return interval.concatenate((cunk, unk_last))




# this function is designed to find contractions for a single row.
# Starting at the function X_dot = F(X) + G(X)*U
# We can find X_DOT - F(X) = G(X)*U
# To simplify notation, assume X_DOT - F(X) = X and G(X)*U = G*U
# thus we have X = G*U
# Given values for X, U and an estimate for G, we can contract the approximation for G
# For jit purposes, we want to be able to do this rowwise. Row wise, we get the following for each row:
# X_i = Sum for m = 1 to the length(U) of (G_approximate_i_m * U_m )
# Solving for each value of G, we get
# new_G_approx_i_a = (X_i - Sum for m = 1 to the length(U) except m=a of (G_approximate_i_m * U_m )) / U_a
# This function expects X to be a scalar, G_approx to be a vector of intervals, and U to be a vector of scalars
# Returns a contract vector of intervals representing G
'''def contract_row_wise(X, G_approx: Interval, U):
    assert len(G_approx) == len(U)

    # create space for result, and find dot product so we only have to compute it once
    new_G_approx = copy.deepcopy(G_approx)

    # for each element in the vector
    for i in range(len(U)):
        # if u is 0, then we get no information on G. Estimate is -inf to inf
        if U[i] == 0.0:
            continue
        else:
            # calculate the sum of G*U of all elements except the g we are trying to calculate
            sum = Interval(0.0,0.0)
            for j in range(len(U)):
                if j != i:
                    sum += new_G_approx[j] * U[j]

            # use the sum (which is the dot product of all elements except the one we are trying to approximate)
            # plus the X result and the U which corresponds to the G we are trying to predict
            approx = (X - sum) / U[i]
            new_G_approx.ub[i] = approx.ub
            new_G_approx.lb[i] = approx.lb
            new_G_approx = G_approx & new_G_approx

    return G_approx & new_G_approx # intersect old estimate and new'''