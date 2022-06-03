 # has to go in this order because of circular dependencies....
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
            bound_f = Interval(np.full((self.number_states, 1), -1000000.0), np.full((self.number_states, 1), 1000000.0))
        if bound_g is None:
            bound_g = Interval(np.full((self.number_states, self.number_actions), -1000000.0), np.full((self.number_states, 1), 1000000.0))
        self.f_approximation = LipschitzApproximator((self.number_states, 1), (self.number_states, 1),                   lipschitz_f, bound_f)
        self.g_approximation = LipschitzApproximator((self.number_states, 1), (self.number_states, self.number_actions), lipschitz_g, bound_g)
        self.known_f = known_f if known_f is not None else lambda x: np.zeros((self.number_states, 1))
        self.known_g = known_g if known_g is not None else lambda x: np.zeros((self.number_states, self.number_actions))
        self.data_collected = 0

    # recall the equation is X_dot = (F_known(X) + F_unknown(X)) + (G_known(X) + G_unknown(X)) * U
    # therefore, F_unknown(X) = (X_dot - (G_known(X) + G_unknown(X)) * U) - F_known(X)
    # therefore, G_unknown(X) = (   (X_dot - (F_known(X) + F_unknown(X)))    * inverse(u)      ) - G_known(X)
    def update(self, state, action, state_dot):
        # reshape to be arrays of Nx1 instead of vectors. Just to make numpy happy
        state = np.reshape(state, (len(state), 1))
        action = np.reshape(action, (len(action), 1))
        state_dot = np.reshape(state_dot, (len(state_dot), 1))

        # find current approximations
        f_approx = self.f_approximation.approximate(state)
        g_approx = self.g_approximation.approximate(state)

        # calculate new bounds for F at the current state
        g_total = np.add(g_approx, self.known_g(state))
        g_times_u = np.matmul(g_total, action)
        new_f_interval = np.subtract(np.subtract(state_dot, g_times_u), self.known_f(state))

        # intersect with current approximation
        new_f_interval = new_f_interval & f_approx

        # update the F approximator
        self.f_approximation.add_data(state, new_f_interval)

        # verify action matrix is not all 0. If it is all 0, we cannot update G
        # if even a single element is non-zero, we can update G
        is_non_zero = np.any(action)
        if is_non_zero:
            # calculate new bounds for G at the current state
            # TODO this is broken
            new_g_interval = np.subtract(np.matmul(np.subtract(state_dot, np.add(f_approx, self.known_f(state))), np.linalg.pinv(action)), self.known_g(state))

            # intersect with current approximation
            new_g_interval = new_g_interval & g_approx

            # update the G approximator
            self.g_approximation.add_data(state, new_g_interval)

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


