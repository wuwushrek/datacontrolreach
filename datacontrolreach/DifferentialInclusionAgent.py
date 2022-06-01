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
    def __init__(self, state_space, action_space, lipschitz_f, lipschitz_g, known_f=None, known_g=None):
        """ Initialize an interval. The lower bound must always be greater that the upper bound
            :param lb : The lower bound of the interval (vector or matrix)
            :param ub : The upper bound of the interval (vector or matrix)
        """
        self.number_states = len(state_space)
        self.number_actions = len(action_space)
        self.state_space = state_space   # maybe dont need this
        self.action_space = action_space  # only used during excitation
        self.f_approximation = LipschitzApproximator((self.number_states,), (self.number_states,), lipschitz_f)
        self.g_approximation = LipschitzApproximator((self.number_states,), (self.number_states, self.number_actions), lipschitz_g)
        self.known_f = known_f if known_f is not None else lambda: np.zeros((self.number_states,))
        self.known_g = known_g if known_g is not None else lambda: np.zeros((self.number_states, self.number_actions))
        self.data_collected = 0

    def update(self, state, action, state_dot):
        # calculate new bounds for F and G at the current state
        new_f_interval = np.subtract(state_dot, np.add(self.g_approximation, self.known_g) * action)
        new_g_interval = 0  # ??? TODO

        # update the approximators
        self.f_approximation.add_data(state, new_f_interval)
        self.g_approximation.add_data(state, new_g_interval)

        # keep track of how many data points we have found
        self.data_collected += 1

    def act(self, state):
        # exploration
        if self.data_collected <= self.number_actions:
            return self.excitation(state)
        # exploitation
        else:
            return self.control_theory(state)

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
        pass  # todo




