 # has to go in this order because of circular dependencies....
import copy
import datacontrolreach.jumpy as jp
from datacontrolreach import interval
from datacontrolreach.HObject import HObject
import jax
import random


class DifferentialInclusionAgent:
    """ A class to approximate a function given Lipschitz constants for each dimension and data in the form x, Interval of F(x)
    """
    def __init__(self, state_space, action_space, HObject):
        self.number_states = state_space.shape[0]
        self.number_actions = action_space.shape[0]
        self.state_space = state_space   # maybe dont need this
        self.action_space = action_space  # only used during excitation
        self.hobject = HObject
        self.data_collected = 0

    # recall the equation is X_dot = (F_known(X) + F_unknown(X)) + (G_known(X) + G_unknown(X)) * U
    # therefore, F_unknown(X) = (X_dot - (G_known(X) + G_unknown(X)) * U) - F_known(X)
    # therefore, G_unknown(X) = (   (X_dot - (F_known(X) + F_unknown(X)))    * inverse(u)      ) - G_known(X)
    def update(self, state, action, state_dot):
        # find current approximations
        self.hobject.contract(state, action, state_dot)

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
        ret = jp.zeros((self.number_actions, ))
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
        # return self.action_space.sample()
        raise NotImplementedError

    # action is an interval, so it can either find state dot for a given action or for a set of possible actions
    def predict_next_state(self, state, action):
        # find x dot
        state_dot_prediction = self.hobject.get_x_dot(state, action)

        # convert x dot to next state
        # TODO
        raise NotImplementedError

