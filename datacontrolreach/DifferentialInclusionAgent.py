 # has to go in this order because of circular dependencies....
import copy
import datacontrolreach.jumpy as jp
from datacontrolreach.interval import Interval
from datacontrolreach.HObject import HObject, contract, init_HObject, get_x_dot
import jax
import random
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
import timeit
from functools import partial
class DifferentialInclusionAgent:
    """ A class to approximate a function given Lipschitz constants for each dimension and data in the form x, Interval of F(x)
    """
    # cost function should take in (x, u, x+1) and return a scalar cost. We will try to minimize this eventually.
    def __init__(self, state_space, action_space, HObject, dt, cost_function, look_ahead_steps, descent_steps, learning_rate, exploration_steps = None, discount_rate = 1.0):
        assert discount_rate <= 1.0
        self.number_states = state_space.shape[0]
        self.number_actions = action_space.shape[0]
        self.state_space = state_space   # maybe dont need this
        self.action_space = action_space  # only used during excitation
        self.hobject = HObject
        self.data_collected = 0
        self.dt = dt
        self.exploration_steps = exploration_steps if exploration_steps is not None else self.number_actions + 1
        self.cost_function = cost_function
        self.look_ahead_steps = look_ahead_steps
        self.descent_steps = descent_steps
        self.learning_rate = learning_rate
        self.states_future = None
        self.actions_future = None
        self.discount_rate = discount_rate

    # recall the equation is X_dot = (F_known(X) + F_unknown(X)) + (G_known(X) + G_unknown(X)) * U
    # therefore, F_unknown(X) = (X_dot - (G_known(X) + G_unknown(X)) * U) - F_known(X)
    # therefore, G_unknown(X) = (   (X_dot - (F_known(X) + F_unknown(X)))    * inverse(u)      ) - G_known(X)
    def update(self, state, action, state_dot):
        # find current approximations
        self.hobject = contract(self.hobject, state, action, state_dot)

        # keep track of how many data points we have found
        self.data_collected += 1

    # the main action method. Chooses between exploration (which is done via excitation)
    # or exploitation. This decision is made based on the amount of data we have already collected.
    def act(self, state):
        # exploration if data collected is small
        if self.data_collected < self.exploration_steps:
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
            ret = ret.at[index_to_change].set(self.action_space.sample()[index_to_change]) # weird format for setting index, required by jax
            return ret
        else:
            index_to_change = random.randint(0, self.number_actions)
            ret = ret.at[index_to_change].set(self.action_space.sample()[index_to_change]) # weird format for setting index, required by jax
            return ret

    def control_theory(self, state):
        actions, states = pick_actions(state, self.look_ahead_steps, self.number_actions, self.hobject.known_functions, self.hobject.unknown_approximations, self.hobject.H, self.dt, self.cost_function, self.discount_rate, self.descent_steps, self.learning_rate, self.action_space.low, self.action_space.high)
        self.states_future = states
        self.actions_future = actions
        return actions[0, :] # returns the first action we have planned

    # returns future states given future actions. This is from our computation assuming we have used control theory.
    # during excitation, returns none, none.
    # Does not compute anything new
    def get_future(self):
        return self.states_future, self.actions_future


def apriori_enclosure(knowns, unknowns, H, x, u_interval, dt, fixpointWidenCoeff=0.2,
                        zeroDiameter=1e-5, containTol=1e-4, maxFixpointIter=10):
    """ Compute the a prioir enclosure via either the Theorem in the paper (Gronwall lemma)
        or using a fixpoint iteration scheme
        :param
        :param
        :param
        :param
    """
    # First compute the vector field at the current pont
    x_dot = H(x, u_interval, knowns, unknowns)
    # print("Xdot = ", type(x_dot), jp.shape(x_dot))

    # Initial a priori enclosure using the fixpoint formula
    iv_odt = Interval(0., dt)
    S = x + x_dot * iv_odt
    # print("S = ", type(S), jp.shape(S))

    # Prepare the loop
    def cond_fun(carry):
        isIn, count, _ = carry
        return jnp.logical_and(jnp.logical_not(isIn) , count <= maxFixpointIter)

    def body_fun(carry):
        _, count, _pastS = carry
        _newS = x + H(_pastS, u_interval, knowns, unknowns) * iv_odt

        # Width increment step
        width_pasS = _pastS.width
        radIncr = jp.where( width_pasS <= zeroDiameter, abs(_pastS).ub, width_pasS) * fixpointWidenCoeff
        # CHeck if the fixpoint condition is satisfied
        isIn = _pastS.contains(_newS, tol=containTol)
        _pastS = jp.where(isIn, _newS, _newS + Interval(-radIncr,radIncr))
        return jp.all(isIn), (count+1), _pastS

    _, nbIter, S = jax.lax.while_loop(cond_fun, body_fun, (False, 1, S))
    S_dot = H( S, u_interval, knowns, unknowns)
    return x + S_dot * iv_odt, x_dot, S_dot


def DaTaReach(knowns, unknowns, H, x0, dt, actions,
                fixpointWidenCoeff=0.2, zeroDiameter=1e-5,
                containTol=1e-4, maxFixpointIter=10):
    """ Compute an over-approximation of the reachable set at time
        t0, t0+dt...,t0 + nPoint*dt.

    Parameters
    ----------
    :param dyn : The object representing the dynamics with side information
    :param x0 : Initial state
    :param dt : Integration time
    :param actions : ND array of actions, 1 action vector for each timestep we seek to predict. If M is dimensions of action, and N is number of steps, should be NxM
    :param fixpointWidenCoeff : A coefficient to enlarge the a priori enclosure
    :param containTol : THe tolerance to determine if a fixpoint has been reached
    :param maxFixpointIter : The maximum number of fixpoint iteration

    Returns
    -------
    list
        a list of different time at which the over-approximation is computed
    list
        a list of the over-approximation of the state at the above time
    """
    # Save the tube of over-approximation of the reachable set
    curr_x = Interval(x0)

    # Constant to not compute every time
    dt_2 = (0.5* dt**2)

    # Define the main body to compute the over-approximation
    def DaTaReachLoop(xt, act):
        # Compute the remainder term given the dynamics evaluation at t
        St, _, fst_utdt = apriori_enclosure(knowns, unknowns, H, xt, act, dt, fixpointWidenCoeff, zeroDiameter, containTol, maxFixpointIter)

        # Compute the known term of the dynamics at t
        fxt_ut = H(  xt, act, knowns, unknowns)

        # Define the dynamics function
        dyn_fun = lambda x : H(x, act, knowns, unknowns)

        #St = xt
        #fst_utdt = fxt_ut
        _, rem = jax.jvp(dyn_fun, (St,), (fst_utdt,))
        # rem = 0
        next_x = xt + dt * fxt_ut + dt_2 * rem
        return next_x, next_x

    # Scan over the operations to obtain the reachable set
    _, res = jax.lax.scan(DaTaReachLoop, curr_x, xs=actions)
    return jp.vstack((curr_x,res))

# Returns a list of states, starting with the initial state, of predicted states given the actions
def predict_n_states(initial_state, actions, knowns, unknowns, H, dt):
        # convert x dot to next state. Uses second order approximation
        states = DaTaReach(knowns, unknowns, H, initial_state, dt, actions)
        return states

def compute_trajectory_cost(actions, states, cost_function, discount_rate):
    # Carry is initially 0. Is incredemented by the cost for each state, action, next_state pair
    # Thus the result is the sum of costs at each timestep
    def compute_cost_of_SAS(carry, x):
        state, action, next_state = x
        total_cost, discount = carry
        total_cost += cost_function(state, action, next_state)
        return (total_cost, discount * discount_rate), 0
    carry, _ = jax.lax.scan(compute_cost_of_SAS, (0.0, 1.0), (states[:-1, :], actions, states[1:, :])) # todo carry interavl ro scalar?
    return carry[0]

def predict_n_cost(actions, initial_state, knowns, unknowns, H, dt, cost_function, discount_rate):
    states = predict_n_states(initial_state, actions,  knowns, unknowns, H, dt)
    costs = compute_trajectory_cost(actions, states, cost_function, discount_rate)
    return costs

# @partial(jax.jit, static_argnums=[1, 2, 3, 5, 7, 9, 10])
def pick_actions(initial_state, look_ahead_steps, action_dims, knowns, unknowns, H, dt, cost_function, discount_rate, descent_steps, learning_rate, action_bounds_low, action_bounds_high):
    # init random? actions
    actions = jp.zeros((look_ahead_steps, action_dims))

    def vanilla_gradient_descent(acts, x):
        # for some reason this returns a tuple???????????
        derivative_cost_wrt_actions = jax.jacfwd(predict_n_cost)(acts, initial_state, knowns, unknowns, H, dt, cost_function, discount_rate)

        # cost = predict_n_cost(acts, initial_state, knowns, unknowns, H, dt, cost_function, discount_rate)
        # id_print(cost)

        # descent
        # print("deriv = ", type(derivative_cost_wrt_actions), derivative_cost_wrt_actions.shape)
         #print("acts = ", type(acts), acts.shape)
        # id_print(derivative_cost_wrt_actions)
        #a = 0.3
        #der = derivative_cost_wrt_actions.lb * a + derivative_cost_wrt_actions.ub * (1.0-a)
        #acts -= der * learning_rate
        acts -= derivative_cost_wrt_actions * learning_rate # TODO mid

        #id_print(acts)
        # id_print(derivative_cost_wrt_actions)
        #id_print(cost)

        # move back into bounds if needed
        acts = jp.minimum(jp.maximum(acts, action_bounds_low), action_bounds_high)
        return acts, 0

    actions, _ = jax.lax.scan(vanilla_gradient_descent, actions, jp.zeros((descent_steps, )))


    # predict state with these actions just so we know what we are trying to do
    states = predict_n_states(initial_state, actions, knowns, unknowns, H, dt)
    return actions, states

