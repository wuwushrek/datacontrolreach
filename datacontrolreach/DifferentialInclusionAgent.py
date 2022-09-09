import jax
import jax.numpy as jnp

import datacontrolreach.jumpy as jp
from datacontrolreach.jumpy import Interval

import numpy as np

from datacontrolreach.HObject import init_HObject
from datacontrolreach.cost_optimizers import initialize_optax_solver

from jax.experimental.host_callback import id_print

import copy

class DifferentialInclusionAgent:
    """ This class provides functions for optimal control and for reachable sets
        computation in a closed loop manner
    """
    def __init__(self, action_space, dt, cost_function, approx_builder, params):
        # Save the set of parameters
        self.params = copy.deepcopy(params)

        # Store the contraction function alongside the side information
        self.approx_builder = approx_builder

        # Store the action space and number of actions
        self.action_space = action_space
        self.number_actions = action_space.shape[0]

        self.data_collected = 0
        self.dt = dt

        # Initially, number of steps for excitation-based control
        exploration_steps = self.params.get('exploration_steps', None)
        self.exploration_steps = exploration_steps if exploration_steps is not None \
                                                   else self.number_actions + 1

        # Store if the intial exploration step is already done
        self.exploration_done = False

        # Store the cost function
        self.cost_function = cost_function

        # Store the future states and actions estimates
        self.states_future = None
        self.actions_future = None
        self.opt_cost = None

        # Now ready! Initialize function to pick actions
        # function to contract, the optimizer parameters,
        # and the approximation parameters
        self.opt_state, self.approx_params, self.pick_actions, self.contract = \
            initialize_solver(self.params, self.approx_builder, self.dt,
                                self.cost_function, self.action_space
                             )

    def update(self, state, action, state_dot):
        """ Update the current approximation of the vector field based on
            LipschitzApproximator

        Args:
            state (TYPE): The current state of the system
            action (TYPE): The action performed
            state_dot (TYPE): And measurement of the state derivative
                              (it could be finite difference)
        """
        # Perform contractions wit respect to the current x, xdot and u
        self.approx_params = self.contract(self.approx_params, state, action, state_dot)
        # keep track of how many data points we have found
        self.data_collected += 1


    def act(self, state):
        """ The main action method. Chooses between exploration
            (which is done via excitation)vor exploitation.
            This decision is made based on the amount of data we have
            already collected or the width of the uncertaincy in
            the over-approximations of the next states

        Args:
            state (TYPE): Current observation/state of the system

        Returns:
            TYPE: Action to perform (numpy array)
        """
        # Exploration if data collected is small
        # Plus, exploration if the width of future states is big beyond a threshold
        if (self.data_collected < self.exploration_steps and not self.exploration_done) \
            or (False) :
            return self.excitation(state)
        else:
            self.exploration_done = True
            return self.control_theory(state)

    def excitation(self, state):
        """ This method returns an action based on excitation.
            The action is 0 in all dimensions except for 1, in which it is random
            and small. This is to explore how the action space affects the environment.
            Note the first action we ever take is all 0s

        Args:
            state (TYPE): The current state of the system

        Returns:
            TYPE: Action to perform (numpy array)
        """
        ret = np.zeros((self.number_actions, ))
        # Generate a random control signal
        rand_u = self.action_space.sample() * self.params['excitation_mag']
        if self.data_collected == 0:
            return ret
        elif 0 < self.data_collected <= self.number_actions:
            index_to_change = self.data_collected-1
            ret[index_to_change] = rand_u[index_to_change]
            return ret
        else:
            index_to_change = np.random.randint(0, self.number_actions)
            ret[index_to_change] = rand_u[index_to_change]
            return ret

    def control_theory(self, state):
        """Synthesize near-optimal control values according to the reachset
            computation and a well-tuned gradient-based mpc descent

        Args:
            state (TYPE): The current state of the system

        Returns:
            TYPE: Action to perform (numpy array)
        """
        # Compute the optimal state
        uopt, self.opt_state, self.opt_cost, self.states_future = \
            self.pick_actions(state, self.opt_state, self.approx_params)

        # Save the set of optimal actions
        self.actions_future = uopt

        return np.array(uopt[0])


    def get_future(self):
        """This function returns future states given future actions.
            This is from our computation assuming we have used control theory.
            During excitation, returns none, none. Does not compute anything new

        Returns:
            TYPE: 2D array of dim state, 2D array of dim action
        """
        return self.states_future, self.actions_future


def initialize_solver(params, approx_builder, dt, cost_function, action_space):
    """Summary

    Args:
        params (TYPE): Description
        approx_builder (TYPE): Description
        dt (TYPE): Description
        cost_function (TYPE): Description
        action_space (TYPE): Description
    """
    # Extract the side information
    H_fun, contractions = approx_builder

    # We start by setting up the approximation functions
    lipparams = [] if params['known_dynamics'] else params['LipApprox']
    approx_p, get_xdot, contract = init_HObject(H_fun, lipparams, contractions)

    # Now we define the horizon and discount factor array
    horizon = params['look_ahead_steps']
    discount = params['discount']
    discount_array = jnp.array([discount**i for i in range(horizon)])

    # Define the initial guess for control and the projection function
    initial_control_guess = jnp.ones((horizon, action_space.shape[0])) * 0.0
    projection_1d = lambda act : jnp.minimum(jnp.maximum(act, action_space.low),
                                                action_space.high)
    projection_fn = jax.vmap(projection_1d)

    # Initialize the optimizer for control actions
    init_opt_state, synth_control =  initialize_optax_solver(initial_control_guess,
                        params['mpc_optimizer'], projection_fn=projection_fn)

    # Define the main function to predict cost and performing the action pick
    def pick_actions(x, opt_state, approx_params):
        # Define the cost function
        cost_fn = lambda act : predict_cost_and_states(act, x, approx_params, get_xdot,
                            cost_function, dt, discount_array, params['enclosure'])
        u_opt, next_opt_state, (opt_cost, states) = synth_control(opt_state, cost_fn)
        return u_opt, next_opt_state, opt_state, states

    # Return also a function to initialize the gradient optimizer
    if params['jit_fn']:
        return init_opt_state, approx_p, jax.jit(pick_actions), jax.jit(contract)
    else:
        return init_opt_state, approx_p, pick_actions, contract


def predict_cost_and_states(actions, initial_state, approx_params, get_xdot,
                            cost_function, dt, discount_array, encl_params):
    """ Given a set of future actions and the current state, predict the cost function
        and the next states of the systems
    """
    # First predict the next states
    states = DaTaReach(approx_params, get_xdot, initial_state, actions, dt, encl_params)
    # Now compute the cost function
    costs = jax.vmap(cost_function)(states[:-1], actions, states[1:])
    # Discount the cost function if required
    costs = costs * discount_array
    opt_cost = jp.sum(costs)
    # This is a trick to obtain the optcost when doing grad wit aux varibales
    return opt_cost, (opt_cost, states)

def DaTaReach(approx_params, get_xdot, x0, actions, dt, encl_params):
    """Compute N-step reachable sets given the current state and a set of actions

    Args:
        approx_params (TYPE): The parameters of the LipschitzApproximator
        get_xdot (TYPE): A function to compute the differential inclusion
        x (TYPE): The current state
        actions (TYPE): The current control input
        dt (TYPE): The time step
        encl_params (TYPE): Dictionary of parameters for the apriori enclosure computation

    Returns:
        TYPE: Description
    """
    # Save the tube of over-approximation of the reachable set
    curr_x = Interval(x0)

    # Constant to not compute every time
    dt_2 = (0.5* dt**2)

    # Define the main body to compute the over-approximation
    def op(xt, act):
        # Compute the remainder term given the dynamics evaluation at t
        St, fxt_ut, fst_utdt = apriori_enclosure(approx_params, get_xdot, xt,
                                                    act, dt, **encl_params)

        # Define the dynamics function
        dyn_fun = lambda x : get_xdot(approx_params, x, act)

        _, rem = jax.jvp(dyn_fun, (St,), (fst_utdt,))
        next_x = xt + dt * fxt_ut + dt_2 * rem
        return next_x, next_x

    # Scan over the operations to obtain the reachable set
    _, res = jax.lax.scan(op, curr_x, xs=actions)
    return jp.vstack((curr_x,res))

def apriori_enclosure(approx_params, get_xdot, x, u, dt, fixpointWidenCoeff=0.2,
                        zeroDiameter=1e-5, containTol=1e-4, maxFixpointIter=10):
    """Compute an apriori rough enclosure St of the next state at time t + dt.
        This term is used to over-approximate the second order term in the Taylor expansion
    Args:
        approx_params (TYPE): The parameters of the LipschitzApproximator
        get_xdot (TYPE): A function to compute the differential inclusion
        x (TYPE): The current state
        u (TYPE): The current control input
        dt (TYPE): The time step
        fixpointWidenCoeff (float, optional): Description
        zeroDiameter (float, optional): Description
        containTol (float, optional): Description
        maxFixpointIter (int, optional): Description

    Returns:
        TYPE: Description
    """
    # First compute the vector field at the current pont
    x_dot = get_xdot(approx_params, x, u)

    # Initial a priori enclosure using the fixpoint formula
    iv_odt = Interval(0., dt)
    S = x + x_dot * iv_odt

    # Prepare the loop
    def cond_fun(carry):
        isIn, count, _ = carry
        return jnp.logical_and(jnp.logical_not(isIn) , count <= maxFixpointIter)

    def body_fun(carry):
        _, count, _pastS = carry
        _newS = x + get_xdot(approx_params, _pastS, u) * iv_odt
        # Width increment step
        width_pasS = _pastS.width
        radIncr = jp.where( width_pasS <= zeroDiameter, jp.abs(_pastS.ub), width_pasS)
        radIncr = radIncr * fixpointWidenCoeff
        # Check if the fixpoint condition is satisfied
        isIn = _pastS.contains(_newS, tol=containTol)
        _pastS = jp.where(isIn, _newS, _newS + Interval(-radIncr,radIncr))
        return jp.all(isIn), (count+1), _pastS

    init_carry = (jnp.array(False), jnp.array(1), S)
    _, nbIter, S = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    # print('Num Iter : ')
    # id_print((nbIter, S, x))
    S_dot = get_xdot(approx_params, S, u)
    return x + S_dot * iv_odt, x_dot, S_dot
