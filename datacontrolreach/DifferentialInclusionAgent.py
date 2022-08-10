 # has to go in this order because of circular dependencies....
import copy
import datacontrolreach.jumpy as jp
from datacontrolreach.interval import Interval
from datacontrolreach.HObject import HObject
import jax
import random
import jax.numpy as jnp


class DifferentialInclusionAgent:
    """ A class to approximate a function given Lipschitz constants for each dimension and data in the form x, Interval of F(x)
    """
    def __init__(self, state_space, action_space, HObject, dt, exploration_steps = None):
        self.number_states = state_space.shape[0]
        self.number_actions = action_space.shape[0]
        self.state_space = state_space   # maybe dont need this
        self.action_space = action_space  # only used during excitation
        self.hobject = HObject
        self.data_collected = 0
        self.dt = dt
        self.exploration_steps = exploration_steps if exploration_steps is not None else self.number_actions + 1

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


    # action is an interval, so it can either find state dot for a given action or for a set of possible actions
    def predict_n_states(self, initial_state, actions_function, actions_function_derivative, current_time):
        # find x dot
        #state_dot = self.hobject.get_x_dot(state, action)
        # find state dot derivative
        #state_dot_derivative = something

        # convert x dot to next state. Uses second order approximation
        number_states = 10
        states, times = DaTaReach(self.hobject, initial_state, current_time, number_states, self.dt, actions_function, actions_function_derivative)
        return states


    def control_theory(self, state):
        action = lambda x : Interval(jp.zeros(self.action_space.shape))
        states = self.predict_n_states(state, action, action, 0.0)
        print("states= ", states)


        # return self.action_space.sample()
        raise NotImplementedError



def apriori_enclosure(hobject, x, u_interval, dt, fixpointWidenCoeff=0.2,
                        zeroDiameter=1e-5, containTol=1e-4, maxFixpointIter=10):
    """ Compute the a prioir enclosure via either the Theorem in the paper (Gronwall lemma)
        or using a fixpoint iteration scheme
        :param
        :param
        :param
        :param
    """
    # First compute the vector field at the current pont
    x_dot = hobject.get_x_dot( x, u_interval)

    # Initial a priori enclosure using the fixpoint formula
    iv_odt = Interval(0., dt)
    S = x + x_dot * iv_odt

    # Prepare the loop
    def cond_fun(carry):
        isIn, count, _ = carry
        return jnp.logical_and(jnp.logical_not(isIn) , count <= maxFixpointIter)

    def body_fun(carry):
        _, count, _pastS = carry
        _newS = x + hobject.get_x_dot(_pastS, u_interval) * iv_odt
        # Width increment step
        width_pasS = _pastS.width
        radIncr = jp.where( width_pasS <= zeroDiameter, jp.abs(_pastS.ub), width_pasS) * fixpointWidenCoeff
        # CHeck if the fixpoint condition is satisfied
        isIn = _pastS.contains(_newS, tol=containTol)
        _pastS = jp.where(isIn, _newS, _newS + Interval(-radIncr,radIncr))
        return jp.all(isIn), (count+1), _pastS

    _, nbIter, S = jax.lax.while_loop(cond_fun, body_fun, (False, 1, S))
    S_dot = hobject.get_x_dot( S, u_interval)
    return x + S_dot * iv_odt, x_dot, S_dot


def DaTaReach(hobject, x0, t0, nPoint, dt, uOver, uDer,
                fixpointWidenCoeff=0.2, zeroDiameter=1e-5,
                containTol=1e-4, maxFixpointIter=10):
    """ Compute an over-approximation of the reachable set at time
        t0, t0+dt...,t0 + nPoint*dt.

    Parameters
    ----------
    :param dyn : The object representing the dynamics with side information
    :param x0 : Intial state
    :param t0 : Initial time
    :param nPoint : Number of point
    :param dt : Integration time
    :param uOver : interval extension of the control signal u
    :param uDer : Interval extension of the derivative of the control signal u
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

    # Save the integration time
    integTime = jp.array([t0 + i*dt for i in range(nPoint+1)])

    # Save the tube of over-approximation of the reachable set
    curr_x = Interval(x0)

    # Constant to not compute every time
    dt_2 = (0.5* dt**2)

    # Define the main body to compute the over-approximation
    def op(xt, tval):
        # Over approximation of the control signal between t and t+dt
        ut = uOver(tval)
        ut_dt = uOver(Interval(tval, tval+dt))
        uder_t_dt = uDer(Interval(tval, tval+dt))


        # Compute the remainder term given the dynamics evaluation at t
        St, _, fst_utdt = apriori_enclosure(hobject, xt, ut_dt, dt,
            fixpointWidenCoeff, zeroDiameter, containTol, maxFixpointIter)

        # Compute the known term of the dynamics at t
        fxt_ut = hobject.get_x_dot( xt, ut)

        # Define the dynamics function
        dyn_fun = lambda x : hobject.get_x_dot(x, ut_dt)
        dyn_fun_u = lambda u : hobject.get_x_dot(St, u)

        _, rem = jax.jvp(dyn_fun, (St,), (fst_utdt,))
        _, remu = jax.jvp(dyn_fun_u, (ut_dt,), (uder_t_dt,))
        next_x = xt + dt * fxt_ut + dt_2 * (rem + remu)
        return next_x, next_x

    # Scan over the operations to obtain the reachable set
    _, res = jax.lax.scan(op, curr_x, integTime[:-1])
    return jp.vstack((curr_x,res)), integTime