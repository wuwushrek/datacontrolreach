import yaml
import argparse

import jax
import jax.numpy as jnp

import numpy as np

import datacontrolreach.jumpy as jp
from datacontrolreach.jumpy import Interval

from datacontrolreach.HObject import inverse_contraction_B
from datacontrolreach.HObject import inverse_contraction_C
from datacontrolreach.LipschitzApproximator import approximate

from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent

# import gym
# import time

from UnicycleMDP.UnicycleMDP import UnicycleMDP


def main_fn(config):
    """ Main function in the main. Look up some configuration file
        for user setting and perform control based on that
    """
    # Set up the general seed for this experiments for reproducibility
    np.random.seed(config['seed'])

    # Pull out some setting from the configuration file
    known_dynamics = config['known_dynamics']
    render = config['render']
    graph = config.get('graph', False)

    # Define the time step -> SMALLER ARE BETTER FOR CONTROL
    time_step = 0.01
    Tlimit = 1000 # Time limit of the simulation

    # Load the environment
    # [TODO] Refactor environments to be load/made by name such that
    # it can be added to the configuration file????
    # env = gym.make("Pendulum-v1")
    env = UnicycleMDP(seed=config['seed'], dt=time_step, t_limit=Tlimit)

    # Extract the number of states and the number of actions
    numStates = env.observation_space.shape[0]
    numActions = env.action_space.shape[0]
    print("[Unicycle] Num states = {} | Num actions = {}".format(numStates, numActions))

    # Define the H function and contractions depending on if dynamics known or not
    if known_dynamics:
        # approx_info = H, contractions, approx_params
        approx_builder = H_known, []
    else:
        assert len(config['LipApprox']) == len(m_contractions), \
            'Number of Contractions and LipApprox parameters should match...'
        approx_builder = H_sideinfo, m_contractions

    # Construct the Differential inclusion agent
    # action_space, dt, cost_function, approx_builder, params
    agent = DifferentialInclusionAgent(env.action_space, env.dt, cost_function,
                    approx_builder, config)

    observation, info = env.reset(return_info=True)

    for _ in range(Tlimit):
        # Get action from agent
        action = agent.act(observation)

        # for our sake
        if render:
            future_states, future_actions = agent.get_future()  # fetch op, no computation
            env.render(predictions=future_states, sleep=0.01)
            if graph:
                env.plot(future_states)

        # get result of action
        new_observation, reward, done, observation_dot = env.step(action)
        print(_, observation, action)
        # print(_, {k : v for k, v in agent.opt_state.items() if k != 'uopt'})

        # update agent
        agent.update(observation, action, observation_dot)

        # continue to next state
        observation = new_observation

    env.close()



########## Setting for the unknown case #########
center_term = lambda x, u : jnp.zeros(x.shape)
def affine_term(x, u):
    # Will only work fine if u is not an interval
    # [TODO] work around for u being an interval by overloading jp.array
    H_mat = [[ 1.0, 0.0, 0.0, u[0], u[1], 0.0, 0.0, 0.0, 0.0],
             [ 0.0, 1.0, 0.0, 0.0, 0.0, u[0], u[1], 0.0, 0.0],
             [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, u[0], u[1]]
            ]
    return jnp.array(H_mat)

# H function with side information
def H_sideinfo(x, u, unknown):
    return center_term(x,u) + jp.matmul(affine_term(x, u), approximate(unknown[0], x))

# COntraction function in this control-affine setting
def contract_aff(x, u, xdot, unknown):
    rhs = xdot - center_term(x,u)
    affine_val = affine_term(x,u)
    return inverse_contraction_C(rhs, affine_val, approximate(unknown[0], x))
m_contractions = [contract_aff]
##################################################


########## Setting for the known case ###########
def transition(x,u):
    x_dot = u[0] * jp.cos(x[2])
    y_dot = u[0] * jp.sin(x[2])
    theta_dot = Interval(u[1])
    return Interval(jp.array([x_dot.lb, y_dot.lb, theta_dot.lb]),
                    jp.array([x_dot.ub, y_dot.ub, theta_dot.ub]))
H_known = lambda x, u, unknown: transition(x, u)
#################################################


########## Cost function definition #############
def cost_function (x, u, x_future):
    # x^2 + y^2, minimal = go to 0,0
    cost =  x_future[0] ** 2 + x_future[1] ** 2
    return cost.ub # + 0.1 * u[1]**2  # cost.ub
#################################################


if __name__ == "__main__":
    # Parse the command line argument
    parser = argparse.ArgumentParser('Controlling the Unicyle System')
    parser.add_argument('--config',  type=str, default='unicycle_config.yaml',
                                        help='Model config file path')
    args = parser.parse_args()

    # Open the yaml file containing the configuration to train the model
    yml_file = open(args.config)
    yml_byte = yml_file.read()
    m_config = yaml.load(yml_byte, yaml.SafeLoader)
    yml_file.close()

    # Print the configuration file
    print(m_config)

    # Simulate the Unicycle dynamics
    main_fn(m_config)
