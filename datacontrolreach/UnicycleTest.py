import math

from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent
from datacontrolreach.interval import Interval
import gym
import numpy as np
import jax
import datacontrolreach.jumpy as jp
from UnicycleMDP.UnicycleMDP import UnicycleMDP
from datacontrolreach.HObject import HObject, inverse_contraction_B, inverse_contraction_C, init_HObject
from datacontrolreach.LipschitzApproximator import LipschitzApproximator, init_LipschitzApproximator, approximate
import time

############ User settings ##########
known_dynamics = False
look_ahead_steps = 10
descent_steps = 1000
learning_rate = 0.1
render = False
graph = False
#####################################


# env = gym.make("Pendulum-v1")
env = UnicycleMDP(seed = 2)
print("Num states = ", env.observation_space.shape[0], ", Num actions = ",env.action_space.shape[0])

# Create an HObject which can predict the next_state_dot.s Can take advantage of any known information
if not known_dynamics:
    k = 9 # unknown terms
    known_functions = [lambda x, u: jp.zeros(env.observation_space.shape),
                       lambda x, u: jp.array([  [ 1.0, 0.0, 0.0, u[0], u[1], 0.0, 0.0, 0.0, 0.0],
                                                 [ 0.0, 1.0, 0.0, 0.0, 0.0, u[0], u[1], 0.0, 0.0],
                                                 [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, u[0], u[1]]])
                       ]
    unknown_approximations = [ init_LipschitzApproximator( shapeOfInputs=env.observation_space.shape, shapeOfOutputs=(k,),
                                                          # lipschitzConstants=jp.array([1.01, 1.01, 1.01, 1.1, 1.1, 1.1, 1.1, 1.01, 1.01]),
                                                          lipschitzConstants=jp.array([0.01, 0.01, 0.01, 1.1, 1.1, 1.1, 1.1, 0.01, 0.01]),
                                                          boundsOnFunctionValues=Interval(jp.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,-10.0, ]),
                                                                                          jp.array([10.0,   10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0, 10.0])),
                                                          max_data_size=100
                                                        )
                             ]
    H = lambda x, u, known, unknown: known[0](x, u) + jp.matmul(known[1](x, u), approximate(unknown[0], x))
    contractions = [    lambda x, u, xdot, known, unknown: inverse_contraction_C(xdot - known[0](x, u), known[1](x, u), approximate(unknown[0], x))        ]
    h = HObject( env.observation_space.shape, env.action_space.shape, known_functions, unknown_approximations, H, contractions)

# The case of all known transitions:
if known_dynamics:
    def transition(x,u):
        x_dot = u[0] * jp.cos(x[2])
        y_dot = u[0] * jp.sin(x[2])
        theta_dot = Interval(u[1])
        return Interval(jp.array([x_dot.lb, y_dot.lb, theta_dot.lb]),
                        jp.array([x_dot.ub, y_dot.ub, theta_dot.ub]))
    known_functions = [ transition]
    unknown_approximations = []
    H = lambda x,u,known,unknown: known[0](x,u)
    contractions = []
    h = init_HObject( env.observation_space.shape, env.action_space.shape, known_functions, unknown_approximations, H, contractions)


# define the cost function, which we will try to minimize
def cost_function (x, u, x_future):
    cost =  x_future[0] ** 2 + x_future[1] ** 2 # x^2 + y^2, minimal = go to 0,0
    return cost.ub


# create agent with any side info we may need
agent = DifferentialInclusionAgent(env.observation_space, env.action_space, h, env.dt, cost_function, look_ahead_steps=look_ahead_steps, descent_steps=descent_steps, learning_rate=learning_rate)

# reset env, begin looping
observation, info = env.reset(seed=10, return_info=True)
for _ in range(100):
    # Get action from agent
    action = agent.act(observation)

    # for our sake
    if render:
        future_states, future_actions = agent.get_future()  # fetch op, no computation
        env.render(predictions=future_states, sleep=1.0)
        if graph:
            env.plot(future_states)

    # get result of action
    new_observation, reward, done, observation_dot = env.step(action)
    print(_, observation, action)

    # update agent
    agent.update(observation, action, observation_dot)

    # continue to next state
    observation = new_observation


env.close()

