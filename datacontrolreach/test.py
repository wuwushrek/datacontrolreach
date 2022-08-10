import math

from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent
from datacontrolreach.interval import Interval
import gym
import numpy as np
import jax
import datacontrolreach.jumpy as jp
from UnicycleMDP.UnicycleMDP import UnicycleMDP
from datacontrolreach.HObject import HObject, inverse_contraction_B, inverse_contraction_C
from datacontrolreach.LipschitzApproximator import LipschitzApproximator
import time


# env = gym.make("Pendulum-v1")
env = UnicycleMDP()

# Print info just for our knowledge
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print("Num states = ", num_states, ", Num actions = ",num_actions)
k = 9
print("Unknown terms = ", k, "\n\n")

# Create an HObject which can predict the next_state_dot.s Can take advantage of any known information
known_functions = [lambda x, u: jp.zeros(env.observation_space.shape),
                   lambda x, u: jp.array([  [ 1.0, 0.0, 0.0, u[0], u[1], 0.0, 0.0, 0.0, 0.0],
                                             [ 0.0, 1.0, 0.0, 0.0, 0.0, u[0], u[1], 0.0, 0.0],
                                             [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, u[0], u[1]]])
                   ]
unknown_approximations = [ LipschitzApproximator( shapeOfInputs=env.observation_space.shape, shapeOfOutputs=(k,),
                                                  lipschitzConstants=jp.array([1.01, 1.01, 1.01, 1.1, 1.1, 1.1, 1.1, 1.01, 1.01]),
                                                  # lipschitzConstants=jp.array([0.01, 0.01, 0.01, 1.1, 1.1, 1.1, 1.1, 0.01, 0.01]),
                                                  boundsOnFunctionValues=Interval(jp.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,-10.0, ]),
                                                                                  jp.array([10.0,   10.0,  10.0,  10.0,  10.0,  10.0,  10.0,  10.0, 10.0])),
                                                )
                         ]



H = lambda x, u, known, unknown: known[0](x, u) + jp.matmul(known[1](x, u), unknown[0](x))
contractions = [    lambda x, u, xdot, known, unknown: inverse_contraction_C(xdot - known[0](x, u), known[1](x, u), unknown[0](x))        ]
h = HObject( env.observation_space.shape, env.action_space.shape, known_functions, unknown_approximations, H, contractions)

# define the cost function, which we will try to minimize
def cost_function (x, u, x_future):
    return jp.sqrt(x[0].mid * x[0].mid + x[1].mid * x[1].mid) # x^2 + y^2, minimal = go to 0,0

# create agent with any side info we may need
agent = DifferentialInclusionAgent(env.observation_space, env.action_space, h, env.dt, cost_function, look_ahead_steps=10, descent_steps=10, learning_rate=0.01)

# reset env, begin looping
observation, info = env.reset(seed=10, return_info=True)
for _ in range(1000):
    # Get action from agent
    action = agent.act(observation)

    # get result of action
    new_observation, reward, done, observation_dot = env.step(action)
    print(_, observation, action)

    # retrieves a copy, does not recompute. This is only needed if we want to visualize
    future_states, future_actions = agent.get_future()

    # update agent
    agent.update(observation, action, observation_dot)

    # for our sake
    env.render(predictions=future_states, sleep=0.01)
    # env.plot(states)

    # continue to next state
    observation = new_observation


env.close()

