from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent
from datacontrolreach.interval import Interval
import gym
import numpy as np
import jax
import datacontrolreach.jumpy as jp
from UnicycleMDP.UnicycleMDP import UnicycleMDP

# env = gym.make("Pendulum-v1")
env = UnicycleMDP()

# Print info just for our knowledge
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print("Num states = ", num_states, ", Num actions = ",num_actions)

# create agent with any side info we may need
agent = DifferentialInclusionAgent(env.observation_space,
                                   env.action_space,
                                   np.full((num_states, ), 100.0),
                                   np.full((num_states, num_actions), 100.0)
                                   )

# reset env, begin looping
observation, info = env.reset(seed=10, return_info=True)
for _ in range(1000):
    action = agent.act(observation) # agent getAction(state) goes here
    # print(_, observation, action)

    new_observation, reward, done, info = env.step(action)
    env.render()
    observation_dot = env.get_state_dot()
    agent.update(observation.astype('float64'), action, observation_dot.astype('float64'))
    observation = new_observation

env.close()

