from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent
from datacontrolreach.interval import Interval
import gym
import numpy as np
import jax
import datacontrolreach.jumpy as jp

i = Interval(np.array([[1, 2],[2, 3],[3, 4]]), np.array([[4, 5],[5, 6],[6, 7]]))
print(type(i), type(i.lb), type(i.ub))

#print(i, jp.shape(i), type(i))
#print(i[0, :], jp.shape(i[0, :]), type(i[0, :]))
#print(i[0], jp.shape(i[0]), type(i[0]))


env = gym.make("Pendulum-v1")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print("Num states = ", num_states, ", Num actions = ",num_actions)


agent = DifferentialInclusionAgent(env.observation_space,
                                   env.action_space,
                                   np.full((num_states, ), 100.0),
                                   np.full((num_states, num_actions), 100.0)
                                   )

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    action = agent.act(observation) # agent getAction(state) goes here
    # print(_, observation, action)

    new_observation, reward, done, info = env.step(action)
    # env.render()
    observation_dot = new_observation - observation
    agent.update(observation.astype('float64'), action, observation_dot.astype('float64'))
    observation = new_observation
    # if done:
    #    observation, info = env.reset(return_info=True)

env.close()

