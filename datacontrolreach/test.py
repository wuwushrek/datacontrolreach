from datacontrolreach.DifferentialInclusionAgent import DifferentialInclusionAgent
import gym
import numpy as np

env = gym.make("Pendulum-v1")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print("Num states = ", num_states, ", Num actions = ",num_actions)


agent = DifferentialInclusionAgent(env.observation_space,
                                   env.action_space,
                                   np.full((num_states, 1), 100),
                                   np.full((num_states, num_actions), 100)
                                   )

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    action = agent.act(observation) # agent getAction(state) goes here
    new_observation, reward, done, info = env.step(action)
    # env.render()
    observation_dot = new_observation - observation
    agent.update(observation, action, observation_dot)
    print(_, observation, action)
    observation = new_observation
    # if done:
    #    observation, info = env.reset(return_info=True)

env.close()
