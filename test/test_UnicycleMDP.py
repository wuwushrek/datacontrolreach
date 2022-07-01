import math
import random

import numpy as np
from  UnicycleMDP.UnicycleMDP import UnicycleMDP

def test_Unicycle(seed):
    env = UnicycleMDP()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    print("Num states = ", num_states, ", Num actions = ", num_actions)


    observation = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        observation_dot = env.get_state_dot()
        print(_, observation, observation_dot, action)
        observation = new_observation
        # if done:
        #    observation, info = env.reset(return_info=True)

    env.close()

# test_Unicycle(201)