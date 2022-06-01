
import gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    action = env.action_space.sample() # agent getAction(state) goes here
    new_observation, reward, done, info = env.step(action)
    env.render()
    # agent update(state, action, next_state) goes here
    observation = new_observation
    if done:
        observation, info = env.reset(return_info=True)
env.close()