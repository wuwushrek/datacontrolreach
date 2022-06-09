import gym
from gym import spaces
import numpy as np
import math
import random

class UnicycleMDP(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, destination=[0.0, 0.0], reached_destination_distance=0.5, dt=0.1, t_limit = 100.0, seed = 123):
    # index 0 = x position
    # index 1 = y position
    # index 2 = direction in radians. 0 radians = points along positive x axis
    self.state = np.array([0.0, 0.0, 0.0])
    self.state_dot = np.array([0.0, 0.0, 0.0])

    # time variables. Starts at 0, increments by dt every timestep, stops at t_limit
    self.dt = dt
    self.t_limit = t_limit
    self.t = 0.0

    # destination for unicycle. Default is origin. Reaches destination when its cartesian distance is less than the specified reached_destination_distance
    self.destination = destination
    self.reached_destination_distance = reached_destination_distance

    # create state and action space
    self.observation_space = spaces.Box(np.array([float('-inf'), float('-inf'), float('-inf')]),
                                        np.array([float('inf'), float('inf'), float('inf')]))
    self.action_space = spaces.Box(np.array([-3.0, -math.pi]),
                                   np.array([3.0, math.pi]))

    # seed random for reproducibility, and reset to a random state
    random.seed(123)
    self.reset()

  # resets to a random state, sets t to 0 again
  def reset(self):
      self.state = np.array([random.uniform(-5.0, 5.0),
                             random.uniform(-5.0, 5.0),
                             random.uniform(0.0, 2.0 * math.pi)])
      self.t = 0.0
      return self.state, self.t

  # takes the given action and applies it to the env. Changes state
  # Updates t. Returns current t as info
  def step(self, action):
      x = self.state[0]
      y = self.state[1]
      theta = self.state[2]

      f = [0.0, 0.0, 0.0]
      g = [[ math.cos(theta), 0.0],
           [ math.sin(theta), 0.0],
           [             0.0, 1.0]]

      self.state_dot = f + np.matmul(g, action)
      self.state = self.state + self.state_dot * self.dt

      reward = 0 # TODO
      self.t += self.dt
      return self.state, reward, self.is_terminated(), self.t

  # directly return the state_dot from the previous update step.
  def get_state_dot(self):
      return self.state_dot

  # measure if our current position is less than the acceptable distance from the destination. If it is, we can terminate.
  # otherwise, terminate when time has expired.
  def is_terminated(self):
      distance = math.sqrt((self.state[0] - self.destination[0])**2 + (self.state[1] - self.destination[1])**2)
      return  distance < self.reached_destination_distance or self.t >= self.t_limit
