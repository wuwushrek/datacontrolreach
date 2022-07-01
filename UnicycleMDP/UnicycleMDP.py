import time

import gym
from gym import spaces

import numpy as np
import math
import random
from typing import Optional
from os import path

class UnicycleMDP(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, destination=[0.0, 0.0], reached_destination_distance=0.5, dt=0.1, t_limit = 100.0, seed = 123, render_mode: Optional[str] = None,):
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

    # renderer stuff
    self.viewer = None

    # seed random for reproducibility, and reset to a random state
    random.seed(seed)
    self.reset()

  # resets to a random state, sets t to 0 again
  def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
      random.seed(seed)

      # get random initial state
      self.state = np.array([random.uniform(-5.0, 5.0),
                             random.uniform(-5.0, 5.0),
                             random.uniform(0.0, 2.0 * math.pi)])
      self.t = 0.0

      # return observation
      if return_info:
        return self.state, self.t
      else:
        return self.state

  # takes the given action and applies it to the env. Changes state
  # Updates t. Returns current t as info
  def step(self, action):
      theta = self.state[2]

      # state dot does not depend on current state
      f = [0.0, 0.0, 0.0]

      # Follows transition given by paper
      g = [[ math.cos(theta), 0.0],
           [ math.sin(theta), 0.0],
           [             0.0, 1.0]]

      # update state and state dot
      self.state_dot = f + np.matmul(g, action)
      self.state = self.state + self.state_dot * self.dt

      # reward is just negative distance to destination
      reward = -self.distance_to_destination()
      self.t += self.dt

      return self.state, reward, self.is_terminated(), self.t

  # directly return the state_dot from the previous update step.
  def get_state_dot(self):
      return self.state_dot

  # measure if our current position is less than the acceptable distance from the destination. If it is, we can terminate.
  # otherwise, terminate when time has expired.
  def is_terminated(self):
      return  self.distance_to_destination() < self.reached_destination_distance or self.t >= self.t_limit

  # helper function
  def distance_to_destination(self):
      return math.sqrt((self.state[0] - self.destination[0])**2 + (self.state[1] - self.destination[1])**2)

  def render(self, mode="human"):
      if self.viewer is None:
          from gym.envs.classic_control import rendering

          # make viewer, set bounds
          self.viewer = rendering.Viewer(500, 500)
          self.viewer.set_bounds(-5.0, 5.0, -5.0, 5.0)

          # make the unicycle image
          fname = path.join(path.dirname(__file__), "unicycle.png") # found at https://pixabay.com/vectors/unicycle-bike-wheel-sport-fun-310174/ for free use via google
          width = 1.0
          self.img = rendering.Image(fname, width, width * 1280.0/712.0)
          self.imgtrans = rendering.Transform()
          self.img.add_attr(self.imgtrans)

          # make the destination point
          destination = rendering.make_circle(0.25)
          destination.set_color(255, 0, 0)
          destination.translation = (self.destination[0], self.destination[1])
          self.viewer.add_geom(destination)


      # move the unicycle as needed
      self.viewer.add_onetime(self.img)
      self.imgtrans.translation = (self.state[0], self.state[1])

      return self.viewer.render(return_rgb_array=mode == "rgb_array")

  def close(self):
      if self.viewer:
          self.viewer.close()
          self.viewer = None

