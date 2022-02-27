from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from path_reward import PathReward


class PathEnv:
    def __init__(self, orders, motors, actions, time_matrix, distance_matrix):
        self.orders = orders
        self.motor = motors
        self.actions = actions
        self.time_matrix = time_matrix
        self.distance_matrix = distance_matrix
        self._episode_ended = False
        self.max_step = 20
        self.step_count = 0

    def reset(self):
        pass

    def update_state(self, eqp_id, job_name):
        pass

    def step(self, action):
        pass
