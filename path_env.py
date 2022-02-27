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
        # self._state = np.full((self.eqps_count, self.jobs_count), -1)
        self._state = self.job_info.get_observation()
        self._episode_ended = False
        self.max_step = 20
        self.step_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        self.job_info = Jobs(self.jobs_run_time)
        self._state = self.job_info.get_observation()
        self._episode_ended = False
        self.step_count = 0
        self.assigned_job = []
        return ts.restart(np.array(self._state, dtype=np.float32))

    def update_state(self, eqp_id, job_name):
        pass

    def step(self, action):
        pass
