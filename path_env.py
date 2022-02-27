from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.order import AllOrders
import torch


class PathEnv:
    def __init__(self, orders: AllOrders, motors, actions, time_matrix, distance_matrix):
        self.orders = orders
        self.motor = motors
        self.actions = actions
        self.time_matrix = time_matrix
        self.distance_matrix = distance_matrix
        self._episode_ended = False
        self.max_step = 20
        self.step_count = 0
        self._state = None

    def reset(self):
        self.orders.reset()
        self._state = self.orders.get_observations()
        self._episode_ended = False
        self.step_count = 0
        return torch.tensor(self._state)

    def update_state(self, motor_id, order_id):
        pass

    def step(self, action):
        pass
