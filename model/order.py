import numpy as np
from typing import NamedTuple, List


class Order(NamedTuple):
    order_id: str
    product: str
    quantity: int
    product_idx: int


class AllOrders:
    def __init__(self, orders: List[Order], product_count: int):
        self.orders = orders
        self.observation = [OrderObservation(order, product_count) for order in self.orders]
        self.product_count = product_count
    def get_observations(self):
        return np.transpose(np.array([ob.get_observation() for ob in self.observation], dtype=float))

    def reset(self):
        self.observation.clear()
        self.observation = [OrderObservation(order, self.product_count) for order in self.orders]


class OrderObservation:
    def __init__(self, order_info: Order, product_count):
        self.order_id = order_info.order_id
        self.product = order_info.product
        self.quantity = order_info.quantity
        self.product_idx = order_info.product_idx
        self.is_delivered = False
        self.delivered_time = 0
        self.product_count = product_count

    def get_observation(self):
        which_product = np.zeros(self.product_count).tolist()
        which_product[self.product_idx] = 1

        observation = [self.is_delivered, self.quantity, self.delivered_time] + list(which_product)
        return np.array(observation, dtype=float)
