import numpy as np


class AllOrders:
    def __init__(self, order_id, product, quantity):
        pass

    def get_observations(self):
        pass


class Order:
    def __init__(self, order_id, product, quantity):
        self.order_id = order_id
        self.product = product
        self.quantity = quantity
        self.is_delivered = False

    def get_observation(self):
        which_product = np.zeros(len(self.product)).tolist()
        pass
