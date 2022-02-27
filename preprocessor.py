import pandas as pd
from model.order import Order, AllOrders
import config
from itertools import product


class Preprocessor:
    def __init__(self):
        self.motors = []
        self.orders = pd.DataFrame
        self.time_matrix = pd.DataFrame
        self.distance_matrix = pd.DataFrame
        self.actions = []
        self.all_order = None

    def process(self):
        try:
            orders = pd.read_csv(config.ORDER_PATH)
            time = pd.read_csv(config.TIME_PATH)
            distance = pd.read_csv(config.DISTANCE_PATH)
        except Exception as e:
            raise e
        products = orders['產品名稱'].unique().tolist()
        self.orders = [Order(orders.loc[idx, "配送單號"], orders.loc[idx, "產品名稱"],
                             orders.loc[idx, "需求桶數"], products.index(orders.loc[idx, "產品名稱"]))
                       for idx in orders.index]

        self.all_order = AllOrders(self.orders, len(products))
        # self.all_orders =
        self.time_matrix = time
        self.distance_matrix = distance

        self.motors = config.MORTOS
        self.actions = list(product(self.motors, list([order.order_id for order in self.orders])))
