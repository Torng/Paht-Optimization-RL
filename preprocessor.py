import pandas as pd
from model.order import Order


class Preprocessor:
    def __init__(self):
        self.orders = pd.DataFrame

    def process(self):
        try:
            orders = pd.read_csv("data/orders_v2.csv")
            time = pd.read_csv("data/time_matrix_v2.csv")
            distance = pd.read_csv("data/distance_matrix_v2.csv")
        except Exception as e:
            raise e

        self.orders = {
            orders.loc[idx, "配送單號"]: Order(orders.loc[idx, "配送單號"], orders.loc[idx, "產品名稱"], orders.loc[idx, "需求桶數"])
            for idx in orders.index}
