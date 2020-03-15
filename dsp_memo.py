import networkx as nx

class DspMemo:
    def __init__(self):
        self.memo = {}

    def getDsp(self, g, route, sensor_name, gateway_name):
        key = "-".join([route, sensor_name, gateway_name])
        value = self.memo.get(key, False)
        if value:
            return value

        self.memo[key] = nx.single_source_dijkstra(g, sensor_name, gateway_name, weight='length')

        return self.memo[key]
