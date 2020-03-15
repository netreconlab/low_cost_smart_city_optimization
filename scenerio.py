import numpy as np
from sensor import *
import sim
import networkx as nx
from helper_func import *
from dsp_memo import DspMemo

class Scenerio:

    def __init__(self,
                 graph,
                 all_stops,
                 all_routes,
                 routes_per_stop,
                 sensor_count,
                 ):
        self.G = graph
        self.all_routes = all_routes
        self.sensor_count = sensor_count
        self.routes_per_stop = routes_per_stop
        self.error = 0
        self.routes_per_gateway = None
        self.gateways_per_route = None
        self.route_subgraphs = None
        self.all_gateways = None
        self.all_sensors = None
        self.sensor_objects = {}
        self.dsp_memo = DspMemo()
        self.place_sensor(all_stops)


    def place_sensor(self, all_stops, sensors=None):
        if sensors is None:
            self.randomly_select_sensor_locations(all_stops)

        self.assign_sensors_to_nodes()
        self.generate_sensors()
        self.generate_route_subgraphs()


    def calculate_penalty_reduction(self, gateways):
        self.place_gateways(gateways)
        return self.run_simulation()

    def place_gateways(self, gateways):
        # gateways here is asumed to be G.name_node
        self.all_gateways = {get_stopid(gateway) for gateway in gateways}
        self.routes_per_gateway = {gateway: self.routes_per_stop[gateway] for gateway in self.all_gateways} #defaultdict(set)
        self.gateways_per_route = invert_dict(self.routes_per_gateway)
        self.assign_gateways_to_nodes(gateways)



    def assign_gateways_to_nodes(self, gateways):

        attr = {gw: True for gw in gateways}
        nx.set_node_attributes(self.G, name='is_gateway', values=attr)

    def randomly_select_sensor_locations(self, all_stops):
        self.all_sensors = np.random.choice(all_stops, size=self.sensor_count, replace=False)

    def assign_sensors_to_nodes(self):
        attr = {sensor: True for sensor in self.all_sensors}
        nx.set_node_attributes(self.G, name='is_sensor', values=attr)

    def generate_sensors(self):
        msg_gen_rate = np.random.randint(low=sim.msg_gen_rate_range[0], high=sim.msg_gen_rate_range[1],
                                         size=self.sensor_count)  # 10mins to 12 hours
        start_time = np.random.randint(low=sim.msg_gen_rate_range[0], high=sim.msg_gen_rate_range[1],
                                       size=self.sensor_count)  # 0 to 1 hour
        np.random.shuffle(start_time)

        for i, sensor_name in enumerate(self.all_sensors):
            # TODO:: add routes_per_stop
            r = self.routes_per_stop[get_stopid(sensor_name)]

            s = OnRouteSensor(name=sensor_name, routes=r, start_time=start_time[i],
                              msg_gen_rate=msg_gen_rate[i], msg_ttl=None, data_size=None)
            self.sensor_objects[sensor_name] = s





        self.assign_sensors_to_nodes()
        self.generate_route_subgraphs()

    def generate_route_subgraphs(self):

        self.route_subgraphs = {}
        stops_per_route = invert_dict(self.routes_per_stop)

        for r in self.all_routes:
            sub_nodes = [namify_stop(self.G.name, s) for s in stops_per_route[r]]
            sub_graph = self.G.subgraph(sub_nodes).copy()
            self.route_subgraphs[r] = sub_graph


    def run_simulation(self):
        total_delay = 0
        total_generated = 0

        for time in range(int(sim.start / 60), sim.duration + 1):
            for name, sensor in self.sensor_objects.items():
                if sensor.generate_msg(time):
                    total_generated += 1
                    routes = self.routes_per_stop[get_stopid(sensor.name)]
                    # change time to secs
                    delay = self.calculate_delay(routes, sensor, time * 60)

                    if delay is None:
                        total_delay += sim.upper_bound_delay
                    else:
                        total_delay += delay

        return total_delay/total_generated


        #print(self.error)

    def calculate_delay(self, routes, sensor, time):
        """
        find shortest path from sensor node to a gateway node in the graph, weight is edge cost,
        while factoring in duration from current time to next next dept time for that edge.

        save gen_time and latency to sensor object

        remember departure time, distance is in seconds
        while "time", gen_time,start_time is in minutes.
        so remember to convert it.
        """
        # print(routes)
        # print(self.all_gateways)
        # print(self.routes_per_gateway)
        # print(self.gateways_per_route)
        # return 0
        import sys
        
        if not self.all_gateways:
            print("no gateways selected")
            return None
        
        waiting_time = None
        shortest_distance, shortest_path = sys.float_info.max, None  # to any gateway

        for r in routes:
            for gateway in self.gateways_per_route[r]:
                g = self.route_subgraphs[r].copy()

                wait_time = None

                try:
                    distance, path = self.dsp_memo.getDsp(g, r, sensor.name, namify_stop(self.G.name, gateway))
                    # distance, path = nx.single_source_dijkstra(g, sensor.name, namify_stop(self.G.name, gateway),
                    #                                            weight='length')
                except Exception as e:
                    continue

                while len(path) > 1:
                    '''
                    make sure then you limit duration to 24 hours. later if time is greater than 24
                    message is not delivered
                    '''

                    # TODO:: error rate too high.. fix it.
                    # print(path)
                    departure_list = g[sensor.name][path[1]][0]['departure_time'].get(r, None)

                    # print(departure_list)
                    if departure_list == None:
                        # print("no departure time found")
                        break
                        # g.remove_node(path[1])
                        # continue

                    else:
                        wait_time = get_time_to_next_departure(current_time=time, departure_list=departure_list)
                        break

                if wait_time != None:
                    #print (distance, wait_time)
                    if distance + wait_time < shortest_distance:
                        shortest_distance, shortest_path = distance + wait_time, path
                        waiting_time = wait_time
                        # break

        if waiting_time == None:
            shortest_distance = None
            self.error += 1

        # sensor.gen_times.append(time)  # in sec
        # sensor.msg_latencies.append(shortest_distance)  # in sec
        # sensor.waiting_time.append(waiting_time)
        # sensor.hops.append(shortest_path)
        #print(shortest_distance, self.error)
        return shortest_distance

