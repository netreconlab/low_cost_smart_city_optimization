from collections import defaultdict, OrderedDict, Counter
import networkx as nx
import numpy as np
import peartree as pt
import random

################## MY MODULES ##################
import sim
from sensor import *





################## GLOBAL VARIABLES ##################

time_table = None
feed = None
G = None
stop_times = None
routes = None
trips = None

error = 0


all_routes = None
all_trips = None

stop_times_dict = None
stop_trips = None
routes_per_stop = None
stop_ranks = None


routes_per_gateway = None
gateways_per_route = None
all_gateways = None


edge_departures = None

all_sensors = None
sensor_count = None

sensor_objects = None

route_subgraphs = None

################## HELPER FUNCTIONS ##################

def get_stopid(node_name):
    return node_name.split('_')[-1]


def namify_stop(g_name,stop_id):
    return "{0}_{1}".format(g_name,stop_id)


def invert_dict(d):
    inverted_d = defaultdict(set)
    for k in d.keys():
        for v in d[k]:
            inverted_d[v].add(k)
    return inverted_d

# I don't think this is useful
def get_routes_per_stop_id(stop_id):
    for stop_id in time_table.stop_id.unique():
            routes = time_table[time_table.stop_id == stop_id].route_id.unique()
            return set(routes)


def get_time_to_next_departure(current_time, departure_list):
    try:
        next_departure = min(v for v in departure_list if v >= current_time)
        wait_time = next_departure - current_time
    except:
        wait_time = None

    return wait_time


######################################################




def load_network():
    global feed,G

    feed = pt.get_representative_feed('data/gtfs/' + sim.network_file)
    G = pt.load_feed_as_graph(feed, sim.start, sim.end, interpolate_times=True)



def load_stop_times():
    global stop_times, routes, trips, time_table


    stop_times = feed.stop_times
    routes = feed.routes
    trips = feed.trips

    stoptimes_trips = stop_times.merge(trips, left_on='trip_id', right_on='trip_id')
    stoptimes_trips_routes = stoptimes_trips.merge(routes, left_on='route_id', right_on='route_id')

    columns = ['route_id',
               'service_id',
               'trip_id',
               #'trip_headsign',
               'direction_id',
               #'block_id',
               #'shape_id',
               #'route_short_name',
               #'route_long_name',
               'route_type',
               'arrival_time',
               'departure_time',
               'stop_id',
               'stop_sequence']

    time_table = stoptimes_trips_routes[columns]



def format_stop_times():
    global time_table, all_trips, all_routes

    #time_table = pt.summarizer._trim_stop_times_by_timeframe(time_table, sim.start, sim.end)

    time_table = time_table[~time_table['route_id'].isnull()]

    time_table = pt.summarizer._linearly_interpolate_infill_times(
        time_table,
        use_multiprocessing=True)

    if 'direction_id' in time_table:
        # If there is such column then check if it contains NaN
        has_nan = time_table['direction_id'].isnull()
        if sum(has_nan) > 0:
            # If it has no full coverage in direction_id, drop the column
            time_table.drop('direction_id', axis=1, inplace=True)

    # all_routes = set(feed.routes.route_id.values)
    all_routes = set(time_table.route_id.unique())
    all_trips = set(time_table.trip_id.unique())


def analyze_stops():
    global stop_times_dict, stop_tripsm, routes_per_stop, stop_ranks
    stop_times_dict = defaultdict(dict)
    stop_trips = defaultdict(set)
    routes_per_stop = defaultdict(set)
    stop_ranks = OrderedDict()

    for i,row in time_table.iterrows():
       stop_trips[row.stop_id].add(row.trip_id)
       routes_per_stop[row.stop_id].add(row.route_id)

    d = {}
    for k,v in routes_per_stop.items():
        d[k] = len(v)

    for k in sorted(d, key=d.get, reverse=True):
        stop_ranks[k] = d[k]
    #stop_ranks = {k:d[k] for k in sorted(d, key=d.get, reverse=True)}


def assign_gateways_to_nodes():
    global all_gateways #input
    global G #output

    attr = {gw:True for gw in all_gateways}
    nx.set_node_attributes(G, name='is_gateway', values=attr)

    return G


#### Add departure times of source node to edges
def get_departure_times_per_edge_per_route():
    import pandas as pd

    global time_table  # input
    global edge_departures  # output

    has_dir_col = 'direction_id' in time_table.columns.values

    all_deps = []
    all_route_ids = []
    all_from_stop_ids = []
    all_to_stop_ids = []

    for trip_id in time_table.trip_id.unique():

        tst_sub = time_table[time_table.trip_id == trip_id]
        route = tst_sub.route_id.values[0]

        # Just in case both directions are under the same trip id
        for direction in [0, 1]:
            # Support situations where direction_id is absent from the
            # GTFS data. In such situations, include all trip and stop
            # time data, instead of trying to split on that column
            # (since it would not exist).
            if has_dir_col:
                dir_mask = (tst_sub.direction_id == direction)
                tst_sub_dir = tst_sub[dir_mask]
            else:
                tst_sub_dir = tst_sub.copy()

            tst_sub_dir = tst_sub_dir.sort_values('stop_sequence')
            deps = tst_sub_dir.departure_time[:-1]

            # Add each resulting list to the running array totals
            all_deps += list(deps)

            from_ids = tst_sub_dir.stop_id[:-1].values
            all_from_stop_ids += list(from_ids)

            to_ids = tst_sub_dir.stop_id[1:].values
            all_to_stop_ids += list(to_ids)

            all_route_ids.extend([route] * len(deps))

    # Only return a dataframe if there is contents to populate
    # it with
    if len(all_deps) > 0:
        # Now place results in data frame
        edge_departures = pd.DataFrame({
            'from_stop_id': all_from_stop_ids,
            'to_stop_id': all_to_stop_ids,
            'deps': all_deps,
            'route_id': all_route_ids})


def add_departure_to_edge():
    global edge_departures  # input
    global G  # output

    for i, row in edge_departures.drop_duplicates(['from_stop_id', 'to_stop_id']).iterrows():
        u,v = row.from_stop_id, row.to_stop_id

        dep_mask = (edge_departures['from_stop_id'] == u) & (edge_departures['to_stop_id'] == v)
        #dep_list = edge_deps[dep_mask].deps.values
        dep_list = edge_departures[dep_mask][['route_id', 'deps']].sort_values(['deps'])

        dep_per_route = dep_list.groupby('route_id')['deps'].apply(lambda x: x.tolist()).to_dict(into=OrderedDict)

        u,v =  namify_stop(G.name,u), namify_stop(G.name,v)

        #TODO:: find out why you have to do thsi
        if u in G and v in G[u]:
            G[u][v][0]['departure_time'] = dep_per_route


    #test to make sure all edges is serviced
    for x in G.edges(keys=True,data=True):
        if 'departure_time' not in x[3]:
            print(x)


## Randomly selects stops to serve as sensors
def randomly_select_sensor_locations():
    global G # input
    global all_sensors, sensor_count # output


    all_stops = set(G.nodes)
    sensor_count = round(len(all_stops) * sim.pct_stops_as_sensors / 100)

    eligible_stops = list(all_stops - set(all_gateways)) #remove gateways from the list
    all_sensors = np.random.choice(eligible_stops, size=sensor_count, replace=False)


## Mark selected nodes as sensors
def assign_sensors_to_nodes():
    global all_sensors  # input
    global G  # output

    attr = {sensor:True for sensor in all_sensors}

    nx.set_node_attributes(G, name='is_sensor', values=attr)


def generate_sensors():
    global all_sensors, routes_per_stop # input
    global sensor_objects # output

    sensor_objects = {}

    msg_gen_rate = np.random.randint(low = sim.msg_gen_rate_range[0], high= sim.msg_gen_rate_range[1], size=len(all_sensors)) # 10mins to 12 hours
    start_time = np.random.randint(low = sim.msg_gen_rate_range[0], high=sim.msg_gen_rate_range[1], size=len(all_sensors)) # 0 to 1 hour
    np.random.shuffle(start_time)

    print(sum(msg_gen_rate), sum(start_time))

    #exit()


    for i,sensor_name in enumerate(all_sensors):
        #print(i,sensor_name)
        #r = get_routes_per_stop_id(get_stopid(sensor_name))
        r = routes_per_stop[get_stopid(sensor_name)]

        s = OnRouteSensor(name=sensor_name, routes=r, start_time=start_time[i], msg_gen_rate=msg_gen_rate[i], msg_ttl=None, data_size=None)
        sensor_objects[sensor_name]=s



def generate_route_subgraphs():
    global G,routes_per_stop,all_routes # input
    global route_subgraphs, stops_per_route # output

    route_subgraphs = {}
    stops_per_route = invert_dict(routes_per_stop)


    for r in all_routes:
        sub_nodes = [namify_stop(G.name, s) for s in stops_per_route[r]]
        # G.remove_nodes_from([n for n in G if n not in set(nodes)])
        sub_graph = G.subgraph(sub_nodes).copy()
        route_subgraphs[r] = sub_graph
    # print(list(nx.simple_cycles(sub_graph)))
    # print(r)


def get_and_store_message_delay(routes, sensor, time):
    """
    find shortest path from sensor node to a gateway node in the graph, weight is edge cost,
    while factoring in duration from current time to next next dept time for that edge.

    save gen_time and latency to sensor object

    remember departure time, distance is in seconds
    while "time", gen_time,start_time is in minutes.
    so remember to convert it.
    """
    global G, route_subgraphs, gateways_per_route  # inputs

    global error

    import sys
    waiting_time = None
    shortest_distance, shortest_path = sys.float_info.max, None  # to any gateway


    for r in routes:

        for gateway in gateways_per_route[r]:

            #get_shortest_path(r,)

            g = route_subgraphs[r].copy()

            wait_time = None


            try:
                distance, path = nx.single_source_dijkstra(g, sensor.name, namify_stop(G.name, gateway), weight='length')
            except:
                continue

            while len(path) > 1:
                '''
                make sure then you limit duration to 24 hours. later if time is greater than 24
                message is not delivered
                '''


                # TODO:: error rate too high.. fix it.
                #print(path)
                departure_list = g[sensor.name][path[1]][0]['departure_time'].get(r, None)

                #print(departure_list)
                if departure_list == None:
                    # print("no departure time found")
                    break
                    #g.remove_node(path[1])
                    #continue

                else:
                    wait_time = get_time_to_next_departure(current_time=time, departure_list=departure_list)
                    break


            if wait_time != None:

                if distance + wait_time < shortest_distance:
                    shortest_distance, shortest_path = distance + wait_time, path
                    waiting_time = wait_time
                    #break

    if waiting_time == None:
        shortest_distance = None
        error +=1


    sensor.gen_times.append(time)  # in sec
    sensor.msg_latencies.append(shortest_distance)  # in sec
    sensor.waiting_time.append(waiting_time)
    sensor.hops.append(shortest_path)



def store_results():
    import json
    from collections import defaultdict
    final_result = defaultdict(list)

    final_result['sim_time'] = sim.duration

    # print(sensor_objects.values())
    # type(sensor_objects.values()[0])

    for s in sensor_objects.values():

        data = {
            'delivery_rate': None,
            'no_of_routes': len(s.routes),
            'all_latencies': s.msg_latencies,
            'all_waiting_times': s.waiting_time ,
            'all_gen_times': s.gen_times,
            'all_hops': s.hops,
            'delivered_latencies': [],
            'delivered_gen_times': [],
            'delivered_waiting_times':[],
            'delivered_hops':[],
        }

        for i in range(len(s.msg_latencies)):
            if (s.msg_latencies[i] != None) and (s.gen_times[i] + s.msg_latencies[i] < sim.duration * 60):
                data['delivered_latencies'].append(s.msg_latencies[i])
                data['delivered_gen_times'].append(s.gen_times[i])

                data['delivered_waiting_times'].append(s.waiting_time[i])
                data['delivered_hops'].append(s.hops[i])

        # print(len(s.gen_times))

        if (len(s.gen_times) != 0):
            data['delivery_rate'] = len(data['delivered_latencies']) / len(s.gen_times)

        final_result['ons'].append(data)

    with open('results/{0}_data_{1}.txt'.format(sim.network_file, sim.seed), 'w') as outfile:
        json.dump(final_result, outfile, indent=True)

    print("Results Stored!")



def run_simulation():
    global sensor_objects, routes_per_stop
    global error

    for time in range(int(sim.start/60), sim.duration + 1):
        for name, sensor in sensor_objects.items():
            if sensor.generate_msg(time):
                routes = routes_per_stop[get_stopid(sensor.name)]
                # change time to secs
                get_and_store_message_delay(routes, sensor, time * 60)

    print("Simulation Completed! for seed_{0}".format(sim.seed))
    print(error)



def reset_sim():
    time_table = None
    feed = None
    G = None
    stop_times = None
    routes = None
    trips = None

    error = 0

    all_routes = None
    all_trips = None

    stop_times_dict = None
    stop_trips = None
    routes_per_stop = None
    stop_ranks = None

    routes_per_gateway = None
    gateways_per_route = None
    all_gateways = None

    edge_departures = None

    all_sensors = None
    sensor_count = None

    sensor_objects = None

    route_subgraphs = None



def select_gateways1():
    # minimal based on ranks
    global all_routes, routes_per_stop, stop_ranks #inputs
    global routes_per_gateway, gateways_per_route, all_gateways #outputs


    selected_gw = []
    routes_per_gateway = defaultdict(set)
    unserved_routes = all_routes.copy()

    for stop_id,rank in stop_ranks.items():

        rs = routes_per_stop[stop_id]
        route_to_serve = rs.intersection(unserved_routes)

        if len(route_to_serve) != 0:

            selected_gw.append(stop_id)
            unserved_routes.difference_update(route_to_serve)


            routes_per_gateway[stop_id].update(routes_per_stop[stop_id])

            if len(unserved_routes) == 0:
                break

    #routes_per_gateway = select_optimal_gateways(all_routes, routes_per_stop, stop_ranks)
    gateways_per_route = invert_dict(routes_per_gateway)
    all_gateways = set(namify_stop(G.name, x) for x in routes_per_gateway.keys())
    #print(all_gateways, '\n', gateways_per_route, '\n', routes_per_gateway, '\n', end='\n')

def select_gateways2():
    # minimal based on ranks
    global all_routes, stops_per_route, routes_per_stop  # inputs
    global all_gateways, gateways_per_route, routes_per_gateway   # outputs

    all_gateways = set()
    routes_per_gateway = defaultdict(set)

    for stops in stops_per_route.values():
        # print(stops)
        gw = np.random.choice(list(stops), random.randint(1, 3), replace=False)

        for stop_id in gw:
            #print(routes_per_gateway, routes_per_stop)
            routes_per_gateway[stop_id].update(routes_per_stop[stop_id])

        all_gateways.update(namify_stop(G.name, gw))

    gateways_per_route = invert_dict(routes_per_gateway)


def print_stats():
    global all_routes, all_gateways, stop_ranks
    print("{} Routes, {} Gateways, {} stops".format(len(all_routes), len(all_gateways), len(stop_ranks)))

if __name__ == '__main__':
    #import importlib

    print("Simulation Started!")


    for network in sim.network_file_list:
        for seed in range(0, sim.no_of_seeds):

            #importlib.reload(gen)

            sim.seed = seed
            sim.network_file = network

            np.random.seed(sim.seed)
            random.seed(sim.seed)


            load_network()
            load_stop_times()
            format_stop_times()
            analyze_stops()
            generate_route_subgraphs()
            select_gateways1()
            print_stats()
            assign_gateways_to_nodes()
            get_departure_times_per_edge_per_route()
            add_departure_to_edge()
            randomly_select_sensor_locations()
            assign_sensors_to_nodes()
            generate_sensors()
            generate_route_subgraphs()
            run_simulation()
            store_results()

            reset_sim()


    #a = edge_departures['route_id'].dtype = int

    #print(a.route_id.dtype)
