class Sensor:
    class_counter = 0

    def __init__(self, name, data_size, msg_gen_rate, start_time, msg_ttl):

        self.id = 'S' + str(Sensor.class_counter)
        self.name = name
        self.data_size = data_size
        self.msg_gen_rate = msg_gen_rate
        self.start_time = start_time
        self.msg_ttl = msg_ttl
        self.gen_times = []
        self.msg_latencies = []
        self.hops = []
        self.waiting_time = []

        Sensor.class_counter += 1

    def generate_msg(self, time):

        if time == 0:
            if self.start_time  == 0:
                return True
            else:
                return False


       # elif (time % self.msg_gen_rate) - self.start_time == 0:
        elif (time % self.msg_gen_rate) == 0:
            return True

        else:
            return False


class OnRouteSensor(Sensor):

    def __init__(self, name, routes, start_time, msg_gen_rate, msg_ttl=None, data_size=None):

        Sensor.__init__(self, name, data_size, msg_gen_rate, start_time, msg_ttl)
        self.routes = set()


    def set_route(self, Route):
        self.routes.append(Route)