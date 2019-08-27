# sim.py
start = 0 * 60 * 60
end = 24 * 60 * 60

duration = 24 * 60  # 26 hours
pct_stops_as_sensors = 30
seed = None
no_of_seeds = 100
msg_gen_rate_range = (1, 2*60) #in minutes
network_file_list = [
    'louisville.zip',
    #'seattle.zip',
    #'mta.zip',
    #'lextran.zip',
    #'cht.zip'
    ]
network_file = None

