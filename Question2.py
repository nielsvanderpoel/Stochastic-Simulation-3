import networkx as nx
import numpy as np
import random
import math
from statistics import mean, stdev
import matplotlib.pyplot as plt
from scipy.stats import t


GML_FILE = "networkAssignment.gml"

# Group 10: City A = Rotterdam, City B = Eindhoven
# In the GML, these junctions appear as "Knooppunt Terbregseplein" (Rotterdam) 
# and "Knooppunt Leenderheide" (Eindhoven).
CITY_A_NAME = "Knooppunt Terbregseplein"
CITY_B_NAME = "Knooppunt Leenderheide"

# Speeds in km/h
CAR_SPEED = 100.0   
TRUCK_SPEED = 80.0  


CAR_FRACTION = 0.90


HOURLY_RATES = [
    314.2, 162.4, 138.6, 148.8, 273.2, 1118.8, 2773.8, 4036.2,
    4237.4, 3277.0, 2843.0, 2876.4, 3143.0, 3277.8, 3546.2, 4335.0,
    4945.4, 4525.8, 2847.8, 1828.0, 1378.4, 1271.2, 1171.2, 767.6
]


NUM_REPLICATIONS = 5


random.seed(42)
np.random.seed(42)


def get_node_id_by_name(graph, city_name):
    for node_id in graph.nodes:
        if graph.nodes[node_id].get("name", "") == city_name:
            return node_id
    return None

def sample_nonhomogeneous_poisson_arrivals(hourly_rates):
    arrival_times = []
    for hour in range(24):
        lam = hourly_rates[hour]
        #Poison number of arrivals in this hour
        num_arrivals = np.random.poisson(lam)
        #uniformly distribute these arrivals in [hour, hour+1) => [hour*60, (hour+1)*60) in minutes
        for _ in range(num_arrivals):
            arrival_minute = hour * 60 + np.random.uniform(0, 60)
            arrival_times.append(arrival_minute)
    arrival_times.sort()
    return arrival_times

def compute_route_length_km(graph, path):
    length_km = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        length_km += graph[u][v]['length']
    return length_km

def simulate_travel_time_minutes(graph, origin, destination, is_car):
    v_max = CAR_SPEED if is_car else TRUCK_SPEED

    try:
        path_nodes = nx.shortest_path(graph, source=origin, target=destination, weight='length')
    except nx.NetworkXNoPath:
        return None

    total_time_min = 0.0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        dist_km = graph[u][v]['length'] 
        mean_min = (dist_km / v_max) * 60.0
        std_min = mean_min / 20.0
        travel_time = np.random.normal(loc=mean_min, scale=std_min)
        if travel_time < 0:
            travel_time = 0.0
        total_time_min += travel_time

    return total_time_min

def run_simulation_one_day(graph, cityA_id, cityB_id):
    arrival_times = sample_nonhomogeneous_poisson_arrivals(HOURLY_RATES)
    total_vehicles = len(arrival_times)

    all_times = []
    car_times = []
    truck_times = []
    AtoB_car_times = []

    all_nodes = list(graph.nodes())

    for _ in arrival_times:
        origin, destination = np.random.choice(all_nodes, 2, replace=False)
        is_car = (random.random() < CAR_FRACTION)  # 90% cars

        tt_min = simulate_travel_time_minutes(graph, origin, destination, is_car)
        if tt_min is None:
            continue

        all_times.append(tt_min)
        if is_car:
            car_times.append(tt_min)
            if origin == cityA_id and destination == cityB_id:
                AtoB_car_times.append(tt_min)
        else:
            truck_times.append(tt_min)

    return {
        "total_vehicles": total_vehicles,
        "all_times": all_times,
        "car_times": car_times,
        "truck_times": truck_times,
        "AtoB_car_times": AtoB_car_times
    }


def get_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        if n == 1:
            return (data[0], data[0])
        else:
            return (0.0, 0.0)
    m = np.mean(data)
    s = np.std(data, ddof=1)
    t_crit = t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_crit * (s / math.sqrt(n))
    return (m - margin, m + margin)

def main():
    graph = nx.read_gml(GML_FILE)
    cityA_id = get_node_id_by_name(graph, CITY_A_NAME)
    cityB_id = get_node_id_by_name(graph, CITY_B_NAME)
    if cityA_id is None or cityB_id is None:
        raise ValueError("One or both citi names were not found in the graph. Check node names.")

    try:
        pathAtoB = nx.shortest_path(graph, source=cityA_id, target=cityB_id, weight='length')
        route_length_km = compute_route_length_km(graph, pathAtoB)
    except nx.NetworkXNoPath:
        route_length_km = 0.0

    replication_data = []
    for r in range(NUM_REPLICATIONS):
        stats_r = run_simulation_one_day(graph, cityA_id, cityB_id)
        replication_data.append(stats_r)


    def replication_means(key):
        means_list = []
        for rep in replication_data:
            values = rep[key]
            if len(values) > 0:
                means_list.append(np.mean(values))
            else:
                means_list.append(0.0)
        return means_list

    total_veh_list = [rep["total_vehicles"] for rep in replication_data]
    all_times_means = replication_means("all_times")
    car_times_means = replication_means("car_times")
    truck_times_means = replication_means("truck_times")
    AB_car_times_means = replication_means("AtoB_car_times")

    def summarize(label, data):
        m = np.mean(data)
        s = np.std(data, ddof=1)
        c_low, c_high = get_confidence_interval(data)
        print(f"{label:<42}  mean={m:7.2f}, std={s:7.2f}, 95%CI=[{c_low:6.2f}, {c_high:6.2f}]")

    print("\n----------------- Simulation Results (Group 10, Q2) -----------------\n")
    print(f"City A = {CITY_A_NAME} (Rotterdam), City B = {CITY_B_NAME} (Eindhoven)")
    print(f"Number of replications: {NUM_REPLICATIONS}")
    print("\nPerformance Measures (times in minutes, route length in km):")
    print("-------------------------------------------------------------------")

    summarize("Total number of vehicles", total_veh_list)
    summarize("Travel time (arbitrary vehicle) [min]", all_times_means)
    summarize("Travel time (truck) [min]", truck_times_means)
    summarize("Travel time (car) [min]", car_times_means)
    summarize(f"Travel time A->B (car) [min]", AB_car_times_means)
    print(f"Route length A->B [km]: {route_length_km:.2f}")

    all_AB_car_times = []
    for rep in replication_data:
        all_AB_car_times.extend(rep["AtoB_car_times"])
    if len(all_AB_car_times) > 0:
        plt.figure(figsize=(8,5))
        plt.hist(all_AB_car_times, bins=20, edgecolor='black')
        plt.title(f"Distribution of Car Travel Times (A->B): {CITY_A_NAME} to {CITY_B_NAME}")
        plt.xlabel("Travel time [minutes]")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No A->B car trips were observed in these replications.")

if __name__ == "__main__":
    main()
