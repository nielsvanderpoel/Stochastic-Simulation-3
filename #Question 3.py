#Question 3

import heapq
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter


GML_FILE = r"C:\Users\maart\Downloads\networkAssignment.gml"
CAR_SPEED = 100.0
TRUCK_SPEED = 80.0
CAR_FRACTION = 0.9
SIMULATION_DURATION_MIN = 24 * 60

HOURLY_RATES = [
    314.2, 162.4, 138.6, 148.8, 273.2, 1118.8, 2773.8, 4036.2,
    4237.4, 3277.0, 2843.0, 2876.4, 3143.0, 3277.8, 3546.2, 4335.0,
    4945.4, 4525.8, 2847.8, 1828.0, 1378.4, 1271.2, 1171.2, 767.6
]

hourly_incident_rates = incident_rate_per_hour.sort_values(by='Hour')['incident_rate'].tolist()
def incident_duration_sampler():
    return np.random.gamma(shape=1.19, scale=6.09)

event_counter = 0
FES = []
vehicle_id_counter = 0
vehicle_stats = []
delayed_vehicle_count = [0] * SIMULATION_DURATION_MIN
active_incidents = set()
active_incident_count = [0] * 1440
ab_car_stats = []  # For Q3.5
CITY_A_NAME = "Knooppunt Terbregseplein"
CITY_B_NAME = "Knooppunt Leenderheide"
CITY_A_ID = None
CITY_B_ID = None

class Vehicle:
    def __init__(self, id, origin, destination, is_car, start_time):
        self.id = id
        self.origin = origin
        self.destination = destination
        self.is_car = is_car
        self.path = []
        self.current_index = 0
        self.start_time = start_time
        self.total_time = 0.0
        self.delay_time = 0.0
        self.incidents = 0
        self.is_AB_car = False

vehicles = {}

def schedule_event(time, event_type, data):
    global event_counter
    heapq.heappush(FES, (time, event_counter, event_type, data))
    event_counter += 1


def generate_daily_incidents(graph):
    incidents = []
    edges = list(graph.edges())
    for hour in range(24):
        lam = hourly_incident_rates[hour]
        num_incidents = np.random.poisson(lam)
        for _ in range(num_incidents):
            start_min = hour * 60 + np.random.uniform(0, 60)
            duration = incident_duration_sampler()
            end_min = start_min + duration
            edge = random.choice(edges)
            incidents.append((start_min, "incident_start", edge))
            incidents.append((end_min, "incident_end", edge))
            for minute in range(int(start_min), int(end_min)):
                if 0 <= minute < 1440:
                    active_incident_count[minute] += 1
    return incidents


def process_event(time, event_type, data, graph):
    global vehicle_id_counter

    if event_type == "vehicle_arrival":
        origin, destination = data
        is_car = random.random() < CAR_FRACTION
        v = Vehicle(vehicle_id_counter, origin, destination, is_car, time)
        if origin == CITY_A_ID and destination == CITY_B_ID and is_car:
            v.is_AB_car = True
        vehicle_id_counter += 1

        try:
            path = nx.shortest_path(graph, origin, destination, weight='length')
            v.path = path
            vehicles[v.id] = v
            schedule_event(time, "enter_edge", {'vehicle_id': v.id})
        except:
            return

    elif event_type == "enter_edge":
        v = vehicles[data['vehicle_id']]
        if v.current_index >= len(v.path) - 1:
            v.total_time = time - v.start_time
            stat = {
                'travel_time': v.total_time,
                'delay_time': v.delay_time,
                'incidents': v.incidents
            }
            vehicle_stats.append(stat)
            if v.is_AB_car:
                ab_car_stats.append(stat)
            return

        u, v_next = v.path[v.current_index], v.path[v.current_index + 1]
        edge = (u, v_next)
        edge_length = graph[u][v_next]['length'] / 1000.0
        speed = CAR_SPEED if v.is_car else TRUCK_SPEED
        travel_time = max(np.random.normal((edge_length / speed) * 60, (edge_length / speed) * 3), 0.1)

        if edge in active_incidents:
            delay = np.random.uniform(5, 15)
            v.delay_time += delay
            v.incidents += 1
            for m in range(int(time), int(time + delay)):
                if 0 <= m < len(delayed_vehicle_count):
                    delayed_vehicle_count[m] += 1
            travel_time += delay

        v.current_index += 1
        schedule_event(time + travel_time, "enter_edge", {'vehicle_id': v.id})

    elif event_type == "incident_start":
        active_incidents.add(data)

    elif event_type == "incident_end":
        active_incidents.discard(data)


def run_discrete_event_sim(graph):
    global CITY_A_ID, CITY_B_ID
    CITY_A_ID = next((n for n, d in graph.nodes(data=True) if d.get('name') == CITY_A_NAME), None)
    CITY_B_ID = next((n for n, d in graph.nodes(data=True) if d.get('name') == CITY_B_NAME), None)

    all_nodes = list(graph.nodes())
    for hour in range(24):
        lam = HOURLY_RATES[hour]
        num_arrivals = np.random.poisson(lam)
        for _ in range(num_arrivals):
            t = hour * 60 + np.random.uniform(0, 60)
            origin, destination = np.random.choice(all_nodes, 2, replace=False)
            schedule_event(t, "vehicle_arrival", (origin, destination))

    for inc_time, inc_type, edge in generate_daily_incidents(graph):
        schedule_event(inc_time, inc_type, edge)

    while FES:
        time, _, event_type, data = heapq.heappop(FES)
        process_event(time, event_type, data, graph)

    return vehicle_stats, delayed_vehicle_count

#Question 3.1
def q3_1(vehicle_stats):
    print("Question 3.1")
    travel_times = [v['travel_time'] for v in vehicle_stats]
    delay_times = [v['delay_time'] for v in vehicle_stats]
    incident_counts = [v['incidents'] for v in vehicle_stats]


    print(f"Total vehicles: {len(vehicle_stats)}")
    print(f"Average travel time: {np.mean(travel_times):.2f} min (std: {np.std(travel_times):.2f})")
    print(f"Average delay time:  {np.mean(delay_times):.2f} min (std: {np.std(delay_times):.2f})")
    print(f"Average incidents encountered: {np.mean(incident_counts):.2f} (std: {np.std(incident_counts):.2f})")

    plt.figure(figsize=(8, 4))
    plt.hist(travel_times, bins=30, color='skyblue', edgecolor='black')
    plt.title("Travel Time Distribution")
    plt.xlabel("Minutes")
    plt.ylabel("Amount of Vehicles")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(delay_times, bins=30, color='salmon', edgecolor='black')
    plt.title("Delay Time Distribution")
    plt.xlabel("Minutes Delayed")
    plt.ylabel("Amount of Vehicles")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(incident_counts, bins=range(0, max(incident_counts) + 2), color='gray', edgecolor='black')
    plt.title("Amount of Incidents Per Vehicle")
    plt.xlabel("Amount of Incidents")
    plt.ylabel("Amount of Vehicles")
    plt.grid(True)
    plt.show()

#Question 3.2
def q3_2(delayed_vehicle_count):
    print("Question 3.2")
    total_minutes = len(delayed_vehicle_count)
    delay_distribution = Counter(delayed_vehicle_count)
    fraction_per_k = {k: v / total_minutes for k, v in sorted(delay_distribution.items())}

    for k, frac in fraction_per_k.items():
        print(f"{k:<22} {frac:.4f}")

    plt.figure(figsize=(8, 4))
    plt.bar(fraction_per_k.keys(), fraction_per_k.values(), color='teal', edgecolor='black')
    plt.title("Percentage Time with k amount Delayed Vehicles")
    plt.xlabel("Amount of Delayed Vehicles (k)")
    plt.ylabel("Percentage Time")
    plt.grid(True)
    plt.show()

#Question 3.3
def q3_3(active_incident_count):
    print("Question 3.3")
    total_minutes = len(active_incident_count)
    incident_distribution = Counter(active_incident_count)
    fraction_per_k = {k: v / total_minutes for k, v in sorted(incident_distribution.items())}

    for k, f in fraction_per_k.items():
        print(f"{k:<22} {f:.4f}")

    plt.figure(figsize=(8, 4))
    plt.bar(fraction_per_k.keys(), fraction_per_k.values(), color='orange', edgecolor='black')
    plt.title("Percentage Time with k Active Incidents")
    plt.xlabel("Amount Active Incidents (k)")
    plt.ylabel("Percentage Time")
    plt.grid(True)
    plt.show()

#Question 3.4
def q3_4(delayed_vehicle_count):
    print("Question 3.4")
    hourly_means = []
    for hour in range(24):
        start_min = hour * 60
        end_min = (hour + 1) * 60
        hour_values = delayed_vehicle_count[start_min:end_min]
        avg_delayed = np.mean(hour_values)
        hourly_means.append(avg_delayed)
        print(f"Hour {hour:02d}: {avg_delayed:.2f} vehicles delayed")

    plt.figure(figsize=(10, 5))
    plt.plot(range(24), hourly_means, marker='o', linestyle='-', color='crimson')
    plt.title("Average Amount Delayed Vehicles Per Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Amount Delayed Vehicles")
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Question 3.5
def q3_5(ab_car_stats, ab_car_baseline=None):
    print("Question 3.5")
    travel_times = [v['travel_time'] for v in ab_car_stats]
    delay_times = [v['delay_time'] for v in ab_car_stats]


    print(f"Trips from A to B: {len(travel_times)}")
    print(f"Mean travel time: {np.mean(travel_times):.2f} min (std: {np.std(travel_times):.2f})")
    print(f"Mean delay time:  {np.mean(delay_times):.2f} min (std: {np.std(delay_times):.2f})")

    plt.figure(figsize=(10, 4))
    plt.hist(travel_times, bins=30, color='orange', edgecolor='black', alpha=0.7, label='With Incidents')

    if ab_car_baseline:
        baseline_times = ab_car_baseline
        plt.hist(baseline_times, bins=30, color='skyblue', edgecolor='black', alpha=0.5, label='Without Incidents')

    plt.title("Travel Time A to B")
    plt.xlabel("Minutes")
    plt.ylabel("Number Trips")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    G = nx.read_gml(GML_FILE)
    vehicle_stats, delayed_vehicle_count = run_discrete_event_sim(G)


    q3_1(vehicle_stats)
    q3_2(delayed_vehicle_count)
    q3_3(active_incident_count)
    q3_4(delayed_vehicle_count)
    q3_5(ab_car_stats)