
"""
Dutch Highway Traffic Simulation (24 h, No Incidents)
=====================================================

This script implements a *Future‑Event‑Set* (FES) based discrete‑event simulation
for one day (24 hours) of traffic on the Dutch highway network provided in
`networkAssignment.gml`.  It reproduces the model specified in Assignment 3
(Question 2) of the Stochastic Simulation & Modelling course:

* Piece‑wise constant Poisson vehicle arrivals with hourly rates  λₜ (t = 0,…,23).
* Random origin–destination pairs (uniform over the network’s junctions).
* Vehicle type: 90 % cars (v_max=100 km/h), 10 % trucks (v_max=80 km/h).
* Route choice: shortest path by distance (no incidents => no re‑routing).
* Link travel time ~ 𝒩(μ = ℓ / v_max,  σ = μ / 20)   (ℓ in km).

For the **group‑10** pair of cities:
    – Rotterdam  (Junction “Knooppunt Terbregseplein”)
    – Eindhoven  (Junction “Knooppunt Leenderheide”)

After *N_RUNS* independent replications (default 30) it prints Table 1 with
means, standard deviations, and 95 % confidence intervals, and saves a histogram
of Rotterdam→Eindhoven car travel times.

"""

import heapq
import math
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Model parameters
# ──────────────────────────────────────────────────────────────────────────────
# Hourly arrival rates λₜ  (vehicles per hour, t = 0…23)
HOURLY_RATES = [
    314.2, 162.4, 138.6, 148.8, 273.2, 1118.8, 2773.8, 4036.2,
    4237.4, 3277.0, 2843.0, 2876.4, 3143.0, 3277.8, 3546.2, 4335.0,
    4945.4, 4525.8, 2847.8, 1828.0, 1378.4, 1271.2, 1171.2, 767.6,
]

CAR_FRACTION         = 0.9       # probability a vehicle is a car
CAR_VMAX_KMH         = 100.0     # maximum speed car  (km/h)
TRUCK_VMAX_KMH       = 80.0      # maximum speed truck (km/h)
STD_COEFF            = 1.0 / 20  # σ = μ / 20  on every link
SIM_DURATION_MIN     = 24 * 60   # 24 h expressed in minutes

# Monte‑Carlo settings
N_RUNS               = 30        # number of independent 24‑hour replications
RANDOM_SEED          = 42        # base RNG seed (different per run via offset)

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def sample_link_travel_time(length_km: float, vmax_kmh: float) -> float:
    """Return a single normal sample for travel time (minutes) on one link."""
    mu = length_km / vmax_kmh * 60.0                # convert h → min
    sigma = mu * STD_COEFF
    tt = random.gauss(mu, sigma)
    return max(0.01, tt)    # truncate at tiny positive value

def poisson(lmbda: float) -> int:
    """Draw from Poisson(λ) using numpy for convenience."""
    return int(np.random.poisson(lmbda))

def t_conf_interval(data: List[float], alpha=0.05) -> Tuple[float, float]:
    """Return the half‑width of the (1‑alpha) confidence interval for the mean."""
    if len(data) < 2:
        return math.nan, math.nan
    mean = statistics.mean(data)
    sd   = statistics.stdev(data, xbar=mean)
    half_width = 1.96 * sd / math.sqrt(len(data))   # Normal approx (n≥30)
    return mean - half_width, mean + half_width

# ──────────────────────────────────────────────────────────────────────────────
# Simulation of one 24‑hour period
# ──────────────────────────────────────────────────────────────────────────────
def simulate_one_day(G: nx.Graph,
                     node_ids: List[int],
                     rotterdam_id: int,
                     eindhoven_id: int,
                     run_seed: int = 0) -> Dict[str, float]:
    """
    Run one replication of the *no‑incidents* model and return performance stats.
    Times are recorded in minutes, distances in km.
    """
    random.seed(run_seed)
    np.random.seed(run_seed)

    # Event set: (time_min, event_type, payload)
    # event_type: 'ARRIVAL' | 'DEPARTURE'
    fes: List[Tuple[float, str, tuple]] = []

    # Pre‑schedule ARRIVAL events for each hour (Poisson count, uniform within hour)
    for hour, rate in enumerate(HOURLY_RATES):
        n_arr = poisson(rate)
        for _ in range(n_arr):
            t_arr = hour * 60.0 + random.uniform(0.0, 60.0)   # minutes since 0:00
            heapq.heappush(fes, (t_arr, 'ARRIVAL', None))

    # Stats storage
    total_vehicles            = 0
    travel_times_all:  List[float] = []
    travel_times_cars: List[float] = []
    travel_times_trucks: List[float] = []
    trip_lengths_km:   List[float] = []
    rot_ehv_car_times: List[float] = []

    # Main event‑processing loop
    while fes:
        time_min, ev_type, _ = heapq.heappop(fes)
        if time_min > SIM_DURATION_MIN:
            break

        if ev_type == 'ARRIVAL':
            # Determine vehicle characteristics
            is_car  = (random.random() < CAR_FRACTION)
            vmax    = CAR_VMAX_KMH if is_car else TRUCK_VMAX_KMH

            # Random OD pair (ensure origin ≠ destination)
            origin, dest = random.sample(node_ids, 2)

            # Compute shortest route (by length) once
            route_nodes = nx.shortest_path(G, origin, dest, weight='length')
            route_len_m = nx.shortest_path_length(G, origin, dest, weight='length')
            route_len_km = route_len_m / 1_000.0

            # Sample travel time along each link on the route
            route_tt_min = 0.0
            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                length_km = G.edges[(u, v)]['length'] / 1_000.0
                route_tt_min += sample_link_travel_time(length_km, vmax)

            dep_time = time_min + route_tt_min
            heapq.heappush(fes, (dep_time, 'DEPARTURE',
                                 (is_car, route_tt_min, route_len_km,
                                  origin, dest)))

        elif ev_type == 'DEPARTURE':
            is_car, tt_min, len_km, origin, dest = _
            # Collect statistics
            total_vehicles += 1
            travel_times_all.append(tt_min)
            trip_lengths_km.append(len_km)
            if is_car:
                travel_times_cars.append(tt_min)
                # Check if specific OD pair (Rotterdam → Eindhoven) *car* trip
                if origin == rotterdam_id and dest == eindhoven_id:
                    rot_ehv_car_times.append(tt_min)
            else:
                travel_times_trucks.append(tt_min)

    # Aggregate daily stats
    stats = {
        'total_vehicles'          : total_vehicles,
        'mean_tt_all'             : statistics.mean(travel_times_all),
        'mean_tt_car'             : statistics.mean(travel_times_cars),
        'mean_tt_truck'           : statistics.mean(travel_times_trucks),
        'mean_len_km'             : statistics.mean(trip_lengths_km),
        'rot_ehv_car_times'       : rot_ehv_car_times,  # list for histogram later
        'mean_rot_ehv_car_tt'     : (statistics.mean(rot_ehv_car_times)
                                     if rot_ehv_car_times else math.nan),
        # store std‑devs too for Table 1
        'std_tt_all'              : statistics.stdev(travel_times_all)
                                      if len(travel_times_all) > 1 else 0.0,
        'std_tt_car'              : statistics.stdev(travel_times_cars)
                                      if len(travel_times_cars) > 1 else 0.0,
        'std_tt_truck'            : statistics.stdev(travel_times_trucks)
                                      if len(travel_times_trucks) > 1 else 0.0,
        'std_len_km'              : statistics.stdev(trip_lengths_km)
                                      if len(trip_lengths_km) > 1 else 0.0,
    }
    return stats

# ──────────────────────────────────────────────────────────────────────────────
# Main driver: multiple replications + summary table + histogram
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Load the assignment graph (undirected)
    G = nx.read_gml('networkAssignment.gml')
    node_ids = list(G.nodes)

    # Look up junction IDs by name
    name_to_id = {G.nodes[n]['name']: n for n in G.nodes}

    rotterdam_name = 'Knooppunt Terbregseplein'
    eindhoven_name = 'Knooppunt Leenderheide'

    try:
        rotterdam_id = name_to_id[rotterdam_name]
        eindhoven_id = name_to_id[eindhoven_name]
    except KeyError as e:
        raise KeyError(f"Could not find junction name {e} in GML file")

    # For confidence‑interval calculation, store run‑level means
    run_stats: List[Dict[str, float]] = []

    # Also collect all Rot→Ehv car times across ALL runs for the histogram
    all_rot_ehv_car_times: List[float] = []

    for run in range(N_RUNS):
        seed = RANDOM_SEED + run   # different seed for each replication
        s = simulate_one_day(G, node_ids, rotterdam_id, eindhoven_id, seed)
        run_stats.append(s)
        all_rot_ehv_car_times.extend(s['rot_ehv_car_times'])

        # Simple progress indicator
        print(f"Run {run+1}/{N_RUNS}  –  vehicles: {s['total_vehicles']}")

    # Build Table 1
    def col(name_mean, name_std):
        values_mean = [r[name_mean] for r in run_stats]
        values_std  = [r[name_std]  for r in run_stats]
        mean_of_means = statistics.mean(values_mean)
        sd_of_means   = statistics.stdev(values_mean)
        ci_low, ci_up = t_conf_interval(values_mean)
        return mean_of_means, sd_of_means, ci_low, ci_up

    rows = [
        ('Total number of vehicles',
            statistics.mean([r['total_vehicles'] for r in run_stats]),
            statistics.stdev([r['total_vehicles'] for r in run_stats]),
         *t_conf_interval([r['total_vehicles'] for r in run_stats])
        ),
        ('Travel time (arbitrary vehicle) [min]', *col('mean_tt_all', 'std_tt_all')),
        ('Travel time car [min]',                 *col('mean_tt_car', 'std_tt_car')),
        ('Travel time truck [min]',               *col('mean_tt_truck', 'std_tt_truck')),
        ('Travel time Rot→Ehv (car) [min]',       *col('mean_rot_ehv_car_tt', 'mean_rot_ehv_car_tt')),
        ('Route length [km]',                     *col('mean_len_km', 'std_len_km')),
    ]

    # Print nicely
    print("\n\nTable 1 – Simulation results over "
          f"{N_RUNS} runs (24 h each, no incidents)\n"
          "(95 % confidence intervals for the mean)\n")
    header = f"{'Performance measure':37s}  {'Mean':>9s}  {'SD':>9s}  {'95% CI low':>12s}  {'95% CI up':>11s}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r[0]:37s}  {r[1]:9.2f}  {r[2]:9.2f}  {r[3]:12.2f}  {r[4]:11.2f}")

    # Histogram for Rot→Ehv car travel times
    plt.figure(figsize=(7, 4))
    plt.hist(all_rot_ehv_car_times, bins=20, edgecolor='black')
    plt.title('Histogram of Car Travel Times\nRotterdam → Eindhoven (N runs)')
    plt.xlabel('Travel time [min]')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rotterdam_eindhoven_hist.png', dpi=300)
    print("\nHistogram saved as 'rotterdam_eindhoven_hist.png'")

if __name__ == '__main__':
    main()
