#=== Installation done hai final vala ===

!pip install simpy networkx matplotlib pandas

# === Imports ===

import simpy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd  # Import pandas for the table

# === Step 1: Add all station groups ===

station_groups = {}
connections = {}
from_stations = {}
group_names = []

print("\u27a1\ufe0f Step 1: Add all station groups (e.g., 'ICT', 'AOI', etc.)")
while True:
    name = input("Enter station group name (or 'done' to finish): ").strip().upper()
    if name == 'DONE':
        break
    if name in group_names:
        print("\u26a0\ufe0f Group already added.")
    else:
        group_names.append(name)

# === Step 2: Configure each station group ===

print("\n\u27a1\ufe0f Step 2: Configure each group.")
for name in group_names:
    print(f"\n--- Configuring Group: {name} ---")

    num_equipments = int(input(f"How many equipments in group '{name}'? "))
    station_groups[name] = {}

    for i in range(1, num_equipments + 1):
        eq_name = f"{name} - EQ{i}"
        while True:
            try:
                ct = float(input(f"Cycle time for {eq_name} (sec): "))
                break
            except ValueError:
                print("\u274c Enter a valid number.")
        station_groups[name][eq_name] = ct

    print("From which station group(s) does it receive products? (comma-separated or 'none')")
    print("Options: ['none'] +", group_names)
    from_input = input(f"{name} receives from: ").strip().upper()
    if from_input == 'NONE':
        from_stations[name] = []
    else:
        sources = [s.strip() for s in from_input.split(',') if s.strip() in group_names]
        while not sources:
            print("\u274c Invalid input. Try again.")
            from_input = input(f"{name} receives from: ").strip().upper()
            sources = [s.strip() for s in from_input.split(',') if s.strip() in group_names]
        from_stations[name] = sources

    print("To which group(s) does it send products? (comma-separated or 'stop')")
    print("Options: ['stop'] +", group_names)
    to_input = input(f"{name} sends to: ").strip().upper()
    if to_input == 'STOP':
        connections[name] = []
    else:
        targets = [s.strip() for s in to_input.split(',') if s.strip() in group_names]
        while not targets:
            print("\u274c Invalid input. Try again.")
            to_input = input(f"{name} sends to: ").strip().upper()
            if to_input == 'STOP':
                targets = []
            else:
                targets = [s.strip() for s in to_input.split(',') if s.strip() in group_names]
        connections[name] = targets

print("\n\u2705 Station group configuration complete.\n")

# === Simulation Class ===

class FactorySimulation:
    def __init__(self, env, station_groups, duration, connections, from_stations):
        self.env = env
        self.station_groups = station_groups
        self.connections = connections
        self.from_stations = from_stations
        self.duration = duration

        self.buffers = defaultdict(lambda: simpy.Store(self.env))
        self.resources = {eq: simpy.Resource(self.env, capacity=1)
                          for group in station_groups.values()
                          for eq in group}

        self.cycle_times = {eq: ct for group in station_groups.values() for eq, ct in group.items()}
        self.equipment_to_group = {eq: group for group, eqs in station_groups.items() for eq in eqs}

        self.throughput_in = defaultdict(int)
        self.throughput_out = defaultdict(int)
        self.wip_over_time = defaultdict(list)
        self.time_points = []
        self.board_id = 1

        # Utilization tracking
        self.equipment_busy_time = defaultdict(float)

        self.wip_interval = 5
        env.process(self.track_wip())

    def equipment_worker(self, eq):
        group = self.equipment_to_group[eq]
        while True:
            board = yield self.buffers[group].get()
            self.throughput_in[eq] += 1
            print(f"{self.env.now:.2f}s - {board} enters {eq}")

            with self.resources[eq].request() as req:
                yield req
                start = self.env.now
                yield self.env.timeout(self.cycle_times[eq])
                end = self.env.now
                self.equipment_busy_time[eq] += (end - start)

            self.throughput_out[eq] += 1
            print(f"{self.env.now:.2f}s - {board} exits {eq}")

            for tgt_group in self.connections.get(group, []):
                yield self.buffers[tgt_group].put(board)

    def feeder(self):
        start_groups = [g for g in self.station_groups if not self.from_stations.get(g)]
        while self.env.now < self.duration:
            for g in start_groups:
                board_name = f"Board-{self.board_id:03d}"
                self.board_id += 1
                yield self.buffers[g].put(board_name)
            yield self.env.timeout(1)

    def track_wip(self):
        while self.env.now < self.duration:
            self.time_points.append(self.env.now)
            for group in self.station_groups:
                prev_out = sum(self.throughput_out[eq] for g in self.from_stations.get(group, []) for eq in self.station_groups[g])
                curr_in = sum(self.throughput_in[eq] for eq in self.station_groups[group])
                wip = max(0, prev_out - curr_in) if self.from_stations.get(group) else 0
                self.wip_over_time[group].append(wip)
            yield self.env.timeout(self.wip_interval)

    def run(self):
        for group in self.station_groups:
            for eq in self.station_groups[group]:
                self.env.process(self.equipment_worker(eq))
        self.env.process(self.feeder())

# === Run Simulation ===

SIM_TIME = float(input("Enter simulation time (seconds): "))
env = simpy.Environment()
sim = FactorySimulation(env, station_groups, SIM_TIME, connections, from_stations)
sim.run()
env.run(until=SIM_TIME)

# === Reporting ===

groups = list(station_groups.keys())
agg = defaultdict(lambda: {'in': 0, 'out': 0, 'wip': 0})
for group in station_groups:
    eqs = station_groups[group]
    agg[group]['in'] = sum(sim.throughput_in[eq] for eq in eqs)
    agg[group]['out'] = sum(sim.throughput_out[eq] for eq in eqs)
    prev_out = sum(sim.throughput_out[eq] for g in from_stations.get(group, []) for eq in station_groups[g])
    curr_in = agg[group]['in']
    agg[group]['wip'] = max(0, prev_out - curr_in)

# Utilization calculation
utilization_vals = []
for group in groups:
    eqs = station_groups[group]
    total_busy = sum(sim.equipment_busy_time[eq] for eq in eqs)
    utilization = total_busy / (len(eqs) * SIM_TIME)
    utilization_vals.append(utilization)

# === Bar Chart for Throughput and WIP ===

in_vals = [agg[g]['in'] for g in groups]
out_vals = [agg[g]['out'] for g in groups]
wip_vals = [agg[g]['wip'] for g in groups]

x = range(len(groups))
bw = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x, in_vals, width=bw, label='In', color='skyblue')
bars2 = ax.bar([i + bw for i in x], out_vals, width=bw, label='Out', color='lightgreen')
bars3 = ax.bar([i + 2 * bw for i in x], wip_vals, width=bw, label='WIP', color='tomato')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Station Group')
ax.set_ylabel('Count')
ax.set_title(f'Throughput & WIP After {SIM_TIME:.0f} Seconds')
ax.set_xticks([i + bw for i in x])
ax.set_xticklabels(groups)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === WIP Over Time Charts ===

for group in groups:
    plt.figure(figsize=(8, 4))
    plt.plot(sim.time_points[:len(sim.wip_over_time[group])], sim.wip_over_time[group], marker='o')
    plt.title(f'WIP Over Time - {group}')
    plt.xlabel('Time (s)')
    plt.ylabel('WIP')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# === Process Layout Diagram ===

G = nx.DiGraph()
for group in station_groups:
    G.add_node(group)
for group, to_groups in connections.items():
    for tgt in to_groups:
        G.add_edge(group, tgt)

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, arrows=True)
plt.title("Production Line Layout")
plt.tight_layout()
plt.show()

# === Final Table with Utilization ===

df = pd.DataFrame({
    'Station Group': groups,
    'In Throughput': in_vals,
    'Out Throughput': out_vals,
    'WIP': wip_vals,
    'Utilization (%)': [f"{u * 100:.1f}%" for u in utilization_vals]
})

from IPython.display import display
display(df)
