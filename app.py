# === PART 1 ===

import streamlit as st
import simpy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd
from io import BytesIO
import zipfile
from graphviz import Digraph

# === Authentication Setup ===
USERNAME = "aksh.fii"
PASSWORD = "foxy123"
st.set_page_config(layout="wide")

def reset_all(skip_password=False):
    for key in list(st.session_state.keys()):
        if key not in ("authenticated", "password_attempted", "skip_password"):
            del st.session_state[key]
    st.session_state.authenticated = skip_password
    st.session_state.password_attempted = skip_password
    st.session_state.skip_password = skip_password

if "skip_password" not in st.session_state:
    st.session_state.skip_password = False

if not st.session_state.skip_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.password_attempted = False

    if not st.session_state.authenticated:
        user = st.text_input("👤 Username")
        pwd = st.text_input("🔐 Password", type="password")

        if user and pwd and not st.session_state.password_attempted:
            st.session_state.password_attempted = True
            if user == USERNAME and pwd == PASSWORD:
                st.session_state.authenticated = True
            else:
                st.error("❌ Incorrect credentials")
                st.stop()

        if not st.session_state.authenticated:
            st.stop()

st.title("🛠️ Production Line Simulation App (Discrete Event Simulation)")

if st.button("🔄 Reset and Skip Password Next Time"):
    reset_all(skip_password=True)
    st.experimental_rerun()

# === Layout ===
col1, col2 = st.columns(2)

# === Step 1: Define Stations ===
with col1:
    st.header("Step 1: Station Groups")
    num_groups = st.number_input("Number of Station Groups", min_value=1, step=1, value=2)

    if "group_names" not in st.session_state or len(st.session_state.group_names) != num_groups:
        st.session_state.group_names = [""] * num_groups
        st.session_state.station_groups = {}
        st.session_state.conveyor_flags = {}
        st.session_state.entry_intervals = {}

    for i in range(num_groups):
        name = st.text_input(f"Group {i+1} Name", key=f"name_{i}").strip().upper()
        st.session_state.group_names[i] = name

        if name:
            eq_count = st.number_input(f"Number of Equipment in {name}", 1, key=f"eq_count_{i}")
            eq_times = [
                st.number_input(f"Cycle Time for {name} - EQ{j+1} (sec)", 0.1, key=f"ct_{i}_{j+1}")
                for j in range(eq_count)
            ]
            st.session_state.station_groups[name] = {
                f"{name} - EQ{j+1}": ct for j, ct in enumerate(eq_times)
            }

            is_conveyor = st.selectbox(
                f"Does {name} behave like a conveyor?",
                options=["No", "Yes"],
                key=f"conveyor_{i}"
            )
            st.session_state.conveyor_flags[name] = (is_conveyor == "Yes")

            if is_conveyor == "Yes":
                interval = st.number_input(f"Entry interval for {name} (sec)", 0.0, step=0.1, key=f"entryint_{i}")
                st.session_state.entry_intervals[name] = interval
            else:
                st.session_state.entry_intervals[name] = None
# === Run Simulation ===
if st.session_state.get("simulate"):
    station_groups = st.session_state.station_groups
    from_stations = st.session_state.from_stations
    connections = st.session_state.connections
    sim_time = st.session_state.sim_time
    conveyor_config = st.session_state.conveyor_config
    valid_groups = {g: eqs for g, eqs in station_groups.items() if g}

    class FactorySimulation:
        def __init__(self, env, station_groups, duration, connections, from_stations, conveyor_config):
            self.env = env
            self.station_groups = station_groups
            self.connections = connections
            self.from_stations = from_stations
            self.duration = duration
            self.conveyor_config = conveyor_config

            self.buffers = defaultdict(lambda: simpy.Store(env))
            self.resources = {eq: simpy.Resource(env, capacity=1)
                              for group in station_groups.values() for eq in group}
            self.cycle_times = {eq: ct for group in station_groups.values() for eq, ct in group.items()}
            self.equipment_to_group = {eq: group for group, eqs in station_groups.items() for eq in eqs}
            self.throughput_in = defaultdict(int)
            self.throughput_out = defaultdict(int)
            self.wip_over_time = defaultdict(list)
            self.time_points = []
            self.equipment_busy_time = defaultdict(float)
            self.board_id = 1
            self.wip_interval = 5
            env.process(self.track_wip())

        def equipment_worker(self, eq):
            group = self.equipment_to_group[eq]
            is_conveyor = self.conveyor_config.get(group, {}).get('is_conveyor', False)
            entry_interval = self.conveyor_config.get(group, {}).get('entry_interval', 0.0)

            while True:
                board = yield self.buffers[group].get()
                self.throughput_in[eq] += 1

                with self.resources[eq].request() as req:
                    yield req
                    start = self.env.now
                    yield self.env.timeout(self.cycle_times[eq])
                    end = self.env.now
                    self.equipment_busy_time[eq] += (end - start)

                self.throughput_out[eq] += 1
                for tgt in self.connections.get(group, []):
                    yield self.buffers[tgt].put(board)

                # Only delay between board arrivals if station behaves like a conveyor
                if is_conveyor and entry_interval > 0:
                    yield self.env.timeout(entry_interval)

        def feeder(self):
            start_groups = [g for g in self.station_groups if not self.from_stations.get(g)]
            while self.env.now < self.duration:
                for g in start_groups:
                    board = f"Board-{self.board_id:03d}"
                    self.board_id += 1
                    yield self.buffers[g].put(board)
                yield self.env.timeout(1)

        def track_wip(self):
            while self.env.now < self.duration:
                self.time_points.append(self.env.now)
                for group in self.station_groups:
                    prev_out = sum(
                        sim.throughput_out[eq] for g in self.from_stations.get(group, [])
                        for eq in self.station_groups.get(g, [])
                    )
                    curr_in = sum(sim.throughput_in[eq] for eq in self.station_groups[group])
                    wip = max(0, prev_out - curr_in) if self.from_stations.get(group) else 0
                    self.wip_over_time[group].append(wip)
                yield self.env.timeout(self.wip_interval)

        def run(self):
            for group in self.station_groups:
                for eq in self.station_groups[group]:
                    self.env.process(self.equipment_worker(eq))
            self.env.process(self.feeder())

    # === Create and run the SimPy environment ===
    env = simpy.Environment()
    sim = FactorySimulation(env, valid_groups, sim_time, connections, from_stations, conveyor_config)
    sim.run()
    env.run(until=sim_time)
# === Check for Required Variables ===
if 'valid_groups' not in locals() or 'sim' not in locals() or 'from_stations' not in locals() or 'sim_time' not in locals():
    st.warning("❗ Run the simulation first to generate results.")
    st.stop()

# === Results Summary ===
st.markdown("---")
st.subheader("📊 Simulation Results Summary")

groups = list(valid_groups.keys())
agg = defaultdict(lambda: {'in': 0, 'out': 0, 'busy': 0, 'count': 0, 'cycle_times': [], 'wip': 0})

for group in groups:
    eqs = valid_groups[group]
    for eq in eqs:
        agg[group]['in'] += sim.throughput_in.get(eq, 0)
        agg[group]['out'] += sim.throughput_out.get(eq, 0)
        agg[group]['busy'] += sim.equipment_busy_time.get(eq, 0)
        agg[group]['cycle_times'].append(sim.cycle_times.get(eq, 0))
        agg[group]['count'] += 1

    prev_out = sum(sim.throughput_out.get(eq, 0) for g in from_stations.get(group, []) for eq in valid_groups.get(g, []))
    curr_in = agg[group]['in']
    agg[group]['wip'] = max(0, prev_out - curr_in)

# Prepare DataFrame
df = pd.DataFrame([{
    "Station Group": g,
    "Boards In": agg[g]['in'],
    "Boards Out": agg[g]['out'],
    "WIP": agg[g]['wip'],
    "Number of Equipment": agg[g]['count'],
    "Cycle Times (sec)": ", ".join(str(round(ct, 1)) for ct in agg[g]['cycle_times']),
    "Utilization (%)": round((agg[g]['busy'] / (sim_time * agg[g]['count'])) * 100, 1) if agg[g]['count'] > 0 else 0
} for g in groups])

st.dataframe(df, use_container_width=True)

# Excel download
towrite = BytesIO()
df.to_excel(towrite, index=False, sheet_name="Summary")
towrite.seek(0)
st.download_button("📥 Download Summary Excel", data=towrite, file_name="simulation_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# === Charts: Throughput & WIP ===
st.subheader("📈 Throughput & WIP")
fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(groups))
bw = 0.25
in_vals = [agg[g]['in'] for g in groups]
out_vals = [agg[g]['out'] for g in groups]
wip_vals = [agg[g]['wip'] for g in groups]

bars1 = ax.bar(x, in_vals, width=bw, label='In', color='skyblue')
bars2 = ax.bar([i + bw for i in x], out_vals, width=bw, label='Out', color='lightgreen')
bars3 = ax.bar([i + 2 * bw for i in x], wip_vals, width=bw, label='WIP', color='salmon')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=8)

ax.set_xticks([i + bw for i in x])
ax.set_xticklabels(groups)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# Save chart as PNG
buf = BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
st.download_button("📥 Download Chart (PNG)", data=buf, file_name="throughput_wip.png", mime="image/png")

# === WIP Over Time Plots ===
st.subheader("📉 WIP Over Time per Station Group")
fig, axs = plt.subplots(len(groups), 1, figsize=(8, 3 * len(groups)), sharex=True)
if len(groups) == 1:
    axs = [axs]

img_buffers = {}
for ax, group in zip(axs, groups):
    ax.plot(sim.time_points, sim.wip_over_time[group], marker='o')
    ax.set_title(f"WIP Over Time: {group}")
    ax.set_ylabel("WIP (units)")
    ax.grid(True)

    buf = BytesIO()
    fig_single, ax_single = plt.subplots()
    ax_single.plot(sim.time_points, sim.wip_over_time[group], marker='o')
    ax_single.set_title(f"WIP Over Time: {group}")
    ax_single.set_ylabel("WIP (units)")
    ax_single.set_xlabel("Time (seconds)")
    ax_single.grid(True)
    fig_single.savefig(buf, format="png")
    plt.close(fig_single)
    buf.seek(0)
    img_buffers[group] = buf

axs[-1].set_xlabel("Time (seconds)")
st.pyplot(fig)

# Chart downloads
for group, buf in img_buffers.items():
    st.download_button(f"📥 Download WIP Chart PNG - {group}", data=buf, file_name=f"WIP_{group}.png", mime="image/png")

# === Layout Diagram (Linear) ===
st.subheader("🗌 Production Line Layout")
if groups:
    try:
        dot = Digraph()
        dot.attr(rankdir="LR", size="8")
        for group in groups:
            dot.node(group, shape="box", style="filled", fillcolor="lightblue")
        for src, tgts in connections.items():
            for tgt in tgts:
                dot.edge(src, tgt)
        st.graphviz_chart(dot.source)
    except Exception as e:
        st.warning(f"Graphviz layout failed: {e}")
else:
    st.info("ℹ️ Run the simulation to view layout diagram.")

# === Bottleneck Suggestion ===
st.subheader("💡 Bottleneck Suggestion")
min_out = float('inf')
bottleneck_group = None
for group in groups:
    out = agg[group]['out']
    if out < min_out:
        min_out = out
        bottleneck_group = group

if bottleneck_group:
    eqs = valid_groups[bottleneck_group]
    avg_ct = sum(sim.cycle_times[eq] for eq in eqs) / len(eqs)
    base_out = agg[groups[-1]]['out']
    eq_count = len(eqs)
    new_out_bottleneck = (agg[bottleneck_group]['out'] / eq_count) * (eq_count + 1)
    estimated_final_out = base_out + (new_out_bottleneck - agg[bottleneck_group]['out']) * 0.7

    delta_b = round(new_out_bottleneck - agg[bottleneck_group]['out'])
    delta_final = round(estimated_final_out - base_out)

    st.markdown(
        f"If you **add 1 more equipment** to **{bottleneck_group}** (avg CT = {round(avg_ct,1)}s), "
        f"output may increase by **{delta_b} boards** in that group, and **{delta_final}** overall."
    )

# === ZIP Download ===
if st.button("📦 Download All Charts and Tables as ZIP"):
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w") as zf:
        if towrite.getbuffer().nbytes > 0:
            zf.writestr("simulation_summary.xlsx", towrite.getvalue())
        for group, buf in img_buffers.items():
            zf.writestr(f"WIP_{group}.png", buf.getvalue())
        zf.writestr("throughput_wip.png", buf.getvalue())
    mem_zip.seek(0)
    st.download_button(
        "📅 Download All as ZIP",
        data=mem_zip,
        file_name="simulation_results.zip",
        mime="application/zip"
    )
