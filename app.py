import streamlit as st
import simpy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd
from io import BytesIO
import zipfile

# ===============================
# 🔐 Password protection
# ===============================
import streamlit as st

PASSWORD = "foxy123"
st.set_page_config(layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("🔒 Enter password to start", type="password")
    if pwd == PASSWORD:
        st.session_state.authenticated = True
        st.experimental_rerun()  # Ensures a clean refresh after login
    else:
        st.stop()

st.title("🛠️ Factory Simulation App (SimPy + Streamlit)")

# ===============================
# Layout: Step 1 and Step 2 side-by-side
# ===============================
col1, col2 = st.columns(2)

# === Step 1: Define Station Groups ===
with col1:
    st.header("Step 1: Define Station Groups")
    num_groups = st.number_input("Number of Station Groups", min_value=1, step=1, value=2)

    if "group_names" not in st.session_state:
        st.session_state.group_names = []
        st.session_state.station_groups = {}

    for i in range(num_groups):
        with st.expander(f"Configure Group {i+1}"):
            name = st.text_input(f"Name of Station Group {i+1}", key=f"name_{i}").strip().upper()
            if not name:
                continue
            if name not in st.session_state.group_names:
                st.session_state.group_names.append(name)
                eq_count = st.number_input(f"Number of Equipment in {name}", min_value=1, step=1, key=f"eq_count_{i}")
                st.session_state.station_groups[name] = {}
                for j in range(1, eq_count + 1):
                    eq_name = f"{name} - EQ{j}"
                    ct = st.number_input(f"Cycle time for {eq_name} (sec)", min_value=0.1, step=0.1, key=f"ct_{i}_{j}")
                    st.session_state.station_groups[name][eq_name] = ct

# === Step 2: Connect Stations ===
with col2:
    st.header("Step 2: Connect Stations")
    if "from_stations" not in st.session_state:
        st.session_state.from_stations = {}
    if "connections" not in st.session_state:
        st.session_state.connections = {}

    for i, name in enumerate(st.session_state.group_names):
        with st.expander(f"{name} Connections"):
            from_group = st.multiselect(f"{name} receives from:", ['START'] + st.session_state.group_names, key=f"from_{i}")
            st.session_state.from_stations[name] = [] if 'START' in from_group else from_group

            to_group = st.multiselect(f"{name} sends to:", ['STOP'] + st.session_state.group_names, key=f"to_{i}")
            st.session_state.connections[name] = [] if 'STOP' in to_group else to_group

# === Step 3: Duration ===
st.markdown("---")
st.header("Step 3: Enter Simulation Duration")
sim_time = st.number_input("Simulation Time (seconds)", min_value=10, value=100, step=10)
if st.button("▶️ Run Simulation"):
    st.session_state.simulate = True
    st.session_state.sim_time = sim_time
# === Run simulation and display results ===
if st.session_state.get("simulate"):
    station_groups = st.session_state.station_groups
    from_stations = st.session_state.from_stations
    connections = st.session_state.connections
    sim_time = st.session_state.sim_time

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
            self.equipment_busy_time = defaultdict(float)
            self.wip_interval = 5
            env.process(self.track_wip())

        def equipment_worker(self, eq):
            group = self.equipment_to_group[eq]
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

    # Run the simulation
    env = simpy.Environment()
    sim = FactorySimulation(env, station_groups, sim_time, connections, from_stations)
    sim.run()
    env.run(until=sim_time)

    # === Reporting ===
    st.markdown("---")
    st.subheader("📊 Simulation Results Summary")

    groups = list(station_groups.keys())
    agg = defaultdict(lambda: {'in': 0, 'out': 0, 'wip': 0})
    for group in station_groups:
        eqs = station_groups[group]
        agg[group]['in'] = sum(sim.throughput_in[eq] for eq in eqs)
        agg[group]['out'] = sum(sim.throughput_out[eq] for eq in eqs)
        prev_out = sum(sim.throughput_out[eq] for g in from_stations.get(group, []) for eq in station_groups[g])
        curr_in = agg[group]['in']
        agg[group]['wip'] = max(0, prev_out - curr_in)

    utilization_vals = []
    for group in groups:
        eqs = station_groups[group]
        total_busy = sum(sim.equipment_busy_time[eq] for eq in eqs)
        utilization = total_busy / (len(eqs) * sim_time)
        utilization_vals.append(utilization)

    df = pd.DataFrame({
        'Station Group': groups,
        'In Throughput': [agg[g]['in'] for g in groups],
        'Out Throughput': [agg[g]['out'] for g in groups],
        'WIP': [agg[g]['wip'] for g in groups],
        'Utilization (%)': [f"{u * 100:.1f}%" for u in utilization_vals]
    })

    st.dataframe(df)
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    st.download_button("📥 Download Table (Excel)", data=excel_buffer.getvalue(), file_name="simulation_results.xlsx")
    # === Charts ===
    chart_buffers = {}

    st.subheader("📈 Throughput & WIP")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(groups))
    bw = 0.25
    ax.bar(x, [agg[g]['in'] for g in groups], width=bw, label='In', color='skyblue')
    ax.bar([i + bw for i in x], [agg[g]['out'] for g in groups], width=bw, label='Out', color='lightgreen')
    ax.bar([i + 2 * bw for i in x], [agg[g]['wip'] for g in groups], width=bw, label='WIP', color='salmon')
    ax.set_xticks([i + bw for i in x])
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    chart_buffers["throughput_wip.png"] = buf
    st.download_button("📥 Download Chart (PNG)", data=buf.getvalue(), file_name="throughput_wip.png")

    st.subheader("📉 WIP Over Time")
    for group in groups:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sim.time_points[:len(sim.wip_over_time[group])], sim.wip_over_time[group], marker='o')
        ax.set_title(f"WIP Over Time - {group}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("WIP")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        chart_buffers[f"{group}_wip.png"] = buf
        st.download_button(f"📥 Download {group} Chart", data=buf.getvalue(), file_name=f"{group}_wip.png")

    # === Process Layout Diagram ===
    st.subheader("🔁 Process Layout Diagram")
    G = nx.DiGraph()
    for g in station_groups:
        G.add_node(g)
    for src, tgts in connections.items():
        for tgt in tgts:
            G.add_edge(src, tgt)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=2000)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=20, edge_color='black')
    st.pyplot(fig)

    # === ZIP Download for Charts ===
    st.subheader("📦 Download All Charts as ZIP")
    filename = st.text_input("Enter ZIP filename", value="simulation_charts")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        for fname, fbuf in chart_buffers.items():
            zf.writestr(fname, fbuf.getvalue())
    st.download_button("📥 Download ZIP", data=zip_buffer.getvalue(), file_name=f"{filename}.zip")

    # === Reset App ===
    st.markdown("---")
    if st.button("🔄 Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
