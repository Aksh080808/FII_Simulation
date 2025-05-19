import streamlit as st
import simpy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd
from io import BytesIO
import zipfile

# ===============================
# Password and username protection
# ===============================
USERNAME = "aksh.fii"
PASSWORD = "foxy123"

st.set_page_config(layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.password_attempted = False

if not st.session_state.authenticated:
    user = st.text_input("👤 Enter username")
    pwd = st.text_input("🔒 Enter password", type="password")

    if user and pwd and not st.session_state.password_attempted:
        st.session_state.password_attempted = True
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.authenticated = True
        else:
            st.error("❌ Incorrect username or password.")
            st.stop()

    if not st.session_state.authenticated:
        st.stop()

st.title("🛠️ Production Line Simulation App (Discrete Event Simulation)")

# ===============================
# Layout: Step 1 and Step 2 side-by-side
# ===============================
col1, col2 = st.columns(2)

# === Step 1: Define Station Groups ===
with col1:
    st.header("Step 1: Define Station Groups")
    num_groups = st.number_input("Number of Station Groups", min_value=1, step=1, value=2)

    # Initialize or reset group names and station groups based on num_groups
    if "group_names" not in st.session_state or len(st.session_state.group_names) != num_groups:
        st.session_state.group_names = [""] * num_groups
        st.session_state.station_groups = {}

    for i in range(num_groups):
        name = st.text_input(f"Name of Station Group {i+1}", key=f"name_{i}").strip().upper()
        st.session_state.group_names[i] = name

        if name:
            eq_count = st.number_input(f"Number of Equipment in {name}", min_value=1, step=1, key=f"eq_count_{i}")
            eq_times = []
            for j in range(1, eq_count + 1):
                ct = st.number_input(f"Cycle time for {name} - EQ{j} (seconds)", min_value=0.1, step=0.1, key=f"ct_{i}_{j}")
                eq_times.append(ct)

            # Store cycle times per equipment under the station group name
            st.session_state.station_groups[name] = {}
            for j, ct in enumerate(eq_times, start=1):
                eq_name = f"{name} - EQ{j}"
                st.session_state.station_groups[name][eq_name] = ct

# === Step 2: Connect Stations ===
with col2:
    st.header("Step 2: 🔗 Connect Stations")

    # Initialize connection dictionaries if not exist
    if "from_stations" not in st.session_state:
        st.session_state.from_stations = {}
    if "connections" not in st.session_state:
        st.session_state.connections = {}

    # For each group, select its input and output connections
    for i, name in enumerate(st.session_state.group_names):
        if not name:
            continue
        with st.expander(f"{name} Connections"):
            # 'from' stations: multiselect with option START to indicate source of line
            from_options = ['START'] + [g for g in st.session_state.group_names if g and g != name]
            from_selected = st.multiselect(f"{name} receives from:", from_options, key=f"from_{i}")
            # If 'START' is selected, it means no upstream groups
            if 'START' in from_selected:
                st.session_state.from_stations[name] = []
            else:
                st.session_state.from_stations[name] = from_selected

            # 'to' stations: multiselect with option STOP to indicate line end
            to_options = ['STOP'] + [g for g in st.session_state.group_names if g and g != name]
            to_selected = st.multiselect(f"{name} sends to:", to_options, key=f"to_{i}")
            if 'STOP' in to_selected:
                st.session_state.connections[name] = []
            else:
                st.session_state.connections[name] = to_selected

# === Step 3: Simulation Duration ===
st.markdown("---")
st.header("Step 3: ⏱️ Enter Simulation Duration")
sim_time = st.number_input("Simulation Time (seconds)", min_value=10, value=100, step=10)

if st.button("▶️ Run Simulation"):
    # Reset simulation flag to trigger run
    st.session_state.simulate = True
    st.session_state.sim_time = sim_time

# === Simulation and Results Display ===
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

            # Buffers per station group
            self.buffers = defaultdict(lambda: simpy.Store(self.env))

            # Resources per equipment (capacity=1)
            self.resources = {
                eq: simpy.Resource(self.env, capacity=1)
                for group in station_groups.values() for eq in group
            }

            # Cycle times per equipment
            self.cycle_times = {eq: ct for group in station_groups.values() for eq, ct in group.items()}
            # Map equipment back to their station group
            self.equipment_to_group = {eq: group for group, eqs in station_groups.items() for eq in eqs}

            # Throughput counts
            self.throughput_in = defaultdict(int)
            self.throughput_out = defaultdict(int)

            # WIP over time tracking
            self.wip_over_time = defaultdict(list)
            self.time_points = []

            self.board_id = 1

            # Track busy time for utilization
            self.equipment_busy_time = defaultdict(float)

            # Interval to track WIP
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

                # Put board into buffers of connected downstream groups
                for tgt_group in self.connections.get(group, []):
                    yield self.buffers[tgt_group].put(board)

        def feeder(self):
            # Identify groups without any upstream connections
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
                    upstream_groups = self.from_stations.get(group, [])
                    if not upstream_groups:
                        # If no upstream groups, WIP is zero (source)
                        self.wip_over_time[group].append(0)
                        continue
                    # Calculate WIP = upstream total output - current total input
                    prev_out = sum(
                        sum(self.throughput_out[eq] for eq in self.station_groups[up_g])
                        for up_g in upstream_groups if up_g in self.station_groups
                    )
                    curr_in = sum(self.throughput_in[eq] for eq in self.station_groups[group])
                    wip = max(0, prev_out - curr_in)
                    self.wip_over_time[group].append(wip)
                yield self.env.timeout(self.wip_interval)

        def run(self):
            # Start equipment processes
            for group in self.station_groups:
                for eq in self.station_groups[group]:
                    self.env.process(self.equipment_worker(eq))
            # Start feeder process
            self.env.process(self.feeder())

    env = simpy.Environment()
    sim = FactorySimulation(env, station_groups, sim_time, connections, from_stations)
    sim.run()
    env.run(until=sim_time)

    st.markdown("---")
    st.subheader("📊 Simulation Results Summary")

    groups = list(station_groups.keys())
    agg = defaultdict(lambda: {'in': 0, 'out': 0, 'wip': 0})
    for group in groups:
        eqs = station_groups[group]
        agg[group]['in'] = sum(sim.throughput_in[eq] for eq in eqs)
        agg[group]['out'] = sum(sim.throughput_out[eq] for eq in eqs)

        # WIP calculation based on upstream groups
        upstream_groups = from_stations.get(group, [])
        if not upstream_groups:
            agg[group]['wip'] = 0
        else:
            prev_out = sum(
                sum(sim.throughput_out[eq] for eq in station_groups[up_g])
                for up_g in upstream_groups if up_g in station_groups
            )
            curr_in = agg[group]['in']
            agg[group]['wip'] = max(0, prev_out - curr_in)

    utilization_vals = []
    for group in groups:
        eqs = station_groups[group]
        total_busy = sum(sim.equipment_busy_time[eq] for eq in eqs)
        utilization = total_busy / (len(eqs) * sim_time)
        utilization_vals.append(utilization)

    cycle_times_str = []
    num_equip_list = []
    for group in groups:
        eqs = station_groups[group]
        num_equip_list.append(len(eqs))
        cts = [str(ct) for ct in eqs.values()]
        cycle_times_str.append(", ".join(cts))

    df = pd.DataFrame({
        'Station Group': groups,
        'Number of Equipment': num_equip_list,
        'Cycle Times (sec)': cycle_times_str,
        'In Throughput': [agg[g]['in'] for g in groups],
        'Out Throughput': [agg[g]['out'] for g in groups],
        'WIP': [agg[g]['wip'] for g in groups],
        'Utilization': [f"{u*100:.1f}%" for u in utilization_vals],
    })

    st.dataframe(df, use_container_width=True)

    # === WIP over time plots per group ===
    st.subheader("📈 WIP Over Time per Station Group")
    fig, axs = plt.subplots(len(groups), 1, figsize=(8, 3 * len(groups)), sharex=True)
    if len(groups) == 1:
        axs = [axs]

    for ax, group in zip(axs, groups):
        ax.plot(sim.time_points, sim.wip_over_time[group], marker='o')
        ax.set_title(f"WIP Over Time: {group}")
        ax.set_ylabel("WIP (units)")
        ax.grid(True)
    axs[-1].set_xlabel("Time (seconds)")

    st.pyplot(fig)

    # === Process Layout Diagram with NetworkX ===
    st.subheader("🗺️ Production Line Layout")

    G = nx.DiGraph()
    for g in groups:
        G.add_node(g)

    for src, tgt_list in connections.items():
        for tgt in tgt_list:
            if tgt in groups:
                G.add_edge(src, tgt)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20)
    plt.axis('off')
    st.pyplot(plt.gcf())

    # === Export charts as zip ===
    if st.button("📥 Download Charts as ZIP"):
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w") as zf:
            # Save WIP plots
            for group in groups:
                plt.figure()
                plt.plot(sim.time_points, sim.wip_over_time[group], marker='o')
                plt.title(f"WIP Over Time: {group}")
                plt.xlabel("Time (seconds)")
                plt.ylabel("WIP (units)")
                plt.grid(True)
                img_buf = BytesIO()
                plt.savefig(img_buf, format='png')
                plt.close()
                img_buf.seek(0)
                zf.writestr(f"WIP_{group}.png", img_buf.read())

            # Save layout plot
            plt.figure(figsize=(8, 6))
            nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=2000)
            nx.draw_networkx_labels(G, pos, font_size=12)
            nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20)
            plt.axis('off')
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close()
            img_buf.seek(0)
            zf.writestr("Production_Line_Layout.png", img_buf.read())

        mem_zip.seek(0)
        st.download_button(
            label="Download ZIP file",
            data=mem_zip,
            file_name="production_line_charts.zip",
            mime="application/zip"
        )
