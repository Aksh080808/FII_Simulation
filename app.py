import streamlit as st
import simpy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd
from io import BytesIO
import zipfile

# ===== PASSWORD & RESET LOGIC =====
USERNAME = "aksh.fii"
PASSWORD = "foxy123"

st.set_page_config(layout="wide")

def reset_all(skip_password=False):
    # Clear all inputs but keep skip_password flag for next login
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

# Reset Button - clears all inputs & skips password on reload
if st.button("🔄 Reset and Skip Password Next Time"):
    reset_all(skip_password=True)
    st.experimental_rerun()

# ===== Layout: Step 1 + Step 2 side by side =====
col1, col2 = st.columns(2)

# ===== Step 1: Define Station Groups =====
with col1:
    st.header("Step 1: Define Station Groups")
    num_groups = st.number_input("Number of Station Groups", min_value=1, step=1, value=2)

    # Reset group names and station_groups if num_groups changes
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

            # Store cycle times per equipment
            st.session_state.station_groups[name] = {}
            for j, ct in enumerate(eq_times, start=1):
                eq_name = f"{name} - EQ{j}"
                st.session_state.station_groups[name][eq_name] = ct

# ===== Step 2: Connect Stations =====
with col2:
    st.header("Step 2: 🔗 Connect Stations")

    # Init connection dicts if missing
    if "from_stations" not in st.session_state:
        st.session_state.from_stations = {}
    if "connections" not in st.session_state:
        st.session_state.connections = {}

    # For each group, select its inputs and outputs
    for i, name in enumerate(st.session_state.group_names):
        if not name:
            continue
        with st.expander(f"{name} Connections"):
            from_options = ['START'] + [g for g in st.session_state.group_names if g and g != name]
            from_selected = st.multiselect(f"{name} receives from:", from_options, key=f"from_{i}")
            if 'START' in from_selected:
                st.session_state.from_stations[name] = []
            else:
                st.session_state.from_stations[name] = from_selected

            to_options = ['STOP'] + [g for g in st.session_state.group_names if g and g != name]
            to_selected = st.multiselect(f"{name} sends to:", to_options, key=f"to_{i}")
            if 'STOP' in to_selected:
                st.session_state.connections[name] = []
            else:
                st.session_state.connections[name] = to_selected

# ===== Step 3: Simulation Duration =====
st.markdown("---")
st.header("Step 3: ⏱️ Enter Simulation Duration")
sim_time = st.number_input("Simulation Time (seconds)", min_value=10, value=100, step=10)

# Clear previous sim results if inputs change
if "simulate" not in st.session_state:
    st.session_state.simulate = False

if st.button("▶️ Run Simulation"):
    st.session_state.simulate = True
    st.session_state.sim_time = sim_time

# ===== Simulation Class & Logic =====
if st.session_state.simulate:

    station_groups = st.session_state.station_groups
    from_stations = st.session_state.from_stations
    connections = st.session_state.connections
    sim_time = st.session_state.sim_time

    # Filter out empty or deleted stations just in case
    valid_groups = {g: eqs for g, eqs in station_groups.items() if g}

    class FactorySimulation:
        def __init__(self, env, station_groups, duration, connections, from_stations):
            self.env = env
            self.station_groups = station_groups
            self.connections = connections
            self.from_stations = from_stations
            self.duration = duration

            self.buffers = defaultdict(lambda: simpy.Store(self.env))
            self.resources = {
                eq: simpy.Resource(self.env, capacity=1)
                for group in station_groups.values() for eq in group
            }
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
                    upstream_groups = self.from_stations.get(group, [])
                    if not upstream_groups:
                        self.wip_over_time[group].append(0)
                        continue
                    prev_out = sum(
                        sum(self.throughput_out[eq] for eq in self.station_groups[up_g])
                        for up_g in upstream_groups if up_g in self.station_groups
                    )
                    curr_in = sum(self.throughput_in[eq] for eq in self.station_groups[group])
                    wip = max(0, prev_out - curr_in)
                    self.wip_over_time[group].append(wip)
                yield self.env.timeout(self.wip_interval)

        def run(self):
            for group in self.station_groups:
                for eq in self.station_groups[group]:
                    self.env.process(self.equipment_worker(eq))
            self.env.process(self.feeder())

    env = simpy.Environment()
    sim = FactorySimulation(env, valid_groups, sim_time, connections, from_stations)
    sim.run()
    env.run(until=sim_time)
    st.markdown("---")
    st.subheader("📊 Simulation Results Summary")

    groups = list(valid_groups.keys())
    agg = defaultdict(lambda: {'in': 0, 'out': 0, 'busy': 0, 'count': 0})

    for group in groups:
        eqs = valid_groups[group]
        for eq in eqs:
            agg[group]['in'] += sim.throughput_in[eq]
            agg[group]['out'] += sim.throughput_out[eq]
            agg[group]['busy'] += sim.equipment_busy_time[eq]
            agg[group]['count'] += 1

    data = []
    for group in groups:
        utilization = agg[group]['busy'] / (sim_time * agg[group]['count']) if agg[group]['count'] > 0 else 0
        data.append({
            "Station Group": group,
            "Boards In": agg[group]['in'],
            "Boards Out": agg[group]['out'],
            "Utilization (%)": round(utilization * 100, 1),
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Excel download after table
    towrite = BytesIO()
    df.to_excel(towrite, index=False, sheet_name="Summary")
    towrite.seek(0)
    st.download_button("📥 Download Summary Excel", data=towrite, file_name="simulation_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # === WIP Over Time Plots ===
    st.subheader("📈 WIP Over Time per Station Group")
    fig, axs = plt.subplots(len(groups), 1, figsize=(8, 3 * len(groups)), sharex=True)
    if len(groups) == 1:
        axs = [axs]

    img_buffers = {}

    for ax, group in zip(axs, groups):
        ax.plot(sim.time_points, sim.wip_over_time[group], marker='o')
        ax.set_title(f"WIP Over Time: {group}")
        ax.set_ylabel("WIP (units)")
        ax.grid(True)

        # Save individual chart to buffer for download
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

    # Download PNG after each group plot
    for group, buf in img_buffers.items():
        st.download_button(f"📥 Download WIP Chart PNG - {group}", data=buf, file_name=f"WIP_{group}.png", mime="image/png")

    # === Process Layout Diagram ===
    st.subheader("🗺️ Production Line Layout")
    G = nx.DiGraph()
    for g in groups:
        G.add_node(g)
    for src, tgt_list in connections.items():
        for tgt in tgt_list:
            if tgt in groups:
                G.add_edge(src, tgt)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20)
    plt.axis('off')
    st.pyplot(plt.gcf())

    # Save layout chart buffer for download
    buf_layout = BytesIO()
    plt.savefig(buf_layout, format="png")
    plt.close()
    buf_layout.seek(0)

    st.download_button("📥 Download Production Line Layout PNG", data=buf_layout, file_name="Production_Line_Layout.png", mime="image/png")

    # === ZIP Download of All Charts ===
    if st.button("📦 Download All Charts as ZIP"):
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w") as zf:
            # Add Excel
            zf.writestr("simulation_summary.xlsx", towrite.getvalue())
            # Add WIP charts
            for group, buf in img_buffers.items():
                zf.writestr(f"WIP_{group}.png", buf.getvalue())
            # Add Layout chart
            zf.writestr("Production_Line_Layout.png", buf_layout.getvalue())
        mem_zip.seek(0)
        st.download_button("📥 Download ZIP file", data=mem_zip, file_name="production_line_charts.zip", mime="application/zip")

else:
    st.info("⚠️ Click **Run Simulation** to generate results and charts.")
