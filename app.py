import streamlit as st
import simpy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd
from io import BytesIO
import zipfile

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
        pwd = st.text_input("🔒 Password", type="password")

        if user and pwd and not st.session_state.password_attempted:
            st.session_state.password_attempted = True
            if user == USERNAME and pwd == PASSWORD:
                st.session_state.authenticated = True
            else:
                st.error("❌ Incorrect credentials")
                st.stop()

        if not st.session_state.authenticated:
            st.stop()

st.title("🛠️ Production Line Simulation App (SimPy)")

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

    for i in range(num_groups):
        name = st.text_input(f"Group {i+1} Name", key=f"name_{i}").strip().upper()
        st.session_state.group_names[i] = name

        if name:
            eq_count = st.number_input(f"Number of Equipment in {name}", 1, key=f"eq_count_{i}")
            eq_times = [st.number_input(f"Cycle Time for {name} - EQ{j+1} (sec)", 0.1, key=f"ct_{i}_{j+1}") for j in range(eq_count)]
            st.session_state.station_groups[name] = {f"{name} - EQ{j+1}": ct for j, ct in enumerate(eq_times)}

# === Step 2: Connections ===
with col2:
    st.header("Step 2: Connect Stations")
    if "from_stations" not in st.session_state:
        st.session_state.from_stations = {}
    if "connections" not in st.session_state:
        st.session_state.connections = {}

    for i, name in enumerate(st.session_state.group_names):
        if not name:
            continue
        with st.expander(f"{name} Connections"):
            from_options = ['START'] + [g for g in st.session_state.group_names if g and g != name]
            to_options = ['STOP'] + [g for g in st.session_state.group_names if g and g != name]

            from_selected = st.multiselect(f"{name} receives from:", from_options, key=f"from_{i}")
            to_selected = st.multiselect(f"{name} sends to:", to_options, key=f"to_{i}")

            st.session_state.from_stations[name] = [] if "START" in from_selected else from_selected
            st.session_state.connections[name] = [] if "STOP" in to_selected else to_selected

# === Step 3: Duration ===
st.markdown("---")
st.header("Step 3: ⏱️ Enter Simulation Duration")
sim_time = st.number_input("Simulation Time (seconds)", min_value=10, value=100, step=10)

if st.button("▶️ Run Simulation"):
    st.session_state.simulate = True
    st.session_state.sim_time = sim_time

# === Run Simulation ===
if st.session_state.get("simulate"):
    station_groups = st.session_state.station_groups
    from_stations = st.session_state.from_stations
    connections = st.session_state.connections
    sim_time = st.session_state.sim_time
    valid_groups = {g: eqs for g, eqs in station_groups.items() if g}

    class FactorySimulation:
        def __init__(self, env, station_groups, duration, connections, from_stations):
            self.env = env
            self.station_groups = station_groups
            self.connections = connections
            self.from_stations = from_stations
            self.duration = duration

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

    env = simpy.Environment()
    sim = FactorySimulation(env, valid_groups, sim_time, connections, from_stations)
    sim.run()
    env.run(until=sim_time)

    # === Results Summary ===
    st.markdown("---")
    st.subheader("📊 Simulation Results Summary")
    groups = list(valid_groups.keys())
    agg = defaultdict(lambda: {'in': 0, 'out': 0, 'busy': 0, 'count': 0, 'cycle_times': [], 'wip': 0})

    for group in groups:
        eqs = valid_groups[group]
        for eq in eqs:
            agg[group]['in'] += sim.throughput_in[eq]
            agg[group]['out'] += sim.throughput_out[eq]
            agg[group]['busy'] += sim.equipment_busy_time[eq]
            agg[group]['cycle_times'].append(sim.cycle_times[eq])
            agg[group]['count'] += 1
        prev_out = sum(sim.throughput_out[eq] for g in from_stations.get(group, []) for eq in valid_groups.get(g, []))
        curr_in = agg[group]['in']
        agg[group]['wip'] = max(0, prev_out - curr_in)

    df = pd.DataFrame([{
        "Station Group": g,
        "Boards In": agg[g]['in'],
        "Boards Out": agg[g]['out'],
        "WIP": agg[g]['wip'],
        "Number of Equipment": agg[g]['count'],
        "Cycle Times (sec)": ", ".join(str(round(ct, 1)) for ct in agg[g]['cycle_times']),
        "Utilization (%)": round((agg[g]['busy'] / (sim_time * agg[g]['count'])) * 100, 1)
    } for g in groups])

    st.dataframe(df, use_container_width=True)


    # Excel download
    towrite = BytesIO()
    df.to_excel(towrite, index=False, sheet_name="Summary")
    towrite.seek(0)
    st.download_button("📥 Download Summary Excel", data=towrite, file_name="simulation_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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

from graphviz import Digraph

# === Process Layout Diagram (Linear Format using Graphviz) ===
st.subheader("🗌 Production Line Layout (Linear Flow)")

layout_png_buf = BytesIO()
if 'groups' in locals() and groups:
    try:
        dot = Digraph(format="png")
        dot.attr(rankdir="LR", size="8")

        for group in groups:
            dot.node(group, shape="box", style="filled", fillcolor="lightblue")

        for i in range(len(groups) - 1):
            dot.edge(groups[i], groups[i + 1])

        st.graphviz_chart(dot)

        # Save as PNG
        dot.render(filename="layout", directory="/tmp", format="png", cleanup=False)
        with open("/tmp/layout.png", "rb") as f:
            layout_png_buf.write(f.read())
        layout_png_buf.seek(0)

        st.download_button("📅 Download Linear Layout PNG", data=layout_png_buf, file_name="Linear_Production_Layout.png", mime="image/png")

    except Exception as e:
        st.warning(f"Graphviz layout failed: {e}")
else:
    st.info("ℹ️ Run the simulation to view layout diagram.")

# === Bottleneck Detection and Suggestion ===
st.subheader("💡 Bottleneck Analysis and Suggestion")

if 'agg' in locals() and 'valid_groups' in locals():
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
            (
                f"If you **add 1 more equipment** to **{bottleneck_group}** with cycle time = **{round(avg_ct,1)} sec**, "
                f"you may increase its output by approximately **{delta_b} boards**, "
                f"and final output by approximately **{delta_final} boards** over {sim_time} seconds."
            )
        )
else:
    st.info("ℹ️ Run the simulation to get bottleneck suggestions.")

# === ZIP Download of All Charts ===
if st.button("📦 Download All Charts and Tables as ZIP"):
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w") as zf:
        if 'towrite' in locals():
            zf.writestr("simulation_summary.xlsx", towrite.getvalue())
        if 'img_buffers' in locals():
            for group, buf in img_buffers.items():
                zf.writestr(f"WIP_{group}.png", buf.getvalue())
        if layout_png_buf.getbuffer().nbytes > 0:
            zf.writestr("Linear_Production_Layout.png", layout_png_buf.getvalue())

    mem_zip.seek(0)
    st.download_button("📅 Download All as ZIP", data=mem_zip, file_name="simulation_results_bundle.zip", mime="application/zip")
else:
    st.info("⚠️ Click **Run Simulation** to generate results and charts.")

