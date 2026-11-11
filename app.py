"""
app.py

Autonomous Power Converter Design & Simulation Studio.
Custom Web Dashboard combining Natural-Language Prompting, 
LangGraph Multi-Stage Agentic Verification, and Ngspice Simulation.
"""

import os
import sys
import re
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import importlib
import automation.circuit_solver
import automation.agent_graph
importlib.reload(automation.circuit_solver)
importlib.reload(automation.agent_graph)
from automation.agent_graph import create_converter_agent_graph, AgentState
from automation.circuit_solver import CircuitSolver

# Page configuration
st.set_page_config(
    page_title="Power Converter Design Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .prompt-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-title">⚡ Autonomous Power Converter Design Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Agentic Natural-Language Synthesis & Closed-Loop Algorithmic Verification Engine (LangGraph + Ngspice)</div>',
    unsafe_allow_html=True
)

# Comprehensive topology catalogue mapping to templates
TOPOLOGY_CATALOGUE = {
    "Buck Converter (Closed-Loop Step-Down)": "buck",
    "Synchronous Buck Converter": "buck",
    "Boost Converter (Closed-Loop Step-Up)": "boost",
    "SEPIC Converter (Buck-Boost Non-Inverting)": "sepic",
    "Cuk Converter (Inverting Buck-Boost)": "cuk",
    "Four-Switch Buck-Boost (FSBB)": "fsbb",
    "Isolated Flyback Converter": "flyback",
    "Push-Pull Converter": "pushpull",
    "LLC Resonant Half-Bridge Converter": "llc",
    "Active Front End (AFE) Rectifier": "afe",
    "Single-Phase Boost PFC": "boostpfc",
    "Totem-Pole PFC": "totempole",
    "Vienna Rectifier (3-Phase 3-Level)": "vienna",
    "3-Phase Inverter (SPWM)": "dcac"
}


# Natural Language Prompt Parser Helper
def parse_natural_language_prompt(prompt: str):
    p = prompt.lower()
    
    # 1. Detect topology using specific keywords (never generic words like 'converter')
    TOPOLOGY_KEYWORDS = [
        ("sepic", r'\bsepic\b'),
        ("cuk", r'\bcuk\b'),
        ("fsbb", r'\b(?:fsbb|four[- ]switch|4[- ]switch)\b'),
        ("flyback", r'\bflyback\b'),
        ("pushpull", r'\b(?:push[- ]pull|pushpull)\b'),
        ("llc", r'\bllc\b'),
        ("boostpfc", r'\b(?:boost[- ]?pfc|pfc)\b'),
        ("totempole", r'\btotem[- ]?pole\b'),
        ("vienna", r'\bvienna\b'),
        ("dcac", r'\b(?:inverter|spwm|dc[- ]?ac)\b'),
        ("afe", r'\b(?:afe|active[- ]front[- ]end)\b'),
        ("boost", r'\bboost\b'),
        ("buck", r'\bbuck\b'),
    ]
    detected_topo = "buck"
    for key, pat in TOPOLOGY_KEYWORDS:
        if re.search(pat, p):
            detected_topo = key
            break
            
    # 2. Extract Vin
    vin = 120.0
    vin_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:v|volts?)?\s*(?:in|input|vin)', p)
    if not vin_match:
        vin_match = re.search(r'(?:in|input|vin)\s*(?:=|of|is|:)?\s*(\d+(?:\.\d+)?)\s*v?', p)
    if vin_match:
        vin = float(vin_match.group(1))

    # 3. Extract Vout
    vout = 48.0
    vout_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:v|volts?)?\s*(?:out|output|vout)', p)
    if not vout_match:
        vout_match = re.search(r'(?:out|output|vout|to)\s*(?:=|of|is|:)?\s*(\d+(?:\.\d+)?)\s*v?', p)
    if vout_match:
        vout = float(vout_match.group(1))

    # 4. Extract Load Current
    iout = 5.0
    iout_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:a|amps?|amperes?)\s*(?:load|out)?', p)
    if iout_match:
        iout = float(iout_match.group(1))

    # 5. Extract Frequency
    fsw_khz = 100.0
    fsw_match = re.search(r'(\d+(?:\.\d+)?)\s*khz', p)
    if fsw_match:
        fsw_khz = float(fsw_match.group(1))

    return detected_topo, vin, vout, iout, fsw_khz


# ─── Natural Language Prompt Bar (Top of Main Screen) ───
st.markdown("### 💬 Natural Language Circuit Request")
default_prompt = "Design a closed-loop buck converter, 120V in, 48V out, 100kHz, 10A load"
user_prompt = st.text_input(
    "Type your converter specifications in natural English (or configure via sidebar):",
    value=default_prompt,
    help="Examples: 'Design a boost converter, 24V in, 48V out, 50kHz, 5A load', 'Synthesize a closed-loop SEPIC converter, 240V to 64V'"
)

p_topo, p_vin, p_vout, p_iout, p_fsw = parse_natural_language_prompt(user_prompt)

# Sidebar: Fine Controls & Overrides
st.sidebar.header("📐 Extracted & Manual Parameters")

# Topology Dropdown (pre-selected based on prompt)
label_index = 0
for idx, (lbl, k) in enumerate(TOPOLOGY_CATALOGUE.items()):
    if k == p_topo:
        label_index = idx
        break

selected_label = st.sidebar.selectbox("Converter Topology", list(TOPOLOGY_CATALOGUE.keys()), index=label_index)
active_topo = TOPOLOGY_CATALOGUE[selected_label]

st.sidebar.subheader("🎯 Operating Parameters")
vin = st.sidebar.number_input("Input Voltage Vin (V)", min_value=1.0, max_value=1000.0, value=float(p_vin), step=1.0)
vout = st.sidebar.number_input("Target Output Voltage Vout (V)", min_value=-500.0, max_value=1000.0, value=float(p_vout), step=1.0)
iout = st.sidebar.number_input("Rated Load Current Iout (A)", min_value=0.1, max_value=200.0, value=float(p_iout), step=0.5)
fsw_khz = st.sidebar.number_input("Switching Frequency fsw (kHz)", min_value=10.0, max_value=1000.0, value=float(p_fsw), step=10.0)
ripple_pct = st.sidebar.slider("Allowed Voltage Ripple (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

fsw = fsw_khz * 1e3
target_ripple_ratio = ripple_pct / 100.0

st.sidebar.markdown("---")
synthesize_btn = st.sidebar.button("⚡ Synthesize & Verify Circuit", use_container_width=True)

# Also support quick synthesize button right next to the text prompt
col_prompt_btn, _ = st.columns([1, 4])
with col_prompt_btn:
    main_prompt_btn = st.button("🚀 Run Prompt", use_container_width=True)

# Determine active execution values based on which button was clicked
run_topo = p_topo if main_prompt_btn else active_topo
run_vin = float(p_vin if main_prompt_btn else vin)
run_vout = float(abs(p_vout if main_prompt_btn else vout))
run_iout = float(p_iout if main_prompt_btn else iout)
run_fsw = float((p_fsw * 1e3) if main_prompt_btn else fsw)

# Initialize Session State
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None

# Run LangGraph Workflow
if synthesize_btn or main_prompt_btn:
    with st.spinner(f"🤖 LangGraph Agent: Parsing specs, calculating passives, running Ngspice, and checking FFT for {run_topo.upper()}..."):
        graph = create_converter_agent_graph()
        initial_state: AgentState = {
            "topology": run_topo,
            "vin": run_vin,
            "vout": run_vout,
            "iout": run_iout,
            "fsw": run_fsw,
            "target_ripple_v": target_ripple_ratio,
            "params": {},
            "netlist_content": "",
            "netlist_path": "",
            "sim_success": False,
            "output_data_file": "",
            "diagnostics": {},
            "iteration": 0,
            "max_iterations": 4,
            "logs": [],
            "status": "Initialized"
        }

        try:
            result = graph.invoke(initial_state)
            st.session_state.agent_result = result
            st.success("Workflow completed successfully!")
        except Exception as e:
            st.error(f"Error executing agent workflow: {str(e)}")

# Display Results
result = st.session_state.agent_result

if result:
    diag = result.get("diagnostics", {})
    params = result.get("params", {})
    is_stable = diag.get("is_stable", False)
    
    # Top Status Banner
    if is_stable:
        st.success(f"✅ **CIRCUIT VERIFIED STABLE** — Regulated at {diag.get('avg_v', 0):.2f} V with {diag.get('ripple_p2p', 0):.2f} V ripple (Completed in {result.get('iteration', 1)} iteration(s))")
    else:
        st.warning(f"⚠️ **DIAGNOSTIC ALERT** — {diag.get('diagnosis_text', 'Oscillation or error detected')}")

    # Key Performance Indicator Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Avg Output Voltage", f"{diag.get('avg_v', 0):.2f} V", delta=f"{diag.get('avg_v', 0) - abs(vout):.2f} V vs Target")
    with col2:
        st.metric("Peak-to-Peak Ripple", f"{diag.get('ripple_p2p', 0):.3f} V")
    with col3:
        st.metric("Dominant Frequency", f"{diag.get('dom_freq', 0):.1f} Hz")
    with col4:
        st.metric("Calculated Inductor", f"{CircuitSolver.format_si(params.get('L1_val', 0))}H")
    with col5:
        st.metric("Calculated Capacitor", f"{CircuitSolver.format_si(params.get('Cout_val', params.get('C1_val', 0)))}F")

    st.markdown("---")

    # Tabs for organized view
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Waveforms & FFT Diagnostics", 
        "🛠️ Parameter Tuning & Tweaking", 
        "📝 Verified SPICE Netlist", 
        "🤖 Agent Execution Audit Log"
    ])

    # Tab 1: Waveforms and FFT
    with tab1:
        st.subheader("Dual-Domain Spectral & Time Diagnostics")
        plot_path = diag.get("plot_path", "")
        if plot_path and os.path.exists(plot_path):
            st.image(plot_path, caption=f"Steady-State Output Voltage & FFT Spectrum for {run_topo.upper()}", use_container_width=True)
        else:
            st.info("No diagnostic plot file available.")

        st.info(f"**Diagnostic Evaluation:** {diag.get('diagnosis_text', 'No notes available.')}")

        # Multi-Iteration Trail Expander
        with st.expander("🔍 View Multi-Iteration Optimization & Redesign Trail", expanded=True):
            for log in result.get("logs", []):
                st.write(log)

        # General Closed-Loop Redesign Button
        if diag.get("ripple_p2p", 0.0) > 1.0 or not is_stable:
            st.markdown("---")
            st.warning("⚠️ Diagnostic deviation detected (DC voltage shift, frequency resonance, or ripple violation).")
            if st.button("⚡ Run Autonomous Closed-Loop Redesign & Tuning Cycle"):
                with st.spinner("Autonomously solving for corrected passives and controller gains in Ngspice..."):
                    updated_params, explanation = CircuitSolver.diagnose_and_redesign(
                        current_params=result["params"],
                        dom_freq=diag.get("dom_freq", 0.0),
                        peak_to_peak_ripple=diag.get("ripple_p2p", 0.0),
                        target_v=result["vout"],
                        measured_v_avg=diag.get("avg_v", 0.0),
                        target_ripple=result.get("target_ripple_v", 0.01) * result["vout"],
                        fsw=result.get("fsw", 100e3)
                    )
                    netlist_content = CircuitSolver.apply_params_to_netlist(result["netlist_content"], updated_params)
                    with open(result["netlist_path"], "w", encoding="utf-8") as f:
                        f.write(netlist_content)

                    from automation.agent_graph import simulate_node, diagnostics_node
                    dummy_state = dict(result)
                    dummy_state["params"] = updated_params
                    dummy_state["netlist_content"] = netlist_content
                    dummy_state["iteration"] = dummy_state.get("iteration", 1) + 1
                    logs = list(dummy_state.get("logs", []))
                    logs.append(f"🔄 Manual Redesign Cycle (Iter {dummy_state['iteration']}):")
                    for line in explanation.split("\n"):
                        if line.strip():
                            logs.append(f"  💡 {line}")
                    dummy_state["logs"] = logs

                    sim_res = simulate_node(dummy_state)
                    dummy_state.update(sim_res)
                    diag_res = diagnostics_node(dummy_state)
                    dummy_state.update(diag_res)

                    st.session_state.agent_result = dummy_state
                    st.rerun()

    # Tab 2: Manual Tweaking
    with tab2:
        st.subheader("Manual Parameter Adjustment & Re-Simulation")
        st.write("You can tweak passive component values or controller gains to observe physical stability changes:")
        
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            tweak_l_uh = st.number_input("Inductor L1 (uH)", value=float(params.get('L1_val', 100e-6) * 1e6), step=10.0)
            tweak_kp = st.number_input("Proportional Gain (Kp)", value=float(params.get('KP', 0.05)), step=0.01)
        with tcol2:
            tweak_c_uf = st.number_input("Capacitor Cout (uF)", value=float(params.get('Cout_val', params.get('C1_val', 100e-6)) * 1e6), step=10.0)
            tweak_ki = st.number_input("Integral Gain (Ki)", value=float(params.get('KI', 0.001)), step=0.001, format="%.5f")
        with tcol3:
            tweak_r_ohm = st.number_input("Load Resistance (Ohm)", value=float(params.get('Rload_val', 10.0)), step=0.5)

        if st.button("🔄 Re-Simulate with Tweaked Parameters"):
            with st.spinner("Re-running simulation with updated parameters..."):
                updated_params = dict(params)
                updated_params['L1_val'] = tweak_l_uh * 1e-6
                updated_params['Cout_val'] = tweak_c_uf * 1e-6
                updated_params['C1_val'] = tweak_c_uf * 1e-6
                updated_params['Rload_val'] = tweak_r_ohm
                updated_params['KP'] = tweak_kp
                updated_params['KI'] = tweak_ki

                # Update netlist
                netlist_content = CircuitSolver.apply_params_to_netlist(result["netlist_content"], updated_params)
                with open(result["netlist_path"], "w", encoding="utf-8") as f:
                    f.write(netlist_content)

                # Run simulation
                from automation.agent_graph import simulate_node, diagnostics_node
                dummy_state = dict(result)
                dummy_state["params"] = updated_params
                dummy_state["netlist_content"] = netlist_content
                sim_res = simulate_node(dummy_state)
                dummy_state.update(sim_res)
                diag_res = diagnostics_node(dummy_state)
                dummy_state.update(diag_res)

                st.session_state.agent_result = dummy_state
                st.rerun()

    # Tab 3: Netlist
    with tab3:
        st.subheader("Verified Ngspice Netlist (.cir)")
        netlist_str = result.get("netlist_content", "")
        st.code(netlist_str, language="spice")
        
        st.download_button(
            label="💾 Download SPICE Netlist (.cir)",
            data=netlist_str,
            file_name=os.path.basename(result.get("netlist_path", "converter.cir")),
            mime="text/plain"
        )

    # Tab 4: Agent Logs
    with tab4:
        st.subheader("Autonomous StateGraph Decision Audit")
        for log in result.get("logs", []):
            st.text(log)

else:
    # Landing page information when no simulation has been run yet
    st.info("👈 Enter a prompt above or configure parameters on the left sidebar, then click **'⚡ Synthesize & Verify Circuit'**.")
    
    st.markdown("""
    ### System Architecture & Workflow
    
    This interface combines **natural language prompt interpretation** with an **autonomous LangGraph verification workflow**:
    
    1. **Prompt Parsing & Topology Selection**: Extracts design constraints from natural language (or sidebar widgets) and retrieves matching verified topologies from the **14+ converter Circuit Library**.
    2. **Physics Sizing Engine**: Deterministically calculates component values ($L, C, R$) based on volt-second balance and ripple constraints.
    3. **Automated Ngspice Execution**: Runs headless batch simulations directly via your local Ngspice engine.
    4. **Spectral Signal Diagnostics**: Isolates steady-state ripple using Fast Fourier Transforms (FFT) to distinguish switching harmonics from low-frequency loop hunting or LC filter resonance.
    5. **Closed-Loop Correction Loop**: Automatically redesigns passives or auto-tunes $K_p, K_i$ gains until the circuit meets stability and ripple thresholds.
    """)
