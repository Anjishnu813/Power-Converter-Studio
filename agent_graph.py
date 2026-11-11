"""
automation/agent_graph.py

LangGraph-based Autonomous Power Converter Design and Verification Agent.
Implements the multi-step state graph:
Calculate -> Synthesize Netlist -> Ngspice Simulation -> FFT Diagnostics -> (if needed) Redesign Loop
"""

import os
import sys
import re
import math
import subprocess
from typing import Dict, Any, List, Optional, TypedDict

# Fix Windows console UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root and local directories to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from langgraph.graph import StateGraph, END

try:
    from automation.circuit_solver import CircuitSolver
except (ImportError, ModuleNotFoundError):
    try:
        from Automation.circuit_solver import CircuitSolver
    except (ImportError, ModuleNotFoundError):
        from circuit_solver import CircuitSolver


# Define the Agent State
class AgentState(TypedDict):
    topology: str
    vin: float
    vout: float
    iout: float
    fsw: float
    target_ripple_v: float
    params: Dict[str, Any]
    netlist_content: str
    netlist_path: str
    sim_success: bool
    output_data_file: str
    diagnostics: Dict[str, Any]
    iteration: int
    max_iterations: int
    logs: List[str]
    status: str


# Helper to find Ngspice executable across Windows, Linux, and Cloud Containers
def get_ngspice_executable() -> str:
    import shutil
    candidates = [
        r"C:\Spice64\bin\ngspice_con.exe",
        os.path.join(ROOT_DIR, "ngspice-46_64", "Spice64", "bin", "ngspice_con.exe"),
        r"C:\Spice64\bin\ngspice.exe",
        shutil.which("ngspice_con"),
        shutil.which("ngspice"),
        "ngspice_con",
        "ngspice"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return "ngspice" if sys.platform != "win32" else "ngspice_con"


# 1. Node: Calculate Passives
def calculate_passives_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    logs.append(f"🔧 Step 1: Calculating physical passive parameters for {state['topology'].upper()} (Enforcing PCMC)...")
    
    params = CircuitSolver.calculate_passives(
        topology=state["topology"],
        vin=state["vin"],
        vout=state["vout"],
        iout=state["iout"],
        fsw=state["fsw"],
        ripple_ratio_v=state.get("target_ripple_v", 0.01)
    )
    
    # Baseline controller gains for Peak Current Mode Control (safely damped crossover)
    params['KP'] = 0.01
    params['KI'] = 0.02
    params['Kp_val'] = 0.01
    params['Ki_val'] = 0.02
    params['M_SLOPE'] = 2e4
    params['RSENSE'] = 0.05
    
    log_summary = (
        f"  -> Calculated: L={CircuitSolver.format_si(params.get('L1_val', 0))}H, "
        f"C={CircuitSolver.format_si(params.get('Cout_val', 0))}F, "
        f"Rload={params.get('Rload_val', 0)} Ohm, D={params.get('D_val', 0)} (PCMC Active)"
    )
    logs.append(log_summary)
    
    return {
        "params": params,
        "logs": logs,
        "status": "Calculated parameters"
    }


CONTROLLERS_LIB_CONTENT = """* controllers.lib - Generalized Control Modules for SPICE
.subckt PCMC_CONTROLLER v_out v_ref i_sense pwm_out ctrl ramp PARAMS: KP=0.01 KI=0.001 M_SLOPE=0 RSENSE=1 F_SW=100k
Eerr err 0 value = { v(v_ref) - v(v_out) }
Bint 0 int_val I={ v(err) * KI }
Cint int_val 0 100u IC=0.5
Rint int_val 0 10MEG
Dclamp1 int_val clamp_high D_ideal
Dclamp2 clamp_low int_val D_ideal
Vclamp_high clamp_high 0 DC 0.9
Vclamp_low clamp_low 0 DC 0.01
Ectrl ctrl 0 value = { v(err) * KP + v(int_val) < 0.01 ? 0.01 : (v(err) * KP + v(int_val) > 0.9 ? 0.9 : v(err) * KP + v(int_val)) }
Vramp ramp 0 PULSE(0 {M_SLOPE/F_SW} 0 {0.99/F_SW} {0.01/F_SW} 0 {1/F_SW})
E_vc_eff vc_eff 0 value = { v(ctrl) - v(ramp) }
Ecomp comp 0 value = { v(i_sense) * RSENSE >= v(vc_eff) ? 1 : 0 }
Vclk clk 0 PULSE(0 1 0 1n 1n {0.05/F_SW} {1/F_SW})
B_set 0 state I = { (v(clk) > 0.5) * (v(comp) <= 0.5) }
B_res state 0 I = { (v(comp) > 0.5) }
C_state state 0 100p IC=0
D1 state clamp_h D_ideal
D2 clamp_l state D_ideal
V_h clamp_h 0 DC 1
V_l clamp_l 0 DC 0
Epwm pwm_out 0 value = { v(state) > 0.5 ? 1 : 0 }
.model D_ideal D(N=0.001 RS=0.001)
.ends PCMC_CONTROLLER
"""

BUILTIN_BUCK_PCMC = """* Regulated Buck Converter (Closed Loop Peak Current Mode Control - PCMC)
.param Vin_val=120
.param Vout_val=48
.param L1_val=200u
.param Cout_val=100u
.param Rload_val=10
.param Fsw_val=100k
.param KP=0.01
.param KI=0.02
.options method=gear reltol=0.01 abstol=1u chgtol=1e-11 cshunt=10p
.include "controllers.lib"
V1 in 0 DC {Vin_val}
Cin in 0 100u IC={Vin_val}
S1 in sw1 pwm 0 MySwitch
.model MySwitch SW(RON=0.01 ROFF=1MEG VT=0.5 VH=0.1)
D1 0 sw1 D1N5822
.model D1N5822 D(IS=1e-5 RS=0.02 N=1 BV=300)
L1 sw1 l1_out {L1_val} IC=5.0
RL1 l1_out l1_mid 0.02
V_sense l1_mid out DC 0
C1 out c1_esr {Cout_val} IC={Vout_val}
RC1 c1_esr 0 0.03
Rload out 0 {Rload_val}
Vref ref 0 DC {Vout_val}
Bi_sense i_sense 0 V = { i(V_sense) }
Xctrl out ref i_sense pwm ctrl ramp PCMC_CONTROLLER PARAMS: KP={KP} KI={KI} M_SLOPE=2e4 RSENSE=0.05 F_SW={Fsw_val}
.tran 0.5u 25m UIC
.control
run
wrdata output_data.txt v(out) v(in) v(ctrl) v(ramp) i(V_sense)
plot v(out) v(in) title "Buck PCMC Input and Output Voltage"
plot v(ctrl) v(ramp) title "Control and Ramp Voltages"
plot i(V_sense) title "Buck Inductor Current"
.endc
.end
"""

BUILTIN_BOOST_PCMC = """* Regulated Boost Converter (Closed Loop Peak Current Mode Control - PCMC)
.param Vin_val=24
.param Vout_val=48
.param L1_val=100u
.param Cout_val=220u
.param C1_val=220u
.param Rload_val=9.6
.param Fsw_val=100k
.param KP=0.01
.param KI=0.02
.options method=gear reltol=0.01 abstol=1u chgtol=1e-11 cshunt=10p
.include "controllers.lib"
V1 in 0 DC {Vin_val}
Cin in 0 100u IC={Vin_val}
V_sense in L1_in DC 0
L1 L1_in sw {L1_val} IC=10.0
RL1 sw sw_real 0.02
S1 sw_real 0 pwm 0 MySwitch
.model MySwitch SW(RON=0.01 ROFF=1MEG VT=0.5 VH=0.1)
D1 sw_real out D1N5822
.model D1N5822 D(IS=1e-5 RS=0.02 N=1 BV=1000)
C1 out c1_esr {Cout_val} IC={Vout_val}
RC1 c1_esr 0 0.03
Rload out 0 {Rload_val}
Vref ref 0 DC {Vout_val}
Bi_sense i_sense 0 V = { i(V_sense) }
Xctrl out ref i_sense pwm ctrl ramp PCMC_CONTROLLER PARAMS: KP={KP} KI={KI} M_SLOPE=2e4 RSENSE=0.05 F_SW={Fsw_val}
.tran 0.5u 25m UIC
.control
run
wrdata output_data.txt v(out) v(in) v(ctrl) v(ramp) i(V_sense)
plot v(out) v(in) title "Boost PCMC Input and Output Voltage"
plot v(ctrl) v(ramp) title "Control and Ramp Voltages"
plot i(V_sense) title "Boost Inductor Current"
.endc
.end
"""

BUILTIN_SEPIC_PCMC = """* Regulated SEPIC Converter (Closed Loop Peak Current Mode Control - PCMC)
.param Vin_val=240
.param Vout_val=64
.param L1_val=1.2m
.param L2_val=330u
.param C1_val=4.7u
.param Cout_val=220u
.param Rload_val=12.8
.param Fsw_val=100k
.param KP=0.01
.param KI=0.02
.options method=gear reltol=0.01 abstol=1u chgtol=1e-11 cshunt=10p
.include "controllers.lib"
V1 in 0 DC {Vin_val}
Cin in 0 100u IC={Vin_val}
V_sense in L1_in DC 0
L1 L1_in sw {L1_val} IC=2.0
RL1 sw sw_mid 0.02
C1 sw_mid c1_mid {C1_val} IC={Vin_val}
RC1 c1_mid d_in 0.02
L2 d_in 0 {L2_val} IC=2.0
S1 sw_mid 0 pwm 0 MySwitch
.model MySwitch SW(RON=0.01 ROFF=1MEG VT=0.5 VH=0.1)
D1 d_in out D1N5822
.model D1N5822 D(IS=1e-5 RS=0.02 N=1 BV=500)
Cout out c_esr {Cout_val} IC={Vout_val}
RCout c_esr 0 0.03
Rload out 0 {Rload_val}
Vref ref 0 DC {Vout_val}
Bi_sense i_sense 0 V = { i(V_sense) }
Xctrl out ref i_sense pwm ctrl ramp PCMC_CONTROLLER PARAMS: KP={KP} KI={KI} M_SLOPE=2e4 RSENSE=0.05 F_SW={Fsw_val}
.tran 0.5u 15m UIC
.control
run
wrdata output_data.txt v(out) v(in) v(ctrl) v(ramp) i(V_sense)
plot v(out) v(in) title "SEPIC PCMC Input and Output Voltage"
plot v(ctrl) v(ramp) title "Control and Ramp Voltages"
plot i(V_sense) title "SEPIC Inductor Current"
.endc
.end
"""

BUILTIN_FSBB_PCMC = """* Regulated Four-Switch Buck-Boost Converter (Closed Loop PCMC)
.param Vin_val=24
.param L1_val=100u
.param Cout_val=470u
.param Cin_val=220u
.param Rload_val=24
.param Vout_val=48
.param Fsw_val=100k
.param KP=0.01
.param KI=0.02
.options method=gear reltol=0.01 abstol=1u chgtol=1e-11 cshunt=10p
.include "controllers.lib"
V1 in 0 DC {Vin_val}
Cin in cin_esr {Cin_val} IC={Vin_val}
Rcin cin_esr 0 0.03
S1 in sw1 pwm 0 MySwitch
S2 sw1 0 pwm_n 0 MySwitch
D2 0 sw1 D_ideal
V_sense sw1 L1_in DC 0
L1 L1_in sw2_mid {L1_val} IC=6.0
RL1 sw2_mid sw2 0.05
S3 sw2 0 pwm 0 MySwitch
S4 sw2 out pwm_n 0 MySwitch
D4 sw2 out D_ideal
.model MySwitch SW(RON=0.01 ROFF=1MEG VT=0.5 VH=0.05)
.model D_ideal D(IS=1e-5 RS=0.02 N=1 BV=1000)
C1 out c1_esr {Cout_val} IC={Vout_val}
RC1 c1_esr 0 0.03
Rload out 0 {Rload_val}
Vref ref 0 DC {Vout_val}
Bi_sense i_sense 0 V = { i(V_sense) }
Xctrl out ref i_sense pwm ctrl ramp PCMC_CONTROLLER PARAMS: KP={KP} KI={KI} M_SLOPE=2e4 RSENSE=0.05 F_SW={Fsw_val}
Epwmn pwm_n 0 value = { 1.0 - v(pwm) }
.tran 0.5u 20m UIC
.control
run
wrdata output_data.txt v(out) v(in) v(ctrl) v(ramp) i(V_sense)
plot v(out) v(in) title "FSBB Input and Output Voltage"
plot v(ctrl) v(ramp) title "FSBB Control and Ramp Voltages"
plot i(V_sense) title "FSBB Inductor Current"
.endc
.end
"""

# 2. Node: Generate/Update Netlist
def generate_netlist_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    topo = state["topology"].lower()
    iteration = state.get("iteration", 0) + 1
    logs.append(f"📝 Step 2 (Iter {iteration}): Synthesizing PCMC SPICE Netlist for {state['topology'].upper()}...")

    # Ensure controllers.lib exists in root and outputs directory
    os.makedirs("outputs", exist_ok=True)
    for clib_path in ["controllers.lib", os.path.join(ROOT_DIR, "controllers.lib"), "outputs/controllers.lib"]:
        if not os.path.exists(clib_path):
            try:
                with open(clib_path, "w", encoding="utf-8") as f:
                    f.write(CONTROLLERS_LIB_CONTENT)
            except Exception:
                pass

    # Template selection prioritizing Peak Current Mode Control (PCMC)
    # Search multiple potential directory names robustly across OSes
    possible_lib_dirs = [
        "Circuit Library",
        os.path.join(ROOT_DIR, "Circuit Library"),
        "circuit library",
        "Circuit_Library",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Circuit Library"),
    ]
    lib_dir = "Circuit Library"
    for d in possible_lib_dirs:
        if os.path.isdir(d):
            lib_dir = d
            break
            
    outputs_dir = "outputs"
    
    template_file = None
    if "buck" in topo:
        template_file = os.path.join(lib_dir, "dcdc_buck_pcmc_cl.cir")
    elif "boost" in topo:
        template_file = os.path.join(lib_dir, "dcdc_boost_pcmc_cl.cir")
    elif "sepic" in topo:
        template_file = os.path.join(lib_dir, "dcdc_sepic_pcmc_cl.cir")
    elif "fsbb" in topo:
        template_file = os.path.join(lib_dir, "dcdc_fsbb_pcmc_cl.cir")
    elif "cuk" in topo:
        template_file = os.path.join(outputs_dir, "cuk_cl_120_neg60.cir")

    content = None
    if template_file and os.path.exists(template_file):
        with open(template_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    elif os.path.isdir(lib_dir):
        # Fallback to general library templates if directory exists
        for f in os.listdir(lib_dir):
            if topo in f.lower() and f.endswith(".cir"):
                with open(os.path.join(lib_dir, f), "r", encoding="utf-8", errors="ignore") as cf:
                    content = cf.read()
                break

    # If still not found, use built-in PCMC templates
    if not content:
        if "sepic" in topo:
            content = BUILTIN_SEPIC_PCMC
        elif "boost" in topo:
            content = BUILTIN_BOOST_PCMC
        elif "fsbb" in topo:
            content = BUILTIN_FSBB_PCMC
        else:
            content = BUILTIN_BUCK_PCMC

    # Standardize data export output name to output_data.txt
    content = re.sub(r'wrdata\s+\S+', 'wrdata output_data.txt', content)

    # Apply parameters
    updated_netlist = CircuitSolver.apply_params_to_netlist(content, state["params"])
    
    # Save netlist in outputs/
    os.makedirs("outputs", exist_ok=True)
    out_name = f"outputs/{topo}_cl_{int(state['vin'])}_{int(state['vout'])}.cir"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(updated_netlist)

    logs.append(f"  -> Generated netlist saved to: {out_name}")
    return {
        "netlist_content": updated_netlist,
        "netlist_path": out_name,
        "iteration": iteration,
        "logs": logs,
        "status": "Generated netlist"
    }


# 3. Node: Run Ngspice Simulation
def simulate_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    logs.append("⚡ Step 3: Executing headless Ngspice simulation...")

    ngspice_bin = get_ngspice_executable()
    netlist_path = os.path.abspath(state["netlist_path"])
    data_file = os.path.join(ROOT_DIR, "output_data.txt")
    sim_out = os.path.join(ROOT_DIR, "sim_out.txt")

    # Remove stale data file if exists
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
        except Exception:
            pass

    cmd = [ngspice_bin, "-b", netlist_path, "-o", sim_out]
    try:
        subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, timeout=90)
        success = os.path.exists(data_file) and os.path.getsize(data_file) > 100
    except Exception as e:
        success = False
        logs.append(f"  ❌ Ngspice simulation error: {str(e)}")

    if success:
        logs.append(f"  ✅ Simulation finished. Data logged to output_data.txt")
    else:
        logs.append(f"  ⚠️ Warning: Simulation generated limited or empty output data.")

    return {
        "sim_success": success,
        "output_data_file": data_file,
        "logs": logs,
        "status": "Simulation completed"
    }


# 4. Node: FFT Diagnostics & Stability Evaluation
def diagnostics_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    logs.append("🔍 Step 4: Running automated FFT spectral diagnostics...")
    data_file = state.get("output_data_file", os.path.join(ROOT_DIR, "output_data.txt"))
    target_v = state["vout"]
    topo = state["topology"]

    diag_result = {
        "avg_v": 0.0,
        "ripple_p2p": 0.0,
        "dom_freq": 0.0,
        "dom_mag": 0.0,
        "is_stable": False,
        "diagnosis_text": "No data",
        "plot_path": ""
    }

    if not os.path.exists(data_file):
        logs.append("  ❌ Diagnostic failed: output_data.txt not found.")
        return {"diagnostics": diag_result, "logs": logs, "status": "Diagnostic failed"}

    try:
        lines = [l.strip().split() for l in open(data_file).readlines() if len(l.strip().split()) > 1]
        data = []
        for l in lines:
            try:
                data.append([float(x) for x in l])
            except ValueError:
                pass

        if len(data) < 100:
            logs.append("  ❌ Insufficient data points in simulation output.")
            return {"diagnostics": diag_result, "logs": logs, "status": "Diagnostic failed"}

        data = np.array(data)
        # Focus on settled steady-state (last 40% of transient run)
        start_idx = int(len(data) * 0.6)
        t = data[start_idx:, 0]
        v = data[start_idx:, 1]  # Output voltage

        dt = np.mean(np.diff(t))
        n = len(t)
        v_mean = float(np.mean(v))
        
        # Detrend to remove linear settling slopes and eliminate window leakage artifacts
        from scipy.signal import detrend
        v_ac = detrend(v)

        yf = fft(v_ac)
        xf = fftfreq(n, dt)[:n // 2]
        mag = 2.0 / n * np.abs(yf[0:n // 2])

        # Filter out DC window-boundary leakage (frequencies below 3 / T_window)
        t_span = t[-1] - t[0] if len(t) > 1 else 1.0
        f_min_leakage = max(250.0, 3.0 / t_span)
        valid_mask = xf >= f_min_leakage

        if np.any(valid_mask):
            sub_mag = mag[valid_mask]
            sub_xf = xf[valid_mask]
            idx = np.argmax(sub_mag)
            dom_freq = float(sub_xf[idx])
            dom_mag = float(sub_mag[idx])
        else:
            dom_freq = float(xf[np.argmax(mag)])
            dom_mag = float(np.max(mag))

        ripple_p2p = float(np.max(v) - np.min(v))

        # Stability criteria:
        # Voltage error <= 5% and Ripple <= max(2.5V, 4% of Vout)
        target_ripple_allowed = max(2.5, target_v * 0.04)
        voltage_error = abs(v_mean - target_v)

        is_stable = (ripple_p2p <= target_ripple_allowed) and (voltage_error <= (0.05 * target_v))

        diag_text = "Stable operation within target specification."
        if not is_stable:
            if ripple_p2p > 2.0 and dom_freq < 100:
                diag_text = f"Low frequency oscillation ({dom_freq:.1f}Hz) detected — likely PI Integrator Windup."
            elif ripple_p2p > 2.0 and dom_freq < 5000:
                diag_text = f"Mid frequency resonance ({dom_freq:.1f}Hz) detected — LC tank / crossover instability."
            elif ripple_p2p > target_ripple_allowed:
                diag_text = f"High frequency ripple ({ripple_p2p:.2f}V > {target_ripple_allowed:.2f}V) exceeds target."
            else:
                diag_text = f"Output voltage offset (Avg {v_mean:.2f}V vs Target {target_v:.2f}V)."

        # Generate and save diagnostic plot
        os.makedirs("outputs", exist_ok=True)
        plot_path = f"outputs/{topo}_cl_{int(state['vin'])}_{int(state['vout'])}_fft.png"

        plt.figure(figsize=(9, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t * 1e3, v, color='#1f77b4', lw=1.5, label='V(out)')
        plt.axhline(target_v, color='red', linestyle='--', label=f'Target ({target_v}V)')
        plt.title(f"Steady-State Output Voltage (Avg: {v_mean:.2f}V, Ripple: {ripple_p2p:.2f}V)", fontsize=11)
        plt.xlabel("Time (ms)", fontsize=9)
        plt.ylabel("Voltage (V)", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        plt.subplot(2, 1, 2)
        plt.plot(xf / 1e3, mag, color='#d62728', lw=1.5)
        plt.xlim(0, max(10, (dom_freq * 2.5) / 1e3))
        plt.title(f"Frequency Spectrum (FFT) - Dominant Peak: {dom_freq:.1f} Hz ({dom_mag:.2f} V)", fontsize=11)
        plt.xlabel("Frequency (kHz)", fontsize=9)
        plt.ylabel("Magnitude (V)", fontsize=9)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()

        diag_result = {
            "avg_v": v_mean,
            "ripple_p2p": ripple_p2p,
            "dom_freq": dom_freq,
            "dom_mag": dom_mag,
            "is_stable": is_stable,
            "diagnosis_text": diag_text,
            "plot_path": plot_path
        }

        logs.append(f"  -> Diagnostics Result: Avg={v_mean:.2f}V, Ripple={ripple_p2p:.2f}V, DomFreq={dom_freq:.1f}Hz")
        logs.append(f"  -> Verdict: {'✅ STABLE' if is_stable else '⚠️ UNSTABLE: ' + diag_text}")

    except Exception as e:
        logs.append(f"  ❌ Error computing FFT: {str(e)}")

    return {
        "diagnostics": diag_result,
        "logs": logs,
        "status": "Diagnostics completed"
    }


# 5. Node: Redesign / Auto-Tuning
def redesign_node(state: AgentState) -> Dict[str, Any]:
    logs = list(state.get("logs", []))
    logs.append("🔄 Step 5: Triggering Algorithmic Redesign & Tuning Engine...")
    
    diag = state["diagnostics"]
    target_rip = state.get("target_ripple_v", 0.01) * state["vout"]
    updated_params, explanation = CircuitSolver.diagnose_and_redesign(
        current_params=state["params"],
        dom_freq=diag.get("dom_freq", 0.0),
        peak_to_peak_ripple=diag.get("ripple_p2p", 0.0),
        target_v=state["vout"],
        measured_v_avg=diag.get("avg_v", 0.0),
        target_ripple=target_rip,
        fsw=state.get("fsw", 100e3)
    )

    for line in explanation.split("\n"):
        if line.strip():
            logs.append(f"  💡 {line}")

    return {
        "params": updated_params,
        "logs": logs,
        "status": "Parameters redesigned"
    }


# Conditional Edge Router
def router_check_stability(state: AgentState) -> str:
    is_stable = state.get("diagnostics", {}).get("is_stable", False)
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", 4)

    if is_stable or iteration >= max_iter:
        return "end"
    return "redesign"


# Build the Graph
def create_converter_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node("calculate", calculate_passives_node)
    builder.add_node("generate_netlist", generate_netlist_node)
    builder.add_node("simulate", simulate_node)
    builder.add_node("diagnostics", diagnostics_node)
    builder.add_node("redesign", redesign_node)

    builder.set_entry_point("calculate")
    builder.add_edge("calculate", "generate_netlist")
    builder.add_edge("generate_netlist", "simulate")
    builder.add_edge("simulate", "diagnostics")

    builder.add_conditional_edges(
        "diagnostics",
        router_check_stability,
        {
            "end": END,
            "redesign": "redesign"
        }
    )
    builder.add_edge("redesign", "generate_netlist")

    return builder.compile()


if __name__ == '__main__':
    graph = create_converter_agent_graph()
    initial_state: AgentState = {
        "topology": "buck",
        "vin": 120.0,
        "vout": 48.0,
        "iout": 10.0,
        "fsw": 100e3,
        "target_ripple_v": 0.01,
        "params": {},
        "netlist_content": "",
        "netlist_path": "",
        "sim_success": False,
        "output_data_file": "",
        "diagnostics": {},
        "iteration": 0,
        "max_iterations": 2,
        "logs": [],
        "status": "Initialized"
    }

    final_state = graph.invoke(initial_state)
    print("\n--- Final Agent Execution Summary ---")
    for log in final_state["logs"]:
        print(log)
    print("\nFinal Status:", final_state["diagnostics"])
