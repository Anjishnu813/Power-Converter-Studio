"""
automation/circuit_solver.py

Deterministic Power Stage Parameter Calculation and Resizing Engine.
Calculates and updates R, L, C, and damping snubber values based on 
power electronics physics and FFT diagnostic feedback.
"""

import math
import re
import os
from typing import Dict, Any, Tuple, Optional


class CircuitSolver:
    """
    Solves passive component values (R, L, C, damping network)
    for switch-mode converters based on volt-second balance, 
    charge balance, and stability criteria.
    """

    @staticmethod
    def calculate_passives(
        topology: str,
        vin: float,
        vout: float,
        iout: float,
        fsw: float,
        ripple_ratio_i: float = 0.3,
        ripple_ratio_v: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculate baseline L, C, R values for a target topology.
        
        Args:
            topology: 'buck', 'boost', 'buckboost', 'sepic', 'cuk', 'fsbb'
            vin: Input voltage (V)
            vout: Target output voltage (V)
            iout: Target load current (A)
            fsw: Switching frequency (Hz)
            ripple_ratio_i: Desired inductor current ripple ratio (dIL / IL)
            ripple_ratio_v: Desired output voltage ripple ratio (dVout / Vout)
        """
        topo = topology.lower().strip()
        r_load = vout / iout if iout > 0 else 10.0
        delta_v = max(vout * ripple_ratio_v, 0.05)  # min 50mV ripple target
        ts = 1.0 / fsw

        results: Dict[str, float] = {
            'Vin_val': vin,
            'V1_val': vin,
            'Vout_val': vout,
            'Vref_val': vout,
            'Rload_val': round(r_load, 3),
            'R1_val': round(r_load, 3),
            'Fsw_val': fsw,
            'Ts_val': ts,
            'fs_period': ts,
            'T_period_val': ts,
        }

        if 'buck' in topo and 'boost' not in topo:
            # Buck Converter
            d = vout / vin if vin > 0 else 0.5
            d = min(max(d, 0.05), 0.95)
            delta_il = iout * ripple_ratio_i
            # L = Vout * (1 - D) / (dIL * fsw)
            l_val = (vout * (1.0 - d)) / (delta_il * fsw)
            # C = dIL / (8 * fsw * dVout)
            c_val = delta_il / (8.0 * fsw * delta_v)
            
            # Critical inductance for CCM: L_crit = (1 - D) * R / (2 * fsw)
            l_crit = (1.0 - d) * r_load / (2.0 * fsw)
            l_val = max(l_val, 1.5 * l_crit)

            results.update({
                'D_val': round(d, 4),
                'L1_val': l_val,
                'C1_val': c_val,
                'Cout_val': c_val,
                'L_crit': l_crit
            })

        elif 'boost' in topo and 'buck' not in topo and 'sepic' not in topo:
            # Boost Converter
            d = 1.0 - (vin / vout) if vout > vin else 0.5
            d = min(max(d, 0.05), 0.95)
            i_in = iout / (1.0 - d)
            delta_il = i_in * ripple_ratio_i
            # L = Vin * D / (dIL * fsw)
            l_val = (vin * d) / (delta_il * fsw)
            # C = Iout * D / (fsw * dVout)
            c_val = (iout * d) / (fsw * delta_v)
            
            # Critical inductance for CCM: L_crit = D * (1 - D)^2 * R / (2 * fsw)
            l_crit = (d * (1.0 - d)**2 * r_load) / (2.0 * fsw)
            l_val = max(l_val, 1.5 * l_crit)

            results.update({
                'D_val': round(d, 4),
                'L1_val': l_val,
                'C1_val': c_val,
                'Cout_val': c_val,
                'L_crit': l_crit
            })

        elif 'sepic' in topo or 'fsbb' in topo or 'cuk' in topo:
            # SEPIC / Cuk / 4-Switch Buck-Boost
            d = vout / (vin + vout)
            d = min(max(d, 0.05), 0.95)
            i_in = iout * (d / (1.0 - d))
            delta_il1 = i_in * ripple_ratio_i
            delta_il2 = iout * ripple_ratio_i

            # L1 and L2
            l1_val = (vin * d) / (delta_il1 * fsw)
            l2_val = (vin * d) / (delta_il2 * fsw)
            # Coupling capacitor (dVc1 ~ 5% of Vin)
            c_cp = max((iout * d) / (fsw * max(vin * 0.05, 0.5)), 2.2e-6)
            # Output capacitor: ensure sufficient capacitance for pulsed diode currents and ESR damping
            c_out_calc = (iout * d) / (fsw * delta_v)
            c_out = max(c_out_calc, 150e-6 if iout >= 4.0 else 47e-6)

            results.update({
                'D_val': round(d, 4),
                'L1_val': l1_val,
                'L2_val': l2_val,
                'C1_val': c_cp,
                'Cout_val': c_out,
                'C2_val': c_out,
            })

        else:
            # Generic fallback (assume Buck-like scaling)
            d = min(max(vout / vin if vin > 0 else 0.5, 0.05), 0.95)
            delta_il = iout * ripple_ratio_i
            l_val = (vout * (1.0 - d)) / (delta_il * fsw)
            c_val = max(delta_il / (8.0 * fsw * delta_v), 100e-6)
            results.update({
                'D_val': round(d, 4),
                'L1_val': l_val,
                'C1_val': c_val,
                'Cout_val': c_val,
            })

        # Calculate Middlebrook Damping Network parameters for the output filter
        l_eff = results.get('L1_val', 100e-6)
        c_eff = results.get('Cout_val', results.get('C1_val', 100e-6))
        z0 = math.sqrt(l_eff / c_eff) if c_eff > 0 else 1.0
        results['Rd_damp'] = round(z0, 2)
        results['Cd_damp'] = 4.0 * c_eff

        # Robust baseline control parameters for Peak Current Mode Control (PCMC)
        results['KP'] = 0.01
        results['KI'] = 0.02
        results['Kp_val'] = 0.01
        results['Ki_val'] = 0.02
        results['M_SLOPE'] = 2e4
        results['RSENSE'] = 0.05

        return results

    @staticmethod
    def diagnose_and_redesign(
        current_params: Dict[str, float],
        dom_freq: float,
        peak_to_peak_ripple: float,
        target_v: float,
        measured_v_avg: float,
        target_ripple: Optional[float] = None,
        fsw: Optional[float] = None
    ) -> Tuple[Dict[str, float], str]:
        """
        Takes diagnostic measurements from FFT and produces physically corrected
        R, L, C, and control parameters.
        
        Returns:
            (updated_params, explanation_of_correction)
        """
        updated = dict(current_params)
        explanation = []
        target_ripple = target_ripple or max(target_v * 0.015, 0.5)
        f_sw = fsw or updated.get('Fsw_val', 100e3)

        # Case 1: Mid-Frequency LC Tank Resonance or Crossover Instability (100 Hz <= dom_freq < 0.4 * f_sw)
        if (peak_to_peak_ripple > target_ripple or peak_to_peak_ripple > 1.5) and (100 <= dom_freq < 0.4 * f_sw):
            l_key = 'L1_val'
            c_key = 'Cout_val' if 'Cout_val' in updated else 'C1_val'
            l_val = updated.get(l_key, 100e-6)
            old_c = updated.get(c_key, 100e-6)
            
            # Increase output capacitance by 1.5x to reduce tank Q and lower impedance
            new_c = old_c * 1.5
            updated[c_key] = new_c
            if 'C2_val' in updated:
                updated['C2_val'] = new_c
            if 'Cout_val' in updated:
                updated['Cout_val'] = new_c
                
            # Recalculate Middlebrook optimal damping
            z0 = math.sqrt(l_val / new_c) if new_c > 0 else 1.0
            rd = round(z0, 2)
            cd = 4.0 * new_c
            updated['Rd_damp'] = rd
            updated['Cd_damp'] = cd

            # Aggressively reduce Kp and Ki to back crossover away from resonant peak
            old_kp = updated.get('KP', updated.get('Kp_val', 0.01))
            new_kp = max(old_kp * 0.5, 0.005)
            updated['KP'] = new_kp
            updated['Kp_val'] = new_kp

            old_ki = updated.get('KI', updated.get('Ki_val', 0.02))
            new_ki = max(old_ki * 0.5, 0.005)
            updated['KI'] = new_ki
            updated['Ki_val'] = new_ki

            # Increase slope compensation to quell subharmonic tendencies
            old_slope = updated.get('M_SLOPE', 2e4)
            updated['M_SLOPE'] = max(old_slope * 1.5, 3e4)

            explanation.append(
                f"Mid-frequency loop/LC resonance detected at {dom_freq:.1f}Hz (Ripple: {peak_to_peak_ripple:.2f}V). "
                f"Stabilized loop: Adjusted Kp ({old_kp:.4f} -> {new_kp:.4f}) and Ki ({old_ki:.5f} -> {new_ki:.5f}), "
                f"boosted Cout ({old_c*1e6:.1f}uF -> {new_c*1e6:.1f}uF), and updated Middlebrook damping (Rd={rd} Ohm, Cd={cd*1e6:.1f}uF)."
            )

        # Case 2: High-Frequency Switching Ripple (dom_freq >= 0.4 * f_sw)
        elif peak_to_peak_ripple > target_ripple and dom_freq >= 0.4 * f_sw:
            scale_factor = min(max(peak_to_peak_ripple / target_ripple, 1.5), 5.0)
            c_key = 'Cout_val' if 'Cout_val' in updated else 'C1_val'
            old_c = updated.get(c_key, 100e-6)
            new_c = old_c * scale_factor
            updated[c_key] = new_c
            if 'C2_val' in updated:
                updated['C2_val'] = new_c
            if 'Cout_val' in updated:
                updated['Cout_val'] = new_c
                
            explanation.append(
                f"High-frequency switching ripple ({peak_to_peak_ripple:.2f}V > {target_ripple:.2f}V) detected at {dom_freq:.1f}Hz. "
                f"Scaled output capacitance {c_key} from {old_c*1e6:.1f}uF to {new_c*1e6:.1f}uF (x{scale_factor:.2f})."
            )

        # Case 3: Very Low Frequency Integrator Windup / Hunting (dom_freq < 100 Hz with high ripple)
        elif peak_to_peak_ripple > 2.5 and 0.5 <= dom_freq < 100:
            old_ki = updated.get('KI', updated.get('Ki_val', 0.02))
            new_ki = max(old_ki * 0.5, 0.005)
            updated['KI'] = new_ki
            updated['Ki_val'] = new_ki
            explanation.append(
                f"Low-frequency loop hunting ({dom_freq:.1f}Hz, Ripple: {peak_to_peak_ripple:.2f}V) detected. "
                f"Adjusted integral gain Ki from {old_ki:.4f} to {new_ki:.5f}."
            )

        # Case 4: Output voltage offset (DC shift / Steady-state error)
        v_error = abs(measured_v_avg - target_v)
        if v_error > (0.03 * target_v) and measured_v_avg > 0.5:
            old_ki = updated.get('KI', updated.get('Ki_val', 0.01))
            new_ki = min(max(old_ki * (1.0 + (v_error / target_v)), 0.01), 0.1)
            updated['KI'] = new_ki
            updated['Ki_val'] = new_ki
            explanation.append(
                f"Steady-state DC error is {v_error:.2f}V (Measured {measured_v_avg:.2f}V vs Target {target_v:.2f}V). "
                f"Adjusted Ki to {new_ki:.5f} to eliminate DC offset."
            )

        # Fallback if ripple is still high but wasn't caught above
        if not explanation and peak_to_peak_ripple > target_ripple:
            c_key = 'Cout_val' if 'Cout_val' in updated else 'C1_val'
            old_c = updated.get(c_key, 100e-6)
            new_c = old_c * 2.0
            updated[c_key] = new_c
            explanation.append(
                f"Ripple exceeds target ({peak_to_peak_ripple:.2f}V > {target_ripple:.2f}V). Doubled capacitance to {new_c*1e6:.1f}uF."
            )

        if not explanation:
            explanation.append("Circuit operating within stable envelope. No passive component redesign needed.")

        return updated, "\n".join(explanation)

    @staticmethod
    def apply_params_to_netlist(netlist_content: str, params: Dict[str, Any]) -> str:
        """
        Substitutes calculated values into the SPICE netlist .param lines
        and injects/updates damping elements.
        """
        updated_netlist = netlist_content

        # Update .param statements
        for key, value in params.items():
            formatted_val = CircuitSolver.format_si(value) if isinstance(value, (int, float)) else str(value)
            pattern = rf'(\.param\s+{re.escape(key)}\s*=)[^\s\n\r;]+'
            if re.search(pattern, updated_netlist, re.IGNORECASE):
                updated_netlist = re.sub(pattern, rf'\g<1>{formatted_val}', updated_netlist, flags=re.IGNORECASE)

        # Inject or update Middlebrook damping network if required
        if 'Rd_damp' in params and 'Cd_damp' in params:
            rd_val = CircuitSolver.format_si(params['Rd_damp'])
            cd_val = CircuitSolver.format_si(params['Cd_damp'])
            if 'Rd_damp' in updated_netlist:
                updated_netlist = re.sub(r'(Rd_damp\s+\S+\s+\S+\s+)[^\s\n\r]+', rf'\g<1>{rd_val}', updated_netlist)
                updated_netlist = re.sub(r'(Cd_damp\s+\S+\s+\S+\s+)[^\s\n\r]+', rf'\g<1>{cd_val}', updated_netlist)
            else:
                damping_block = (
                    f"\n* --- Middlebrook LC Damping Branch (Auto-Injected for Stability) ---\n"
                    f"Rd_damp out damp_mid {rd_val}\n"
                    f"Cd_damp damp_mid 0 {cd_val}\n"
                )
                if '.control' in updated_netlist:
                    updated_netlist = updated_netlist.replace('.control', f"{damping_block}\n.control")
                elif '.tran' in updated_netlist:
                    updated_netlist = updated_netlist.replace('.tran', f"{damping_block}\n.tran")
                else:
                    updated_netlist += f"\n{damping_block}\n"

        return updated_netlist

    @staticmethod
    def format_si(val: float) -> str:
        """Formats a float into an engineering notation SPICE suffix (u, m, k, MEG)."""
        if abs(val) < 1e-12:
            return "0"
        abs_v = abs(val)
        if abs_v >= 1e6:
            return f"{val/1e6:.4g}MEG"
        elif abs_v >= 1e3:
            return f"{val/1e3:.4g}k"
        elif abs_v >= 1.0:
            return f"{val:.4g}"
        elif abs_v >= 1e-3:
            return f"{val*1e3:.4g}m"
        elif abs_v >= 1e-6:
            return f"{val*1e6:.4g}u"
        elif abs_v >= 1e-9:
            return f"{val*1e9:.4g}n"
        elif abs_v >= 1e-12:
            return f"{val*1e12:.4g}p"
        return f"{val:.4e}"


if __name__ == '__main__':
    # Self-test
    solver = CircuitSolver()
    buck_params = solver.calculate_passives('buck', vin=120, vout=48, iout=10, fsw=100e3)
    print("Calculated Buck Passives:")
    for k, v in buck_params.items():
        print(f"  {k} = {solver.format_si(v) if isinstance(v, (int, float)) else v}")

    # Test redesign on resonance
    redesigned, note = solver.diagnose_and_redesign(
        buck_params, dom_freq=1200.0, peak_to_peak_ripple=4.5, target_v=48.0, measured_v_avg=47.8
    )
    print("\nRedesign Diagnostic Note:")
    print(note)
    print("\nUpdated Damping:", redesigned.get('Rd_damp'), redesigned.get('Cd_damp'))
