import csv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PLOTTING & ANALYSIS CONFIGURATION
# ==========================================
USE_MARKERS_FOR_RAW_DATA = True 
USE_LOG_SCALE = False 

# Drift thresholds for flagging defective devices (%)
DEFECT_THRESHOLD_SET_PCT = 15.0   
DEFECT_THRESHOLD_RESET_PCT = 30.0 

# Target device for the R-Cycle Endurance plot
TARGET_WL = 5
TARGET_BL = 0

# ------------------------------------------
# 1a. FONT SIZE CONFIGURATION (applies to ALL generated plots)
# ------------------------------------------
FONTSIZE_TITLE       = 25   # plt.title(...)
FONTSIZE_AXIS_LABEL  = 20   # plt.xlabel / plt.ylabel
FONTSIZE_TICK_LABEL  = 20   # x/y tick numbers
FONTSIZE_LEGEND      = 20   # legend entries

# Apply tick-label size globally (title/axis/legend are passed explicitly below)
plt.rcParams['xtick.labelsize'] = FONTSIZE_TICK_LABEL
plt.rcParams['ytick.labelsize'] = FONTSIZE_TICK_LABEL

# ==========================================
# 2. DEFECT STATE CONFIGURATION
# ==========================================
DEFECT_STATE = "NOMINAL" 
CONC_MULT = 1e26 

defect_profiles = {
    "LOWEST_DEFECTIVE": {"r_fil": 10e-9, "N_min": 0.001 * CONC_MULT, "N_max": 1.5 * CONC_MULT},
    "NOMINAL":          {"r_fil": 45e-9, "N_min": 0.008 * CONC_MULT, "N_max": 20.0 * CONC_MULT},
    "LOWER_DEFECT_NMAX":{"r_fil": 45e-9, "N_min": 0.008 * CONC_MULT, "N_max": 4.5 * CONC_MULT},
    "HIGHEST_DEFECTIVE":{"r_fil": 90e-9, "N_min": 4.5 * CONC_MULT,  "N_max": 20.0 * CONC_MULT}
}

current_profile = defect_profiles[DEFECT_STATE]
r_fil, N_min, N_max = current_profile["r_fil"], current_profile["N_min"], current_profile["N_max"]

# ==========================================
# 3. JART PHYSICS PARAMETERS
# ==========================================
l_cell = 3e-9; l_det = 0.5e-9; q = 1.602e-19; un = 8.5e-6; zvo = 2; k_B = 1.381e-23
Nplug = 20.0; A_fil = np.pi * (r_fil**2)
RTiOx = 650; R0 = 719.24; alphaline = 3.92e-3; Rthline = 90471.47; R_contact = 8000
T0 = 293; Rth0 = 15.72e6; Rtheff_scaling = 0.27

R_disc_hrs = l_det / (N_min * zvo * q * un * A_fil)
R_plug_hrs = (l_cell - l_det) / (N_max * zvo * q * un * A_fil)
R_series_static = RTiOx + R0 + R_contact
RHRS_nominal = R_disc_hrs + R_plug_hrs + R_series_static
RLRS_nominal = (l_cell / (N_max * zvo * q * un * A_fil)) + R_series_static

# ==========================================
# 4. 1T1R SIMULATION PARAMETERS
# ==========================================
VWL = 2.0; W_trans = 3e-6; L_trans = 0.3e-6; UN_COX = 120e-6; V_TH = 0.3              

# ==========================================
# 5. JART-ENHANCED 1T1R BEHAVIORAL SIMULATOR
# ==========================================
V_BL_SET = 0.5; V_BL_RESET = 0.45; V_smooth = 0.12       
k_s = 2.5e27; k_r = 15e27; Ea_thermal = 0.30       
SET_EXPONENT = 1.8; dt_total = 0.005; SUB_STEPS = 8          

def soft_ramp(x, x0, w):
    d = x - x0
    if d < -5 * w: return 0.0
    if d > 5 * w: return d
    arg = np.clip(d / max(w, 1e-8), -20, 20)
    return max(d, 0) / (1 + np.exp(-arg))

def simulate_1T1R_curve(v_min, v_max):
    npts = 400
    vbl_sweep = np.concatenate((np.linspace(0, v_max, npts), np.linspace(v_max, 0, npts),
                                np.linspace(0, v_min, npts), np.linspace(v_min, 0, npts)))
    
    if VWL > V_TH:
        I_comp = 0.5 * UN_COX * (W_trans / L_trans) * (VWL - V_TH)**2
        R_on_nmos = 1 / (UN_COX * (W_trans / L_trans) * (VWL - V_TH))
    else: I_comp = 1e-9; R_on_nmos = 1e9
    
    dt = dt_total / SUB_STEPS
    N_current, T_local = N_min, T0
    i_sim = []
    
    for v in vbl_sweep:
        for _ in range(SUB_STEPS):
            R_disc = l_det / (N_current * zvo * q * un * A_fil)
            R_plug = (l_cell - l_det) / (Nplug * 1e26 * zvo * q * un * A_fil)
            R_active = R_disc + R_plug
            R_series = RTiOx + R0 + R_contact
            R_total = R_active + R_series + R_on_nmos
            I_expected = v / R_total
            
            if v >= 0 and I_expected > I_comp: I_expected = I_comp
            elif v < 0 and I_expected < -I_comp: I_expected = -I_comp
            
            V_active = I_expected * R_active
            P_active = abs(V_active * I_expected)
            Rtheff = Rth0 * (Rtheff_scaling if v < 0 else 1.0)
            T_local = min(T0 + P_active * Rtheff, 1500)
            
            R_series = RTiOx + R0 * (1 + R0 * alphaline * I_expected**2 * Rthline) + R_contact
            temp_factor = min(np.exp(Ea_thermal * q / k_B * (1/T0 - 1/T_local)), 30) if T_local > T0 else 1.0
            
            Flim_set   = max(1.0 - (N_current / N_max)**10, 0.0)
            Flim_reset = max(1.0 - (N_min / max(N_current, 1e10))**10, 0.0)
            
            set_drive_raw = soft_ramp(v, V_BL_SET, V_smooth)
            set_drive = set_drive_raw ** SET_EXPONENT if set_drive_raw > 0 else 0.0
            dN_set = k_s * set_drive * (N_max - N_current) / N_max * temp_factor * Flim_set * dt
            
            reset_drive = soft_ramp(-v, V_BL_RESET, V_smooth)
            dN_reset = k_r * reset_drive * (N_current - N_min) / N_max * temp_factor * Flim_reset * dt
            
            N_current = np.clip(N_current + dN_set - dN_reset, N_min, N_max)
        i_sim.append(I_expected)
    return vbl_sweep, np.array(i_sim)

def calc_R(v, i_uA):
    with np.errstate(divide='ignore', invalid='ignore'): 
        return np.abs(v / (i_uA * 1e-6))

# ==========================================
# 6. DATA EXTRACTION & READ-RESISTANCE SAMPLING
# ==========================================
all_set_x, all_set_y, all_reset_x, all_reset_y = [], [], [], []
device_data = {} 
global_max_v, global_min_v = 1.2, -1.8 
print("Parsing files and extracting cycle-by-cycle read resistance after switching...")

for k in range(0, 20):
    for i in range(0, 7):
        for j in range(0, 7):
            if (i, j) not in device_data:
                device_data[(i, j)] = {'set_x': [], 'set_y': [], 'reset_x': [], 'reset_y': [],
                                       'R_HRS_cycles': [], 'R_LRS_cycles': [], 
                                       'cycles_set': [], 'cycles_reset': []}
                
            # --- SET SWEEP (Extracting LRS at the END of the sweep) ---
            try:
                set_filename = f'../rram_measurement_data_old/2022-09-20/TEST/Set_QS_WL{i}_BL{j}_{k}.csv'
                with open(set_filename, 'r') as f:
                    reader = csv.reader(f); next(reader)
                    x_set, y_set = [], []
                    for line in reader:
                        y_set.append(float(line[1]) * 1e6) 
                        x_set.append(float(line[4]))
                    
                    if x_set and y_set:
                        global_max_v = max(global_max_v, max(x_set))
                        all_set_x.append(np.array(x_set)); all_set_y.append(np.array(y_set))
                        device_data[(i, j)]['set_x'].append(np.array(x_set))
                        device_data[(i, j)]['set_y'].append(np.array(y_set))
                        
                        # Calculate R_LRS (R_SET) at the END of the SET sweep (return ramp down to 0V)
                        v_arr = np.array(x_set); i_arr = np.array(y_set) * 1e-6
                        half_idx = max(1, len(v_arr) // 2)
                        mask = (v_arr[half_idx:] >= 0.05) & (v_arr[half_idx:] <= 0.25)
                        r_lrs = np.nan
                        if np.any(mask):
                            with np.errstate(divide='ignore', invalid='ignore'):
                                r_vals = np.abs(v_arr[half_idx:][mask] / i_arr[half_idx:][mask])
                                r_vals = r_vals[np.isfinite(r_vals)]
                                if len(r_vals) > 0: r_lrs = np.median(r_vals)
                        
                        device_data[(i, j)]['R_LRS_cycles'].append(r_lrs)
                        device_data[(i, j)]['cycles_set'].append(k)
            except FileNotFoundError: pass

            # --- RESET SWEEP (Extracting HRS at the END of the sweep) ---
            try:
                reset_filename = f'../rram_measurement_data_old/2022-09-20/TEST/Reset_QS_WL{i}_BL{j}_{k}.csv'
                with open(reset_filename, 'r') as f:
                    reader = csv.reader(f); next(reader) 
                    x_reset, y_reset = [], []
                    for line in reader:
                        y_reset.append(float(line[3]) * 1e6) 
                        x_reset.append(-float(line[6]))
                    
                    if x_reset and y_reset:
                        global_min_v = min(global_min_v, min(x_reset))
                        all_reset_x.append(np.array(x_reset)); all_reset_y.append(np.array(y_reset))
                        device_data[(i, j)]['reset_x'].append(np.array(x_reset))
                        device_data[(i, j)]['reset_y'].append(np.array(y_reset))

                        # Calculate R_HRS (R_RESET) at the END of the RESET sweep (return ramp up to 0V)
                        v_arr = np.array(x_reset); i_arr = np.array(y_reset) * 1e-6
                        half_idx = max(1, len(v_arr) // 2)
                        mask = (v_arr[half_idx:] <= -0.05) & (v_arr[half_idx:] >= -0.25)
                        r_hrs = np.nan
                        if np.any(mask):
                            with np.errstate(divide='ignore', invalid='ignore'):
                                r_vals = np.abs(v_arr[half_idx:][mask] / i_arr[half_idx:][mask])
                                r_vals = r_vals[np.isfinite(r_vals)]
                                if len(r_vals) > 0: r_hrs = np.median(r_vals)
                        
                        device_data[(i, j)]['R_HRS_cycles'].append(r_hrs)
                        device_data[(i, j)]['cycles_reset'].append(k)
            except FileNotFoundError: pass


# ==========================================
# 7. GLOBAL CALCULATIONS & AVERAGING
# ==========================================
# 7.1 Global Measured Average Traces (for plotting IV/RV lines)
set_x_avg, set_y_avg = None, None
reset_x_avg, reset_y_avg = None, None

if all_set_x and all_set_y:
    min_len_set = min(len(arr) for arr in all_set_y)
    set_y_array = np.array([arr[:min_len_set] for arr in all_set_y])
    set_y_avg = np.mean(set_y_array, axis=0)
    set_x_avg = all_set_x[0][:min_len_set] 

if all_reset_x and all_reset_y:
    min_len_reset = min(len(arr) for arr in all_reset_y)
    reset_y_array = np.array([arr[:min_len_reset] for arr in all_reset_y])
    reset_y_avg = np.mean(reset_y_array, axis=0)
    reset_x_avg = all_reset_x[0][:min_len_reset]

# 7.2 Global Median States (for device defect comparison)
all_dev_hrs, all_dev_lrs = [], []

for (i, j), data in device_data.items():
    hrs_arr = np.array(data['R_HRS_cycles'])
    lrs_arr = np.array(data['R_LRS_cycles'])
    
    if len(hrs_arr) > 0 and not np.all(np.isnan(hrs_arr)):
        data['avg_R_HRS'] = np.nanmean(hrs_arr)
        all_dev_hrs.append(data['avg_R_HRS'])
    else: data['avg_R_HRS'] = np.nan
        
    if len(lrs_arr) > 0 and not np.all(np.isnan(lrs_arr)):
        data['avg_R_LRS'] = np.nanmean(lrs_arr)
        all_dev_lrs.append(data['avg_R_LRS'])
    else: data['avg_R_LRS'] = np.nan

# R_HRS_measured = np.nanmedian(all_dev_hrs) if all_dev_hrs else None
# R_LRS_measured = np.nanmedian(all_dev_lrs) if all_dev_lrs else None

R_HRS_measured = np.nanmean(all_dev_hrs) if all_dev_hrs else None
R_LRS_measured = np.nanmean(all_dev_lrs) if all_dev_lrs else None

# R_HRS_measured = 50000
# R_LRS_measured = 6500

sim_x, sim_y = simulate_1T1R_curve(global_min_v, global_max_v)
sim_y_uA = sim_y * 1e6 


# ==========================================
# 8. CYCLE-AVERAGED DEFECT FLAGGING
# ==========================================
if R_HRS_measured and R_LRS_measured:
    print("\n" + "="*55)
    print("      DEFECTIVE DEVICE LOG (CYCLE-AVERAGED DRIFT)")
    print("="*55)
    print(f"Global Median R_SET  (LRS): {R_LRS_measured:.2f} Ohm")
    print(f"Global Median R_RESET(HRS): {R_HRS_measured:.2f} Ohm")
    print(f"Thresholds -> SET: ±{DEFECT_THRESHOLD_SET_PCT}%, RESET: ±{DEFECT_THRESHOLD_RESET_PCT}%")
    print("-" * 55)

    for (i, j), data in sorted(device_data.items()):
        dev_R_HRS = data.get('avg_R_HRS', np.nan)
        dev_R_LRS = data.get('avg_R_LRS', np.nan)
        
        if np.isnan(dev_R_HRS) or np.isnan(dev_R_LRS): continue

        drift_set_pct = ((dev_R_LRS - R_LRS_measured) / R_LRS_measured) * 100
        drift_reset_pct = ((dev_R_HRS - R_HRS_measured) / R_HRS_measured) * 100

        is_set_defective = abs(drift_set_pct) > DEFECT_THRESHOLD_SET_PCT
        is_reset_defective = abs(drift_reset_pct) > DEFECT_THRESHOLD_RESET_PCT

        if is_set_defective or is_reset_defective:
            flags = []
            if is_set_defective: flags.append("SET")
            if is_reset_defective: flags.append("RESET")
            print(f"Device WL{i}_BL{j} flagged for {' & '.join(flags)}:")
            
            if is_set_defective:
                sign = "+" if drift_set_pct > 0 else ""
                print(f"  -> R_SET drift:   {sign}{drift_set_pct:.2f}% (Avg: {dev_R_LRS:.2f} Ohm)")
            if is_reset_defective:
                sign = "+" if drift_reset_pct > 0 else ""
                print(f"  -> R_RESET drift: {sign}{drift_reset_pct:.2f}% (Avg: {dev_R_HRS:.2f} Ohm)")
    print("="*55 + "\n")


# ==========================================
# 9. PLOT 1: I-V HYSTERESIS
# ==========================================
plt.figure(figsize=(8, 6))

for x, y in zip(all_set_x, all_set_y):
    y_plot = np.abs(y) if USE_LOG_SCALE else y
    if USE_MARKERS_FOR_RAW_DATA: plt.plot(x, y_plot, color='blue', marker='o', markerfacecolor='none', linestyle='None', alpha=0.1)
    else: plt.plot(x, y_plot, color='blue', linestyle='-', alpha=0.1)

for x, y in zip(all_reset_x, all_reset_y):
    y_plot = np.abs(y) if USE_LOG_SCALE else y
    if USE_MARKERS_FOR_RAW_DATA: plt.plot(x, y_plot, color='red', marker='o', markerfacecolor='none', linestyle='None', alpha=0.1)
    else: plt.plot(x, y_plot, color='red', linestyle='-', alpha=0.1)

if set_x_avg is not None:
    y_plot_avg = np.abs(set_y_avg) if USE_LOG_SCALE else set_y_avg
    plt.plot(set_x_avg, y_plot_avg, color='cyan', linewidth=2.5, label='Average Measured Set')
if reset_x_avg is not None:
    y_plot_avg = np.abs(reset_y_avg) if USE_LOG_SCALE else reset_y_avg
    plt.plot(reset_x_avg, y_plot_avg, color='black', linewidth=2.5, label='Average Measured Reset')

sim_y_plot = np.abs(sim_y_uA) if USE_LOG_SCALE else sim_y_uA
# plt.plot(sim_x, sim_y_plot, color='lime', linewidth=3, linestyle='--', label=f'JART Sim ({DEFECT_STATE})')

plt.xlabel('Voltage [V]', fontsize=FONTSIZE_AXIS_LABEL)
if USE_LOG_SCALE:
    plt.yscale('log')
    plt.ylabel('|Current| [uA]', fontsize=FONTSIZE_AXIS_LABEL)
    plt.title(f'I-V Hysteresis: Measured vs 1T1R Simulated (Log)\nDefect Profile: {DEFECT_STATE}', fontsize=FONTSIZE_TITLE)
    out_iv = f'IV_hysteresis_{DEFECT_STATE}_log.png'
else:
    plt.ylabel('Current [uA]', fontsize=FONTSIZE_AXIS_LABEL)
    # plt.title(f'I-V Hysteresis: Measured vs 1T1R Simulated (Linear)\nDefect Profile: {DEFECT_STATE}', fontsize=FONTSIZE_TITLE)
    plt.title(f'Measured I-V Hysteresis', fontsize=FONTSIZE_TITLE)
    out_iv = f'IV_hysteresis_{DEFECT_STATE}_linear.png'

plt.grid(True, which="both", ls="--", alpha=0.3)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=FONTSIZE_LEGEND)
plt.tight_layout(); plt.savefig(out_iv, dpi=300); plt.close() 

# ==========================================
# 10. PLOT 2: R-V HYSTERESIS 
# ==========================================
plt.figure(figsize=(8, 6))

for x, y in zip(all_set_x, all_set_y):
    r_val = calc_R(x, y)
    if USE_MARKERS_FOR_RAW_DATA: plt.plot(x, r_val, color='blue', marker='o', markerfacecolor='none', linestyle='None', alpha=0.1)
    else: plt.plot(x, r_val, color='blue', linestyle='-', alpha=0.1)

for x, y in zip(all_reset_x, all_reset_y):
    r_val = calc_R(x, y)
    if USE_MARKERS_FOR_RAW_DATA: plt.plot(x, r_val, color='red', marker='o', markerfacecolor='none', linestyle='None', alpha=0.1)
    else: plt.plot(x, r_val, color='red', linestyle='-', alpha=0.1)

if R_HRS_measured is not None and R_LRS_measured is not None:
    plt.axhline(y=R_HRS_measured, color='black', linestyle='-.', linewidth=2.5, label='Median $R_{RESET}$ (Measured)')
    plt.axhline(y=R_LRS_measured, color='cyan', linestyle='-.', linewidth=2.5, label='Median $R_{SET}$ (Measured)')

plt.axhline(y=RHRS_nominal, color='lime', linestyle=':', linewidth=2.5, label=f'Sim $R_{{RESET}}$ ({DEFECT_STATE})')
plt.axhline(y=RLRS_nominal, color='orange', linestyle=':', linewidth=2.5, label=f'Sim $R_{{SET}}$ ({DEFECT_STATE})')

plt.xlabel('Voltage [V]', fontsize=FONTSIZE_AXIS_LABEL)
plt.ylabel('Resistance [Ohm]', fontsize=FONTSIZE_AXIS_LABEL)

if USE_LOG_SCALE:
    plt.yscale('log')
    plt.title(f'R-V Characteristics: Measured vs Dynamic JART (Log Scale)\nDefect Profile: {DEFECT_STATE}', fontsize=FONTSIZE_TITLE)
    plt.ylim(min(1e3, RLRS_nominal * 0.5), max(1e7, RHRS_nominal * 2)) 
    out_rv = f'RV_characteristics_{DEFECT_STATE}_log.png'
else:
    plt.title(f'R-V Characteristics: Measured vs Dynamic JART (Linear Scale)\nDefect Profile: {DEFECT_STATE}', fontsize=FONTSIZE_TITLE)
    plt.ylim(0, RHRS_nominal * 1.5) 
    out_rv = f'RV_characteristics_{DEFECT_STATE}_linear.png'

plt.grid(True, which="both", ls="--", alpha=0.3)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=FONTSIZE_LEGEND)
plt.tight_layout(); plt.savefig(out_rv, dpi=300); plt.close()

# ==========================================
# 11. PLOT 3: R-CYCLE ENDURANCE (TARGET DEVICE)
# ==========================================
if (TARGET_WL, TARGET_BL) in device_data:
    data = device_data[(TARGET_WL, TARGET_BL)]
    c_set = data['cycles_set']
    r_hrs = data['R_HRS_cycles']
    c_reset = data['cycles_reset']
    r_lrs = data['R_LRS_cycles']
    
    if c_set and c_reset:
        plt.figure(figsize=(8, 6))
        plt.plot(c_set, r_hrs, marker='o', color='black', label='$R_{RESET}$ (HRS)', linewidth=2, markersize=8)
        plt.plot(c_reset, r_lrs, marker='s', color='cyan', label='$R_{SET}$ (LRS)', linewidth=2, markersize=8)
        
        plt.xlabel('Cycle Number', fontsize=FONTSIZE_AXIS_LABEL)
        
        if USE_LOG_SCALE:
            plt.yscale('log')
            plt.ylabel('Resistance [Ohm] (Log Scale)', fontsize=FONTSIZE_AXIS_LABEL)
            plt.title(f'R-Cycle Endurance Plot (Log Scale)\nTarget Device: WL{TARGET_WL}_BL{TARGET_BL}', fontsize=FONTSIZE_TITLE)
            out_rcycle = f'R_cycle_WL{TARGET_WL}_BL{TARGET_BL}_log.png'
        else:
            plt.ylabel('Resistance [Ohm] (Linear Scale)', fontsize=FONTSIZE_AXIS_LABEL)
            plt.title(f'R-Cycle Endurance Plot (Linear Scale)\nTarget Device: WL{TARGET_WL}_BL{TARGET_BL}', fontsize=FONTSIZE_TITLE)
            out_rcycle = f'R_cycle_WL{TARGET_WL}_BL{TARGET_BL}_linear.png'
            
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='best', fontsize=FONTSIZE_LEGEND)
        plt.tight_layout()
        plt.savefig(out_rcycle, dpi=300)
        plt.close()
        print(f"-> Generated R-Cycle Endurance Plot for WL{TARGET_WL}_BL{TARGET_BL}")
else:
    print(f"-> Target Device WL{TARGET_WL}_BL{TARGET_BL} data not found for R-Cycle plotting.")