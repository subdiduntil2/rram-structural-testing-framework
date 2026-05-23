import pandas as pd
import json
import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict, Counter
from scipy.optimize import linprog
import json
import sys
import matplotlib
matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         22,
    'axes.labelsize':    26,
    'axes.titlesize':    26,
    'xtick.labelsize':   20,
    'ytick.labelsize':   20,
    'axes.linewidth':    1.2,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'savefig.dpi':       300,
    'figure.dpi':        150,
})

# # ETS26 dat (not really working, works activation_function_dat.py but don't need it) 
# FOLDER_LOCATION = "For_ETS26_v1" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 1 # or 1 for better results
# DETECTION_MODE = "mapped_digital" # only
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc26_DA_1.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc26_DA_1.txt")

# ITC25_v4 (old)
# FOLDER_LOCATION = "For_ITC25_v4" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")

# DETECTION_THRESHOLD = 2
# DETECTION_MODE = "mapper_digital" # only
# ANALOG_DETECTION_THRESHOLD = 0.0005   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json

# --- Pipeline Execution Options ---
ENABLE_MATRIX_GEN = True       # Set to True to apply thresholds and create the detectability .xlsx matrix
ENABLE_PLOTTING = True       # Set to True to generate the visual grids and summary CSVs
ENABLE_TPG = True

# --- Configuration ---
# First ETS25
# FOLDER_LOCATION = "For_ETS_25_results" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 2 # not used
# DETECTION_MODE = "direct_analog" # only
# ANALOG_DETECTION_THRESHOLD = 0.001   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_ets25.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_ets25.txt")

# ITC25_v4_test (ITC25 2T2R res w+r)
# FOLDER_LOCATION = "For_ITC25_v4_test" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper_corrected_final_ohms_fix.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 2 
# DETECTION_MODE = "mapped_digital" 
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc25_res_2t2r.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc25_res_2t2r.txt")

# ITC25 8x2 VCOs
# FOLDER_LOCATION = "For_ITC26_all_vcos_ITC25" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "combined_CVCO5_measurements.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 1 # or 1 for better results
# DETECTION_MODE = "mapper_digital" # only
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc25_vcos.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc25_vcos.txt")

# ETS26 DAT 32x2
# FOLDER_LOCATION = "For_ETS26_v2" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper_exp_filtered.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 1
# DETECTION_MODE = "mapped_digital"
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_ets26_DAT32.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_ets26_DAT32.txt") 

# 2T2R 32x2 unbalanced
# FOLDER_LOCATION = "For_ITC26_v1" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper_itc26_v2.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 1
# DETECTION_MODE = "mapped_digital" # only
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc26_2t2r_res_unbalanced.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc26_2t2r_res_unbalanced.txt")

# ITC26 V0+V0.5 corrected 32x2
# FOLDER_LOCATION = "For_ITC26_vco_32x2_corrected" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper_itc26_vco_32x2.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 1
# DETECTION_MODE = "mapper_digital" # only
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc26_vco_small_1.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc26_vco_small_1.txt")

# ITC26 final VCO 32x2 corrected (has all ITC25 8x2 data that are not in the 32x2-corrected)
# Have to turn patterns 8x2->32x2 and combine with resimulated 32x2
# FOLDER_LOCATION = "For_ITC26_vco_final" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "combined_CVCO5_measurements_pruned.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_pruned.json")
# DETECTION_THRESHOLD = 1 # more realistic with 2
# DETECTION_MODE = "mapper_digital" # only
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = False  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc26_vco_pruned.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc26_vco_pruned.txt") 

# Choose from: "merged", "read-only", or "write-based" or "forming-1" or "forming-2"
MIXED_DATA_MODE = "merged"

# to be compared with For_ITC26_v1 
# FOLDER_LOCATION = "For_ITC26_part_balanced" 
# MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper_itc26_part_balanced_v2.csv")
# JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
# DETECTION_THRESHOLD = 2
# DETECTION_MODE = "mapped_digital" # only
# ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
# ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
# TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc26_2t2r_all_part_balanced.txt")
# VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc26_2t2r_all_part_balanced.txt") 

FOLDER_LOCATION = "For_ITC26_full_balanced" 
MAPPER_FILE = os.path.join(FOLDER_LOCATION, "mapper_itc26_full_balanced.csv")
JSON_FILE = os.path.join(FOLDER_LOCATION, "fault_dict_corrected.json")
DETECTION_THRESHOLD = 1
DETECTION_MODE = "mapped_digital" # only
ANALOG_DETECTION_THRESHOLD = 0.002   # Used ONLY if mode or fallback hits 'direct_analog'
ENABLE_FAULT_DICT_GEN = True  # Set to True to rescan .scs files and build fault_dict.json
TPG_LOG_FILE = os.path.join(FOLDER_LOCATION, "tpg_solver_log_itc26_2t2r_all_full_balanced.txt")
VENN_LOG_FILE = os.path.join(FOLDER_LOCATION, "venn_log_itc26_2t2r_all_full_balanced.txt") 

# --- Auto-Generated Paths & Settings ---
DEFECTIVE_CELLS_DIR = os.path.join(FOLDER_LOCATION, "defective_cells")
OUTPUT_FILE = os.path.join(FOLDER_LOCATION, "final_combined_results.csv")
RECREATED_RESULT_FILE = os.path.join(FOLDER_LOCATION, "recreated_result_file.xlsx")

ADC_MAPPING_FILE ="adc_df.csv"
DEFECT_STRENGHT_NUM = 10
PLOT_MODE = "combined" # "combined" or "adc_diff"

# Choose from: "mapper_digital", "direct_analog", or "mapped_digital"
print("detection threshold is set to => ", DETECTION_THRESHOLD)
def generate_fault_dict(defective_cells_dir, output_json_path):
    print(f"Scanning .scs files in {defective_cells_dir} to generate fault dictionary...")
    fault_dict = {}
    missing_rdef_count = 0
    scs_files = glob.glob(os.path.join(defective_cells_dir, '*.scs'))
    
    if not scs_files:
        print(f"Warning: No .scs files found in {defective_cells_dir}.")
        return

    for filename in scs_files:
        adc_mac_match = re.search(r'(ADC_MAC_1n1m_st_schematic_\d+|ADC_MAC_1n1m_st_rev_schematic_\d+|CVCO5_V1_\d+|CVCO5_V3_\d+|CVCO5_V0_\d+|CVCO5_V05_\d+)', filename)
        if not adc_mac_match:
            continue
            
        cell_name = adc_mac_match.group(0)
        is_golden = cell_name.endswith('_0')
        
        # Auto-assign Mixed Data Non-Resistive Properties for the specific range of ADC_MAC and CVCO5 cells
        num_match = re.search(r'_(\d+)$', cell_name)
        cell_num = int(num_match.group(1)) if num_match else -1
        
        if 181 <= cell_num <= 210:
            if 181 <= cell_num <= 190:
                fault_dict[cell_name] = ["r_def", "Ndiscmin", "Ndiscmin", cell_num - 180]
            elif 191 <= cell_num <= 200:
                fault_dict[cell_name] = ["r_def", "rdet", "rdet", cell_num - 190]
            elif 201 <= cell_num <= 210:
                fault_dict[cell_name] = ["r_def", "Ndiscmax", "Ndiscmax", cell_num - 200]
            continue  # Skip standard r_def parsing for this range

        defect_detection_flag = False
        
        with open(filename, 'r') as f:
            for line in f:
                if 'r_def' in line and 'resistor r=' in line:
                    try:
                        nets_for_defect = list(set(re.findall(r'\((.*?)\)', line)[0].split()))
                        value_str = line.split("resistor r=", 1)[1].strip()
                        value = int(float(value_str))
                        
                        fault_dict[cell_name] = ["r_def", nets_for_defect[0], nets_for_defect[1], value]
                        defect_detection_flag = True
                        break
                    except Exception as e:
                        print(f"Error parsing r_def in {filename}: {e}")
        
        if not defect_detection_flag:
            fault_dict[cell_name] = ["r_def", "None", "None", 0]
            if not is_golden and "ADC_MAC" in cell_name:
                missing_rdef_count += 1

    if missing_rdef_count > 1:
        print(f"Detected {missing_rdef_count} ADC_MAC cells missing 'r_def'. Activating DA modeling creation...")
        fault_dict = {} 
        for prefix in ["ADC_MAC_1n1m_st_schematic", "ADC_MAC_1n1m_st_rev_schematic"]:
            for i in range(31):
                key = f"{prefix}_{i}"
                if i == 0:
                    fault_dict[key] = ["r_def", "None", "None", 0]
                elif 1 <= i <= 10:
                    fault_dict[key] = ["r_def", "Ndiscmin", "Ndiscmin", i]
                elif 11 <= i <= 20:
                    fault_dict[key] = ["r_def", "rdet", "rdet", i - 10]
                elif 21 <= i <= 30:
                    fault_dict[key] = ["r_def", "Ndiscmax", "Ndiscmax", i - 20]

    with open(output_json_path, 'w') as f:
        json.dump(fault_dict, f, indent=4)
        
    print(f"Successfully created {output_json_path} with {len(fault_dict)} elements.\n")

def check_resistive_difference(init_rp, init_rn, final_rp, final_rn):
    init_state = 0
    if (init_rp == 20.0 or init_rp == 20) and init_rn in [0.01, 0.008]:
        init_state = 1
    elif init_rp in [0.01, 0.008] and (init_rn == 20.0 or init_rn == 20):
        init_state = -1
        
    final_state = 0
    if final_rp < 10000 and final_rn > 30000:
        final_state = 1
    elif final_rp > 30000 and final_rn < 10000:
        final_state = -1
            
    return final_state - init_state

def encode_combined_state(diff, detect):
    return int((diff + 2) * 2 + detect)

def calculate_input_score(input_combo):
    match = re.search(r'rp_([0-9\.]+)_rn_([0-9\.]+)_inp_([0-9\.]+)_inn_([0-9\.]+)_neighs_([A-Z0-9]{4})', input_combo)
    if not match: return 0 
        
    rp, rn = float(match.group(1)), float(match.group(2))
    inp, inn = float(match.group(3)), float(match.group(4))
    neighs = match.group(5)
    
    victim_score = 0
    if np.isclose(rp, rn, atol=1e-4) or (np.isclose(inp, 0.55) and np.isclose(inn, 0.55)):
        victim_score = 0
    elif (rp == 20.0 or rp == 20) and (rn in [0.01, 0.008]) and (inp == 0.85) and (inn == 0.25):
        victim_score = 1
    elif (rp in [0.01, 0.008]) and (rn == 20.0 or rn == 20) and (inp == 0.25) and (inn == 0.85):
        victim_score = 1
    elif (rp == 20.0 or rp == 20) and (rn in [0.01, 0.008]) and (inp == 0.25) and (inn == 0.85):
        victim_score = -1
    elif (rp in [0.01, 0.008]) and (rn == 20.0 or rn == 20) and (inp == 0.85) and (inn == 0.25):
        victim_score = -1

    char_map = {'P': 1, 'N': -1, '0': 0}
    n_score = sum(char_map.get(neighs[i], 0) for i in range(3))
    n_score += char_map.get(neighs[3], 0) * 5 
    
    return victim_score + n_score

def get_group_keys(s):
    key_3 = 0
    
    # Extract rp and rn safely
    match = re.search(r'rp_([0-9\.]+)_rn_([0-9\.]+)', s)
    if match:
        rp = float(match.group(1))
        rn = float(match.group(2))
        
        # Use mathematical comparison instead of strict strings
        if rp > rn: 
            key_3 = 1
        elif rp < rn: 
            key_3 = 2
        elif np.isclose(rp, rn, atol=1e-4): 
            key_3 = 3

    key_2 = 0
    if "_ninit" in s:
        if "ninit_20" in s or "ninit_20.0" in s: key_2 = 0
        elif "ninit_0" in s or "ninit_0.0" in s: key_2 = 1

    try:
        extracted = s.split("neighs_")[1][:4]
        key_1 = extracted[0:3].count('0')
        key_0 = extracted[3].count('0')
    except:
        key_1, key_0 = 0, 0
        
    return key_3, key_2, key_1, key_0

def generate_plots(df_final, fault_dict, has_ninit, mode="combined"):
    num_rows = 6 if has_ninit else 3
    print(f"Generating Plots (Mode: {mode.upper()} | ninit detected: {has_ninit} -> {num_rows * 8} Tiles)...")
    
    inputs = df_final.columns.tolist()
    
    grouped_inputs = defaultdict(list)
    for inp in inputs:
        k3, k2, k1, k0 = get_group_keys(inp)
        if k3 == 0: k3 = 1 
        
        if has_ninit:
            row = (k3 - 1) * 2 + k2 
        else:
            row = (k3 - 1)
            
        col = k1 * 2 + k0       
        grouped_inputs[(row, col)].append(inp)
    
    defect_groups = defaultdict(list)
    for defect_id, props in fault_dict.items():
        if defect_id in df_final.index:
            arch = defect_id.rsplit('_', 1)[0] 
            group_key = (arch, props[0], props[1], props[2]) 
            defect_groups[group_key].append({'id': defect_id, 'res': props[3]})
        
    if mode == "combined":
        colors = ['#fbb4b9', '#c51b8a', '#fdd49e', '#d95f0e', '#e0e0e0', '#525252', '#c6dbef', '#3182bd', '#c7e9c0', '#31a354']
        cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(-0.5, 10.5, 1) 
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
    elif mode == "binary":
        colors = ['#e0e0e0', '#d95f0e']
        cmap = mcolors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
    else: 
        cmap = plt.cm.plasma
        vmax = max(1, df_final.values.max()) 
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    
    if has_ninit:
        row_labels = ["rp > rn\nninit_20", "rp > rn\nninit_0", 
                      "rp < rn\nninit_20", "rp < rn\nninit_0", 
                      "rp = rn\nninit_20", "rp = rn\nninit_0"]
        
        row_labels = ["1w1", "0w1", 
                      "0w-1", "-1w-1", 
                      "0w0", "-1w0"]
    else:
        row_labels = ["rp > rn", "rp < rn", "rp = rn"]
        row_labels = ["1", "-1", "0"]
        
    col_labels_top = ["***", "**0", "*00", "000"]
    
    cbar_labels = [
        '(-2) Masked', '(-2) Detected',
        '(-1) Masked', '(-1) Detected',
        '(0) Masked', '(0) Detected',
        '(+1) Masked', '(+1) Detected',
        '(+2) Masked', '(+2) Detected'
    ]

    for group_key, items in defect_groups.items():
        arch, def_type, node1, node2 = group_key
        
        items_sorted = sorted(items, key=lambda x: x['res'])
        defect_ids = [item['id'] for item in items_sorted]
        res_values = [item['res'] for item in items_sorted]
        
        fig_height = 18 if has_ninit else 11
        fig, axes = plt.subplots(num_rows, 8, figsize=(30, fig_height), sharex=True)
        
        title_suffix = "48 Categories Grid" if has_ninit else "24 Categories Grid"
        if mode == "combined":
            title_text = f"Cell: {arch} | Defect: {def_type} | Nodes: {node1} - {node2}\nDiff & Detection {title_suffix} (Hue=Diff, Shade=Detect)"
            # title_text = f"Cell: VCO ADC | Defect: Resistive ({node1} - {node2})"
            # title_text = f"Cell: RRAM | Defect: Resistive (Nodes: {node1} - {node2})"
            # title_text = f"Cell: RRAM | Defect: Device (Reset)"
        elif mode == "binary":
            title_text = f"Cell: {arch} | Defect: {def_type} | Nodes: {node1} - {node2}\nBinary Detection {title_suffix}"
        else:
            title_text = f"Cell: {arch} | Defect: {def_type} | Nodes: {node1} - {node2}\nAbsolute ADC Difference {title_suffix}"
            
        fig.suptitle(title_text, fontsize=35, fontweight='bold', y=0.95)
        
        for row in range(num_rows):
            for col in range(8):
                ax = axes[row, col]
                
                tile_inputs = grouped_inputs.get((row, col), [])
                tile_inputs = sorted(tile_inputs, key=lambda x: (calculate_input_score(x), x))
                
                if not tile_inputs:
                    ax.axis('off')
                    continue
                
                plot_data = np.zeros((len(tile_inputs), len(defect_ids)))
                for i, inp_name in enumerate(tile_inputs):
                    for j, def_id in enumerate(defect_ids):
                        plot_data[i, j] = df_final.at[def_id, inp_name]
                        
                cax = ax.imshow(plot_data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest', origin='upper')
                ax.set_yticks([]) 
                
                if row == num_rows - 1:
                    ax.set_xticks(np.arange(len(res_values)))
                    ax.set_xticklabels([f"{r:,}" for r in res_values], rotation=90, ha='center', fontsize=25)
                
                if row == 0:
                    first_3_str = col_labels_top[col//2]
                    last_1_str = "0" if col%2==1 else "*"
                    ax.set_title(f"{first_3_str}|{last_1_str}", fontsize=30, pad=10)
                
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=30, fontweight='bold', rotation=0, labelpad=50, va='center')

        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
        if mode == "combined":
            cbar = fig.colorbar(cax, cax=cbar_ax, ticks=np.arange(10))
            cbar.ax.set_yticklabels(cbar_labels, fontsize=25)
            cbar.ax.tick_params(length=0)
        elif mode == "binary":
            cbar = fig.colorbar(cax, cax=cbar_ax, ticks=[0, 1])
            cbar.ax.set_yticklabels(['Undetected (0)', 'Detected (1)'], fontsize=25)
        else:
            cbar = fig.colorbar(cax, cax=cbar_ax)
            cbar.ax.set_ylabel("Absolute ADC Difference", rotation=270, labelpad=30, fontweight='bold', fontsize=24)
        
        plt.subplots_adjust(left=0.09, right=0.89, top=0.90 if has_ninit else 0.86, bottom=0.10, wspace=0.12, hspace=0.25)
        
        safe_name = f"plot_{arch}_{def_type}_{node1}_{node2}_{mode}_grid.png".replace(" ", "")
        plt.savefig(os.path.join(FOLDER_LOCATION, safe_name), dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        
    print(f"Success! {len(defect_groups)} grid plots generated and saved.\n")

def get_homogeneity_features(pattern):
    bulk = '0'
    match_bulk = re.search(r'rp_([0-9\.]+)_rn_([0-9\.]+)', pattern)
    if match_bulk:
        rp = float(match_bulk.group(1))
        rn = float(match_bulk.group(2))
        if rp > rn:
            bulk = 'X' 
        elif rp < rn:
            bulk = 'X' 
        else:
            bulk = '0'
            
    neighs = ['0', '0', '0', '0']
    match_neighs = re.search(r'neighs_([A-Z0-9]{4})', pattern)
    if match_neighs:
        n_str = match_neighs.group(1)
        for i in range(4):
            if n_str[i] in ['P', 'N']:
                neighs[i] = 'X' 
            else:
                neighs[i] = '0'
                
    return (bulk, neighs[0], neighs[1], neighs[2], neighs[3])

def calculate_homogeneity_score(solution):
    if not solution: return 0
    features = [get_homogeneity_features(p) for p in solution]
    
    score = 0
    for col in range(5):
        col_vals = [f[col] for f in features]
        counts = Counter(col_vals)
        most_common_freq = counts.most_common(1)[0][1]
        score += most_common_freq
        
    for row_features in features:
        counts = Counter(row_features)
        most_common_freq = counts.most_common(1)[0][1]
        score += most_common_freq
        
    return score

def generate_test_patterns(excel_file, adc_diff_file, combined_file, log_file, venn_log_file, defect_names_map=None):    
    print(f"\n{'='*60}")
    print(f"Starting Test Pattern Generation Solver")
    print(f"{'='*60}")
    
    # vvv Changed this block to open both files and define a second print helper vvv
    with open(log_file, "w") as f_log, open(venn_log_file, "w") as f_venn:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, file=f_log, **kwargs)

        def venn_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, file=f_venn, **kwargs)

        try:
            df = pd.read_excel(excel_file, index_col=0)
        except Exception as e:
            log_print(f"Failed to load {excel_file} for TPG solver: {e}")
            return
            
        log_print(f"df before solver => \n{df}")
        
        matrix = np.array(df.values)
        matrix[matrix < 0.5] = int(0)
        matrix[matrix >= 0.5] = int(1)
        
        num_rows, num_cols = matrix.shape
        total_defects = num_cols
        
        undetectable_defects = np.where(np.all(matrix == 0, axis=0))[0]
        undetectable_names = df.iloc[:, undetectable_defects].columns.tolist()
        undetectable_count = len(undetectable_defects)
        
        log_print(f"undetectable defs =>  {undetectable_defects} {np.shape(undetectable_defects)}")
        log_print(f"lets see {undetectable_names}")
        
        detectable_matrix = np.delete(matrix, undetectable_defects, axis=1)
        log_print(f"detectable matrix characteristics are =>  \n{detectable_matrix} {np.shape(detectable_matrix)} {type(detectable_matrix)}")
        
        column_sums = np.sum(detectable_matrix, axis=0)
        valid_columns = column_sums >= 1
        detectable_matrix_filtered = detectable_matrix[:, valid_columns]
        log_print(f"filtered data shape is =>  {np.shape(detectable_matrix_filtered)}")
        
        num_detectable_cols = detectable_matrix_filtered.shape[1]
        
        if num_detectable_cols == 0:
            log_print("No detectable defects found. Exiting solver.")
            return
            
        solutions = []
        excluded_solutions = []
        max_num_sols = 5  # Keeping this low so the total run time is manageable
        
        for i in range(max_num_sols):
            c = np.ones(num_rows)
            A = -detectable_matrix_filtered.T
            b = -np.ones(num_detectable_cols)
            
            for sol in excluded_solutions:
                A_new = np.zeros(num_rows)
                A_new[sol] = 1
                A = np.vstack([A, A_new])
                b = np.append(b, len(sol) - 1)
                
            integrality = np.ones(num_rows) 
            bounds = [(0, 1) for _ in range(num_rows)] 
            
            log_print(f"Running solver iteration {i+1}/{max_num_sols}...")
            result = linprog(
                c=c, 
                integrality=integrality, 
                A_ub=A, 
                b_ub=b, 
                bounds=bounds, 
                method='highs',
                options={
                    'time_limit': 60,       # Max 60 seconds allowed per iteration
                    'mip_rel_gap': 0.05     # Accept a solution within 5% of the absolute mathematical best
                }
            )
            
            for sol in excluded_solutions:
                A_new = np.zeros(num_rows)
                A_new[sol] = 1
                A = np.vstack([A, A_new])
                b = np.append(b, len(sol) - 1)
                
            integrality = np.ones(num_rows) 
            bounds = [(0, 1) for _ in range(num_rows)] 
            
            result = linprog(c=c, integrality=integrality, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if result.success:
                selected_rows = np.where(result.x.round() > 0.1)[0].tolist()
                excluded_solutions.append(selected_rows)
                
                boolean_test = np.column_stack([detectable_matrix_filtered[row, :] for row in selected_rows]).any(axis=1)
                percentage_ = boolean_test.sum() / num_detectable_cols
                
                if 0.9 <= percentage_ <= 1.0:
                    strings_candidates = df.iloc[[row for row in selected_rows]].index.tolist()
                    solutions.append(strings_candidates)
            else:
                log_print(f"No more good sols found. Terminating solver at iteration {i}.")
                break
                
        log_print(f"\nFound {len(solutions)} minimal-length, 100% coverage test pattern solutions. Finding maximum homogeneity...")
        
        if not solutions:
            log_print("No solutions available to score.")
            return

        scored_solutions = []
        for sol in solutions:
            score = calculate_homogeneity_score(sol)
            scored_solutions.append((score, sol))
            
        scored_solutions.sort(key=lambda x: x[0], reverse=True)
        max_score_achieved = scored_solutions[0][0]
        best_solutions = [sol for score, sol in scored_solutions if score == max_score_achieved]
        
        log_print("\n======================================================================")
        log_print("=== Optimal Test Patterns (Min-Length, Max-Coverage, Max-Homogeneity) ===")
        log_print(f"Found {len(best_solutions)} optimal solution(s) with the maximum homogeneity score of {max_score_achieved}.")
        log_print("======================================================================\n")
        
        for rank, sol in enumerate(best_solutions, 1):
            max_possible_score = 10 * len(sol) 
            log_print(f"Optimal Solution {rank} (Score: {max_score_achieved}/{max_possible_score}):")
            for pattern in sol:
                features = get_homogeneity_features(pattern)
                log_print(f"  {pattern}  --> Features (Bulk, N1, N2, N3, N4): {features}")
            log_print("-" * 60)
        
        # --- NEW: Unique Defect Location Calculation ---
        # --- NEW: Unique Defect Location & ATPG Validation Calculation ---
        def get_defect_location(defect_id):
            """Parses the defect ID to extract its base physical location or group."""
            defect_str = str(defect_id)
            parts = defect_str.rsplit('_', 1)
            
            if len(parts) == 2 and parts[1].isdigit():
                num = int(parts[1])
                base_name = parts[0]
                
                # Catch the ETS26_v2 / device-aware dataset
                if base_name.endswith('schematic') or base_name.endswith('r_def') or base_name.endswith('def'):
                    group_num = (num - 1) // 10 + 1
                    return f"{base_name}_group_{group_num}"
                else:
                    return base_name
            return defect_str

        # Identify all theoretically detectable defects across ALL patterns
        detectable_cols = [col for col in df.columns if df[col].sum() > 0]
        
        # Check what the optimal ATPG patterns ACTUALLY cover
        if best_solutions:
            first_sol = best_solutions[0]
            valid_patterns = [p for p in first_sol if p in df.index]
            
            # Keep only the defects that trigger at least one '1' across the selected ATPG patterns
            actually_covered_cols = [col for col in detectable_cols if df.loc[valid_patterns, col].sum() > 0]
        else:
            actually_covered_cols = []
            
        # Calculate actual ATPG Success Rate
        atpg_success_pct = (len(actually_covered_cols) / len(detectable_cols)) * 100 if len(detectable_cols) > 0 else 0.0
        
        # Calculate Location Groupings based on ACTUAL coverage
        all_locations = set(get_defect_location(d) for d in df.columns)
        detectable_locations = set(get_defect_location(d) for d in detectable_cols) # Theoretical max
        covered_locations = set(get_defect_location(d) for d in actually_covered_cols) # Actually hit by ATPG
        
        # Calculate the specific undetected locations ---
        undetectable_locations = sorted(list(all_locations - detectable_locations))

        total_locations = len(all_locations)
        detectable_loc_count = len(detectable_locations)
        covered_loc_count = len(covered_locations)
        undetectable_loc_count = len(undetectable_locations)
        
        # Final Percentages
        overall_coverage_pct = (len(actually_covered_cols) / total_defects) * 100 if total_defects > 0 else 0.0
        unique_loc_coverage_pct = (covered_loc_count / total_locations) * 100 if total_locations > 0 else 0.0

        # Calculate Detectable but Missed Defects by ATPG ---
        missed_by_atpg = sorted(list(set(detectable_cols) - set(actually_covered_cols)))

        # --- Final Coverage Report ---
        log_print("\n======================================================================")
        log_print("=== Final Test Accuracy & Coverage Report ===")
        log_print(f"Total Evaluated Defects:     {total_defects}")
        log_print(f"Undetectable Defects:        {undetectable_count}")
        
        # --- NEW: Explicitly list the specific undetectable defects ---
        if undetectable_count > 0:
            log_print(f"  -> Undetectable Defect IDs: {undetectable_names}")
            
        log_print(f"Detectable Defects Covered:  {len(actually_covered_cols)} / {len(detectable_cols)}")
        
        # --- NEW: Explicitly list defects the optimal patterns failed to cover ---
        if missed_by_atpg:
            log_print(f"  -> Detectable but Missed by ATPG: {len(missed_by_atpg)} {missed_by_atpg}")
            
        log_print(f"ATPG Success on Detectables: {atpg_success_pct:.2f}%")
        log_print(f"Overall Final Test Accuracy: {overall_coverage_pct:.2f}%")
        log_print("----------------------------------------------------------------------")
        log_print(f"Total Defect Locations:      {total_locations}")
        
        # --- Print the array of undetected locations ---
        if undetectable_loc_count > 0:
            log_print(f"Undetectable Locations:      {undetectable_loc_count} {undetectable_locations}")
        else:
            log_print(f"Undetectable Locations:      0 []")
        
        log_print(f"Detectable Locations:        {detectable_loc_count}")
        log_print(f"Actually Covered Locations:  {covered_loc_count}")
        log_print(f"Unique Location Coverage:    {unique_loc_coverage_pct:.2f}%")
        log_print("======================================================================\n")
        
        # --- NEW: Individual Pattern Contribution Analysis ---
        if best_solutions:
            log_print("======================================================================")
            log_print("=== Individual Pattern Coverage Contribution (Optimal Solution 1) ===")
            log_print("======================================================================")
            
            first_sol = best_solutions[0]
            
            for pattern in first_sol:
                if pattern in df.index:
                    # 1. Calculate Total Defects Detected by this specific pattern
                    detected_by_pattern = df.columns[df.loc[pattern] > 0].tolist()
                    num_total_detected = len(detected_by_pattern)
                    
                    # 2. Calculate Unique Locations Detected
                    locations_by_pattern = set(get_defect_location(d) for d in detected_by_pattern)
                    num_unique_locations = len(locations_by_pattern)
                    
                    # 3. Log the metrics
                    log_print(f"Pattern: {pattern}")
                    log_print(f"  -> Total Defects Covered:   {num_total_detected:>3} / {total_defects}")
                    log_print(f"  -> Unique Locations Hit:    {num_unique_locations:>3} / {total_locations}")
                    log_print("-" * 60)
            log_print("\n")
            
        if best_solutions:
            first_sol = best_solutions[0]
            valid_patterns = [p for p in first_sol if p in df.index]
            
            # Slice the original binary dataframe to keep only the selected optimal patterns
            optimal_submatrix = df.loc[valid_patterns]
            
            # Determine save directory based on the input excel file location
            out_dir = os.path.dirname(excel_file)
            submatrix_out_path = os.path.join(out_dir, "optimal_binary_submatrix.csv")
            
            # Transpose (.T) so Defects are rows and Patterns are columns, matching your other exports
            optimal_submatrix.T.to_csv(submatrix_out_path)
            log_print(f"Exported Optimal Binary Submatrix (Defects x Selected Patterns) to: {submatrix_out_path}\n")
        
        # ADC Difference Distribution on the Submatrix (Placed AFTER the report) ---
        if best_solutions:
            venn_print("======================================================================")
            venn_print("=== ADC Difference Distribution (Optimal Solution 1) ===")
            venn_print("======================================================================")
            try:
                # Read the auto-logged ADC matrix and transpose it so inputs are rows, defects are columns
                df_adc = pd.read_csv(adc_diff_file, index_col=0).T 
                
                # Select ONLY the first optimal solution
                first_solution = best_solutions[0]
                
                venn_print(f"Calculating offsets for the {len(first_solution)} patterns in Optimal Solution 1.")
                venn_print(f"Submatrix size: {len(first_solution)} patterns x {df_adc.shape[1]} total defects.")
                venn_print("-" * 60)
                
                # Filter the dataframe to only keep the rows of our selected test patterns
                valid_patterns = [p for p in first_solution if p in df_adc.index]
                submatrix = df_adc.loc[valid_patterns]
                
                # Dictionary of sets to automatically deduplicate defect/offset pairs
                offset_to_defects = defaultdict(set)
                
                # Iterate through every defect (column) and map it to its corresponding offsets
                for defect_col in submatrix.columns:
                    for val in submatrix[defect_col]:
                        offset_to_defects[float(val)].add(defect_col)
                
                # Print the sorted results (including 0.0)
                for val in sorted(offset_to_defects.keys()):
                    unique_defects = sorted(list(offset_to_defects[val]))
                    count = len(unique_defects)
                    venn_print(f"  ADC Offset {val:>6.1f} : {count:>6} unique defects: {unique_defects}")
                    
            except Exception as e:
                venn_print(f"  Could not calculate ADC offset distributions. Error: {e}")
            venn_print("\n")

            # Combined State Distribution on the Submatrix (Dynamically Calculated) ---
            venn_print("======================================================================")
            venn_print("=== Combined State Distribution (Optimal Solution 1) ===")
            venn_print("======================================================================")
            try:
                # Read the pre-calculated combined states (0-9) used for plotting
                df_combined = pd.read_csv(combined_file, index_col=0)
                
                state_to_defects = defaultdict(set)
                
                # Filter to only look at the patterns in our optimal solution
                valid_patterns_in_df = [p for p in valid_patterns if p in df_combined.columns]
                sub_combined = df_combined[valid_patterns_in_df]
                
                # Iterate through all defects natively using the combined file's own index
                for defect_id, row in sub_combined.iterrows():
                    for pattern, val in row.items():
                        if pd.notna(val):
                            state = int(val)
                            # Translate the JSON key to the descriptive name ---
                            mapped_name = defect_names_map.get(str(defect_id), str(defect_id)) if defect_names_map else str(defect_id)
                            state_to_defects[state].add(mapped_name)
                
                labels_map = {
                    0: "Diff -2 | Undetected", 1: "Diff -2 | Detected  ",
                    2: "Diff -1 | Undetected", 3: "Diff -1 | Detected  ",
                    4: "Diff  0 | Undetected", 5: "Diff  0 | Detected  ",
                    6: "Diff +1 | Undetected", 7: "Diff +1 | Detected  ",
                    8: "Diff +2 | Undetected", 9: "Diff +2 | Detected  "
                }
                
                for state in range(10):
                    unique_defects = sorted(list(state_to_defects.get(state, [])))
                    count = len(unique_defects)
                    
                    if count > 0:
                        venn_print(f"  {labels_map[state]}: {count:>6} unique defects: {unique_defects}")
                    else:
                        venn_print(f"  {labels_map[state]}: {count:>6} unique defects")
                    
            except Exception as e:
                venn_print(f"  Could not calculate Combined state distributions. Error: {e}")
            venn_print("\n")
            # ---------------------------------------------------------
        
            # ====================================================================
            # --- NEW SECTION 3: Per-Pattern ADC Offset & Defect Mapping ---
            # ====================================================================
            venn_print("======================================================================")
            venn_print("=== SECTION 3: Per-Pattern ADC Offset & Defect Distribution ===")
            venn_print("======================================================================")
            try:
                for pattern in valid_patterns:
                    venn_print(f"Pattern: {pattern}")
                    pattern_row = submatrix.loc[pattern]
                    
                    # Dictionary to group defect column names by the ADC offset they triggered
                    offset_to_defects_pattern = defaultdict(list)
                    
                    for col_name, offset_val in pattern_row.items():
                        if offset_val != 0.0:  # Only care about sensitized defects
                            offset_to_defects_pattern[offset_val].append(col_name)
                    
                    if not offset_to_defects_pattern:
                        venn_print("  -> No defects sensitized by this pattern.")
                    else:
                        for offset in sorted(offset_to_defects_pattern.keys()):
                            # Get unique, sorted defect column names for this offset
                            unique_defects = sorted(list(set(offset_to_defects_pattern[offset])))
                            count = len(unique_defects)
                            
                            venn_print(f"  -> ADC Offset {offset:>6.1f} : Sensitized {count:>4} defects: {unique_defects}")
                    venn_print("-" * 60)
            except Exception as e:
                venn_print(f"  Could not calculate per-pattern defect distribution. Error: {e}")
            venn_print("\n")

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w") # Use "w" to overwrite each run, or "a" to append

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensures data is written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Enable Global Dual Logging ---
    log_path = os.path.join(FOLDER_LOCATION, "fault_log.txt")
    sys.stdout = DualLogger(log_path)
    print(f"--- Terminal output is now also being logged to: {log_path} ---")
    
    global PLOT_MODE
    
    # Initialize mapping dict so it's available for the TPG step ---
    generated_cols = {}
    
    # Block 1: Dictionary Generation
    if ENABLE_FAULT_DICT_GEN:
        print("\n--- Running Fault Dictionary Generation ---")
        generate_fault_dict(DEFECTIVE_CELLS_DIR, JSON_FILE)
    else:
        print("\n--- Fault Dictionary Generation is DISABLED ---")

    # Block 2 & 3: Matrix Generation and Plotting
    if ENABLE_MATRIX_GEN or ENABLE_PLOTTING:
        print("\n--- Loading Datasets ---")
        try:
            with open(JSON_FILE, 'r') as f:
                fault_dict = json.load(f)
        except Exception as e:
            print(f"Error loading {JSON_FILE}. Please run with ENABLE_FAULT_DICT_GEN=True first. Error: {e}")
            return
            
        # Robust CSV Loading with Auto-Padding for Failed Simulations ---
        print("Reading mapper CSV with ragged-row handling...")
        rows_raw = []
        max_cols = 0
        with open(MAPPER_FILE, 'r') as f_map:
            for line in f_map:
                # Auto-detect delimiter: try tab first (most common for .measure files),
                fields = line.rstrip('\n\r').split('\t')
                if len(fields) <= 1:
                    fields = line.rstrip('\n\r').split(',')
                rows_raw.append(fields)
                if len(fields) > max_cols:
                    max_cols = len(fields)

        # Normalize all rows to the same width (pad short rows with empty strings)
        for i in range(len(rows_raw)):
            if len(rows_raw[i]) < max_cols:
                rows_raw[i] += [''] * (max_cols - len(rows_raw[i]))

        df_mapper = pd.DataFrame(rows_raw)

        # Convert numeric columns (cols 1 onward) from strings to proper types.
        # Non-numeric / empty cells become NaN, which downstream code already handles.
        for col_idx in range(1, len(df_mapper.columns)):
            df_mapper[col_idx] = pd.to_numeric(df_mapper[col_idx], errors='coerce')

        print(f"  Mapper loaded: {len(df_mapper)} rows x {len(df_mapper.columns)} columns")

        # Auto-pad failed simulation rows.
        def extract_input_combo_from_path(filepath_str):
            """Extract the input combination key from a measurement filepath."""
            if pd.isna(filepath_str):
                return None
            fp = str(filepath_str).replace("\\", "/")
            fn = fp.split("/")[-1]
            clean = os.path.splitext(fn)[0]
            if "_rp_" not in clean:
                return None
            return "rp_" + clean.split("_rp_", 1)[1]

        last_valid = {}          # input_combo -> list of column values (row data)
        padded_count = 0
        failed_no_donor = 0

        for idx in range(len(df_mapper)):
            filepath_val = df_mapper.iat[idx, 0]
            analog_val   = df_mapper.iat[idx, 1]

            input_combo = extract_input_combo_from_path(filepath_val)
            if input_combo is None:
                continue

            # Check whether this row is a failed simulation (analog output == -1)
            is_failed = (analog_val == -1) or (pd.isna(analog_val) and str(df_mapper.iat[idx, 1]).strip() == '-1')

            if not is_failed and not pd.isna(analog_val):
                # Valid result: cache it as donor for future failed rows
                last_valid[input_combo] = df_mapper.iloc[idx, 1:].values.copy()
            elif is_failed and input_combo in last_valid:
                # Failed result with a donor available: forward-fill from donor
                df_mapper.iloc[idx, 1:] = last_valid[input_combo]
                padded_count += 1
            elif is_failed:
                failed_no_donor += 1

        print(f"  Auto-padded {padded_count} failed-simulation rows from previous valid results.")
        if failed_no_donor > 0:
            print(f"  Warning: {failed_no_donor} failed rows had no prior valid donor for their input combo.")

        adc_df = pd.read_csv(ADC_MAPPING_FILE)
        json_defect_ids = set(fault_dict.keys())
        
        has_final_states = len(df_mapper.columns) >= 5
        if not has_final_states and PLOT_MODE == "combined":
            print("Notice: Mapper file lacks final resistance state data. 'combined' mode will default to Binary Detection (0/1).")
        
        defect_list = []
        for defect_id, properties in fault_dict.items():
            arch_prefix = defect_id.rsplit('_', 1)[0]
            defect_list.append({
                'defect_id': defect_id, 'arch': arch_prefix, 'type': properties[0], 
                'node1': properties[1], 'node2': properties[2], 'res': properties[3]
            })
        df_defects = pd.DataFrame(defect_list)

        generated_cols = {}
        groups = df_defects.groupby(['arch', 'type', 'node1', 'node2'])
        for (arch, t, n1, n2), group in groups:
            sorted_group = group.sort_values('res', ascending=True)
            for i, (idx, row) in enumerate(sorted_group.iterrows()):
                defect_id = row['defect_id']
                col_name = f"{arch}_{t}_{n1}_{n2}_{i+1}"
                generated_cols[defect_id] = col_name

        print(f"Parsing mapper data (Pre-computing all mappings for Mode: {DETECTION_MODE})...")
        
        # Check for mixed ninit data & Initialize Combiner Logic 
        has_ninit_overall = False
        has_non_ninit = False
        for idx, row in df_mapper.iterrows():
            if pd.isna(row[0]): continue
            clean_name = str(row[0])
            if "_rp_" in clean_name:
                if "_ninit_" in clean_name:
                    has_ninit_overall = True
                else:
                    has_non_ninit = True
        
        mixed_ninit = has_ninit_overall and has_non_ninit
        
        # Mixed Data Filtering Logic ---
        if mixed_ninit:
            print(f"\nMixed ninit data detected. Applying MIXED_DATA_MODE: '{MIXED_DATA_MODE}'")
            
            if MIXED_DATA_MODE == "read-only":
                # Keep only rows where the filename DOES NOT contain '_ninit_'
                df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_ninit_" not in str(x))]
                mixed_ninit = False  # Turn off normalization so names stay original
                print(f"  -> 'read-only' mode: Excluded ninit patterns. Remaining rows: {len(df_mapper)}\n")
                
            elif MIXED_DATA_MODE == "write-based":
                # Keep rows that contain '_ninit_' (or aren't simulation pattern files, to be safe)
                df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_rp_" not in str(x) or "_ninit_" in str(x))]
                mixed_ninit = False  # Turn off normalization since the dataset is now purely ninit
                print(f"  -> 'write-based' mode: Excluded non-ninit patterns. Remaining rows: {len(df_mapper)}\n")
            
            elif MIXED_DATA_MODE == "forming-1":
                # Keep all non-ninit (read ops) OR specific ninit_0.01 / ninit_0.008 write ops
                df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_ninit_" not in str(x) or "_ninit_0.01" in str(x) or "_ninit_0.008" in str(x))]
                # Note: We leave mixed_ninit = True so the normalizer can still fuse the reads with the remaining writes
                print(f"  -> 'forming-1' mode: Kept reads + ninit_0.01/0.008 writes. Remaining rows: {len(df_mapper)}\n")

            elif MIXED_DATA_MODE == "forming-2":
                # Keep all non-ninit (read ops) OR specific ninit_20 / ninit_20.0 write ops
                df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_ninit_" not in str(x) or "_ninit_20" in str(x))]
                # Note: We leave mixed_ninit = True so the normalizer can still fuse the reads with the remaining writes
                print(f"  -> 'forming-2' mode: Kept reads + ninit_20 writes. Remaining rows: {len(df_mapper)}\n")

            elif MIXED_DATA_MODE == "merged":
                print("  -> 'merged' mode: Keeping all patterns. Normalization logic will combine them.\n")
        
        def normalize_input_combo(combo):
            """Integrates non-ninit forms into ninit equivalence blocks automatically."""
            if mixed_ninit and "_ninit" not in combo:
                match = re.search(r'rp_([0-9\.]+)_rn_([0-9\.]+)', combo)
                if match:
                    rp_val = float(match.group(1))
                    if rp_val == 20.0 or rp_val == 20:
                        return combo + "_ninit_20"
                    elif rp_val in [0.01, 0.008]:
                        if rp_val == 0.01:
                            return combo + "_ninit_0.01"
                        elif rp_val == 0.008:
                            return combo + "_ninit_0.008"
            return combo

        parsed_adc_codes = defaultdict(dict)
        unique_defect_ids = set()
        golden_ids = set()
        
        for idx, row in df_mapper.iterrows():
            if pd.isna(row[0]): continue 
            try:
                analog_val = float(row[1])
            except ValueError:
                continue
                
            filepath = str(row[0]).replace("\\", "/") 
            filename = filepath.split("/")[-1] 
            clean_name = os.path.splitext(filename)[0]
            
            if "_rp_" not in clean_name: continue
                
            parts = clean_name.split("_rp_")
            left_part = parts[0]
            input_combo = "rp_" + parts[1]
            input_combo = normalize_input_combo(input_combo) # ADDED
            
            try:
                mapper_digital_code = float(row[2])
            except:
                mapper_digital_code = -1 
                
            closest_idx = (adc_df['adc_in'] - analog_val).abs().idxmin()
            mapped_digital_code = adc_df.loc[closest_idx, 'adc_out']
            is_mapper_valid = (0 <= mapper_digital_code <= 256)

            run_data = {
                'analog': analog_val,
                'mapper_digital': mapper_digital_code,
                'mapped_digital': mapped_digital_code,
                'is_mapper_valid': is_mapper_valid
            }
            
            if left_part.endswith("_0") or "None_None" in left_part or "schematic_0" in left_part:
                golden_ids.add(left_part)
                parsed_adc_codes[input_combo][left_part] = run_data
                continue
                
            defect_id = None
            for k in json_defect_ids:
                if left_part.endswith(k):
                    defect_id = k
                    break
                    
            if defect_id:
                parsed_adc_codes[input_combo][defect_id] = run_data
                unique_defect_ids.add(defect_id)

        def get_golden_id(def_id, g_ids):
            def_base = def_id.rsplit('_', 1)[0] 
            exact_golden = f"{def_base}_0"
            if exact_golden in g_ids: return exact_golden
            for g in g_ids:
                if g.startswith(def_base + "_"): return g
            for g in g_ids:
                if "None_None" in g or "schematic_0" in g: return g
            return list(g_ids)[0] if g_ids else None

        print("Calculating detections and ADC Differences based on selected mode...")
        detection_matrix = defaultdict(dict)
        adc_diff_matrix = defaultdict(dict)
        
        for input_combo, defects in parsed_adc_codes.items():
            for defect_id in unique_defect_ids:
                defective_data = defects.get(defect_id)
                if defective_data is None: continue
                    
                golden_id = get_golden_id(defect_id, golden_ids)
                golden_data = defects.get(golden_id)
                
                if golden_data is None:
                    detect_flag = 0
                    diff = 0
                else:
                    if DETECTION_MODE == "direct_analog":
                        # 1. Calculate the raw, signed difference
                        raw_analog_diff = defective_data['analog'] - golden_data['analog']
                        
                        # 2. Use the absolute value ONLY to check if it crosses the detection threshold
                        detect_flag = 1 if abs(raw_analog_diff) >= ANALOG_DETECTION_THRESHOLD else 0
                        
                        # 3. Store the signed difference for the Venn distribution log
                        diff = round(raw_analog_diff / ANALOG_DETECTION_THRESHOLD)
                    elif DETECTION_MODE == "mapped_digital":
                        # We store the signed difference for the occurrence report
                        diff = defective_data['mapped_digital'] - golden_data['mapped_digital']
                        # We use the absolute difference ONLY to determine if it's detected
                        detect_flag = 1 if abs(diff) >= DETECTION_THRESHOLD else 0
                    elif DETECTION_MODE == "mapper_digital":
                        if defective_data['is_mapper_valid'] and golden_data['is_mapper_valid']:
                            diff = defective_data['mapper_digital'] - golden_data['mapper_digital']
                        else:
                            diff = defective_data['mapped_digital'] - golden_data['mapped_digital']
                        detect_flag = 1 if abs(diff) >= DETECTION_THRESHOLD else 0
                    
                col_name = generated_cols.get(defect_id)
                if col_name:
                    detection_matrix[input_combo][col_name] = detect_flag
                    adc_diff_matrix[input_combo][col_name] = diff

        # Create Core DataFrames
        df_result = pd.DataFrame.from_dict(detection_matrix, orient='index').fillna(0).astype(int)
        df_result.index = df_result.index.astype(str).str.strip("'").str.strip()
        df_result.index.name = "Input_Combination" 
        
        df_adc_diff = pd.DataFrame.from_dict(adc_diff_matrix, orient='index').fillna(0)
        df_adc_diff.index = df_adc_diff.index.astype(str).str.strip("'").str.strip()

        # Phase 2a: Matrix Generation (Export)
        if ENABLE_MATRIX_GEN:
            print("\n--- Running Matrix Generation ---")
            df_result.to_excel(RECREATED_RESULT_FILE)
            print(f"Exported perfectly binary (0/1) recreated result dataset to {RECREATED_RESULT_FILE}")
            
            # --- NEW: Export matrices independent of plotting ---
            binary_out_path = os.path.join(FOLDER_LOCATION, "final_binary_matrix.csv")
            adc_diff_out_path = os.path.join(FOLDER_LOCATION, "final_adc_diff_matrix.csv")
            
            # # We transpose (.T) them before saving so Defects are rows and Inputs are columns
            df_result.T.to_csv(binary_out_path)
            df_adc_diff.T.to_csv(adc_diff_out_path)
            
            print(f"Exported Binary Matrix CSV to: {binary_out_path}")
            print(f"Exported ADC Diff Matrix CSV to: {adc_diff_out_path}")

        else:
            print("\n--- Matrix Generation (.xlsx export) is DISABLED ---")

        # Phase 2b: Plotting Logic
        if ENABLE_PLOTTING:
            print("\n--- Running Fault Map Plotting Pipeline ---")
            mapper_records = {}
            unique_inputs = set()
            
            for idx, row in df_mapper.iterrows():
                if pd.isna(row[0]): continue
                filepath = str(row[0]).replace("\\", "/") 
                filename = filepath.split("/")[-1] 
                clean_name = os.path.splitext(filename)[0]
                if "_rp_" not in clean_name: continue
                    
                parts = clean_name.split("_rp_")
                left_part = parts[0]
                input_combo = "rp_" + parts[1]
                
                input_combo = normalize_input_combo(input_combo) # ADDED
                
                defect_id = None
                for k in json_defect_ids:
                    if left_part.endswith(k):
                        defect_id = k
                        break
                if not defect_id: continue
                    
                unique_inputs.add(input_combo)
                init_match = re.search(r'rp_([0-9\.]+)_rn_([0-9\.]+)', input_combo)
                if not init_match: continue
                init_rp, init_rn = float(init_match.group(1)), float(init_match.group(2))
                
                if has_final_states and not pd.isna(row[3]) and not pd.isna(row[4]):
                    try:
                        final_rp, final_rn = float(row[3]), float(row[4])
                        state_diff = check_resistive_difference(init_rp, init_rn, final_rp, final_rn)
                    except ValueError:
                        state_diff = 0
                else:
                    state_diff = 0
                
                if defect_id not in mapper_records: mapper_records[defect_id] = {}
                mapper_records[defect_id][input_combo] = state_diff

            has_ninit = any("_ninit" in combo for combo in unique_inputs)
            col_mapping = {}
            
            groups = df_defects.groupby(['arch', 'type', 'node1', 'node2'])
            for (arch, t, n1, n2), group in groups:
                sorted_group = group.sort_values('res', ascending=True)
                search_str_1 = f"{arch}_{t}_{n1}_{n2}"
                search_str_2 = f"{arch}_{t}_{n2}_{n1}"
                matching_cols = [c for c in df_result.columns if search_str_1 in c or search_str_2 in c]
                
                def extract_idx(col_name):
                    m = re.search(r'_(\d+)$', col_name)
                    return int(m.group(1)) if m else 0
                    
                matching_cols_sorted = sorted(matching_cols, key=extract_idx)
                for i, (idx, row) in enumerate(sorted_group.iterrows()):
                    if i < len(matching_cols_sorted):
                        col_mapping[row['defect_id']] = matching_cols_sorted[i]

            final_data_combined = {}
            final_data_adc = {}
            final_data_binary = {}
            
            for _, defect_row in df_defects.iterrows():
                defect_id = defect_row['defect_id']
                final_data_combined[defect_id] = {}
                final_data_adc[defect_id] = {}
                final_data_binary[defect_id] = {}
                excel_col = col_mapping.get(defect_id)
                
                if defect_id in mapper_records:
                    for input_combo in unique_inputs:
                        res_diff_val = mapper_records[defect_id].get(input_combo, 0)
                        
                        d_val = 0
                        adc_diff_val = 0
                        if excel_col and input_combo in df_result.index:
                            d_val = int(df_result.at[input_combo, excel_col])
                            adc_diff_val = abs(float(df_adc_diff.at[input_combo, excel_col]))
                        
                        combined_val = encode_combined_state(res_diff_val, d_val)
                        final_data_combined[defect_id][input_combo] = combined_val
                        final_data_adc[defect_id][input_combo] = adc_diff_val
                        final_data_binary[defect_id][input_combo] = d_val

            df_final_combined = pd.DataFrame.from_dict(final_data_combined, orient='index')
            df_final_adc = pd.DataFrame.from_dict(final_data_adc, orient='index')
            df_final_binary = pd.DataFrame.from_dict(final_data_binary, orient='index')
            
            df_defects_sorted = df_defects.sort_values(by=['arch', 'type', 'node1', 'node2', 'res'])
            sorted_index = [d for d in df_defects_sorted['defect_id'] if d in df_final_combined.index]
            
            df_final_combined = df_final_combined.reindex(sorted_index)
            df_final_adc = df_final_adc.reindex(sorted_index)
            df_final_binary = df_final_binary.reindex(sorted_index)
            
            df_final_combined.index.name = "Defect_ID"
            df_final_binary.index.name = "Defect_ID"
            df_final_adc.index.name = "Defect_ID"
            
            # Always export Binary and ADC Diff matrices independently ---
            binary_out_path = os.path.join(FOLDER_LOCATION, "final_binary_matrix.csv")
            
            if PLOT_MODE == "combined":
                if has_final_states:
                    df_final_combined.to_csv(OUTPUT_FILE)
                    print(f"Exported combined dataset to {OUTPUT_FILE}")
                    print("\n=== Combined Results Summary ===")
                    val_counts = df_final_combined.unstack().dropna().value_counts()
                    labels_map = {
                        0: "Diff -2 | Undetected", 1: "Diff -2 | Detected  ",
                        2: "Diff -1 | Undetected", 3: "Diff -1 | Detected  ",
                        4: "Diff  0 | Undetected", 5: "Diff  0 | Detected  ",
                        6: "Diff +1 | Undetected", 7: "Diff +1 | Detected  ",
                        8: "Diff +2 | Undetected", 9: "Diff +2 | Detected  "
                    }
                    total = val_counts.sum()
                    if total == 0: total = 1 
                    for state in range(10):
                        count = val_counts.get(state, 0)
                        pct = (count / total) * 100
                        print(f"{labels_map[state]}: {int(count):>6} instances ({pct:>5.2f}%)")
                    generate_plots(df_final_combined, fault_dict, has_ninit, mode="combined")
                else:
                    df_final_binary.to_csv(OUTPUT_FILE)
                    print(f"Exported binary dataset to {OUTPUT_FILE}")
                    print("\n=== Binary Detection Results Summary (No Final States) ===")
                    val_counts = df_final_binary.unstack().dropna().value_counts()
                    total = val_counts.sum()
                    if total == 0: total = 1 
                    for state in [0, 1]:
                        count = val_counts.get(state, 0)
                        pct = (count / total) * 100
                        label = "Detected  " if state == 1 else "Undetected"
                        print(f"{label}: {int(count):>6} instances ({pct:>5.2f}%)")
                    generate_plots(df_final_binary, fault_dict, has_ninit, mode="binary")
                
            elif PLOT_MODE == "adc_diff":
                df_final_adc.to_csv(OUTPUT_FILE)
                print(f"Exported ADC difference dataset to {OUTPUT_FILE}")
                print("\n=== ADC Difference Results Summary ===")
                max_diff = df_final_adc.values.max()
                avg_diff = df_final_adc.values.mean()
                print(f"Maximum Difference Observed: {max_diff}")
                print(f"Average Difference Across Grid: {avg_diff:.2f}")
                
                # Categorical fix removed - generate_plots handles sorting internally
                generate_plots(df_final_adc, fault_dict, has_ninit, mode="adc_diff")
        else:
            print("\n--- Fault Map Plotting Pipeline is DISABLED ---")
    else:
        print("\n--- Matrix Generation and Plotting are DISABLED ---")

    # Block 4: Test Pattern Generation
    if ENABLE_TPG:
        print("\n--- Running Test Pattern Generation (TPG) Pipeline ---")
        adc_diff_out_path = os.path.join(FOLDER_LOCATION, "final_adc_diff_matrix.csv")
        
        if not os.path.exists(RECREATED_RESULT_FILE) or not os.path.exists(adc_diff_out_path):
            print(f"ERROR: TPG pipeline is enabled, but required matrices could not be found.")
            print("Please ensure ENABLE_MATRIX_GEN ran first to create the files.")
            return

        # Pass OUTPUT_FILE (the combined matrix) as the 3rd argument
        generate_test_patterns(RECREATED_RESULT_FILE, adc_diff_out_path, OUTPUT_FILE, TPG_LOG_FILE, VENN_LOG_FILE, generated_cols)
        print(f"Test Pattern Generation completed.")
        print(f"General details logged to: {TPG_LOG_FILE}")
        print(f"Distribution details logged to: {VENN_LOG_FILE}")
    else:
        print("\n--- Test Pattern Generation (TPG) Pipeline is DISABLED ---")

if __name__ == "__main__":
    main()