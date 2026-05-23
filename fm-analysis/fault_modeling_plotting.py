import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

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

def generate_plots(df_final, fault_dict, has_ninit, mode="combined", folder_location="."):
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
        plt.savefig(os.path.join(folder_location, safe_name), dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        
    print(f"Success! {len(defect_groups)} grid plots generated and saved.\n")

def generate_matrices_and_plots(fault_dict, df_mapper, adc_mapping_file, detection_mode, detection_threshold, analog_detection_threshold, mixed_data_mode, plot_mode, enable_matrix_gen, enable_plotting, recreated_result_file, folder_location, output_file):
    adc_df = pd.read_csv(adc_mapping_file)
    json_defect_ids = set(fault_dict.keys())
    
    has_final_states = len(df_mapper.columns) >= 5
    if not has_final_states and plot_mode == "combined":
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

    print(f"Parsing mapper data (Pre-computing all mappings for Mode: {detection_mode})...")
    
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
        print(f"\nMixed ninit data detected. Applying MIXED_DATA_MODE: '{mixed_data_mode}'")
        
        if mixed_data_mode == "read-only":
            # Keep only rows where the filename DOES NOT contain '_ninit_'
            df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_ninit_" not in str(x))]
            mixed_ninit = False  # Turn off normalization so names stay original
            print(f"  -> 'read-only' mode: Excluded ninit patterns. Remaining rows: {len(df_mapper)}\n")
            
        elif mixed_data_mode == "write-based":
            # Keep rows that contain '_ninit_' (or aren't simulation pattern files, to be safe)
            df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_rp_" not in str(x) or "_ninit_" in str(x))]
            mixed_ninit = False  # Turn off normalization since the dataset is now purely ninit
            print(f"  -> 'write-based' mode: Excluded non-ninit patterns. Remaining rows: {len(df_mapper)}\n")
        
        elif mixed_data_mode == "forming-1":
            # Keep all non-ninit (read ops) OR specific ninit_0.01 / ninit_0.008 write ops
            df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_ninit_" not in str(x) or "_ninit_0.01" in str(x) or "_ninit_0.008" in str(x))]
            # Note: We leave mixed_ninit = True so the normalizer can still fuse the reads with the remaining writes
            print(f"  -> 'forming-1' mode: Kept reads + ninit_0.01/0.008 writes. Remaining rows: {len(df_mapper)}\n")

        elif mixed_data_mode == "forming-2":
            # Keep all non-ninit (read ops) OR specific ninit_20 / ninit_20.0 write ops
            df_mapper = df_mapper[df_mapper[0].fillna("").apply(lambda x: "_ninit_" not in str(x) or "_ninit_20" in str(x))]
            # Note: We leave mixed_ninit = True so the normalizer can still fuse the reads with the remaining writes
            print(f"  -> 'forming-2' mode: Kept reads + ninit_20 writes. Remaining rows: {len(df_mapper)}\n")

        elif mixed_data_mode == "merged":
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
                if detection_mode == "direct_analog":
                    # 1. Calculate the raw, signed difference
                    raw_analog_diff = defective_data['analog'] - golden_data['analog']
                    
                    # 2. Use the absolute value ONLY to check if it crosses the detection threshold
                    detect_flag = 1 if abs(raw_analog_diff) >= analog_detection_threshold else 0
                    
                    # 3. Store the signed difference for the Venn distribution log
                    diff = round(raw_analog_diff / analog_detection_threshold)
                elif detection_mode == "mapped_digital":
                    # We store the signed difference for the occurrence report
                    diff = defective_data['mapped_digital'] - golden_data['mapped_digital']
                    # We use the absolute difference ONLY to determine if it's detected
                    detect_flag = 1 if abs(diff) >= detection_threshold else 0
                elif detection_mode == "mapper_digital":
                    if defective_data['is_mapper_valid'] and golden_data['is_mapper_valid']:
                        diff = defective_data['mapper_digital'] - golden_data['mapper_digital']
                    else:
                        diff = defective_data['mapped_digital'] - golden_data['mapped_digital']
                    detect_flag = 1 if abs(diff) >= detection_threshold else 0
                
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
    if enable_matrix_gen:
        print("\n--- Running Matrix Generation ---")
        df_result.to_excel(recreated_result_file)
        print(f"Exported perfectly binary (0/1) recreated result dataset to {recreated_result_file}")
        
        # --- NEW: Export matrices independent of plotting ---
        binary_out_path = os.path.join(folder_location, "final_binary_matrix.csv")
        adc_diff_out_path = os.path.join(folder_location, "final_adc_diff_matrix.csv")
        
        # # We transpose (.T) them before saving so Defects are rows and Inputs are columns
        df_result.T.to_csv(binary_out_path)
        df_adc_diff.T.to_csv(adc_diff_out_path)
        
        print(f"Exported Binary Matrix CSV to: {binary_out_path}")
        print(f"Exported ADC Diff Matrix CSV to: {adc_diff_out_path}")
    else:
        print("\n--- Matrix Generation (.xlsx export) is DISABLED ---")

    # Phase 2b: Plotting Logic
    if enable_plotting:
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
        
        if plot_mode == "combined":
            if has_final_states:
                df_final_combined.to_csv(output_file)
                print(f"Exported combined dataset to {output_file}")
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
                generate_plots(df_final_combined, fault_dict, has_ninit, mode="combined", folder_location=folder_location)
            else:
                df_final_binary.to_csv(output_file)
                print(f"Exported binary dataset to {output_file}")
                print("\n=== Binary Detection Results Summary (No Final States) ===")
                val_counts = df_final_binary.unstack().dropna().value_counts()
                total = val_counts.sum()
                if total == 0: total = 1 
                for state in [0, 1]:
                    count = val_counts.get(state, 0)
                    pct = (count / total) * 100
                    label = "Detected  " if state == 1 else "Undetected"
                    print(f"{label}: {int(count):>6} instances ({pct:>5.2f}%)")
                generate_plots(df_final_binary, fault_dict, has_ninit, mode="binary", folder_location=folder_location)
            
        elif plot_mode == "adc_diff":
            df_final_adc.to_csv(output_file)
            print(f"Exported ADC difference dataset to {output_file}")
            print("\n=== ADC Difference Results Summary ===")
            max_diff = df_final_adc.values.max()
            avg_diff = df_final_adc.values.mean()
            print(f"Maximum Difference Observed: {max_diff}")
            print(f"Average Difference Across Grid: {avg_diff:.2f}")
            
            generate_plots(df_final_adc, fault_dict, has_ninit, mode="adc_diff", folder_location=folder_location)
    else:
        print("\n--- Fault Map Plotting Pipeline is DISABLED ---")
        
    return generated_cols