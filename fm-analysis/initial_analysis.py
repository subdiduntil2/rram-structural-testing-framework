import os
import re
import glob
import json
import sys
import pandas as pd

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

def load_mapper_data(mapper_file):
    print("Reading mapper CSV with ragged-row handling...")
    rows_raw = []
    max_cols = 0
    with open(mapper_file, 'r') as f_map:
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
        
    return df_mapper