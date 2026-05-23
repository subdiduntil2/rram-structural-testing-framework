import os
import sys
import json
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

# Import functional pipeline dependencies
from initial_analysis import DualLogger, generate_fault_dict, load_mapper_data
from fault_modeling_plotting import generate_matrices_and_plots
from atpg_generation import generate_test_patterns

# Choose from: "mapper_digital", "direct_analog", or "mapped_digital"
print("detection threshold is set to => ", DETECTION_THRESHOLD)

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
            
        df_mapper = load_mapper_data(MAPPER_FILE)
        
        generated_cols = generate_matrices_and_plots(
            fault_dict=fault_dict,
            df_mapper=df_mapper,
            adc_mapping_file=ADC_MAPPING_FILE,
            detection_mode=DETECTION_MODE,
            detection_threshold=DETECTION_THRESHOLD,
            analog_detection_threshold=ANALOG_DETECTION_THRESHOLD,
            mixed_data_mode=MIXED_DATA_MODE,
            plot_mode=PLOT_MODE,
            enable_matrix_gen=ENABLE_MATRIX_GEN,
            enable_plotting=ENABLE_PLOTTING,
            recreated_result_file=RECREATED_RESULT_FILE,
            folder_location=FOLDER_LOCATION,
            output_file=OUTPUT_FILE
        )
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