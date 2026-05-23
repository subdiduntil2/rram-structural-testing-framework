import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.optimize import linprog

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