import re
from collections import defaultdict

# 1. Define the input files and output file
input_files = [
    "For_ITC26_vco_32x2_corrected\\venn_log_itc26_vco_small_1.txt",
    "For_ITC26_full_balanced\\venn_log_itc26_2t2r_all_full_balanced_r.txt",
    "For_ITC26_vco_final\\venn_log_itc26_vco_pruned.txt"
]
output_file = "all_finals\\hybrid\\merged_venn_final_r.txt"

# 2. Data structures for the merge
merged_adc_data = defaultdict(set)
merged_diff_data = {"Undetected": defaultdict(set), "Detected": defaultdict(set)}
total_patterns = 0
total_defects = 0

# Regex patterns for extracting data
header_pattern_re = re.compile(r"Calculating offsets for the (\d+) patterns")
header_defects_re = re.compile(r"Submatrix size: \d+ patterns x (\d+) total defects")
data_line_re = re.compile(r"ADC Offset\s+([-\d\.]+)\s+:\s+\d+\s+unique defects:\s+\[(.*?)\]")

# Added regex for Diff extraction (Handles multiline and single line logs seamlessly)
diff_pattern_re = re.compile(r"Diff\s+([-\d]+)\s*\|\s*(Undetected|Detected)\s*:\s*\d+\s*unique defects(?:.*?:\s*\[(.*?)\])?")

# 3. Read and aggregate data
for file_name in input_files:
    with open(file_name, 'r') as f:
        content = f.read()
        
        # Parse header totals
        for match in header_pattern_re.findall(content):
            total_patterns += int(match)
            
        for match in header_defects_re.findall(content):
            total_defects += int(match)
            
        # Parse ADC individual offset rows
        for match in data_line_re.finditer(content):
            offset_val = float(match.group(1))
            defects_str = match.group(2)
            
            # Split, clean, and add defects to the set for this offset
            defects_list = [d.strip().strip("'") for d in defects_str.split("',") if d.strip()]
            for defect in defects_list:
                # Clean up trailing quotes if any
                clean_defect = defect.strip("'")
                if clean_defect:
                    merged_adc_data[offset_val].add(clean_defect)

        # Parse Diff Detected/Undetected rows
        for match in diff_pattern_re.finditer(content):
            diff_val = int(match.group(1))
            state = match.group(2).strip()
            defects_str = match.group(3)
            
            # If there are defects, split and extract them
            if defects_str:
                defects_list = [d.strip().strip("'") for d in defects_str.split("',") if d.strip()]
                for defect in defects_list:
                    clean_defect = defect.strip("'")
                    if clean_defect:
                        merged_diff_data[state][diff_val].add(clean_defect)
            else:
                # Initialize the empty set to ensure '0 unique defects' cases get logged
                if diff_val not in merged_diff_data[state]:
                    merged_diff_data[state][diff_val] = set()

# 4. Sort the offsets
sorted_adc_offsets = sorted(merged_adc_data.keys())

# 5. Write the merged data in the exact example format
with open(output_file, 'w') as f:
    f.write("======================================================================\n")
    f.write("=== ADC Difference Distribution (Optimal Solution 1) ===\n")
    f.write("======================================================================\n")
    f.write(f"Calculating offsets for the {total_patterns} patterns in Optimal Solution 1.\n")
    f.write(f"Submatrix size: {total_patterns} patterns x {total_defects} total defects.\n")
    f.write("------------------------------------------------------------\n")
    
    for offset in sorted_adc_offsets:
        # Sort the defects alphabetically for consistent formatting
        defects_list = sorted(list(merged_adc_data[offset]))
        defects_formatted = ", ".join(f"'{d}'" for d in defects_list)
        f.write(f"  ADC Offset {offset:>6.1f} : {len(defects_list):>6} unique defects: [{defects_formatted}]\n")

    # Write out the newly merged Combined State Distributions
    f.write("\n\n======================================================================\n")
    f.write("=== Combined State Distribution (Optimal Solution 1) ===\n")
    f.write("======================================================================\n")

    # Collect all existing integer diff values and sort them
    all_diff_vals = set(merged_diff_data["Undetected"].keys()).union(merged_diff_data["Detected"].keys())
    sorted_diff_vals = sorted(all_diff_vals)
    
    for val in sorted_diff_vals:
        for state in ["Undetected", "Detected"]:
            defects_set = merged_diff_data[state].get(val, set())
            
            if len(defects_set) == 0:
                f.write(f"  Diff {val:>2} |\n{state:<10}:      0 unique defects\n")
            else:
                defects_list = sorted(list(defects_set))
                defects_formatted = ", ".join(f"'{d}'" for d in defects_list)
                f.write(f"  Diff {val:>2} |\n{state:<10}: {len(defects_list):>6} unique defects: [{defects_formatted}]\n")

print(f"Successfully merged {total_patterns} patterns and {total_defects} defects into {output_file}")