import os
import re
from collections import OrderedDict

def generate_unified_tpg_report(input_files, output_file):
    # Data structures for aggregated data
    all_undetectable_defs = set()
    source_patterns = OrderedDict()
    source_stats = OrderedDict()
    combined_patterns = OrderedDict()
    
    for filepath in input_files:
        # Extract a clean source name from the file (e.g. "itc26_vco_pruned_1")
        fname = os.path.basename(filepath)
        source_name = fname.replace('tpg_solver_log_', '').replace('.txt', '')
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 1. Parse Undetectable Defects Array ("lets see [...]")
        lets_see_match = re.search(r"lets see \[(.*?)\]", content, re.DOTALL)
        if lets_see_match:
            defs_str = lets_see_match.group(1).replace('\n', '').replace('\r', '')
            defs = [d.strip(" '") for d in defs_str.split(',')]
            defs = [d for d in defs if d]  # Filter empty strings
            all_undetectable_defs.update(defs)
            
        # 2. Parse Optimal Solution 1 Patterns
        opt_block_match = re.search(r"Optimal Solution 1 \(Score:.*?\):\n(.*?)(?:------------------------------------------------------------)", content, re.DOTALL)
        patterns = []
        if opt_block_match:
            lines = opt_block_match.group(1).strip().split('\n')
            for line in lines:
                if '-->' in line:
                    pattern_part, feature_part = line.split('-->')
                    pattern = pattern_part.strip()
                    features = feature_part.strip().replace('Features (Bulk, N1, N2, N3, N4): ', '').strip()
                    patterns.append((pattern, features))
                    
                    # Deduplicate into combined dictionary while preserving first-seen source
                    if pattern not in combined_patterns:
                        combined_patterns[pattern] = (features, source_name)
        source_patterns[source_name] = patterns
        
        # 3. Parse Final Test Accuracy & Coverage Block
        stats_block = re.search(r"=== Final Test Accuracy & Coverage Report ===\n(.*?)(?:======================================================================)", content, re.DOTALL)
        if stats_block:
            stats_text = stats_block.group(1)
            stats = {}
            
            def ext_int(regex):
                m = re.search(regex, stats_text)
                return int(m.group(1)) if m else 0
                
            def ext_float(regex):
                m = re.search(regex, stats_text)
                return float(m.group(1)) if m else 0.0

            stats['total_eval'] = ext_int(r"Total Evaluated Defects:\s+(\d+)")
            stats['undetectable'] = ext_int(r"Undetectable Defects:\s+(\d+)")
            
            det_cov_m = re.search(r"Detectable Defects Covered:\s+(\d+)\s+/\s+(\d+)", stats_text)
            stats['det_cov'] = int(det_cov_m.group(1)) if det_cov_m else 0
            stats['det_total'] = int(det_cov_m.group(2)) if det_cov_m else 0
            
            stats['atpg_success'] = ext_float(r"ATPG Success on Detectables:\s+([\d.]+)%")
            stats['overall_acc'] = ext_float(r"Overall Final Test Accuracy:\s+([\d.]+)%")
            
            stats['loc_total'] = ext_int(r"Total Defect Locations:\s+(\d+)")
            stats['loc_undetect'] = ext_int(r"Undetectable Locations:\s+(\d+)")
            
            loc_und_list_m = re.search(r"Undetectable Locations:\s+\d+\s+\[(.*?)\]", stats_text)
            stats['loc_undetect_list'] = []
            if loc_und_list_m and loc_und_list_m.group(1).strip():
                clean_list = [l.strip(" '") for l in loc_und_list_m.group(1).split(',')]
                stats['loc_undetect_list'] = [l for l in clean_list if l]
                
            stats['loc_det'] = ext_int(r"Detectable Locations:\s+(\d+)")
            stats['loc_cov'] = ext_int(r"Actually Covered Locations:\s+(\d+)")
            stats['loc_cov_pct'] = ext_float(r"Unique Location Coverage:\s+([\d.]+)%")
            
            source_stats[source_name] = stats

    # Write Merged Output
    with open(output_file, 'w') as f:
        # Header
        f.write("======================================================================\n")
        f.write("=== Unified TPG Solver Report ===\n")
        f.write(f"=== Merged from {len(input_files)} source files ===\n")
        f.write("======================================================================\n\n")
        
        # Undetectable Defs
        sorted_undetectable = sorted(list(all_undetectable_defs))
        f.write(f"undetectable defs => ({len(sorted_undetectable)})\n")
        formatted_defs = ", ".join(f"'{d}'" for d in sorted_undetectable)
        f.write(f"lets see [{formatted_defs}]\n\n")
        
        # Combined Patterns
        f.write("======================================================================\n")
        f.write("=== Combined Optimal Test Patterns (Solution 1, Deduplicated) ===\n")
        f.write(f"=== {len(combined_patterns)} unique patterns from {sum(len(p) for p in source_patterns.values())} total ===\n")
        f.write("======================================================================\n\n")
        
        f.write("Combined Solution 1:\n")
        for pat, (feat, src) in combined_patterns.items():
            f.write(f"  {pat}  --> Features (Bulk, N1, N2, N3, N4): {feat}  [from: {src}]\n")
        f.write("------------------------------------------------------------\n\n")
        
        # Per-Source Pattern Breakdown
        f.write("======================================================================\n")
        f.write("=== Per-Source Pattern Breakdown ===\n")
        f.write("======================================================================\n\n")
        
        for src, patterns in source_patterns.items():
            f.write(f"  Source: {src} ({len(patterns)} patterns)\n")
            for pat, feat in patterns:
                f.write(f"    {pat}  --> {feat}\n")
            f.write("\n")
            
        # Per-Source Test Accuracy & Coverage
        f.write("======================================================================\n")
        f.write("=== Per-Source Test Accuracy & Coverage ===\n")
        f.write("======================================================================\n\n")
        
        for src, stats in source_stats.items():
            f.write(f"  Source: {src}\n")
            f.write(f"    Total Evaluated Defects:     {stats['total_eval']}\n")
            f.write(f"    Undetectable Defects:        {stats['undetectable']}\n")
            f.write(f"    Detectable Defects Covered:  {stats['det_cov']} / {stats['det_total']}\n")
            f.write(f"    ATPG Success on Detectables: {stats['atpg_success']:.2f}%\n")
            f.write(f"    Overall Final Test Accuracy: {stats['overall_acc']:.2f}%\n")
            f.write(f"    Total Defect Locations:      {stats['loc_total']}\n")
            und_locs_str = ", ".join(f"'{l}'" for l in stats['loc_undetect_list'])
            f.write(f"    Undetectable Locations:      {stats['loc_undetect']} [{und_locs_str}]\n")
            f.write(f"    Detectable Locations:        {stats['loc_det']}\n")
            f.write(f"    Actually Covered Locations:  {stats['loc_cov']}\n")
            f.write(f"    Unique Location Coverage:    {stats['loc_cov_pct']:.2f}%\n")
            f.write("  ------------------------------------------------------------\n")
            
        f.write("\n")
        
        # Calculates Averages & Sums
        if source_stats:
            avg_atpg = sum(s['atpg_success'] for s in source_stats.values()) / len(source_stats)
            avg_overall = sum(s['overall_acc'] for s in source_stats.values()) / len(source_stats)
            avg_loc_cov = sum(s['loc_cov_pct'] for s in source_stats.values()) / len(source_stats)
            
            sum_eval = sum(s['total_eval'] for s in source_stats.values())
            sum_undetectable = sum(s['undetectable'] for s in source_stats.values())
            sum_det_cov = sum(s['det_cov'] for s in source_stats.values())
            sum_det_total = sum(s['det_total'] for s in source_stats.values())
            
            sum_loc_total = sum(s['loc_total'] for s in source_stats.values())
            sum_loc_undetect = sum(s['loc_undetect'] for s in source_stats.values())
            sum_loc_det = sum(s['loc_det'] for s in source_stats.values())
            sum_loc_cov = sum(s['loc_cov'] for s in source_stats.values())
            
            # Combine undetectable locations
            combined_und_locs = set()
            for s in source_stats.values():
                combined_und_locs.update(s['loc_undetect_list'])
            sorted_combined_und_locs = sorted(list(combined_und_locs))
            
            # Calculate final total percentages
            final_atpg_success = (sum_det_cov / sum_det_total * 100) if sum_det_total > 0 else 100.0
            final_overall_acc = (sum_det_cov / sum_eval * 100) if sum_eval > 0 else 100.0
            final_loc_cov = (sum_loc_cov / sum_loc_det * 100) if sum_loc_det > 0 else 100.0
            
            f.write("======================================================================\n")
            f.write("=== Average Test Accuracy & Coverage (Across All Sources) ===\n")
            f.write("======================================================================\n")
            f.write(f"  Avg ATPG Success on Detectables: {avg_atpg:.2f}%\n")
            f.write(f"  Avg Overall Final Test Accuracy: {avg_overall:.2f}%\n")
            f.write(f"  Avg Unique Location Coverage:    {avg_loc_cov:.2f}%\n\n")
            
            f.write("======================================================================\n")
            f.write("=== Combined Test Accuracy & Coverage (Summed Across Sources) ===\n")
            f.write("======================================================================\n")
            f.write(f"  Total Evaluated Defects:     {sum_eval}\n")
            f.write(f"  Undetectable Defects:        {sum_undetectable}\n")
            f.write(f"  Detectable Defects Covered:  {sum_det_cov} / {sum_det_total}\n")
            f.write(f"  ATPG Success on Detectables: {final_atpg_success:.2f}%\n")
            f.write(f"  Overall Final Test Accuracy: {final_overall_acc:.2f}%\n")
            f.write("  ------------------------------------------------------------\n")
            f.write(f"  Total Defect Locations:      {sum_loc_total}\n")
            formatted_und_locs = ", ".join(f"'{l}'" for l in sorted_combined_und_locs)
            f.write(f"  Undetectable Locations:      {sum_loc_undetect} [{formatted_und_locs}]\n")
            f.write(f"  Detectable Locations:        {sum_loc_det}\n")
            f.write(f"  Actually Covered Locations:  {sum_loc_cov}\n")
            f.write(f"  Unique Location Coverage:    {final_loc_cov:.2f}%\n")
            f.write("======================================================================\n")

if __name__ == '__main__':
    files_to_merge = [
        "For_ITC26_vco_32x2_corrected\\tpg_solver_log_itc26_vco_small_1.txt",
        "For_ITC26_full_balanced\\tpg_solver_log_itc26_2t2r_all_full_balanced_w+r.txt",
        "For_ITC26_vco_final\\tpg_solver_log_itc26_vco_pruned.txt"
    ]
    
    generate_unified_tpg_report(files_to_merge, "all_finals\\hybrid\\merged_tpg_final_w+r.txt")
    
    print("Files successfully merged into merged_tpg_final_part_1.txt!")