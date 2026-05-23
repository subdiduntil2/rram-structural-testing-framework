import pandas as pd
import warnings
import os
import re
import ast
from collections import defaultdict

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Defect-name canonicalization
_ADC_MAC_PREFIX = 'ADC_MAC_1n1m_st_schematic_r_def_'
_ADC_MAC_SIGNALS = frozenset({
    'BL', 'WL', 'SL', 'VDD', 'VSS', 'RRAM_down',
    'BL_internal', 'WL_internal', 'SL_internal', 'RRAM_down_internal',
    'Ndiscmax', 'Ndiscmin', 'rdet',
})


def _canonicalize_defect_name(name):
    """Return the canonical label for an ADC_MAC bridge/short defect."""
    if not isinstance(name, str) or not name.startswith(_ADC_MAC_PREFIX):
        return name
    core = name[len(_ADC_MAC_PREFIX):]
    body, _, tail = core.rpartition('_')
    if not tail.isdigit():
        return name
    parts = body.split('_')
    for k in range(1, len(parts)):
        a = '_'.join(parts[:k])
        b = '_'.join(parts[k:])
        if a in _ADC_MAC_SIGNALS and b in _ADC_MAC_SIGNALS:
            s1, s2 = sorted([a, b])
            return f"{_ADC_MAC_PREFIX}{s1}_{s2}_{tail}"
    return name


def _get_base_location(defect_name):
    """Strip the trailing '_<N>' index from a defect name so that all N
    variants of the same physical site collapse to a single location label
    """
    parts = defect_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return defect_name


class ATPGCoverageEvaluator:
    def __init__(self, file_r, file_wr, file_vco_p, file_vco_s):
        print("Loading CSV matrices into memory...")
        
        self.file_r = file_r
        self.file_wr = file_wr
        self.file_vco_p = file_vco_p
        self.file_vco_s = file_vco_s
        
        self.df_r = pd.read_csv(file_r, index_col=0)
        self.df_wr = pd.read_csv(file_wr, index_col=0)
        self.df_vco_p = pd.read_csv(file_vco_p, index_col=0)
        self.df_vco_s = pd.read_csv(file_vco_s, index_col=0)

        # Canonicalize row labels and standardize column names
        for df in [self.df_r, self.df_wr, self.df_vco_p, self.df_vco_s]:
            df.index = [_canonicalize_defect_name(x) for x in df.index]

        for df in [self.df_r, self.df_wr, self.df_vco_p, self.df_vco_s]:
            df.columns = [self._standardize_name(col) for col in df.columns]

        # Defect pool = union across the 3 physical datasets
        self.all_defects = set(self.df_r.index) | set(self.df_vco_p.index) | set(self.df_vco_s.index)
        self.total_defects = len(self.all_defects)

        self.all_base_locations = {_get_base_location(d) for d in self.all_defects}
        self.total_locations = len(self.all_base_locations)

        self.detected_global = set()

    def _standardize_name(self, name):
        """
        Applies all naming standardizations to a single string so that CSV columns 
        """
        name = name.replace('rp_0.01', 'rp_0.008')
        name = name.replace('rn_0.01', 'rn_0.008')
        name = name.replace('ninit_0.01', 'ninit_0.008')
        name = name.replace('rp_20_', 'rp_20.0_')
        name = name.replace('rn_20_', 'rn_20.0_')
        return name
        
    def _get_mapped_pattern(self, pattern, dataset_name):
        """
        Maps specific patterns to new ones if the dataset is NOT pruned.
        """
        is_pruned = "pruned" in str(dataset_name).lower()
        std_pat = self._standardize_name(pattern)
        
        if is_pruned:
            return std_pat
            
        mapping = {
            "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP": "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_P0PP",
            "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN": "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_NN0N_ninit_20",
            "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000": "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_NNNP"
        }
        
        std_mapping = {self._standardize_name(k): self._standardize_name(v) for k, v in mapping.items()}
        
        return std_mapping.get(std_pat, std_pat)

    def _get_detected_defects(self, df, pattern, source_filename="", warn_on_miss=True):
        search_pattern = self._get_mapped_pattern(pattern, source_filename)
        
        if search_pattern not in df.columns:
            if warn_on_miss:
                print(f"  [warn] column not found in {source_filename}: {search_pattern} (original: {pattern})")
            return set()
        
        detected_series = df[search_pattern]
        return set(detected_series[detected_series == 1].index)

    def evaluate_pattern(self, pattern, is_fast):
        # Fetch hits from fast (r) or slow (w+r) dataset, union with VCO datasets
        if is_fast:
            det_rw = self._get_detected_defects(self.df_r, pattern, source_filename=self.file_r)
        else:
            det_rw = self._get_detected_defects(self.df_wr, pattern, source_filename=self.file_wr)
            
        det_vco_p = self._get_detected_defects(self.df_vco_p, pattern, source_filename=self.file_vco_p)
        det_vco_s = self._get_detected_defects(self.df_vco_s, pattern, source_filename=self.file_vco_s)

        current_pattern_hits = det_rw | det_vco_p | det_vco_s

        new_hits = current_pattern_hits - self.detected_global
        added_defects = len(new_hits)
        
        self.detected_global.update(new_hits)
        
        detected_locs = {_get_base_location(d) for d in self.detected_global}
        print(f"Pattern: {pattern} ({'FAST' if is_fast else 'SLOW'})")
        print(f"  -> Added Defects Covered:  {added_defects}")
        print(f"  -> Total Cumulative Coverage: {len(self.detected_global)} / {self.total_defects} "
              f"({(len(self.detected_global)/self.total_defects)*100:.2f}%)")
        print(f"  -> Unique Locations Covered:  {len(detected_locs)} / {self.total_locations} "
              f"({(len(detected_locs)/self.total_locations)*100:.2f}%)")
        print("-" * 60)

    def run_sequence(self, fast_patterns, slow_patterns):
        print(f"Total Evaluated Defects Pool: {self.total_defects}  "
              f"(unique locations: {self.total_locations})")
        print("=" * 60)
        for pattern in fast_patterns:
            self.evaluate_pattern(pattern, is_fast=True)
        for pattern in slow_patterns:
            self.evaluate_pattern(pattern, is_fast=False)
        self.report_undetected()

    def report_undetected(self):
        undetected = sorted(self.all_defects - self.detected_global)
        detected_locs = {_get_base_location(d) for d in self.detected_global}
        undetected_locs = sorted(self.all_base_locations - detected_locs)
        print("=" * 60)
        print(f"Final coverage            : {len(self.detected_global)} / {self.total_defects} "
              f"({len(self.detected_global)/self.total_defects*100:.2f}%)")
        print(f"Final location coverage   : {len(detected_locs)} / {self.total_locations} "
              f"({len(detected_locs)/self.total_locations*100:.2f}%)")
        print(f"Undetected defect count   : {len(undetected)}")
        print(f"Undetected location count : {len(undetected_locs)}")
        print("Undetected defects:")
        print(undetected)
        return undetected

class ADCOffsetEvaluator:
    def __init__(self, file_full, file_vco, file_vco_32x2):
        print("\nLoading ADC Diff CSV matrices into memory...")
        
        self.file_full = file_full
        self.file_vco = file_vco
        self.file_vco_32x2 = file_vco_32x2
        
        self.df_full = pd.read_csv(file_full, index_col=0)
        self.df_vco = pd.read_csv(file_vco, index_col=0)
        self.df_vco_32x2 = pd.read_csv(file_vco_32x2, index_col=0)

        for df in [self.df_full, self.df_vco, self.df_vco_32x2]:
            df.index = [_canonicalize_defect_name(x) for x in df.index]

        for df in [self.df_full, self.df_vco, self.df_vco_32x2]:
            df.columns = [self._standardize_name(col) for col in df.columns]

    def _standardize_name(self, name):
        name = name.replace('rp_0.01', 'rp_0.008')
        name = name.replace('rn_0.01', 'rn_0.008')
        name = name.replace('ninit_0.01', 'ninit_0.008')
        name = name.replace('rp_20_', 'rp_20.0_')
        name = name.replace('rn_20_', 'rn_20.0_')
        return name

    def _get_mapped_pattern(self, pattern, dataset_name):
        """
        Maps specific patterns to new ones if the dataset is NOT pruned.
        """
        is_pruned = "pruned" in str(dataset_name).lower()
        std_pat = self._standardize_name(pattern)
        
        if is_pruned:
            return std_pat
            
        mapping = {
            "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP": "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_P0PP",
            "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN": "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_NN0N_ninit_20",
            "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000": "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_NNNP"
        }
        
        std_mapping = {self._standardize_name(k): self._standardize_name(v) for k, v in mapping.items()}
        
        return std_mapping.get(std_pat, std_pat)

    def evaluate_pattern_offsets(self, pattern):
        offset_to_defects = {}

        datasets = [
            (self.df_full, self.file_full), 
            (self.df_vco, self.file_vco), 
            (self.df_vco_32x2, self.file_vco_32x2)
        ]

        for df, filename in datasets:
            search_pattern = self._get_mapped_pattern(pattern, filename)
            
            if search_pattern in df.columns:
                series = df[search_pattern]
                series = series[series != 0].dropna()
                for defect, offset in series.items():
                    if offset not in offset_to_defects:
                        offset_to_defects[offset] = set()
                    offset_to_defects[offset].add(defect)

        print(f"Pattern: {pattern}")
        unique_offsets = sorted(offset_to_defects.keys())
        for offset in unique_offsets:
            defects = sorted(list(offset_to_defects[offset]))
            print(f"  -> ADC Offset {offset:>6.1f} : Sensitized {len(defects):>4} defects: {defects}")
            
        return unique_offsets

    def run_sequence(self, patterns):
        print("\n--- Running ADC Offset Breakdown Pipeline ---")
        all_pattern_offsets = []
        for pattern in patterns:
            unique_offsets = self.evaluate_pattern_offsets(pattern)
            all_pattern_offsets.append(unique_offsets)
            
        print("\n--- ADC Offset NumPy Arrays Summary ---")
        for i, offsets in enumerate(all_pattern_offsets):
            items = [str(o) for o in offsets]
            lines = []
            for j in range(0, len(items), 10):
                lines.append("    " + ", ".join(items[j:j+10]))
            
            array_str = f"pattern_{i} = np.array([\n" + ",\n".join(lines) + "\n])\n"
            print(array_str)


class PerLayerDefectCoverageEvaluator:
    def __init__(self, sim_file='hybrid_sim_results.txt', lut_file='combined_offsets_lut_r'):
        self.sim_file = sim_file
        self.lut_file = lut_file
        self.TOTAL_DEFECTS = 775 + 210  # legacy constant; real LUT pool is 986
        self.active_offsets_per_layer = defaultdict(set)
        self.offset_to_defects = defaultdict(set)
        self.layer_to_defects = defaultdict(list)
        self.all_base_defects = set()

    def _total_defects(self):
        """Actual size of the inference defect universe, derived from the LUT."""
        if not self.offset_to_defects:
            return 0
        return len(set().union(*self.offset_to_defects.values()))

    def _clean_text(self, text):
        """Removes any raw source tag artifacts that might have been copied over."""
        return re.sub(r'\\"', "", text)
        
    def _get_base_name(self, defect_name):
        """Extracts the base location name by stripping the numeric suffix (e.g., '_10')."""
        parts = defect_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return defect_name
                      
    def parse_data(self):
        # Simulation results: locate active offsets per layer
        with open(self.sim_file, 'r') as f:
            sim_content = self._clean_text(f.read())
            
        pattern = r'(fc\d):\s*([\-\.\d]+)\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%\s*\|\s*(\d+)'
        for match in re.finditer(pattern, sim_content):
            layer = match.group(1)
            offset = float(match.group(2))
            mismatch = float(match.group(4))
            total_tested = int(match.group(5))
            
            # Active: mismatch > 0% and tested on >= 32 patterns
            if mismatch > 0.0 and total_tested >= 32:
                self.active_offsets_per_layer[layer].add(offset)
                
        # LUT file: map offsets -> defects
        with open(self.lut_file, 'r') as f:
            lut_content = self._clean_text(f.read())
            
        matches = re.finditer(r'ADC Offset\s+([\-\.\d]+)\s*:.*?(\[.*?\])', lut_content, re.DOTALL)
        for match in matches:
            try:
                offset_val = float(match.group(1))
                defects_list_str = match.group(2)
                
                defects = ast.literal_eval(defects_list_str)

                defects = [_canonicalize_defect_name(d) for d in defects]

                self.offset_to_defects[offset_val].update(defects)
                
                for d in defects:
                    self.all_base_defects.add(self._get_base_name(d))
            except Exception:
                pass

    def evaluate_and_report(self):
        self.parse_data()
        
        TOTAL_LOCATIONS = len(self.all_base_defects)
        total_defects = self._total_defects() or 1

        print("\n" + "="*100)
        print("--- Per-Layer Final Defect Coverage Summary ---")
        print("="*100)
        print(f"Inference defect pool (derived): {total_defects}  "
              f"(legacy constant was {self.TOTAL_DEFECTS})")
        print(f"{'Layer':<7} | {'Active Offsets':<15} | {'Overall Defects':<16} | {'Overall %':<10} | {'Unique Locs':<12} | {'Unique Loc %':<12}")
        print("-" * 100)
        
        all_layers = ['fc1', 'fc2', 'fc3', 'fc4']
        
        for layer in all_layers:
            active_offsets = self.active_offsets_per_layer.get(layer, set())
            unique_defects = set()
            unique_locations = set()
            
            for offset in active_offsets:
                defects = self.offset_to_defects[offset]
                unique_defects.update(defects)
                
                for d in defects:
                    unique_locations.add(self._get_base_name(d))
            
            self.layer_to_defects[layer] = sorted(list(unique_defects))
            
            count = len(unique_defects)
            coverage = (count / total_defects) * 100
            
            loc_count = len(unique_locations)
            loc_coverage = (loc_count / TOTAL_LOCATIONS) * 100 if TOTAL_LOCATIONS > 0 else 0
            
            print(f"{layer:<7} | {len(active_offsets):<15} | {count:<16} | {coverage:<9.2f}% | {loc_count:<12} | {loc_coverage:.2f}%")
            
        print("-" * 100)
        
        print("\n" + "="*100)
        print("--- Accumulated Defect Vectors Per Layer ---")
        print("="*100)
        for layer in all_layers:
            print(f"\n{layer}_defects = {self.layer_to_defects[layer]}")


# Combined Per-Layer Defect Coverage: merges ATPG (Class 1) with inference (Class 3)
class CombinedPerLayerCoverageEvaluator:
    """Merge per-layer ATPG evidence (Class 1) with inference evidence (Class 3)."""
    LEGACY_TOTAL_DEFECTS = 775 + 210

    def __init__(
        self,
        file_r, file_wr, file_vco_p, file_vco_s,
        sim_file, lut_file,
        layer_vectors,
    ):
        """
        layer_vectors : dict[str, tuple[list, list]]
            Maps layer name ('fc4', 'fc3', 'fc2', ...) to a
            (fast_patterns, slow_patterns) tuple. 
        """
        self.file_r, self.file_wr = file_r, file_wr
        self.file_vco_p, self.file_vco_s = file_vco_p, file_vco_s
        self.sim_file, self.lut_file = sim_file, lut_file
        self.layer_vectors = layer_vectors

        self.atpg_layer_defects = {}
        self.inference_layer_defects = {}
        self.combined_layer_defects = {}

        self.atpg_pool = set()
        self.inference_pool = set()

    @property
    def total_defects(self):
        """True universal defect pool: union of ATPG CSVs and LUT defects."""
        return len(self.atpg_pool | self.inference_pool)

    def _run_atpg_per_layer(self):
        """Instantiate Class 1 once, rerun with a clean accumulator per layer."""
        print("\n" + "#" * 80)
        print("### Stage A : Class 1 ATPG runs (per layer)")
        print("#" * 80)

        evaluator = ATPGCoverageEvaluator(
            file_r=self.file_r,
            file_wr=self.file_wr,
            file_vco_p=self.file_vco_p,
            file_vco_s=self.file_vco_s,
        )

        self.atpg_pool = set(evaluator.all_defects)

        for layer, (fast_patterns, slow_patterns) in self.layer_vectors.items():
            active_fast = [p for p in fast_patterns if not p.strip().startswith("#")]
            active_slow = [p for p in slow_patterns if not p.strip().startswith("#")]

            print("\n" + "-" * 80)
            print(
                f"[Class 1] layer = {layer.upper()} | "
                f"{len(active_fast)} fast vec, {len(active_slow)} slow vec"
            )
            print("-" * 80)

            # Reset accumulator so each layer's run is independent
            evaluator.detected_global = set()
            evaluator.run_sequence(
                fast_patterns=active_fast,
                slow_patterns=active_slow,
            )
            self.atpg_layer_defects[layer] = set(evaluator.detected_global)

    def _run_inference(self):
        print("\n" + "#" * 80)
        print("### Stage B : Class 3 Per-Layer Inference run (unchanged)")
        print("#" * 80)

        layer_evaluator = PerLayerDefectCoverageEvaluator(
            sim_file=self.sim_file,
            lut_file=self.lut_file,
        )
        layer_evaluator.evaluate_and_report()

        self.inference_pool = set().union(*layer_evaluator.offset_to_defects.values()) \
            if layer_evaluator.offset_to_defects else set()

        self.inference_layer_defects = {
            layer: set(defects)
            for layer, defects in layer_evaluator.layer_to_defects.items()
        }

    def run(self):
        self._run_atpg_per_layer()
        self._run_inference()

        all_layers = ["fc1", "fc2", "fc3", "fc4"]
        for layer in all_layers:
            atpg = self.atpg_layer_defects.get(layer, set())
            inf = self.inference_layer_defects.get(layer, set())
            self.combined_layer_defects[layer] = atpg | inf

        self._report(all_layers)

    def _report(self, all_layers):
        true_total = self.total_defects
        legacy_total = self.LEGACY_TOTAL_DEFECTS

        atpg_locs = {_get_base_location(d) for d in self.atpg_pool}
        inf_locs  = {_get_base_location(d) for d in self.inference_pool}
        total_locs = len(atpg_locs | inf_locs)

        print("\n" + "=" * 120)
        print("--- Stage C : COMBINED Per-Layer Defect Coverage (ATPG UNION Inference) ---")
        print("=" * 120)
        print(f"ATPG defect pool        : {len(self.atpg_pool)}  "
              f"(unique locations: {len(atpg_locs)})")
        print(f"Inference defect pool   : {len(self.inference_pool)}  "
              f"(unique locations: {len(inf_locs)})")
        print(f"Union (true denominator): {true_total}  (unique locations: {total_locs})")
        print(f"Legacy Class 3 constant : {legacy_total}  (kept for reference only)")
        print("-" * 120)
        print(
            f"{'Layer':<6} | {'ATPG-only':<10} | {'Inf-only':<10} | "
            f"{'Overlap':<8} | {'Combined':<9} | {'Cov %':<7} | "
            f"{'UniqLocs':<9} | {'Loc %':<7}"
        )
        print("-" * 120)
        for layer in all_layers:
            atpg = self.atpg_layer_defects.get(layer, set())
            inf = self.inference_layer_defects.get(layer, set())
            combined = self.combined_layer_defects[layer]
            only_atpg = len(atpg - inf)
            only_inf = len(inf - atpg)
            overlap = len(atpg & inf)
            cov = (len(combined) / true_total) * 100.0 if true_total else 0.0

            combined_locs = {_get_base_location(d) for d in combined}
            loc_cov = (len(combined_locs) / total_locs) * 100.0 if total_locs else 0.0

            print(
                f"{layer:<6} | {only_atpg:<10} | {only_inf:<10} | "
                f"{overlap:<8} | {len(combined):<9} | {cov:<7.2f} | "
                f"{len(combined_locs):<9} | {loc_cov:<7.2f}"
            )
        print("-" * 120)

        print("\n" + "=" * 110)
        print("--- Combined Defect Vectors Per Layer ---")
        print("=" * 110)
        for layer in all_layers:
            vec = sorted(self.combined_layer_defects[layer])
            print(f"\n{layer}_combined_defects = {vec}")

if __name__ == "__main__":
    fast_atpg = []
    slow_atpg = []
    #fc4
    # fast_atpg=[]
    # "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_P0PP",
    # "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_NN0N_ninit_20",
    # "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_NNNP",
    # slow_atpg=[
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0"]
    
    slow_atpg=[
        "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
        "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
        "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
        "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
        "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
        "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
        "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN_ninit_0.008",
        "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NNNP_ninit_0.008",
        "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN_ninit_20",
        "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0_ninit_0.008"]
    
    #fc3
    # fast_atpg = [
    #     # "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     # "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     # "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     # "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     # "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     # "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     # "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     # "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0"
    # ]
    
    # slow_atpg = [
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     # "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     # "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     # "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0"
    # ]
    # #fc1
    # fast_atpg = [
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0"
    # ]
    
    # slow_atpg = [
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     # "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     # "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     # "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0"
    # ]
    #fc2
    # fast_atpg = [
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     # "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     # "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     # "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0"
    # ]
    
    # slow_atpg = [
    #     # "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     # "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     # "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     # "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     # "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     # "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     # "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     # "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0"
    # ]

    base_dir = "all_finals/hybrid/"

    try:
        evaluator = ATPGCoverageEvaluator(
            file_r=os.path.join(base_dir, "final_binary_matrix_r.csv"),
            file_wr=os.path.join(base_dir, "final_binary_matrix_w+r.csv"),
            file_vco_p=os.path.join(base_dir, "final_binary_matrix_vco_pruned.csv"),
            file_vco_s=os.path.join(base_dir, "final_binary_matrix_vco_small.csv")
        )

        active_fast = [p for p in fast_atpg if not p.strip().startswith('#')]
        active_slow = [p for p in slow_atpg if not p.strip().startswith('#')]
        evaluator.run_sequence(fast_patterns=active_fast, slow_patterns=active_slow)
    except FileNotFoundError as e:
        print(f"Skipping Hybrid ATPG evaluation due to missing files: {e}")


    # # =========================================================
    # # New Class Evaluation: ADC Offset Breakdown Pipeline
    # # =========================================================
    # atpg_offset = [
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", 
    #     "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", 
    #     "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0",
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP", 
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008", 
    # ]
    
    # try:
    #     offset_evaluator = ADCOffsetEvaluator(
    #         file_full="For_ITC26_full_balanced/final_adc_diff_matrix.csv",
    #         file_vco="For_ITC26_vco_final/final_adc_diff_matrix_pruned.csv",
    #         file_vco_32x2="For_ITC26_vco_32x2_corrected/final_adc_diff_matrix.csv"
    #     )
    #     offset_evaluator.run_sequence(atpg_offset)
    # except FileNotFoundError as e:
    #     print(f"Skipping ADC Offset evaluation due to missing files: {e}")
        
    # try:
    #     layer_evaluator = PerLayerDefectCoverageEvaluator(
    #         sim_file="hybrid_sim_results.txt",
    #         lut_file="combined_offsets_lut_r"
    #     )
    #     layer_evaluator.evaluate_and_report()
    # except FileNotFoundError as e:
    #     print(f"\nSkipping Per-Layer Defect Coverage evaluation due to missing files: {e}")

    # # =========================================================
    # # Class 4 Evaluation: Combined Per-Layer Defect Coverage
    # # (fc4 / fc3 / fc2 ATPG runs + Class 3 inference, merged)
    # # =========================================================

    # # fc4 slow vectors (full complementary set, 11 patterns)
    # slow_fc4 = [
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP",
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN",
    #     "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0",
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0",
    #     "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0",
    # ]

    # # fc3 slow vectors (8 patterns)
    # slow_fc3 = [
    #     "rp_0.01_rn_20.0_inp_0.55_inn_0.55_neighs_PPPP",
    #     "rp_20.0_rn_0.01_inp_0.85_inn_0.25_neighs_NNNN",
    #     "rp_20_rn_20_inp_0.55_inn_0.55_neighs_P000",
    #     "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",
    #     "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN",
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0",
    # ]

    # # fc2 slow vectors (3 patterns)
    # slow_fc2 = [
    #     "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",
    #     "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0",
    # ]

    # layer_vectors = {
    #     "fc4": ([], slow_fc4),
    #     "fc3": ([], slow_fc3),
    #     "fc2": ([], slow_fc2),
    #     # 'fc1' intentionally omitted -> inference-only in the combined report
    # }

    # try:
    #     combined = CombinedPerLayerCoverageEvaluator(
    #         file_r=os.path.join(base_dir, "final_binary_matrix_r.csv"),
    #         file_wr=os.path.join(base_dir, "final_binary_matrix_w+r.csv"),
    #         file_vco_p=os.path.join(base_dir, "final_binary_matrix_vco_pruned.csv"),
    #         file_vco_s=os.path.join(base_dir, "final_binary_matrix_vco_small.csv"),
    #         sim_file="hybrid_sim_results.txt",
    #         lut_file="combined_offsets_lut_r",
    #         layer_vectors=layer_vectors,
    #     )
    #     combined.run()
    # except FileNotFoundError as e:
    #     print(f"Skipping Combined Per-Layer evaluation due to missing files: {e}")
