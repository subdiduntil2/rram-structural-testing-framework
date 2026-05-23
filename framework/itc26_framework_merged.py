import os
try:
    os.add_dll_directory(r"C:\Users\manos\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torchcodec") 
except AttributeError:
    pass 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import shutil
import re
import numpy as np
import copy
import csv
import urllib.request
import tarfile
import wave  # Replaces torchaudio for reading WAVs
import matplotlib.pyplot as plt # For plotting targets

# ==========================================
# 1. User Configuration
# ==========================================
DATASET = 'KWS' # Options: 'MNIST', 'KWS'                   
KWS_FEATURE_TYPE = 'TINYSNS'    # Options: 'MFCC' (traditional) or 'TINYSNS' (16-ch filterbank like tinysns)  
KWS_NUM_CLASSES = 12               # New Option: 2 (Binary) or 12 (Standard KWS multi-class)
INPUT_MIN = -1.0                  
INPUT_MAX = 1.0

MODEL_ARCH = 'PureLinear'         # Options: 'MLP', 'LeNet5', 'PureLinear', 'BinaryCNN'
MLP_NUM_LAYERS = 3                
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is => ", DEVICE)
LOG_DIR = "./rram_fault_logs_csv_pure_batch_final_23_04_26_lenet5"

ANALYSIS_IMAGE_LIMIT = 5000
RUN_FAULT_INJECTION = True
USE_BATCH_NORM = False
RUN_TARGET_GENERATION = True 
FRESH_LOGS = True          
VALIDATE_ON_FULL_DATASET = True 

# --- NEW: Evaluation Mode Settings ---
EVAL_MODE = 'DATASET' # Options: 'DATASET' (Standard MNIST/KWS) or 'SYNTHETIC' (Generated PASS targets)
SYNTHETIC_LOG_DIR = "./rram_fault_logs_csv_pure_batch_15_04_26" 

READOUT_MODE = 'LIF' # LIF or ADC

LIF_TIME_STEPS = 255                # simulation timesteps (0..255 = 8-bit spike counter)
LIF_LEAK = 1.0                      # membrane leak: 1.0 = pure IF, <1.0 = LIF
LIF_THRESHOLDS = (28.0, 18.0, 10.0) # per-hidden-layer firing thresholds (fc1,fc2,fc3)
LIF_SPIKE_BINARIZE_THRESH = 60      # spike-count target matching: count>=this -> 1, else 0

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
USE_SHIFTED_TARGETS = False

# ATPG_STRINGS_FUNCTIONAL = [
#     [
#         "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", #same
#         "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
#         "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
#         "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", #same 
#         "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0",
#         # ADC patterns (constant)
#         "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_P0PP",
#         "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_NN0N_ninit_20",
#         "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_NNNP",
#         "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",  
#         "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
#         "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
#         # RRAM W+R
#         "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN_ninit_0.008", #same
#         "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NNNP_ninit_0.008", 
#         "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN_ninit_20",  
#         "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0_ninit_0.008", #same
#     ]
# ]

ATPG_STRINGS_FUNCTIONAL = [
    [
        "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN", #same
        "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_P0N0", 
        "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_PN00",  
        "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0", #same 
        "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_PPN0",
        # # # ADC patterns (constant)
        "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_P0PP",
        "rp_0.008_rn_20.0_inp_0.85_inn_0.25_neighs_NN0N_ninit_20",
        "rp_20.0_rn_0.008_inp_0.85_inn_0.25_neighs_NNNP",
        "rp_20.0_rn_0.008_inp_0.55_inn_0.55_neighs_PP0N_ninit_0.008",  
        "rp_20_rn_20_inp_0.25_inn_0.85_neighs_PPP0_ninit_0.008",  
        "rp_20_rn_20_inp_0.55_inn_0.55_neighs_000P_ninit_0.008",
        # # W+R
        # "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN_ninit_0.008", #same
        # "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NNNP_ninit_0.008", 
        # "rp_0.008_rn_20.0_inp_0.55_inn_0.55_neighs_PNPN_ninit_20",  
        # "rp_20.0_rn_0.008_inp_0.25_inn_0.85_neighs_NPP0_ninit_0.008", #same
    ]
]

ATPG_STRINGS=ATPG_STRINGS_FUNCTIONAL
# This will be populated dynamically at runtime by the ATPG pipeline
INPUT_TARGETS = [] 
TARGET_CASE_REGISTRY = {}

class SyntheticPassDataset(torch.utils.data.Dataset):
    """Loads synthetic images marked with PASS from CSV files for evaluation."""
    def __init__(self, data_dir=None, file_paths=None):
        self.file_paths = []
        if file_paths is not None:
            self.file_paths = file_paths
        elif data_dir is not None and os.path.exists(data_dir):
            for file_name in os.listdir(data_dir):
                if 'PASS' in file_name and file_name.endswith('.csv'):
                    self.file_paths.append(os.path.join(data_dir, file_name))
        else:
            if data_dir is not None:
                print(f"[Warning] Synthetic directory {data_dir} not found!")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        flat_data = np.loadtxt(self.file_paths[idx], delimiter=",", dtype=np.float32)
        tensor = torch.from_numpy(flat_data)

        t_size = tensor.size(0)
        if t_size == 1024:
            tensor = tensor.view(1, 32, 32) 
        elif t_size == 784:
            # Prevent KWS vectors from being viewed as 28x28 MNIST images
            if DATASET == 'KWS':
                tensor = tensor.view(1, 784) 
            else:
                tensor = tensor.view(1, 28, 28) 
        elif t_size == 1010:
             tensor = tensor.view(1, 1010)  
        elif t_size == 256:
             tensor = tensor.view(1, 256)   
        
        return tensor, torch.tensor(-1)

def evaluate_synthetic_faults(model, synth_loader, case_config):
    """ Evaluates equivalent 'defect accuracy' by comparing Clean vs Faulty inference.
    """
    model.eval()
    total = 0
    matches_0 = 0
    mismatches_1 = 0

    with torch.no_grad():
        for data, _ in synth_loader:
            data = data.to(DEVICE)

            # 1. Clean Pass (Baseline)
            model.reset_all_faults()
            clean_out = model(data)
            clean_preds = clean_out.argmax(dim=1)

            # 2. Faulty Pass
            for layer_name, param in case_config:
                model.configure_faults('offset', (param,), layer_name=layer_name)

            faulty_out = model(data)
            faulty_preds = faulty_out.argmax(dim=1)

            # Compare (0 = match, 1 = mismatch)
            matches_0 += (clean_preds == faulty_preds).sum().item()
            mismatches_1 += (clean_preds != faulty_preds).sum().item()
            total += data.size(0)

    # Clean up faults so they don't leak into the next experiment
    model.reset_all_faults()

    # Equivalent to MNIST accuracy: How many images survived the fault?
    defect_accuracy = 100. * matches_0 / total if total > 0 else 0.0
    mismatch_rate = 100. * mismatches_1 / total if total > 0 else 0.0

    return defect_accuracy, mismatch_rate, total

def evaluate_synthetic_faults_lif(model, synth_loader, case_config,
                                   thresholds=LIF_THRESHOLDS,
                                   time_steps=LIF_TIME_STEPS,
                                   leak=LIF_LEAK):
    model.eval()
    rram_layers = _extract_rram_layers(model)
    th_list = _build_th_list(rram_layers, thresholds)
    engine = LIFReadoutEngine(rram_layers, th_list, time_steps, leak)

    total = 0
    matches_0 = 0
    mismatches_1 = 0

    with torch.no_grad():
        for data, _ in synth_loader:
            data = data.to(DEVICE)
            feats = _to_lif_range(data, time_steps)

            # 1. Clean Pass (Baseline) -- ensure no stale spike faults
            model.reset_all_faults()
            clean_preds = [engine.infer_single(feats[i]) for i in range(feats.size(0))]

            # 2. Faulty Pass -- apply spike faults via layer attributes
            for layer_name, param in case_config:
                model.configure_faults('spike_offset', (param,), layer_name=layer_name)
            faulty_preds = [engine.infer_single(feats[i]) for i in range(feats.size(0))]

            # Compare (0 = match, 1 = mismatch)
            for c, f in zip(clean_preds, faulty_preds):
                if c == f: matches_0 += 1
                else:      mismatches_1 += 1
                total += 1

    # Clean up so faults don't leak into the next experiment
    model.reset_all_faults()

    defect_accuracy = 100. * matches_0 / total if total > 0 else 0.0
    mismatch_rate = 100. * mismatches_1 / total if total > 0 else 0.0
    return defect_accuracy, mismatch_rate, total

class TinySNSFeatureExtractor:
    """16-channel Filterbank extractor updated to match v2 (16ch x 49fr)."""
    def __init__(self, sample_rate=16000, n_channels=16, n_frames=49):
        self.sr = sample_rate
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.filterbank = self._build_filterbank(100, 5000)

    def _build_filterbank(self, f_low, f_high):
        n_fft = 1024 
        n_bins = n_fft // 2 + 1
        freqs = np.geomspace(f_low, f_high, self.n_channels + 2)
        bin_indices = np.floor((n_fft + 1) * freqs / self.sr).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        fb = np.zeros((self.n_channels, n_bins))
        for i in range(self.n_channels):
            left, center, right = bin_indices[i], bin_indices[i+1], bin_indices[i+2]
            if center > left: fb[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
            if right > center: fb[i, center:right + 1] = np.linspace(1, 0, right - center + 1)
        return torch.from_numpy(fb).float()

    def __call__(self, waveform):
        if waveform.shape[1] > 16000: waveform = waveform[:, :16000]
        else: waveform = F.pad(waveform, (0, 16000 - waveform.shape[1]))
        
        # 16000 samples -> 49 frames using 400 sample (25ms) windows and 320 (20ms) hop size
        frames = waveform.unfold(1, 400, 320) 
        window = torch.hamming_window(400).to(waveform.device)
        spec = torch.fft.rfft(frames * window, n=1024).abs()**2
        
        energies = torch.matmul(self.filterbank.to(waveform.device), spec.transpose(1, 2))
        log_energies = torch.log(energies + 1e-10)
        
        # Flatten to [1, 784] (16 channels x 49 frames)
        return log_energies.permute(0, 2, 1).reshape(1, -1)

class SimulatorConfig:
    XB_SIZE = 32
    DAC_MIN = -15
    DAC_MAX = 15
    ADC_MIN = -128
    ADC_MAX = 127

# ==========================================
# Log Directory Management
# ==========================================
print(f"\n[System] Log Directory: {LOG_DIR}")
should_wipe = (str(FRESH_LOGS).lower() == 'true')

if should_wipe:
    if os.path.exists(LOG_DIR):
        print(f"[System] FRESH_LOGS=True. Wiping existing data...")
        shutil.rmtree(LOG_DIR)
    else:
        print(f"[System] FRESH_LOGS=True. Directory does not exist, creating new...")
else:
    if os.path.exists(LOG_DIR):
        num_files = len(os.listdir(LOG_DIR))
        print(f"[System] FRESH_LOGS=False. Preserving {num_files} existing files.")
    else:
        print(f"[System] FRESH_LOGS=False, but directory not found. Creating new...")

os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# 2. Fault Masking Analyzer
# ==========================================
class FaultMaskingAnalyzer:
    def __init__(self):
        self.active = False
        self.mode = 'IDLE' 
        self.clean_tensors = {}
        self.divergence_active = False
        self.start_point = None
        self.end_point = None
        self.mechanism = None
        self.initial_fault_val_clean = None 
        self.initial_fault_val_faulty = None
        self.last_divergent_clean = None
        self.last_divergent_faulty = None
        self.mask_input_clean = None
        self.mask_input_faulty = None
        self.mask_output_clean = None
        self.mask_output_faulty = None

    def start_capture(self):
        self.mode = 'CAPTURE_CLEAN'
        self.clean_tensors = {}
        self.active = True
        self.reset_trace()

    def start_compare(self):
        self.mode = 'COMPARE_FAULTY'
        self.divergence_active = False
        self.start_point = None
        self.end_point = None
        self.mechanism = None
        self.active = True

    def stop(self):
        self.active = False
        self.mode = 'IDLE'
        
    def reset_trace(self):
        self.initial_fault_val_clean = None
        self.initial_fault_val_faulty = None
        self.last_divergent_clean = None
        self.last_divergent_faulty = None
        self.mask_input_clean = None
        self.mask_input_faulty = None
        self.mask_output_clean = None
        self.mask_output_faulty = None

    def log(self, tag, tensor):
        if not self.active: return
        t = tensor.detach()

        if self.mode == 'CAPTURE_CLEAN':
            self.clean_tensors[tag] = t
            
        elif self.mode == 'COMPARE_FAULTY':
            if tag not in self.clean_tensors: return 
            clean_t = self.clean_tensors[tag]
            is_equal = torch.equal(clean_t, t)
            
            if "tile_0_0_out" in tag and self.initial_fault_val_clean is None:
                self.initial_fault_val_clean = clean_t[0, 0].item()
                self.initial_fault_val_faulty = t[0, 0].item()

            if not is_equal:
                self.divergence_active = True
                if self.start_point is None: self.start_point = tag
                self.last_divergent_clean = str(clean_t.flatten().cpu().numpy().tolist()[:20])
                self.last_divergent_faulty = str(t.flatten().cpu().numpy().tolist()[:20])
                
            elif is_equal and self.divergence_active and self.end_point is None:
                self.divergence_active = False
                self.end_point = tag 
                self.mechanism = self._diagnose_masking(tag)
                self.mask_input_clean = self.last_divergent_clean
                self.mask_input_faulty = self.last_divergent_faulty
                self.mask_output_clean = str(clean_t.flatten().cpu().numpy().tolist()[:20])
                self.mask_output_faulty = str(t.flatten().cpu().numpy().tolist()[:20])

    def _diagnose_masking(self, current_tag):
        if "relu" in current_tag: return "ReLU Activation"
        if "quant" in current_tag: return "Input Quantization"
        if "pool" in current_tag: return "Spatial Pooling"
        if "bn" in current_tag: return "Batch Normalization"
        if "pre_mask" in current_tag: return "Crossbar MAC (Zero-Weight/Cancellation)"
        if "tile" in current_tag: return "Tile Accumulation"
        return f"Op: {current_tag}"

ANALYZER = FaultMaskingAnalyzer()

def parse_multiple_atpg_pairs(input_data, counts=(5, 5, 5, 16)): 
    if sum(counts) != 31: raise ValueError("The 'counts' tuple must sum to exactly 31.")
    
    def _process_single_string(string):
        pattern_str = string.strip()
        first_el_inp, first_el_wgt = 0, 0 
        
        # Determine Input for Victim (0th element)
        if "inp_0.55_inn_0.55" in pattern_str:
            first_el_inp = 0
        elif "inp_0.25_inn_0.85" in pattern_str:
            # first_el_inp = -1
            first_el_inp = 1
        elif "inp_0.85_inn_0.25" in pattern_str:
            first_el_inp = 1
            
        # 2. Strictly Determine Weight for Victim (0th element)
        if any(sub in pattern_str for sub in ["rp_20.0_rn_20.0", "rp_20_rn_20", "rp_0.008_rn_0.008", "rp_0.001_rn_0.001"]):
            first_el_wgt = 0
        elif any(sub in pattern_str for sub in ["rp_0.01_rn_20.0", "rp_0.008_rn_20.0"]):
            first_el_wgt = -1
        elif any(sub in pattern_str for sub in ["rp_20.0_rn_0.01", "rp_20.0_rn_0.008"]):
            first_el_wgt = 1
            
        # Determine Neighbors
        match = re.search(r"_neighs_([PN0]{4})", pattern_str)
        n1, n2, n3, n4 = match.group(1) if match else ('0', '0', '0', '0')
        
        val_map_inp = {'P': 1, 'N': 1, '0': 0}
        val_map_wgt = {'P': 1, 'N': -1, '0': 0}
        c1, c2, c3, c4 = counts
        
        rest_inp = ([val_map_inp[n1]] * c1 + [val_map_inp[n2]] * c2 + [val_map_inp[n3]] * c3 + [val_map_inp[n4]] * c4)
        rest_wgt = ([val_map_wgt[n1]] * c1 + [val_map_wgt[n2]] * c2 + [val_map_wgt[n3]] * c3 + [val_map_wgt[n4]] * c4)
        
        return [first_el_inp] + rest_inp, [first_el_wgt] + rest_wgt

    is_nested = isinstance(input_data[0], list) or isinstance(input_data[0], np.ndarray)
    input_data = input_data if is_nested else [input_data]
    
    master_inps, master_wgts = [], []
    for group in input_data:
        g_inp, g_wgt = [], []
        for s in group:
            i, w = _process_single_string(s)
            g_inp.append(i); g_wgt.append(w)
        master_inps.append(np.array(g_inp))
        master_wgts.append(np.array(g_wgt))
        
    if not is_nested: return master_inps[0], master_wgts[0]
    return master_inps, master_wgts

def generate_shifted_pairs(base_inps, base_wgts, enable_shifts=True):
    is_nested = isinstance(base_inps, list)
    inps_to_proc = base_inps if is_nested else [base_inps]
    wgts_to_proc = base_wgts if is_nested else [base_wgts]
    
    shifted_groups = []
    for g_idx in range(len(inps_to_proc)):
        shifted_group_pairs = []
        for v_idx in range(len(inps_to_proc[g_idx])):
            inp_v = inps_to_proc[g_idx][v_idx]
            wgt_v = wgts_to_proc[g_idx][v_idx]
            seen = set()
            shifts = []
            shift_range = range(len(inp_v)) if enable_shifts else [0]
            
            for shift_amount in shift_range:
                s_inp = tuple(np.roll(inp_v, shift_amount).tolist())
                s_wgt = tuple(np.roll(wgt_v, shift_amount).tolist())
                if (s_inp, s_wgt) not in seen:
                    seen.add((s_inp, s_wgt))
                    shifts.append((list(s_inp), list(s_wgt)))
            shifted_group_pairs.append(shifts)
        shifted_groups.append(shifted_group_pairs)
    return shifted_groups if is_nested else shifted_groups[0]

def flatten_and_check_global_pair_duplicates(jagged_shifted_array):
    seen = set()
    unique_inps, unique_wgts = [], []
    for group in jagged_shifted_array:
        for shifts_list in group:
            for inp, wgt in shifts_list:
                key = (tuple(inp), tuple(wgt))
                if key not in seen:
                    seen.add(key)
                    unique_inps.append(inp)
                    unique_wgts.append(wgt)
    return unique_inps, unique_wgts

# ==========================================
# 3. Helper Functions
# ==========================================
def save_target_data(tensor, base_fname):
    """Saves raw CSV, and a PNG representation (image or waveform plot)."""
    img_np = tensor.detach().cpu().numpy().squeeze()
    
    # Always save the CSV logging the raw values
    np.savetxt(base_fname + ".csv", img_np.flatten(), delimiter=",", fmt="%.4f")
    
    if DATASET == 'KWS':
        try:
            plt.figure(figsize=(10, 4))
            if KWS_FEATURE_TYPE == 'MFCC':
                plt.imshow(img_np, aspect='auto', origin='lower')
                plt.title("Generated MFCC Target")
                plt.colorbar()
            else:
                plt.plot(img_np.flatten(), color='blue')
                plt.title("Generated Audio Target")
                plt.ylim(INPUT_MIN, INPUT_MAX)
                plt.grid(True)
            plt.savefig(base_fname + ".png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"      -> [Warning] Could not save PNG plot: {e}")
    else:
        save_image(tensor.detach().clone(), base_fname + ".png", normalize=True, value_range=(INPUT_MIN, INPUT_MAX))

class TernaryWeightFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_weight):
        ctx.save_for_backward(input_weight)
        alpha = input_weight.abs().mean()
        delta = 0.7 * alpha
        output_weight = torch.zeros_like(input_weight)
        output_weight[input_weight > delta] = 1.0
        output_weight[input_weight < -delta] = -1.0
        return output_weight
    @staticmethod
    def backward(ctx, grad_output): return grad_output

class InputQuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.save_for_backward(inputs, alpha)
        x_scaled = inputs / alpha
        x_int = x_scaled.round().clamp(SimulatorConfig.DAC_MIN, SimulatorConfig.DAC_MAX)
        return x_int
    @staticmethod
    def backward(ctx, grad_output):
        inputs, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None

class LeakyClampFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, min_val, max_val):
        ctx.save_for_backward(inputs)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return torch.clamp(inputs, min=min_val, max=max_val)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        grad_input = grad_output.clone()
        out_of_bounds = (inputs < ctx.min_val) | (inputs > ctx.max_val)
        grad_input[out_of_bounds] *= 0.1
        return grad_input, None, None

class OutputClampFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, num_row_blocks):
        max_limit = 1905.0 * num_row_blocks
        min_limit = -1920.0 * num_row_blocks
        ctx.save_for_backward(inputs)
        ctx.min_limit = min_limit
        ctx.max_limit = max_limit
        return torch.clamp(inputs, min=min_limit, max=max_limit)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        grad_input = grad_output.clone()
        out_of_bounds = (inputs < ctx.min_limit) | (inputs > ctx.max_limit)
        grad_input[out_of_bounds] *= 0.1
        return grad_input, None

# ==========================================
# 4. Hardware Mapping
# ==========================================
class TileMap:
    def __init__(self, weight_matrix):
        self.device = weight_matrix.device
        out_ch, in_ch = weight_matrix.shape
        alpha = torch.mean(torch.abs(weight_matrix))
        delta = 0.7 * alpha
        w_int = torch.zeros_like(weight_matrix)
        w_int[weight_matrix > delta] = 1.0
        w_int[weight_matrix < -delta] = -1.0
        
        xb_size = SimulatorConfig.XB_SIZE 
        pad_in = (xb_size - (in_ch % xb_size)) % xb_size
        pad_out = (xb_size - (out_ch % xb_size)) % xb_size
        w_padded = F.pad(w_int, (0, pad_in, 0, pad_out), mode='constant', value=0)
        new_out, new_in = w_padded.shape
        num_row_blocks = new_in // xb_size
        num_col_blocks = new_out // xb_size
        w_blocked = w_padded.view(num_col_blocks, xb_size, num_row_blocks, xb_size)
        self.crossbar_array = w_blocked.permute(0, 2, 3, 1).contiguous().view(-1, xb_size, xb_size)
        self.num_row_blocks = num_row_blocks
        self.num_col_blocks = num_col_blocks
        self.pad_in = pad_in
        self.logical_shape = (out_ch, in_ch)

class RRAMLayerBase(nn.Module):
    def __init__(self, name="layer"):
        super().__init__()
        self.name = name
        self.mode = 'fp32'
        self.tile_map = None
        self.effective_tiles = None 
        self.clean_tiles = None     
        self.adc_offset = 0   
        self.spike_offset = 0
        self.last_x_int = None 
        self.active_defect_rows = {0} 
        
    def map_to_hardware(self, weight_tensor):
        self.tile_map = TileMap(weight_tensor)
        self.pristine_tiles = self.tile_map.crossbar_array.clone() # BACKUP
        self.clean_tiles = self.pristine_tiles.clone()
        self.effective_tiles = self.clean_tiles.clone()
        self.adc_offset = 0
        if hasattr(self, 'layer') and hasattr(self.layer, 'weight'):
            self.pristine_weight_data = self.layer.weight.data.clone() # BACKUP

    def restore_pristine_weights(self):
        if hasattr(self, 'pristine_tiles') and self.pristine_tiles is not None:
            self.clean_tiles = self.pristine_tiles.clone()
            self.effective_tiles = self.clean_tiles.clone()
        if hasattr(self, 'pristine_weight_data'):
            self.layer.weight.data = self.pristine_weight_data.clone()
        
    def clear_hardware_map(self): 
        self.tile_map = None
        self.effective_tiles = None
        self.clean_tiles = None
        self.last_x_int = None
        
    def reset_faults(self):
        if self.clean_tiles is not None: self.effective_tiles = self.clean_tiles.clone()
        self.adc_offset = 0
        self.spike_offset = 0
        self.active_defect_rows = {0}
        
    def apply_adc_offset(self, val): self.adc_offset = float(val)
    def apply_spike_offset(self, val): self.spike_offset = int(val)
    
    def update_fault_metadata(self, fault_mask_indices):
        if len(fault_mask_indices) > 0:
            self.active_defect_rows = set([int(loc[1]) for loc in fault_mask_indices])
    
    def get_dynamic_scale(self, x):
        max_val = x.abs().max()
        if max_val == 0: return torch.tensor(1.0).to(x.device)
        return (max_val / SimulatorConfig.DAC_MAX).detach()

    def rram_forward_matmul(self, input_int, bias_int=None):
            batch_size = input_int.size(0)
            sign = torch.sign(input_int)
            sign[sign == 0] = 1.0
            magnitude_int = torch.abs(input_int).int()
            masks = [1, 2, 4, 8] 
            
            num_row = self.tile_map.num_row_blocks
            num_col = self.tile_map.num_col_blocks
            w_grid = self.effective_tiles.view(num_col, num_row, 32, 32)
            output_accum = torch.zeros((batch_size, num_col, num_row, 32), device=input_int.device)
            
            static_fault_mask = torch.zeros((num_col, num_row, 32), dtype=torch.bool)
            if self.adc_offset != 0 and num_col > 0 and num_row > 0:
                static_fault_mask[0, 0, 0] = True 
                
            defective_locs = static_fault_mask.nonzero(as_tuple=False).cpu().numpy()
            
            if self.adc_offset != 0:
                self.update_fault_metadata(defective_locs)
            
            for bit_pos, m in enumerate(masks):
                bit = (magnitude_int & m).bool().float()
                x_bit = bit * sign
                x_padded = F.pad(x_bit, (0, self.tile_map.pad_in), mode='constant', value=0)
                x_blocked = x_padded.view(batch_size, num_row, 32)
                
                partial_analog = torch.einsum('bkr, jkrc -> bjkc', x_blocked, w_grid)
                adc_raw = partial_analog
                
                if self.adc_offset != 0 and num_col > 0:
                    fault_mask = static_fault_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).to(adc_raw.device)
                    adc_raw[fault_mask] += self.adc_offset
                    over_max_mask = adc_raw > SimulatorConfig.ADC_MAX
                    under_min_mask = adc_raw < SimulatorConfig.ADC_MIN
                    adc_raw[over_max_mask] = adc_raw[over_max_mask] - SimulatorConfig.ADC_MAX
                    adc_raw[under_min_mask] = adc_raw[under_min_mask] - SimulatorConfig.ADC_MIN
                    
                adc_out = LeakyClampFn.apply(adc_raw, SimulatorConfig.ADC_MIN, SimulatorConfig.ADC_MAX)
                shift_factor = (1 << bit_pos) 
                output_accum += adc_out * shift_factor

            if num_col > 0 and num_row > 0:
                ANALYZER.log(f"{self.name}_tile_0_0_out", output_accum[:, 0, 0, :])

            layer_output_accum = output_accum.sum(dim=2)
            layer_output_flat = layer_output_accum.view(batch_size, -1)
            final_out_channels = self.tile_map.logical_shape[0]
            output_cropped = layer_output_flat[:, :final_out_channels]
            if bias_int is not None: output_cropped += bias_int.unsqueeze(0)
            return output_cropped

# ==========================================
# 5. Layers
# ==========================================
class RRAMConv2d(RRAMLayerBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, name="conv"):
        super().__init__(name)
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False) 
        
    def forward(self, x):
        if self.mode == 'fp32': return self.layer(x)
        
        kernel_elements = self.layer.in_channels * self.layer.kernel_size[0] * self.layer.kernel_size[1]
        num_row_blocks = (kernel_elements + SimulatorConfig.XB_SIZE - 1) // SimulatorConfig.XB_SIZE
        
        w_eff = TernaryWeightFn.apply(self.layer.weight)
        scale = self.get_dynamic_scale(x)
        x_q = InputQuantizerFn.apply(x, scale)

        if self.mode == 'ternary':
            # Unfold input so shape matches RRAM mode [B*L, C_flat] for target generation & gradient flow
            x_unfold = F.unfold(x_q, kernel_size=self.layer.kernel_size, stride=self.layer.stride, padding=self.layer.padding)
            B_uf, C_flat, L_uf = x_unfold.shape
            self.last_x_int = x_unfold.transpose(1, 2).reshape(B_uf * L_uf, C_flat)
            
            with torch.no_grad():
                sign = torch.sign(x_q)
                sign[sign == 0] = 1.0
                mag_int = torch.abs(x_q).int()
                
                hw_accum = 0 
                for bit_pos, m in enumerate([1, 2, 4, 8]):
                    bit = (mag_int & m).bool().float()
                    x_bit = bit * sign
                    adc_raw = F.conv2d(x_bit, w_eff, stride=self.layer.stride, padding=self.layer.padding)
                    adc_out = torch.clamp(adc_raw, SimulatorConfig.ADC_MIN, SimulatorConfig.ADC_MAX)
                    hw_accum = hw_accum + adc_out * (1 << bit_pos)

                if self.layer.bias is not None:
                    hw_accum += self.layer.bias.view(1, -1, 1, 1)

            soft_accum = F.conv2d(x_q, w_eff, bias=self.layer.bias, stride=self.layer.stride, padding=self.layer.padding)
            out = soft_accum + (hw_accum - soft_accum).detach()
            return OutputClampFn.apply(out, num_row_blocks)
            
        if self.tile_map is None: 
            w_flat = self.layer.weight.view(self.layer.out_channels, -1)
            self.map_to_hardware(w_flat)
            
        x_int = InputQuantizerFn.apply(x, scale)
        x_unfold = F.unfold(x_int, kernel_size=self.layer.kernel_size, stride=self.layer.stride, padding=self.layer.padding)
        B, C_flat, L = x_unfold.shape
        x_unfold_flat = x_unfold.transpose(1, 2).reshape(B * L, C_flat)
        
        self.last_x_int = x_unfold_flat 
        ANALYZER.log(f"{self.name}_quant_out", x_unfold_flat)
        
        out_int_flat = self.rram_forward_matmul(x_unfold_flat, bias_int=None)
        out_int = out_int_flat.view(B, L, -1).transpose(1, 2)
        
        h_out = (x.shape[2] + 2 * self.layer.padding[0] - self.layer.kernel_size[0]) // self.layer.stride[0] + 1
        w_out = (x.shape[3] + 2 * self.layer.padding[1] - self.layer.kernel_size[1]) // self.layer.stride[1] + 1
        
        out_float = out_int.view(B, -1, h_out, w_out).float()
        if self.layer.bias is not None:
            out_float += self.layer.bias.view(1, -1, 1, 1)
            
        return OutputClampFn.apply(out_float, num_row_blocks)

class RRAMLinear(RRAMLayerBase):
    def __init__(self, in_features, out_features, name="fc", is_output=False):
        super().__init__(name)
        self.is_output = is_output
        self.layer = nn.Linear(in_features, out_features, bias=False) 
        
    def forward(self, x):
        if self.mode == 'fp32': return self.layer(x)
        
        num_row_blocks = (self.layer.in_features + SimulatorConfig.XB_SIZE - 1) // SimulatorConfig.XB_SIZE
        w_eff = TernaryWeightFn.apply(self.layer.weight)
        scale = self.get_dynamic_scale(x)
        
        if self.mode == 'ternary':
            self.last_x_pre_quant = x.detach()  # Tier 1.3 -- pre-quant input for GradOpt penalty
            x_q = InputQuantizerFn.apply(x, scale)
            self.last_x_int = x_q
            
            with torch.no_grad():
                batch_size = x_q.size(0)
                sign = torch.sign(x_q)
                sign[sign == 0] = 1.0
                mag_int = torch.abs(x_q).int()
                
                hw_accum = torch.zeros((batch_size, self.layer.out_features), device=x.device)
                
                for bit_pos, m in enumerate([1, 2, 4, 8]):
                    bit = (mag_int & m).bool().float()
                    x_bit = bit * sign
                    adc_raw = F.linear(x_bit, w_eff)
                    adc_out = torch.clamp(adc_raw, SimulatorConfig.ADC_MIN, SimulatorConfig.ADC_MAX)
                    hw_accum += adc_out * (1 << bit_pos)

                if self.layer.bias is not None: hw_accum += self.layer.bias
            
            soft_accum = F.linear(x_q, w_eff, self.layer.bias)
            out = soft_accum + (hw_accum - soft_accum).detach()
            return OutputClampFn.apply(out, num_row_blocks)
            
        if self.tile_map is None: self.map_to_hardware(self.layer.weight)
        self.last_x_pre_quant = x.detach()  # Tier 1.3 -- pre-quant input for GradOpt penalty
        x_int = InputQuantizerFn.apply(x, scale)
        self.last_x_int = x_int 
        ANALYZER.log(f"{self.name}_quant_out", x_int)
        
        out_int = self.rram_forward_matmul(x_int, bias_int=None)
        
        out_float = out_int.float()
        if self.layer.bias is not None: out_float += self.layer.bias
        return OutputClampFn.apply(out_float, num_row_blocks)

# ==========================================
# 5b. LIF Readout Engine & Full Pipeline
# ==========================================

class LIFReadoutEngine:
    """Layer-by-layer LIF/IF readout through RRAM crossbar tiles.
    """

    def __init__(self, rram_layers, thresholds, time_steps=63, leak=1.0):
        self.layers      = rram_layers
        self.thresholds  = thresholds          # per-layer; None for the output layer
        self.T           = time_steps
        self.leak        = leak

    # ---- 4-bit DAC quantization for inter-layer transfer --------------------
    @staticmethod
    def _quantize_to_4bit(values):
        """Map any numeric vector to the 4-bit signed-magnitude DAC range."""
        v = np.asarray(values, dtype=np.float64)
        mx = float(np.max(np.abs(v)))
        if mx <= 1e-12:
            return np.zeros_like(v, dtype=np.int32)
        scale = float(SimulatorConfig.DAC_MAX) / mx
        q = np.rint(v * scale).astype(np.int32)
        return np.clip(q, SimulatorConfig.DAC_MIN, SimulatorConfig.DAC_MAX)

    # ---- 4-bit DAC bit-serial RRAM MAC (mirrors ADC pipeline) ---------------
    @staticmethod
    def _rram_4bit_mac(layer, x_int):
        """4-bit DAC bit-serial MAC + 8-bit ADC clamp + fault offset.
        x_int: signed integer tensor in [DAC_MIN, DAC_MAX]. Stores layer.last_x_int."""
        with torch.no_grad():
            batch = x_int.size(0)
            sign = torch.sign(x_int.float())
            sign[sign == 0] = 1.0
            magnitude = torch.abs(x_int).int()

            num_row = layer.tile_map.num_row_blocks
            num_col = layer.tile_map.num_col_blocks
            w_grid = layer.effective_tiles.view(num_col, num_row, 32, 32)
            out_accum = torch.zeros((batch, num_col, num_row, 32),
                                    device=x_int.device)

            # Cache integer input for downstream target match checking
            layer.last_x_int = x_int.detach().clone()

            for bit_pos, m_mask in enumerate([1, 2, 4, 8]):
                bit = (magnitude & m_mask).bool().float()
                x_bit = bit * sign
                x_padded = F.pad(x_bit, (0, layer.tile_map.pad_in),
                                 mode='constant', value=0)
                x_blocked = x_padded.view(batch, num_row, 32)

                partial = torch.einsum('bkr, jkrc -> bjkc', x_blocked, w_grid)
                adc_out = torch.clamp(partial,
                                      SimulatorConfig.ADC_MIN,
                                      SimulatorConfig.ADC_MAX)

                if layer.adc_offset != 0 and num_col > 0 and num_row > 0:
                    fm = torch.zeros((num_col, num_row, 32),
                                     dtype=torch.bool, device=x_int.device)
                    fm[0, 0, 0] = True
                    fm = fm.unsqueeze(0).expand(batch, -1, -1, -1)
                    adc_out[fm] += layer.adc_offset
                    over  = adc_out > SimulatorConfig.ADC_MAX
                    under = adc_out < SimulatorConfig.ADC_MIN
                    adc_out[over]  = adc_out[over]  - SimulatorConfig.ADC_MAX
                    adc_out[under] = adc_out[under] - SimulatorConfig.ADC_MIN
                    adc_out = torch.clamp(adc_out,
                                          SimulatorConfig.ADC_MIN,
                                          SimulatorConfig.ADC_MAX)

                shift = (1 << bit_pos)
                out_accum += adc_out * shift

            layer_out = out_accum.sum(dim=2).view(batch, -1)
            return layer_out[:, :layer.tile_map.logical_shape[0]]

    # ---- single hidden-layer LIF simulation (current-driven) ----------------
    def _simulate_hidden(self, layer, input_values, threshold):
        """4-bit DAC MAC produces input current; T-step constant-current LIF."""
        T = self.T
        n_out = layer.tile_map.logical_shape[0]
        # Quantize input to 4-bit DAC
        x_int_np = self._quantize_to_4bit(input_values)
        x_int_t = torch.tensor(x_int_np, dtype=torch.int32,
                               device=DEVICE).unsqueeze(0)
        # Single 4-bit MAC -> input current per neuron
        I_t = self._rram_4bit_mac(layer, x_int_t)
        I = I_t.squeeze(0).cpu().numpy().astype(np.float64)
        # T-step constant-current LIF
        membrane    = np.zeros(n_out, dtype=np.float64)
        spike_count = np.zeros(n_out, dtype=np.float64)
        for _ in range(T):
            if self.leak < 1.0:
                membrane *= self.leak
            membrane += I
            fired = membrane >= threshold
            spike_count[fired] += 1.0
            membrane[fired] -= threshold
        return spike_count

    # ---- output layer accumulation (4-bit DAC MAC, no firing) ---------------
    def _simulate_output(self, layer, input_values):
        """4-bit DAC MAC produces output potentials directly (no LIF firing)."""
        x_int_np = self._quantize_to_4bit(input_values)
        x_int_t = torch.tensor(x_int_np, dtype=torch.int32,
                               device=DEVICE).unsqueeze(0)
        I_t = self._rram_4bit_mac(layer, x_int_t)
        return I_t.squeeze(0).cpu().numpy().astype(np.float64)

    # ---- full inference with per-layer spike-count trace --------------------
    def infer_with_trace(self, features):
        """Returns (pred, [spike_counts_per_hidden_layer], output_potentials)."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        n_layers = len(self.layers)
        n_hidden = n_layers - 1
        vals = features.astype(np.float64)
        traces = []

        for li in range(n_layers):
            layer = self.layers[li]
            if li < n_hidden:
                sc = self._simulate_hidden(layer, vals, self.thresholds[li])
                traces.append(sc.copy())             # trace BEFORE fault
                if layer.spike_offset != 0:
                    sc[0] = np.clip(sc[0] + layer.spike_offset, 0, self.T)
                vals = sc
            else:
                pot = self._simulate_output(layer, vals)
                return int(np.argmax(pot)), traces, pot
        return 0, traces, np.zeros(1)

    def infer_single(self, features):
        pred, _, _ = self.infer_with_trace(features)
        return pred

    def infer_batch(self, features_batch, verbose_every=500):
        if isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.cpu().numpy()
        n = len(features_batch)
        preds = np.empty(n, dtype=int)
        for i in range(n):
            preds[i] = self.infer_single(features_batch[i])
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"      LIF inference: {i+1}/{n}")
        return preds

# ---- shared helpers ---------------------------------------------------------

def _extract_rram_layers(model):
    layers = []
    for _, m in model.named_modules():
        if isinstance(m, RRAMLinear):
            assert m.tile_map is not None, f"{m.name}: tiles not mapped"
            layers.append(m)
    return layers

def _build_th_list(rram_layers, hidden_thresholds):
    return list(hidden_thresholds) + [None] * (len(rram_layers) - len(hidden_thresholds))

def _to_lif_range(data, T):
    """Quantize input to 4-bit signed magnitude DAC range [DAC_MIN, DAC_MAX]."""
    flat = data.view(data.size(0), -1)
    # Linear map [INPUT_MIN, INPUT_MAX] -> [DAC_MIN, DAC_MAX], then round to int
    f01 = (flat - INPUT_MIN) / (INPUT_MAX - INPUT_MIN + 1e-10)
    dac_span = SimulatorConfig.DAC_MAX - SimulatorConfig.DAC_MIN
    return (f01 * dac_span + SimulatorConfig.DAC_MIN).round().clamp(
        SimulatorConfig.DAC_MIN, SimulatorConfig.DAC_MAX)

def _binarize(spike_counts, thresh):
    return (spike_counts >= thresh).astype(int)

def _lif_4bit_match(layer, target):
    """Mirror of ADC bit-cycle check: at layer.last_x_int, return best
    32-row match across the 4 DAC bit-cycles. Used for LIF target validation."""
    if not hasattr(layer, 'last_x_int') or layer.last_x_int is None:
        return 0
    raw = layer.last_x_int.detach()
    if raw.dim() == 2:
        raw = raw[0]
    raw = raw[:32].int()
    if raw.numel() < 32:
        return 0
    mag = torch.abs(raw).int()
    sgn = torch.sign(raw).int()
    sgn[sgn == 0] = 1
    tgt_t = torch.tensor(np.array(target[:32], dtype=int),
                         dtype=torch.int32, device=raw.device)
    best_mc = 0
    for mask in [1, 2, 4, 8]:
        eff = ((mag & mask).bool().int() * sgn)
        mc = int((eff == tgt_t).sum().item())
        if mc > best_mc: best_mc = mc
    return best_mc

def _lif_clean_fn(s):
    return s.replace(" ","_").replace("|","").replace(":","_").replace("&","_")\
            .replace("(","").replace(")","").replace(",","_")


# ---- LIF evaluation --------------------------------------------------------

def evaluate_lif(model, test_loader, thresholds, time_steps=255, leak=1.0,
                 spike_faults=None, n_max=2000, desc="LIF"):
    """LIF evaluation. Returns accuracy on up to n_max samples from test_loader.
    """
    model.eval()
    rram_layers = _extract_rram_layers(model)
    th_list = _build_th_list(rram_layers, thresholds)

    # Backup pre-existing layer spike_offset values, then apply requested faults
    fault_backups = {m.name: m.spike_offset for m in rram_layers}
    try:
        if spike_faults:
            for ln, off in spike_faults.items():
                for m in rram_layers:
                    if m.name == ln:
                        m.spike_offset = int(off)

        engine = LIFReadoutEngine(rram_layers, th_list, time_steps, leak)
        preds_l, labs_l = [], []
        total = 0
        print(f"   [{desc}] Up to {n_max} samples (T={time_steps}, leak={leak}) ...")
        with torch.no_grad():
            for data, target in test_loader:
                if total >= n_max: break
                data = data.to(DEVICE)
                feats = _to_lif_range(data, time_steps)
                for i in range(feats.size(0)):
                    if total >= n_max: break
                    preds_l.append(engine.infer_single(feats[i]))
                    labs_l.append(target[i].item())
                    total += 1
        preds, labs = np.array(preds_l), np.array(labs_l)
        acc = 100.0 * (preds == labs).mean()
        print(f"   [{desc}] Accuracy: {acc:.2f}% ({(preds==labs).sum()}/{len(labs)})")
        return acc
    finally:
        # Restore pre-existing spike_offset values regardless of how we exit
        for m in rram_layers:
            m.spike_offset = fault_backups[m.name]

# ---- LIF target generation (Det -> Dataset -> GradOpt) ---------------------

def check_and_generate_all_16_cases_lif(model, train_loader, test_loader):
    """Spike-count-domain target generation.
    """
    global TARGET_CASE_REGISTRY
    print("\n[Phase 3.5-LIF] Target Generation (Det -> Dataset -> GradOpt, spike-count domain)")
    model.eval()

    rram_layers = _extract_rram_layers(model)
    th_list     = _build_th_list(rram_layers, LIF_THRESHOLDS)
    n_hidden    = len(rram_layers) - 1
    bth         = LIF_SPIKE_BINARIZE_THRESH
    T           = LIF_TIME_STEPS

    # Cover all RRAM layers (incl. output) -- mirrors ADC orchestrator scope.
    target_layer_names = [rram_layers[i].name for i in range(len(rram_layers))]
    for ln in target_layer_names:
        for ti in range(len(INPUT_TARGETS)):
            TARGET_CASE_REGISTRY[(ln, ti)] = "MISS"

    hits    = {(ln, ti): False for ln in target_layer_names for ti in range(len(INPUT_TARGETS))}
    closest = {(ln, ti): {'count': -1, 'img': None}
               for ln in target_layer_names for ti in range(len(INPUT_TARGETS))}

    template_img, _ = next(iter(test_loader))
    template_img = template_img[0:1].to(DEVICE)

    # === STEP 1  Deterministic inversion ===
    print("   [STEP 1-LIF] Deterministic pseudo-inverse ...")
    for li, ln in enumerate(target_layer_names):
        for ti, t_in in enumerate(INPUT_TARGETS):
            if hits[(ln, ti)]: continue
            # --- INJECT UNIQUE TARGET WEIGHTS (parity with ADC pipeline) ---
            model.restore_all_pristine_weights()
            model.inject_target_weights(ln, WEIGHT_TARGETS[ti])
            ok, img, mc = _det_inv_lif(model, rram_layers, th_list, li, t_in,
                                       template_img, T, bth)
            if ok:
                hits[(ln, ti)] = True
                TARGET_CASE_REGISTRY[(ln, ti)] = "PASS (Det-LIF)"
                string_id = REVERSE_MAPPING.get(ti, "Unknown")
                save_target_data(img, os.path.join(LOG_DIR, f"lif_PASS_Det_{ln}_String{string_id}_Shift{ti}"))
            elif img is not None and mc > closest[(ln, ti)]['count']:
                closest[(ln, ti)] = {'count': mc, 'img': img.cpu()}

    # === STEP 2  Dataset search ===
    print("\n   [STEP 2-LIF] Dataset spike-count scan ...")
    with torch.no_grad():
        for li, ln in enumerate(target_layer_names):
            for ti, t_in in enumerate(INPUT_TARGETS):
                if hits[(ln, ti)]: continue

                # --- INJECT UNIQUE TARGET WEIGHTS (parity with ADC pipeline) ---
                model.restore_all_pristine_weights()
                model.inject_target_weights(ln, WEIGHT_TARGETS[ti])
                engine0 = LIFReadoutEngine(rram_layers, th_list, T, LIF_LEAK)

                tgt = np.array(t_in[:32], dtype=int)
                done = False
                for tag, loader in [('Train', train_loader), ('Test', test_loader)]:
                    if done: break
                    for _, (data, _) in enumerate(loader):
                        if done: break
                        data = data.to(DEVICE)
                        feats = _to_lif_range(data, T)
                        for ii in range(feats.size(0)):
                            _, scs, _ = engine0.infer_with_trace(feats[ii])
                            # PASS if any of the 4 DAC bit-cycles matches the target
                            mc = _lif_4bit_match(rram_layers[li], t_in)
                            if mc == 32:
                                hits[(ln, ti)] = True
                                TARGET_CASE_REGISTRY[(ln, ti)] = "PASS (Dataset-LIF)"
                                string_id = REVERSE_MAPPING.get(ti, "Unknown")
                                save_target_data(data[ii], os.path.join(LOG_DIR, f"lif_PASS_Dataset_{ln}_String{string_id}_Shift{ti}"))
                                done = True
                                break
                            elif mc > closest[(ln, ti)]['count']:
                                closest[(ln, ti)] = {'count': mc,
                                                     'img': data[ii:ii+1].cpu()}

    # === STEP 3  Gradient optimisation ===
    print("\n   [STEP 3-LIF] Gradient optimisation for remaining MISSES ...")
    for li, ln in enumerate(target_layer_names):
        for ti, t_in in enumerate(INPUT_TARGETS):
            if hits[(ln, ti)]: continue
            seed = closest[(ln, ti)]['img']
            cnt  = closest[(ln, ti)]['count']
            if seed is None:
                TARGET_CASE_REGISTRY[(ln, ti)] = "SKIPPED (No Seed)"
                continue
            # --- INJECT UNIQUE TARGET WEIGHTS (parity with ADC pipeline) ---
            model.restore_all_pristine_weights()
            model.inject_target_weights(ln, WEIGHT_TARGETS[ti])
            _grad_opt_lif(model, rram_layers, th_list, li, t_in, ti, seed, cnt, T, bth)

    # Restore pristine weights so that subsequent fault-injection phases
    model.restore_all_pristine_weights()
    print("   [System] Restored pristine hardware/software weights for normal operations.")

    # === summary table ===
    print("\n" + "=" * 80)
    print("LIF TARGET GENERATION SUMMARY")
    print("=" * 80)
    hdr = " | ".join([f"Tgt {i}".center(15) for i in range(len(INPUT_TARGETS))])
    print(f"{'Layer':<10} | {hdr}")
    print("-" * (13 + 18 * len(INPUT_TARGETS)))
    for ln in sorted(set(k[0] for k in TARGET_CASE_REGISTRY)):
        row = f"{ln:<10}"
        for ti in range(len(INPUT_TARGETS)):
            row += f" | {TARGET_CASE_REGISTRY.get((ln,ti),'N/A'):^15}"
        print(row)


def _det_inv_lif(model, rram_layers, th_list, target_li, target_in,
                 template_img, T, bth):
    """Deterministic pseudo-inverse for LIF target generation.
    """
    if target_li < 0 or target_li >= len(rram_layers):
        return False, None, 0
    target_layer_name = rram_layers[target_li].name

    # Walk forward to find target_layer + collect preceding layers (mirrors ADC version)
    target_layer = None
    layers_before = []
    for name, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear, nn.BatchNorm1d, nn.BatchNorm2d,
                          nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            if getattr(m, 'name', '') == target_layer_name:
                target_layer = m; break
            layers_before.append(m)
    if target_layer is None:
        return False, None, 0

    # Trigger a forward through the LIF engine to populate last_x_int on target_layer
    eng_probe = LIFReadoutEngine(rram_layers, th_list, T, LIF_LEAK)
    template_dev = template_img.to(DEVICE)
    feats_probe = _to_lif_range(template_dev, T)
    _ = eng_probe.infer_single(feats_probe.squeeze(0))
    if not hasattr(target_layer, 'last_x_int') or target_layer.last_x_int is None:
        return False, None, 0
    full_dim = target_layer.last_x_int.shape[1]
    expected_max = target_layer.last_x_int.abs().max().item()
    target_inflation_factor = expected_max if expected_max > 0 else 15.0

    # CNN guard: only PureLinear/SimpleMLP-style flat-input architectures supported
    is_cnn = (MODEL_ARCH not in ['PureLinear', 'SimpleMLP'])
    if is_cnn:
        return False, None, 0

    target_rows = getattr(target_layer, 'active_defect_rows', {0})
    row_idx = list(target_rows)[0]
    start = row_idx * 32
    end = min(start + 32, full_dim)
    active_len = end - start
    if active_len <= 0: return False, None, 0

    full_target_vector_template = target_layer.last_x_int[0].detach().clone().float()
    is_first_layer = (len(layers_before) == 0)
    if not is_first_layer and -1 in target_in: return False, None, 0

    target_bin = torch.tensor(target_in[:active_len], dtype=torch.int32, device=DEVICE)

    # Trace forward shapes (and natural scale at the target layer) for inversion.
    layer_in_shapes, layer_out_shapes = {}, {}
    x_trace = template_dev.clone()
    for m_trace in layers_before:
        layer_in_shapes[m_trace] = x_trace.shape
        if isinstance(m_trace, RRAMLinear) and x_trace.dim() > 2:
            x_trace = x_trace.view(x_trace.size(0), -1)
        x_trace = m_trace(x_trace)
        layer_out_shapes[m_trace] = x_trace.shape

    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in layers_before)
    natural_max = x_trace.abs().max().item() if x_trace.numel() > 0 and x_trace.abs().max().item() > 0 else 1.0

    # Multi-scale sweep mirrors attempt_deterministic_inversion's scale_sweep.
    if is_first_layer:
        scale_sweep = [15.0 * target_inflation_factor]
    else:
        scale_sweep = [
            15.0 * target_inflation_factor,
            0.5  * natural_max,
            0.25 * natural_max,
            1.0  * natural_max,
        ]
        if has_bn:
            scale_sweep.append(0.005 * natural_max)

    pinv_cache = {}

    def _do_inversion(scale_factor):
        """Pseudo-inverse at one scale; LIF mirror of ADC _do_inversion."""
        full_target_vector = full_target_vector_template.clone()
        mapped = [val * scale_factor for val in target_in[:active_len]]
        full_target_vector[start:end] = torch.tensor(mapped, dtype=torch.float32, device=DEVICE)
        current_target = full_target_vector.unsqueeze(0)
        for m in reversed(layers_before):
            if isinstance(m, nn.BatchNorm1d):
                mean, var, gamma, beta, eps = m.running_mean, m.running_var, m.weight, m.bias, m.eps
                std = torch.sqrt(var + eps)
                current_target = ((current_target - beta) * std / gamma) + mean
            elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
                out_shape = layer_out_shapes[m]
                temp_spatial = current_target.view(out_shape)
                scale_fac = m.kernel_size
                if isinstance(scale_fac, tuple):
                    inverted_spatial = F.interpolate(temp_spatial, size=layer_in_shapes[m][2:], mode='nearest')
                else:
                    inverted_spatial = F.interpolate(temp_spatial, scale_factor=scale_fac, mode='nearest')
                current_target = inverted_spatial.view(current_target.size(0), -1)
            elif isinstance(m, RRAMLinear):
                if m not in pinv_cache:
                    W_eff = TernaryWeightFn.apply(m.layer.weight)
                    pinv_cache[m] = torch.linalg.pinv(W_eff.t())
                pinv_W = pinv_cache[m]
                num_row_blocks = (m.layer.in_features + SimulatorConfig.XB_SIZE - 1) // SimulatorConfig.XB_SIZE
                adc_effective_max = SimulatorConfig.ADC_MAX * 15.0 * num_row_blocks
                adc_effective_min = SimulatorConfig.ADC_MIN * 15.0 * num_row_blocks
                current_target = torch.clamp(current_target, min=adc_effective_min, max=adc_effective_max)
                if m.layer.bias is not None: current_target = current_target - m.layer.bias.unsqueeze(0)
                current_target = current_target @ pinv_W
                if m != layers_before[0]:
                    current_target = torch.clamp(current_target, min=0.0)
                    max_val = current_target.abs().max()
                    if max_val > 0:
                        sc = max_val / 15.0
                        current_target = ((current_target / sc) + 1e-5).round() * sc
        img_raw = current_target.view_as(template_dev)
        img_max_abs = img_raw.abs().max()
        if is_first_layer:
            if img_max_abs > 0:
                return torch.clamp(img_raw / img_max_abs, INPUT_MIN, INPUT_MAX)
            return torch.clamp(img_raw, INPUT_MIN, INPUT_MAX)
        # Deep-layer hard-clamp; BN affine is sensitive to global rescaling.
        if img_max_abs > 50.0 * INPUT_MAX:
            p99 = torch.quantile(img_raw.abs().flatten(), 0.99).item()
            if p99 > INPUT_MAX:
                img_raw = img_raw * (INPUT_MAX / max(p99, 1e-6))
        return torch.clamp(img_raw, INPUT_MIN, INPUT_MAX)

    def _score(img):
        """LIF-engine 4-bit-cycle best match at target_layer (mirror of ADC _score)."""
        eng_s = LIFReadoutEngine(rram_layers, th_list, T, LIF_LEAK)
        feats_s = _to_lif_range(img, T)
        if feats_s.dim() > 1: feats_s = feats_s.squeeze(0)
        _ = eng_s.infer_single(feats_s)
        return _lif_4bit_match(target_layer, target_in)

    best_match_count = -1
    best_img = None
    with torch.no_grad():
        for sf in scale_sweep:
            img_try = _do_inversion(sf)
            score = _score(img_try)
            if score > best_match_count:
                best_match_count = score
                best_img = img_try.clone()
                if best_match_count == 32:
                    break

    if best_match_count == 32:
        return True, best_img, 32
    return False, best_img if best_img is not None else template_dev.clone(), best_match_count


def _lif_match_via_engine(model, rram_layers, th_list, target_layer, img,
                          target_in_tensor_full, start, T):
    """LIF analog of ADC _rram_match_batched: returns per-image best match
    across the 4 DAC bit-cycles as a tensor of shape [B]."""
    eng = LIFReadoutEngine(rram_layers, th_list, T, LIF_LEAK)
    if img.dim() == 1: img = img.unsqueeze(0)
    batch_size = img.size(0)
    fv_batch = _to_lif_range(img, T)
    scores = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    for b in range(batch_size):
        _ = eng.infer_single(fv_batch[b])
        if target_layer.last_x_int is None: continue
        full_dim = target_layer.last_x_int.shape[1]
        end_b = min(start + 32, full_dim)
        active_len_b = end_b - start
        if active_len_b <= 0: continue
        ri = target_layer.last_x_int[:, start:end_b].int()
        rm = torch.abs(ri); sign_ri = torch.sign(ri); sign_ri[sign_ri == 0] = 1
        best_mc = 0
        for mask in [1, 2, 4, 8]:
            rb = ((rm & mask).bool().int() * sign_ri).int()
            mc = (rb == target_in_tensor_full[:active_len_b].unsqueeze(0)).sum(dim=1).max().item()
            if mc > best_mc: best_mc = mc
        scores[b] = best_mc
    return scores


def _grad_opt_lif(model, rram_layers, th_list, target_li, target_in, target_idx,
                  seed_img, seed_count, T, bth):
    """LIF GradOpt: multi-bit-cycle variant.
    """
    global TARGET_CASE_REGISTRY
    target_layer_name = rram_layers[target_li].name
    seed_img = seed_img.to(DEVICE)
 
    target_layer = None
    for _, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)) and m.name == target_layer_name:
            target_layer = m
            break
    if target_layer is None:
        TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = "SKIPPED"
        return
 
    target_rows = getattr(target_layer, 'active_defect_rows', {0})
    row_idx = list(target_rows)[0]
    start = row_idx * 32
    target_in_tensor_full = torch.tensor(target_in, dtype=torch.int32, device=DEVICE)

    # Tier 1.1 -- ReLU-fed layers cannot accept negative inputs.
    RELU_FED_LAYERS = {"fc2", "fc3", "fc4"}
    _active_len_check = min(32, len(target_in))
    if target_layer_name in RELU_FED_LAYERS and any(v < 0 for v in target_in[:_active_len_check]):
        TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = "INFEASIBLE_RELU"
        print(f"      -> LIF GradOpt: SKIP (negative target at ReLU-fed layer)")
        return

    # Tier 3.1 -- pre-fc4 / pre-fc2 leaky-ReLU overrides activate when targeting that layer.
    _USE_PRE_FC4_LEAKY = (target_layer_name == "fc4")
    _pre_fc4_slope = [0.0]
    _USE_PRE_FC2_LEAKY = (target_layer_name == "fc2")
    _pre_fc2_slope = [0.0]

    original_relu = F.relu

    def _rram_match_batched(img):
        """LIF batched matcher; scores under hard ReLU then restores leaky overrides."""
        F.relu = original_relu
        _saved4 = GRADOPT_PRE_LAYER_LEAKY.pop("fc4", None)
        _saved2 = GRADOPT_PRE_LAYER_LEAKY.pop("fc2", None)
        try:
            return _lif_match_via_engine(model, rram_layers, th_list, target_layer,
                                         img, target_in_tensor_full, start, T)
        finally:
            if _saved4 is not None: GRADOPT_PRE_LAYER_LEAKY["fc4"] = _saved4
            if _saved2 is not None: GRADOPT_PRE_LAYER_LEAKY["fc2"] = _saved2

    def _rram_match_single(img):
        """Single-image standalone validator (LIF mirror of ADC _rram_match_single)."""
        if img.dim() == seed_img.dim() and img.size(0) != 1:
            img = img[:1]
        scores = _rram_match_batched(img)
        return int(scores.max().item())

    leaky_slope = [0.15]
    F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=leaky_slope[0], inplace=inplace)
    model.set_mode('ternary')
 
    try:
        model(seed_img)
        full_dim = target_layer.last_x_int.shape[1]
        end = min(start + 32, full_dim)
        active_len = end - start
        if active_len <= 0:
            return
 
        ones_count = sum(1 for v in target_in[:active_len] if v != 0)
        if ones_count > 20:
            batch_size, max_noise, num_of_steps = 64, 1.5, 20000
        elif ones_count > 12:
            batch_size, max_noise, num_of_steps = 32, 1.0, 15000
        else:
            batch_size, max_noise, num_of_steps = 16, 0.5, 10000
 
        rep_shape  = [batch_size] + [1] * (seed_img.dim() - 1)
        view_shape = [-1]         + [1] * (seed_img.dim() - 1)
        img_batch = seed_img.repeat(*rep_shape)
        noise_scales = torch.linspace(0.0, max_noise, batch_size, device=DEVICE).view(*view_shape)
        img_batch = torch.clamp(img_batch + torch.randn_like(img_batch) * noise_scales,
                                INPUT_MIN, INPUT_MAX)
        img_batch.requires_grad_(True)
 
        # ---- LR halved (was 0.08) ---------------------------------------------  CHANGED
        optimizer = torch.optim.AdamW([img_batch], lr=0.04, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2)
 
        target_tensor_for_mask = torch.tensor(target_in[:active_len], device=DEVICE)
        target_bin             = torch.tensor(target_in[:active_len], dtype=torch.int32, device=DEVICE)
 
        best_img = seed_img.detach().clone()
        oob_mask = torch.ones(full_dim, dtype=torch.bool, device=DEVICE)
        oob_mask[start:end] = False
        global_best_match = seed_count
 
        rram_best_match = seed_count
        rram_best_img   = seed_img.detach().clone()
        RRAM_EVAL_INTERVAL = 250
 
        for step in range(num_of_steps):
            # ---- Annealing schedules ------------------------------------------
            progress = min(step / (num_of_steps * 0.8), 1.0)
            leaky_slope[0]       = 0.15 * (1.0 - progress)
            current_temp         = 0.5 + 4.5 * progress      # 0.5 -> 5.0
            current_soft_min_tau = 1.0 - 0.9 * progress      # 1.0 -> 0.1   NEW

            # Tier 3.1 -- pre-fc4 / pre-fc2 leaky slopes decay to 0.
            if _USE_PRE_FC4_LEAKY:
                _pre_fc4_slope[0] = 0.30 * (1.0 - progress)
                GRADOPT_PRE_LAYER_LEAKY["fc4"] = _pre_fc4_slope[0]
            if _USE_PRE_FC2_LEAKY:
                _pre_fc2_slope[0] = 0.30 * (1.0 - progress)
                GRADOPT_PRE_LAYER_LEAKY["fc2"] = _pre_fc2_slope[0]

            optimizer.zero_grad()
            model(img_batch)
 
            x_int = target_layer.last_x_int
            full_dim = x_int.shape[1]
            end = min(start + 32, full_dim)
            active_len = end - start
            if x_int is None or active_len <= 0:
                break
 
            segment_target_full    = x_int[:, start:end]
            target_tensor_for_mask = torch.tensor(target_in[:active_len], device=DEVICE)
            target_bin             = torch.tensor(target_in[:active_len], dtype=torch.int32, device=DEVICE)
 
            # ---- Spatial isolation: best-of-4-cycles per CNN patch  CHANGED ---
            B_size = img_batch.size(0)
            N_size = x_int.size(0)
            L_size = N_size // B_size
 
            if L_size > 1:
                seg_reshaped = segment_target_full.view(B_size, L_size, -1)
                with torch.no_grad():
                    ri = seg_reshaped.int()
                    rm = torch.abs(ri)
                    sign_ri = torch.sign(ri)
                    best_per_patch = torch.zeros(B_size, L_size,
                                                 device=DEVICE, dtype=torch.int32)
                    for mask in (1, 2, 4, 8):
                        rb = ((rm & mask).bool().float() * sign_ri).int()
                        mc = (rb == target_bin.unsqueeze(0).unsqueeze(0)).sum(dim=2)
                        best_per_patch = torch.maximum(best_per_patch, mc.to(torch.int32))
                    best_patch_indices = best_per_patch.argmax(dim=1)
 
                batch_indices = torch.arange(B_size, device=DEVICE)
                segment_target = seg_reshaped[batch_indices, best_patch_indices, :]
            else:
                segment_target = segment_target_full
 
            # ---- Multi-bit-cycle matching loss  CHANGED -----------------------
            loss_bit_match = multibit_loss(
                segment_target, target_tensor_for_mask,
                temperature=current_temp, gamma=2.0,
                soft_min_tau=current_soft_min_tau,
            )
 
            loss_oob = F.smooth_l1_loss(
                x_int[:, oob_mask], torch.zeros_like(x_int[:, oob_mask]))
 
            loss_variance = torch.tensor(0.0, device=DEVICE)
            _mask_ones  = (target_tensor_for_mask == 1) | (target_tensor_for_mask == -1)
            _mask_zeros = (target_tensor_for_mask == 0)
            if _mask_ones.sum() > 1:
                loss_variance = loss_variance + torch.var(segment_target[:, _mask_ones], dim=1).mean()
            if _mask_zeros.sum() > 1:
                loss_variance = loss_variance + torch.var(segment_target[:, _mask_zeros], dim=1).mean()
 
            loss_silence = torch.mean(torch.relu(torch.abs(x_int) - 80.0))
 
            img_reg = torch.norm(img_batch)

            # Tier 1.3 -- penalize pre-quant lanes that overshoot the integer-1 window (FC4 only).
            loss_pre_excess = torch.tensor(0.0, device=DEVICE)
            if target_layer_name == "fc4":
                _pre = getattr(target_layer, 'last_x_pre_quant', None)
                if _pre is not None and _pre.dim() == 2:
                    pre_window = _pre[:, start:end]
                    _scale_now = target_layer.get_dynamic_scale(_pre)
                    _scale_val = float(_scale_now.max().item()) if _scale_now.numel() > 0 else (1.0 / 15.0)
                    excess = F.relu(pre_window.abs() - 1.5 * _scale_val)
                    loss_pre_excess = excess.pow(2).mean()

            # Tier 1.4 -- global lane-sum constraint for high-density targets (FC4 only).
            loss_sum = torch.tensor(0.0, device=DEVICE)
            if target_layer_name == "fc4":
                _nz_target = int((target_tensor_for_mask != 0).sum().item())
                if _nz_target >= 24:
                    _target_sum = float(target_tensor_for_mask.float().sum().item())
                    loss_sum = (segment_target.float().sum(dim=1) - _target_sum).pow(2).mean()

            loss = (loss_bit_match
                    + 0.05 * loss_oob
                    + 0.01 * loss_variance
                    + 0.01 * loss_silence
                    + 1e-5 * img_reg
                    + 0.05 * loss_pre_excess
                    + 0.02 * loss_sum)
 
            # ---- Global-best tracking across 4 cycles  CHANGED ----------------
            with torch.no_grad():
                raw_int  = segment_target.int()
                mag_int  = torch.abs(raw_int)
                sign_int = torch.sign(raw_int)
 
                best_matches = torch.zeros(segment_target.size(0), device=DEVICE)
                for mask in (1, 2, 4, 8):
                    eff_bit = ((mag_int & mask).bool().float() * sign_int).int()
                    m = (eff_bit == target_bin.unsqueeze(0)).sum(dim=1).float()
                    best_matches = torch.maximum(best_matches, m)
                max_match_in_batch = best_matches.max().item()
 
                if max_match_in_batch > global_best_match:
                    global_best_match = int(max_match_in_batch)
                    best_patch_idx = int(best_matches.argmax().item())
                    imgs_per_batch = img_batch.size(0)
                    patches_per_img = segment_target.size(0) // imgs_per_batch
                    best_img_idx = best_patch_idx // patches_per_img if patches_per_img > 0 else best_patch_idx
                    best_img = img_batch[best_img_idx:best_img_idx + 1].detach().clone()
 
            # ---- Periodic LIF-engine validation / early stopping --------------
            if (step + 1) % RRAM_EVAL_INTERVAL == 0 or max_match_in_batch >= 30:
                rms = _rram_match_batched(img_batch.detach())
                # Standalone-validate top-K candidates; batched scores aren't reproducible.
                K_validate = min(4, img_batch.size(0))
                top_vals, top_idx = torch.topk(rms, K_validate)
                for k in range(K_validate):
                    cand = img_batch[top_idx[k]:top_idx[k]+1].detach().clone()
                    true_score = _rram_match_single(cand)
                    if true_score > rram_best_match:
                        rram_best_match = true_score
                        rram_best_img = cand
                    if true_score == 32:
                        break
                if rram_best_match == 32:
                    F.relu = original_relu
                    break
                F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=leaky_slope[0], inplace=inplace)
                model.set_mode('ternary')
 
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                img_batch.data = img_batch.data.clamp(INPUT_MIN, INPUT_MAX)
 
        # ---- Step E. Stochastic polish (single-image-validated, mirrors ADC). ----
        if rram_best_match < 32:
            polish_seed = rram_best_img if rram_best_match >= global_best_match else best_img
            seed_true = _rram_match_single(polish_seed)
            if seed_true > rram_best_match:
                rram_best_match = seed_true
                rram_best_img = polish_seed.clone()

            BATCH_POLISH = 250
            for polish_round in range(5000 // BATCH_POLISH):
                if rram_best_match == 32: break
                decay = max(0.02, 0.3 * (1.0 - (polish_round * BATCH_POLISH) / 5000))
                # Build noisy batch (arch-agnostic repeat) and top-K validate.
                rep = [BATCH_POLISH] + [1] * (rram_best_img.dim() - 1)
                perturbed_batch = rram_best_img.repeat(*rep)
                perturbed_batch = perturbed_batch + torch.randn_like(perturbed_batch) * decay
                perturbed_batch = torch.clamp(perturbed_batch, INPUT_MIN, INPUT_MAX)

                rms = _rram_match_batched(perturbed_batch)
                K_validate = 4
                top_vals, top_idx = torch.topk(rms, min(K_validate, rms.size(0)))
                for k in range(top_vals.size(0)):
                    cand = perturbed_batch[top_idx[k]:top_idx[k]+1].detach().clone()
                    true_score = _rram_match_single(cand)
                    if true_score > rram_best_match:
                        rram_best_match = true_score
                        rram_best_img = cand
                    if true_score == 32:
                        break
 
        # ---- Step F. Per-bit emergency refinement (unchanged) -----------------
        if 30 <= rram_best_match < 32:
            F.relu = original_relu
            _ = _rram_match_batched(rram_best_img)
            with torch.no_grad():
                ri = target_layer.last_x_int[:, start:end].int()
                rm_abs = torch.abs(ri)
                rb = ((rm_abs & 8).bool().float() * torch.sign(ri)).int()
                bit_correct = (rb == target_bin.unsqueeze(0)).float()
                best_patch_idx = bit_correct.sum(dim=1).argmax()
                failing_mask = (bit_correct[best_patch_idx] == 0)
                num_failing = int(failing_mask.sum().item())
 
            if 0 < num_failing <= 4:
                bit_weights = torch.ones(32, device=DEVICE)
                bit_weights[failing_mask] = 10.0
 
                refine_slope = [0.05]
                F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=refine_slope[0], inplace=inplace)
                model.set_mode('ternary')
 
                img_refine = rram_best_img.clone()
                refine_batch = 8
                rep_r  = [refine_batch] + [1] * (img_refine.dim() - 1)
                view_r = [-1]           + [1] * (img_refine.dim() - 1)
                img_r = img_refine.repeat(*rep_r)
                noise_r = torch.randn_like(img_r) * torch.linspace(
                    0.0, 0.15, refine_batch, device=DEVICE).view(*view_r)
                img_r = torch.clamp(img_r + noise_r, INPUT_MIN, INPUT_MAX)
                img_r.requires_grad_(True)
 
                opt_r   = torch.optim.Adam([img_r], lr=0.005)
                sched_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=5000)
 
                for r_step in range(5000):
                    opt_r.zero_grad()
                    model(img_r)
                    x_r = target_layer.last_x_int
                    if x_r is None or end > x_r.shape[1]:
                        break
                    seg_r = x_r[:, start:end]
 
                    abs_x_r = torch.abs(seg_r)
                    p_on = torch.sigmoid(5.0 * (abs_x_r - 7.5))
 
                    m_active = (target_tensor_for_mask == 1) | (target_tensor_for_mask == -1)
                    m_zero   = (target_tensor_for_mask == 0)
                    eps = 1e-6
                    loss_r = torch.tensor(0.0, device=DEVICE)
 
                    if m_active.any():
                        p_a = p_on[:, m_active]
                        w_a = bit_weights[m_active].unsqueeze(0)
                        loss_r = loss_r - torch.mean(w_a * torch.log(p_a + eps))
                        signs_w = target_tensor_for_mask[m_active].float() \
                                                                  .unsqueeze(0) \
                                                                  .expand_as(seg_r[:, m_active])
                        hinge = torch.clamp(8.5 - signs_w * seg_r[:, m_active], min=0.0)
                        loss_r = loss_r + torch.mean(w_a * hinge)
                    if m_zero.any():
                        p_z = p_on[:, m_zero]
                        w_z = bit_weights[m_zero].unsqueeze(0)
                        loss_r = loss_r - torch.mean(w_z * torch.log(1.0 - p_z + eps))
 
                    loss_r = loss_r + 0.05 * F.smooth_l1_loss(
                        x_r[:, oob_mask], torch.zeros_like(x_r[:, oob_mask]))
 
                    loss_r.backward()
                    opt_r.step()
                    sched_r.step()
                    with torch.no_grad():
                        img_r.data = img_r.data.clamp(INPUT_MIN, INPUT_MAX)
 
                    if (r_step + 1) % 200 == 0:
                        F.relu = original_relu
                        with torch.no_grad():
                            for bi in range(img_r.size(0)):
                                cand_r = img_r[bi:bi + 1].detach()
                                true_score = _rram_match_single(cand_r)
                                if true_score > rram_best_match:
                                    rram_best_match = true_score
                                    rram_best_img = cand_r.clone()
                                if true_score == 32:
                                    break
                        if rram_best_match == 32:
                            break
                        F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=refine_slope[0], inplace=inplace)
                        model.set_mode('ternary')
 
        best_img = rram_best_img
 
    finally:
        F.relu = original_relu
        GRADOPT_PRE_LAYER_LEAKY.pop("fc4", None)  # Tier 3.1 -- clear leaky override.
        GRADOPT_PRE_LAYER_LEAKY.pop("fc2", None)
 
    # ---- Final verification + Tier 4.1 mismatched-lane diagnostics. ----------
    _final_active = min(32, len(target_in))
    target_bin_final = torch.tensor(target_in[:_final_active], dtype=torch.int32, device=DEVICE)

    final_outcome = "FAIL"
    best_synth_match = 0
    best_eff_bit = None
    eng_final = LIFReadoutEngine(rram_layers, th_list, T, LIF_LEAK)
    feats_final = _to_lif_range(best_img, T)
    if feats_final.dim() > 1: feats_final = feats_final.squeeze(0)
    with torch.no_grad():
        _ = eng_final.infer_single(feats_final)
        if target_layer.last_x_int is not None:
            full_dim_f = target_layer.last_x_int.shape[1]
            end_f = min(start + 32, full_dim_f)
            raw_int_all = target_layer.last_x_int[:, start:end_f].int()
            sign = torch.sign(raw_int_all)
            mag_int = torch.abs(raw_int_all).int()
            for mask in (1, 2, 4, 8):
                eff_bit = ((mag_int & mask).bool().float() * sign).int()
                matches = (eff_bit == target_bin_final[:end_f-start].unsqueeze(0)).sum(dim=1)
                best_patch_match = matches.max().item()
                if best_patch_match > best_synth_match:
                    best_synth_match = best_patch_match
                    best_eff_bit = eff_bit[matches.argmax().item()]
                if best_patch_match == 32:
                    final_outcome = "PASS"

    if int(rram_best_match) != int(best_synth_match):
        print(f"      -> [WARN] rram_best_match ({rram_best_match}) != final ({best_synth_match}); "
              f"check for model state mutation between validation and final eval.")

    if final_outcome == "FAIL":
        print(f"      -> LIF GradOpt: FAIL - Best Match: {best_synth_match}/32 "
              f"(ternary-opt: {global_best_match}/32, lif: {rram_best_match}/32)")
        if best_eff_bit is not None:
            mismatches = [(int(i), int(target_bin_final[i].item()), int(best_eff_bit[i].item()))
                          for i in (best_eff_bit != target_bin_final[:best_eff_bit.shape[0]]).nonzero().flatten().tolist()]
            print(f"         Mismatched lanes (idx, want, got): {mismatches}")
    else:
        print(f"      -> LIF GradOpt: PASS")

    TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = f"{final_outcome} (GradOpt-LIF)"
    string_id = REVERSE_MAPPING.get(target_idx, "Unknown")
    save_target_data(
        best_img,
        os.path.join(LOG_DIR, f"lif_{final_outcome}_GradOpt_{target_layer_name}_String{string_id}_Shift{target_idx}")
    )

# ---- LIF fault masking analysis --------------------------------------------
def run_lif_masking_analysis(model, test_loader, experiment_desc, spike_faults,
                              n_max=None):
    """Compare clean vs faulty LIF spike-count traces sample-by-sample.
    """
    test_dataset_size = len(test_loader.dataset)
    if n_max is None: n_max = ANALYSIS_IMAGE_LIMIT
    dynamic_limit = min(test_dataset_size, 10000, n_max)
    tag   = _lif_clean_fn(experiment_desc)
    fname = os.path.join(LOG_DIR, f"lif_masking_{tag}.csv")
    print(f"\n   [LIF MASKING] {dynamic_limit} samples from Test Set -> {fname}")
    model.eval()

    rram_layers = _extract_rram_layers(model)
    th_list = _build_th_list(rram_layers, LIF_THRESHOLDS)
    n_hidden = len(rram_layers) - 1
    T = LIF_TIME_STEPS

    # Single engine, fault state lives on the layers (parity with ADC path).
    engine = LIFReadoutEngine(rram_layers, th_list, T, LIF_LEAK)

    # Backup any pre-existing layer-level spike offsets
    fault_backups = {m.name: m.spike_offset for m in rram_layers}

    proc, masked_n, div_n = 0, 0, 0
    try:
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            hdr = ["Idx","Clean_Pred","Faulty_Pred","Masked",
                   "Diverge_Layer","Converge_Layer","Mechanism"]
            hdr += [f"L{i}_delta_mean" for i in range(n_hidden)]
            hdr += ["Out_Clean","Out_Faulty"]
            w.writerow(hdr)

            with torch.no_grad():
                for data, target in test_loader:
                    if proc >= n_max: break
                    data = data.to(DEVICE)
                    feats = _to_lif_range(data, T)
                    for ii in range(feats.size(0)):
                        if proc >= n_max: break
                        fv = feats[ii]

                        # Clean pass: ensure all layer spike_offsets are zero
                        for m in rram_layers: m.spike_offset = 0
                        pc, sc_c, oc = engine.infer_with_trace(fv)

                        # Faulty pass: apply requested spike faults
                        if spike_faults:
                            for ln, off in spike_faults.items():
                                for m in rram_layers:
                                    if m.name == ln:
                                        m.spike_offset = int(off)
                        pf, sc_f, of_ = engine.infer_with_trace(fv)

                        # Reset for next sample
                        for m in rram_layers: m.spike_offset = 0

                        deltas, first_d, last_cv = [], None, None
                        for li in range(min(n_hidden, len(sc_c), len(sc_f))):
                            d = np.abs(sc_c[li] - sc_f[li]).mean()
                            deltas.append(d)
                            if d > 0.01 and first_d is None: first_d = li
                            if d < 0.01 and first_d is not None and last_cv is None:
                                last_cv = li
                        msk = (pc == pf)
                        if first_d is not None: div_n += 1
                        if msk and first_d is not None: masked_n += 1

                        if not msk:          mech = "None"
                        elif last_cv is not None: mech = f"Threshold Quant (L{last_cv})"
                        elif first_d is None:     mech = "No Divergence"
                        else:                     mech = "Output Tolerance"

                        row = [proc, pc, pf, msk, first_d, last_cv, mech]
                        row += [f"{d:.4f}" for d in deltas]
                        row += [str(oc[:5].tolist()), str(of_[:5].tolist())]
                        w.writerow(row)
                        proc += 1
        print(f"      [LIF MASKING] {masked_n}/{div_n} masked ({proc} samples)")
    finally:
        # Restore pre-existing layer spike_offset values regardless of how we exit
        for m in rram_layers:
            m.spike_offset = fault_backups[m.name]


class ModelWrapper(nn.Module):
    def set_mode(self, mode):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)):
                m.mode = mode
                if mode == 'rram' and m.tile_map is None:
                    if isinstance(m, RRAMConv2d):
                        w = m.layer.weight.view(m.layer.out_channels, -1)
                    else: w = m.layer.weight
                    m.map_to_hardware(w)
    def reset_hardware_map(self):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)): m.clear_hardware_map()
    def reset_all_faults(self):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)): m.reset_faults()
    def configure_faults(self, fault_type, values, layer_name):
        found = False
        for name, m in self.named_modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)):
                if m.name == layer_name:
                    if fault_type == 'offset': m.apply_adc_offset(values[0])
                    elif fault_type == 'spike_offset': m.apply_spike_offset(values[0])
                    found = True
        if not found: print(f"[Warning] Layer {layer_name} not found!")
    def restore_all_pristine_weights(self):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)) and hasattr(m, 'restore_pristine_weights'):
                m.restore_pristine_weights()

    def inject_target_weights(self, layer_name, weight_target_list):
        for name, m in self.named_modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)) and m.name == layer_name:
                active_len = min(32, len(weight_target_list))
                
                # 1. Hardware Injection (RRAM mode)
                if m.effective_tiles is not None and m.tile_map is not None:
                    num_col = m.tile_map.num_col_blocks
                    num_row = m.tile_map.num_row_blocks
                    if num_col > 0 and num_row > 0:
                        w_grid = m.effective_tiles.view(num_col, num_row, 32, 32)
                        w_tensor = torch.tensor(weight_target_list[:active_len], dtype=w_grid.dtype, device=w_grid.device)
                        w_grid[0, 0, :active_len, 0] = w_tensor # Column 0, Tile 0
                        m.effective_tiles = w_grid.view(-1, 32, 32)
                        m.clean_tiles = m.effective_tiles.clone() # Prevent fault resets from overwriting this
                        
                # 2. Software Injection (Ternary / FP32 modes)
                if hasattr(m, 'layer') and hasattr(m.layer, 'weight'):
                    with torch.no_grad():
                        if isinstance(m.layer, nn.Linear):
                            # Cap software injection to the actual input features
                            sw_active_len = min(active_len, m.layer.in_features)
                            w_tensor_sw = torch.tensor(weight_target_list[:sw_active_len], dtype=m.layer.weight.dtype, device=m.layer.weight.device)
                            m.layer.weight[0, :sw_active_len] = w_tensor_sw
                            
                        elif isinstance(m.layer, nn.Conv2d):
                            flat_w = m.layer.weight.view(m.layer.out_channels, -1)
                            # Cap software injection to the flattened kernel size (e.g., 25 for 5x5)
                            sw_active_len = min(active_len, flat_w.shape[1])
                            w_tensor_sw = torch.tensor(weight_target_list[:sw_active_len], dtype=m.layer.weight.dtype, device=m.layer.weight.device)
                            flat_w[0, :sw_active_len] = w_tensor_sw
                            m.layer.weight.data = flat_w.view_as(m.layer.weight)

class LeNet5(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.conv1 = RRAMConv2d(1, 6, 5, name="conv1")   
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = RRAMConv2d(6, 16, 5, name="conv2")  
        self.fc1 = RRAMLinear(400, 120, name="fc1")
        self.fc2 = RRAMLinear(120, 84, name="fc2")
        self.fc3 = RRAMLinear(84, 10, name="fc3", is_output=True)
    def forward(self, x):
        x = self.conv1(x)
        ANALYZER.log("conv1_pre_mask", x) 
        x = F.relu(x); ANALYZER.log("conv1_relu", x) 
        x = self.pool(x); ANALYZER.log("conv1_pool1", x) 
        x = self.conv2(x)
        ANALYZER.log("conv2_pre_mask", x)
        x = F.relu(x); ANALYZER.log("conv2_relu", x)
        x = self.pool(x); ANALYZER.log("conv2_pool1", x) 
        x = x.reshape(-1, 400)
        x = self.fc1(x)
        ANALYZER.log("fc1_pre_mask", x)
        x = F.relu(x); ANALYZER.log("fc1_relu", x)
        x = self.fc2(x)
        ANALYZER.log("fc2_pre_mask", x)
        x = F.relu(x); ANALYZER.log("fc2_relu", x)
        x = self.fc3(x)
        ANALYZER.log("final_logits", x)
        return x

class SimpleMLP(ModelWrapper):
    def __init__(self):
        super().__init__()
        layers = []
        self.norms = nn.ModuleList() if USE_BATCH_NORM else None
        input_dim = 784
        hidden_dim = 700
        for i in range(MLP_NUM_LAYERS - 1):
            layer_name = f"fc{i+1}"
            layers.append(RRAMLinear(input_dim, hidden_dim, name=layer_name))
            if USE_BATCH_NORM: self.norms.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        out_layer_name = f"fc{MLP_NUM_LAYERS}"
        layers.append(RRAMLinear(hidden_dim, 10, name=out_layer_name, is_output=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                # 1. Catch Zero-Weight Cancellations
                ANALYZER.log(f"{layer.name}_pre_mask", x) 
                
                if USE_BATCH_NORM and self.norms: 
                    x = self.norms[i](x)
                    # 2. Catch BatchNorm Masking (NEW)
                    ANALYZER.log(f"{layer.name}_bn", x) 
                    
                x = F.relu(x)
                # 3. Catch ReLU Masking
                ANALYZER.log(f"{layer.name}_relu", x) 
                
        ANALYZER.log("final_logits", x)
        return x
    
# Tier 3.1 -- per-layer leaky-ReLU override for the activation feeding the named layer.
# Empty dict by default (no effect on training/eval). GradOpt populates it for fc4 only.
GRADOPT_PRE_LAYER_LEAKY = {}

class PureLinearMLP(ModelWrapper):
    def __init__(self, input_dim=784, num_classes=10, hidden_dim=128):
        super().__init__()
        self.fc1 = RRAMLinear(input_dim, 256, name="fc1")
        self.bn1 = nn.BatchNorm1d(256) if USE_BATCH_NORM else nn.Identity()
        
        self.fc2 = RRAMLinear(256, hidden_dim, name="fc2")
        self.bn2 = nn.BatchNorm1d(hidden_dim) if USE_BATCH_NORM else nn.Identity()
        
        self.fc3 = RRAMLinear(hidden_dim, hidden_dim, name="fc3")
        self.bn3 = nn.BatchNorm1d(hidden_dim) if USE_BATCH_NORM else nn.Identity()
        
        self.fc4 = RRAMLinear(hidden_dim, num_classes, name="fc4", is_output=True)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        ANALYZER.log("fc1_pre_mask", x) # Catches Zero-Weight / MAC cancellations
        x = self.bn1(x)
        ANALYZER.log("fc1_bn", x)       # <--- NEW: Catches BatchNorm masking
        # Tier 3.1 -- if GradOpt is targeting fc2, use leaky-ReLU here (annealed to 0).
        _fc2_slope = GRADOPT_PRE_LAYER_LEAKY.get("fc2", None)
        if _fc2_slope is not None and _fc2_slope > 0.0:
            x = F.leaky_relu(x, negative_slope=_fc2_slope)
        else:
            x = F.relu(x)
        ANALYZER.log("fc1_relu", x)
        
        x = self.fc2(x)
        ANALYZER.log("fc2_pre_mask", x)
        x = self.bn2(x)
        ANALYZER.log("fc2_bn", x)       # <--- NEW
        x = F.relu(x); ANALYZER.log("fc2_relu", x)
        
        x = self.fc3(x)
        ANALYZER.log("fc3_pre_mask", x)
        x = self.bn3(x)
        ANALYZER.log("fc3_bn", x)       # <--- NEW
        # Tier 3.1 -- if GradOpt is targeting fc4, use leaky-ReLU here (annealed to 0).
        _fc4_slope = GRADOPT_PRE_LAYER_LEAKY.get("fc4", None)
        if _fc4_slope is not None and _fc4_slope > 0.0:
            x = F.leaky_relu(x, negative_slope=_fc4_slope)
        else:
            x = F.relu(x)
        ANALYZER.log("fc3_relu", x)
        
        x = self.fc4(x)
        ANALYZER.log("final_logits", x)
        return x
    
class TinySNS_BinaryCNN(ModelWrapper):
    def __init__(self, num_classes=2):
        super().__init__()
        # Input is reshaped to (Batch, 1 Channel, 16 Height, 16 Width)
        self.conv1 = RRAMConv2d(1, 8, kernel_size=3, stride=1, padding=1, name="conv1")
        self.bn1 = nn.BatchNorm2d(8) if USE_BATCH_NORM else nn.Identity()
        self.pool1 = nn.AvgPool2d(2, 2) # Reduces 16x16 to 8x8
        
        self.conv2 = RRAMConv2d(8, 16, kernel_size=3, stride=1, padding=1, name="conv2")
        self.bn2 = nn.BatchNorm2d(16) if USE_BATCH_NORM else nn.Identity()
        self.pool2 = nn.AvgPool2d(2, 2) # Reduces 8x8 to 4x4
        
        # Flattened dimension: 16 channels * 4 height * 4 width = 256
        self.fc1 = RRAMLinear(256, 64, name="fc1") 
        self.bn3 = nn.BatchNorm1d(64) if USE_BATCH_NORM else nn.Identity()
        
        self.fc2 = RRAMLinear(64, num_classes, name="fc2", is_output=True)

    def forward(self, x):
        # 1. Reshape the flat (B, 256) KWS features into a 16x16 "image"
        x = x.view(x.size(0), 1, 16, 16)
        
        x = self.conv1(x)
        ANALYZER.log("conv1_pre_mask", x)
        x = self.bn1(x)
        x = F.relu(x); ANALYZER.log("conv1_relu", x)
        x = self.pool1(x); ANALYZER.log("conv1_pool1", x)
        
        x = self.conv2(x)
        ANALYZER.log("conv2_pre_mask", x)
        x = self.bn2(x)
        x = F.relu(x); ANALYZER.log("conv2_relu", x)
        x = self.pool2(x); ANALYZER.log("conv2_pool2", x)
        
        # 2. Flatten for the linear layers
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        ANALYZER.log("fc1_pre_mask", x)
        x = self.bn3(x)
        x = F.relu(x); ANALYZER.log("fc1_relu", x)
        
        x = self.fc2(x)
        ANALYZER.log("final_logits", x)
        return x

class RawAudioCNNKWS(ModelWrapper):
    def __init__(self, num_classes=12): # <-- Add param
        super().__init__()
        self.conv1 = RRAMConv2d(1, 16, kernel_size=(1, 80), stride=(1, 16), padding=(0, 32), name="conv1")
        self.pool1 = nn.AvgPool2d((1, 4), stride=(1, 4))
        self.conv2 = RRAMConv2d(16, 32, kernel_size=(1, 10), stride=(1, 5), padding=(0, 2), name="conv2")
        self.pool2 = nn.AvgPool2d((1, 2), stride=(1, 2))
        
        self.fc1 = RRAMLinear(768, 128, name="fc1")
        self.fc2 = RRAMLinear(128, num_classes, name="fc2", is_output=True) # <-- Update

    def forward(self, x):
        x = x.unsqueeze(2) 
        x = self.conv1(x)
        ANALYZER.log("conv1_pre_mask", x)
        x = F.relu(x); ANALYZER.log("conv1_relu", x)
        x = self.pool1(x); ANALYZER.log("conv1_pool1", x)
        
        x = self.conv2(x)
        ANALYZER.log("conv2_pre_mask", x)
        x = F.relu(x); ANALYZER.log("conv2_relu", x)
        x = self.pool2(x); ANALYZER.log("conv2_pool2", x)
        
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        ANALYZER.log("fc1_pre_mask", x)
        x = F.relu(x); ANALYZER.log("fc1_relu", x)
        
        x = self.fc2(x)
        ANALYZER.log("final_logits", x)
        return x

class MFCC_CNNKWS(ModelWrapper):
    def __init__(self, num_classes=12): # <-- Add param
        super().__init__()
        self.conv1 = RRAMConv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1, name="conv1")
        self.bn1 = nn.BatchNorm2d(16) if USE_BATCH_NORM else nn.Identity()
        self.pool1 = nn.AvgPool2d((2, 2))
        
        self.conv2 = RRAMConv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1, name="conv2")
        self.bn2 = nn.BatchNorm2d(32) if USE_BATCH_NORM else nn.Identity()
        self.pool2 = nn.AvgPool2d((2, 2))
        
        self.fc1 = RRAMLinear(1600, 128, name="fc1") 
        self.bn3 = nn.BatchNorm1d(128) if USE_BATCH_NORM else nn.Identity()
        self.fc2 = RRAMLinear(128, num_classes, name="fc2", is_output=True) # <-- Update

    def forward(self, x):
        x = self.conv1(x)
        ANALYZER.log("conv1_pre_mask", x)
        x = self.bn1(x)
        x = F.relu(x); ANALYZER.log("conv1_relu", x)
        x = self.pool1(x); ANALYZER.log("conv1_pool1", x)
        
        x = self.conv2(x)
        ANALYZER.log("conv2_pre_mask", x)
        x = self.bn2(x)
        x = F.relu(x); ANALYZER.log("conv2_relu", x)
        x = self.pool2(x); ANALYZER.log("conv2_pool2", x)
        
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        ANALYZER.log("fc1_pre_mask", x)
        x = self.bn3(x)
        x = F.relu(x); ANALYZER.log("fc1_relu", x)
        
        x = self.fc2(x)
        ANALYZER.log("final_logits", x)
        return x

def log_mapped_weights(model):
    print(f"\n   [WEIGHT MAPPING LOG] Saving pristine RRAM 32x32 tiles to CSV...")
    for name, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)):
            if m.tile_map is None: continue
            tiles = m.tile_map.crossbar_array.cpu().detach().numpy()
            num_col_blocks = m.tile_map.num_col_blocks
            num_row_blocks = m.tile_map.num_row_blocks
            
            csv_path = os.path.join(LOG_DIR, f"{m.name}_mapped_weights.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"Layer: {m.name}"])
                tile_idx = 0
                for c in range(num_col_blocks):
                    for r in range(num_row_blocks):
                        writer.writerow([])
                        writer.writerow([f"Tile (Col_Block: {c}, Row_Block: {r})"])
                        tile_weights = tiles[tile_idx]
                        
                        zero_rows = np.where(~tile_weights.any(axis=1))[0].tolist()
                        zero_cols = np.where(~tile_weights.any(axis=0))[0].tolist()
                        
                        writer.writerow(["All-Zero Rows (Indices):", zero_rows if zero_rows else "None"])
                        writer.writerow(["All-Zero Cols (Indices):", zero_cols if zero_cols else "None"])
                        writer.writerow(["Weights (32x32):"])
                        for row_w in tile_weights: writer.writerow(row_w.astype(int).tolist())
                        tile_idx += 1
                        
class GSCDataset(torch.utils.data.Dataset):
    """
    Speech Commands dataset with proper 12-class handling
    """
    def __init__(self, data_path, target_keyword="yes", split='train', num_classes=2):
        self.data_path = data_path
        self.num_classes = num_classes
        self.target_keyword = target_keyword
        self.split = split

        # Standard 10 keywords for 12-way classification
        self.std_keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.word_to_idx = {word: i+2 for i, word in enumerate(self.std_keywords)}

        test_files = set(open(os.path.join(data_path, "testing_list.txt")).read().splitlines())
        val_files = set(open(os.path.join(data_path, "validation_list.txt")).read().splitlines())

        self.file_paths = []     # str path
        self.file_labels = []    # int class
        self.file_starts = []    # start sample for cropping (0 for normal 1s files; varies for noise)

        # First pass: collect keyword and unknown files
        for folder in sorted(os.listdir(data_path)):
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path) or folder.startswith("_"):
                continue

            for file_name in os.listdir(folder_path):
                if not file_name.endswith('.wav'): continue
                rel_path = f"{folder}/{file_name}"

                is_test = rel_path in test_files
                is_val = rel_path in val_files
                is_train = not is_test and not is_val

                if (split == 'test' and is_test) or (split == 'val' and is_val) or (split == 'train' and is_train):
                    if num_classes == 2:
                        label = 1 if folder == self.target_keyword else 0
                    else:
                        label = self.word_to_idx.get(folder, 1)  # 1 = unknown

                    self.file_paths.append(os.path.join(data_path, rel_path))
                    self.file_labels.append(label)
                    self.file_starts.append(0)

        # ------------------------------------------------------------
        # 12-class rebalancing of the "unknown" class (only in train)
        # ------------------------------------------------------------
        if num_classes == 12 and split == 'train':
            rng = np.random.RandomState(42)
            kw_counts = sum(1 for l in self.file_labels if l >= 2)
            target_unk = max(1, kw_counts // 10)  # match average per-keyword count
            unk_idx = [i for i, l in enumerate(self.file_labels) if l == 1]
            if len(unk_idx) > target_unk:
                keep = set(rng.choice(unk_idx, size=target_unk, replace=False).tolist())
                drop = set(unk_idx) - keep
                self.file_paths  = [p for i, p in enumerate(self.file_paths)  if i not in drop]
                self.file_labels = [l for i, l in enumerate(self.file_labels) if i not in drop]
                self.file_starts = [s for i, s in enumerate(self.file_starts) if i not in drop]
                print(f"   -> [GSC] Subsampled 'unknown' from {len(unk_idx)} to {target_unk} for class balance.")

        # ------------------------------------------------------------
        # Silence (idx 0): slice each ~60s background_noise file into
        # many 1-second chunks so we get thousands of silence samples.
        # ------------------------------------------------------------
        if num_classes == 12 and split in ('train', 'val', 'test'):
            bg_path = os.path.join(data_path, "_background_noise_")
            if os.path.exists(bg_path):
                # Match average per-keyword count for train; smaller for val/test
                per_class_target = max(1, kw_counts // 10) if split == 'train' else 200
                bg_files = [f for f in sorted(os.listdir(bg_path)) if f.endswith('.wav')]
                if len(bg_files) > 0:
                    chunks_per_file = max(1, per_class_target // len(bg_files))
                    sr = 16000
                    rng2 = np.random.RandomState(0 if split == 'train' else 1)
                    for fname in bg_files:
                        full_path = os.path.join(bg_path, fname)
                        try:
                            with wave.open(full_path, 'rb') as wf:
                                n_frames = wf.getnframes()
                        except Exception:
                            continue
                        max_start = max(0, n_frames - sr)
                        if max_start == 0:
                            self.file_paths.append(full_path)
                            self.file_labels.append(0)
                            self.file_starts.append(0)
                        else:
                            for _ in range(chunks_per_file):
                                start = int(rng2.randint(0, max_start + 1))
                                self.file_paths.append(full_path)
                                self.file_labels.append(0)
                                self.file_starts.append(start)
                    print(f"   -> [GSC] Added {chunks_per_file*len(bg_files)} silence chunks "
                          f"({chunks_per_file}/file x {len(bg_files)} files) for split={split}.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Return (path, start_sample, label) collate uses start_sample to crop
        return self.file_paths[idx], self.file_starts[idx], self.file_labels[idx]
    
class KWSCollate:
    # Empirical bounds for log-mel of waveforms in [-1,1]
    LOG_MIN = -15.0
    LOG_MAX =   5.0

    def __init__(self, feature_type='TINYSNS'):
        self.feature_type = feature_type
        self.extractor = TinySNSFeatureExtractor() if feature_type == 'TINYSNS' else None

    def _load_wave(self, wav_path, start_sample):
        with wave.open(wav_path, 'rb') as wav_file:
            n_frames = wav_file.getnframes()
            # Seek to start_sample for noise chunks; ignore for normal 1s files
            if start_sample > 0 and start_sample < n_frames:
                wav_file.setpos(start_sample)
                want = min(16000, n_frames - start_sample)
                frames = wav_file.readframes(want)
            else:
                frames = wav_file.readframes(-1)
        # int16 -> float32 in [-1, 1] for stable log-mel range
        waveform_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(waveform_np).unsqueeze(0)

    def __call__(self, batch):
        tensors, targets = [], []
        for item in batch:
            # Backwards compatible: accept (path, label) or (path, start, label)
            if len(item) == 3:
                wav_path, start_sample, label_idx = item
            else:
                wav_path, label_idx = item
                start_sample = 0

            waveform = self._load_wave(wav_path, start_sample)

            if self.feature_type == 'TINYSNS':
                feat = self.extractor(waveform)  # [1, 256] log-mel features

                # Stable fixed-range mapping (NOT per-sample min-max).
                # This preserves absolute-energy info, so silence != speech.
                feat = feat.clamp_(min=self.LOG_MIN, max=self.LOG_MAX)
                feat = 2.0 * (feat - self.LOG_MIN) / (self.LOG_MAX - self.LOG_MIN) - 1.0

                tensors.append(feat.squeeze(0))

            targets.append(torch.tensor(label_idx, dtype=torch.long))

        return torch.stack(tensors).unsqueeze(1), torch.stack(targets)  # [B, 1, 256]
    
def check_and_generate_all_16_cases(model, train_loader, test_loader):
    print("\n[Phase 3.5] Target Generation Pipeline (Det -> MNIST -> GradOpt)")
    model.eval()
    
    target_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)) and m.tile_map is not None:
            target_layers.append(m.name)
            for t_idx in range(len(INPUT_TARGETS)):
                TARGET_CASE_REGISTRY[(m.name, t_idx)] = "MISS"

    hits = {(layer_name, t_idx): False for layer_name in target_layers for t_idx in range(len(INPUT_TARGETS))}
    closest_matches = {(layer_name, t_idx): {'count': -1, 'img': None} for layer_name in target_layers for t_idx in range(len(INPUT_TARGETS))}
    
    # Grab a single template image for shaping the Deterministic Inversion
    template_img, _ = next(iter(test_loader))
    template_img = template_img[0:1].to(DEVICE) 

    # ==============================================================
    # STEP 1: DETERMINISTIC INVERSION (Math First)
    # ==============================================================
    print("   [STEP 1] Running Deterministic Inversion...")
    for layer_name in target_layers:
        for t_idx, t_in_list in enumerate(INPUT_TARGETS):
            
            # --- INJECT UNIQUE TARGET WEIGHTS ---
            model.restore_all_pristine_weights()
            model.inject_target_weights(layer_name, WEIGHT_TARGETS[t_idx])
            
            success, det_img, match_count = attempt_deterministic_inversion(model, layer_name, t_in_list, template_img)
            
            if success:
                hits[(layer_name, t_idx)] = True
                TARGET_CASE_REGISTRY[(layer_name, t_idx)] = "PASS (Det)"
                string_id = REVERSE_MAPPING.get(t_idx, "Unknown")
                base_fname = os.path.join(LOG_DIR, f"item_PASS_Det_{layer_name}_String{string_id}_Shift{t_idx}")
                save_target_data(det_img, base_fname)
            else:
                # Save the deterministic image as a baseline seed if it got ANY bits right
                if match_count > closest_matches[(layer_name, t_idx)]['count']:
                    closest_matches[(layer_name, t_idx)]['count'] = match_count
                    closest_matches[(layer_name, t_idx)]['img'] = det_img.cpu() if det_img is not None else template_img.cpu()

    # ==============================================================
    # STEP 2: DATASET SEARCH (The Net)
    # ==============================================================
    print("\n   [STEP 2] Scanning Dataset for remaining MISSES...")
    model.set_mode('rram') 
    loaders = [('Train', train_loader), ('Test', test_loader)]
    
    with torch.no_grad():
        for layer_name in target_layers:
            for t_idx, t_in_list in enumerate(INPUT_TARGETS):
                if hits[(layer_name, t_idx)]: continue 
                
                # --- INJECT UNIQUE TARGET WEIGHTS ---
                model.restore_all_pristine_weights()
                model.inject_target_weights(layer_name, WEIGHT_TARGETS[t_idx])
                
                for loader_name, loader in loaders:
                    if hits[(layer_name, t_idx)]: break
                    for batch_idx, (data, target) in enumerate(loader):
                        data = data.to(DEVICE)
                        model(data) 
                        
                        m = dict(model.named_modules())[layer_name]
                        if hasattr(m, 'last_x_int') and m.last_x_int is not None:
                            full_dim = m.last_x_int.shape[1]
                            active_len = min(32, full_dim)
                            if active_len <= 0: continue 
                            
                            x_int_chunk = m.last_x_int[:, :active_len] 
                            B, N = data.size(0), m.last_x_int.size(0)
                            L = N // B 
                            
                            t_bin = torch.tensor(t_in_list[:active_len], device=DEVICE, dtype=torch.float32)
                            raw_int = x_int_chunk.int()
                            sign = torch.sign(raw_int)
                            mag = torch.abs(raw_int).int()
                            
                            found_in_batch = False
                            for mask in [1, 2, 4, 8]:
                                bit = (mag & mask).bool().float()
                                eff_bit = bit * sign 
                                row_matches = torch.all(eff_bit == t_bin, dim=1)
                                
                                if row_matches.any():
                                    found_in_batch = True
                                    match_idx = torch.where(row_matches)[0][0].item()
                                    img_idx = match_idx // L
                                    
                                    hits[(layer_name, t_idx)] = True
                                    TARGET_CASE_REGISTRY[(layer_name, t_idx)] = "PASS (MNIST)"
                                    img_tensor = data[img_idx].clone()
                                    
                                    string_id = REVERSE_MAPPING.get(t_idx, "Unknown")
                                    base_fname = os.path.join(LOG_DIR, f"item_PASS_MNIST_{layer_name}_String{string_id}_Shift{t_idx}")
                                    save_target_data(img_tensor, base_fname)
                                    break
                                
                                # Update Closest Seed Matches
                                match_counts = torch.sum(eff_bit == t_bin, dim=1)
                                batch_max_count, batch_max_idx = torch.max(match_counts, dim=0)
                                if batch_max_count.item() > closest_matches[(layer_name, t_idx)]['count']:
                                    closest_matches[(layer_name, t_idx)]['count'] = batch_max_count.item()
                                    best_img_idx = batch_max_idx.item() // L
                                    closest_matches[(layer_name, t_idx)]['img'] = data[best_img_idx:best_img_idx+1].clone().cpu()
                            if found_in_batch: break

    # ==============================================================
    # STEP 3: GRADIENT OPTIMIZATION (The Fallback)
    # ==============================================================
    print("\n   [STEP 3] Running Gradient Optimization for remaining MISSES...")
    for layer_name in target_layers:
        for t_idx, t_in_list in enumerate(INPUT_TARGETS):
            if not hits[(layer_name, t_idx)]:
                best_count = closest_matches[(layer_name, t_idx)]['count']
                best_img = closest_matches[(layer_name, t_idx)]['img']
                
                if best_img is None:
                    TARGET_CASE_REGISTRY[(layer_name, t_idx)] = "SKIPPED (No Seed)"
                    continue
                
                # --- INJECT UNIQUE TARGET WEIGHTS ---
                model.restore_all_pristine_weights()
                model.inject_target_weights(layer_name, WEIGHT_TARGETS[t_idx])
                
                case_desc = f"Synth_{layer_name}_T{t_idx}"
                run_gradient_optimization(
                    model, 
                    layer_name, 
                    t_in_list, 
                    t_idx, 
                    case_desc, 
                    best_img, 
                    best_count
                )
    
    print("\n=========================================================================================")
    print(f"TARGET VECTOR GENERATION SUMMARY (Grouped by ATPG String)")
    print("=========================================================================================")
    
    # Create headers based on the original ATPG Strings
    atpg_keys = list(ATPG_STRING_MAPPING.keys())
    target_headers = " | ".join([f"Str {i}".center(10) for i in range(len(atpg_keys))])
    print(f"{'Layer':<10} | {target_headers}")
    print("-" * (13 + 13 * len(atpg_keys)))
    
    layers = sorted(list(set(k[0] for k in TARGET_CASE_REGISTRY.keys())))
    for layer in layers:
        row_str = f"{layer:<10}"
        for s_idx, atpg_str in enumerate(atpg_keys):
            mapped_indices = ATPG_STRING_MAPPING[atpg_str]
            
            # Fetch the status for all binary vectors tied to this string
            statuses = [TARGET_CASE_REGISTRY.get((layer, t_idx), "MISS") for t_idx in mapped_indices]
            
            # Group Logic: Only PASS if ALL binary vectors passed
            if all("PASS" in s for s in statuses):
                group_status = "PASS"
            elif any("PASS" in s for s in statuses):
                group_status = "PARTIAL"
            else:
                group_status = "FAIL"
                
            row_str += f" | {group_status:^10}"
        print(row_str)
    print("-" * (13 + 13 * len(atpg_keys)))
    
    model.restore_all_pristine_weights()
    print("   [System] Restored pristine hardware/software weights for normal operations.")


def attempt_deterministic_inversion(model, target_layer_name, target_in, template_img):
    """Standalone Phase 1: Purely Mathematical Pseudo-Inverse."""
    target_layer = None
    layers_before = []
    
    for name, m in model.named_modules():
        # FIX: Include standard PyTorch layers so they aren't skipped in the inverse trace
        if isinstance(m, (RRAMConv2d, RRAMLinear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            # Safely check for .name (standard nn.Modules won't have it)
            if getattr(m, 'name', '') == target_layer_name: 
                target_layer = m
                break
            layers_before.append(m)
            
    if target_layer is None: return False, None, 0

    model.set_mode('ternary')
    with torch.no_grad():
        model(template_img)
        if not hasattr(target_layer, 'last_x_int') or target_layer.last_x_int is None: return False, None, 0
        full_dim = target_layer.last_x_int.shape[1]
        expected_max = target_layer.last_x_int.abs().max().item()
        target_inflation_factor = expected_max if expected_max > 0 else 15.0

    is_cnn = (MODEL_ARCH not in ['PureLinear', 'SimpleMLP'])
    if is_cnn: 
        model.set_mode('rram') # Restore state before bailing out
        return False, None, 0  # <--- Move check UP to prevent tensor crashes

    target_rows = getattr(target_layer, 'active_defect_rows', {0})
    row_idx = list(target_rows)[0]
    start = row_idx * 32
    
    # Dynamically size the target bounds
    end = min(start + 32, full_dim)
    active_len = end - start
    
    if active_len <= 0: return False, None, 0

    full_target_vector_template = target_layer.last_x_int[0].detach().clone().float()
    is_first_layer = (len(layers_before) == 0)

    if not is_first_layer and -1 in target_in: return False, None, 0

    target_bin = torch.tensor(target_in[:active_len], dtype=torch.int32, device=DEVICE)

    # Trace forward to collect shapes and the natural pre-quant scale at the target layer
    layer_in_shapes, layer_out_shapes = {}, {}
    x_trace = template_img.clone().to(DEVICE)
    for m_trace in layers_before:
        layer_in_shapes[m_trace] = x_trace.shape
        if isinstance(m_trace, RRAMLinear) and x_trace.dim() > 2:
            x_trace = x_trace.view(x_trace.size(0), -1)
        x_trace = m_trace(x_trace)
        layer_out_shapes[m_trace] = x_trace.shape

    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in layers_before)
    natural_max = x_trace.abs().max().item() if x_trace.numel() > 0 and x_trace.abs().max().item() > 0 else 1.0

    # ==================================================================
    # MULTI-SCALE SWEEP
    # ==================================================================
    if is_first_layer:
        scale_sweep = [15.0 * target_inflation_factor]   # original behaviour, unchanged
    else:
        # Mix of (a) the original aggressive inflation that the existing
        # code happens to rely on, and (b) sensible BN-aware scales.
        scale_sweep = [
            15.0 * target_inflation_factor,   # original (often best for clustered patterns)
            0.5  * natural_max,               # mid-bit-3 target (~8 quantised)
            0.25 * natural_max,               # low bit target (~4 quantised)
            1.0  * natural_max,               # high target (~15, saturating)
        ]
        if has_bn:
            scale_sweep.append(0.005 * natural_max)   # the (originally dead) BN-aware scale

    best_match_count = -1
    best_img = None
    pinv_cache = {}

    def _do_inversion(scale_factor):
        """Return the inverted image (clamped to INPUT range)."""
        full_target_vector = full_target_vector_template.clone()
        mapped = [val * scale_factor for val in target_in[:active_len]]
        full_target_vector[start:end] = torch.tensor(mapped, dtype=torch.float32, device=DEVICE)
        current_target = full_target_vector.unsqueeze(0)

        for m in reversed(layers_before):
            if isinstance(m, nn.BatchNorm1d):
                mean, var, gamma, beta, eps = m.running_mean, m.running_var, m.weight, m.bias, m.eps
                std = torch.sqrt(var + eps)
                current_target = ((current_target - beta) * std / gamma) + mean

            elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
                out_shape = layer_out_shapes[m]
                temp_spatial = current_target.view(out_shape)
                scale_fac = m.kernel_size
                if isinstance(scale_fac, tuple):
                    inverted_spatial = F.interpolate(temp_spatial, size=layer_in_shapes[m][2:], mode='nearest')
                else:
                    inverted_spatial = F.interpolate(temp_spatial, scale_factor=scale_fac, mode='nearest')
                current_target = inverted_spatial.view(current_target.size(0), -1)

            elif isinstance(m, RRAMLinear):
                if m not in pinv_cache:
                    W_eff = TernaryWeightFn.apply(m.layer.weight)
                    pinv_cache[m] = torch.linalg.pinv(W_eff.t())
                pinv_W = pinv_cache[m]
                num_row_blocks = (m.layer.in_features + SimulatorConfig.XB_SIZE - 1) // SimulatorConfig.XB_SIZE
                adc_effective_max = SimulatorConfig.ADC_MAX * 15.0 * num_row_blocks
                adc_effective_min = SimulatorConfig.ADC_MIN * 15.0 * num_row_blocks
                current_target = torch.clamp(current_target, min=adc_effective_min, max=adc_effective_max)

                if m.layer.bias is not None: current_target = current_target - m.layer.bias.unsqueeze(0)
                current_target = current_target @ pinv_W

                if m != layers_before[0]:
                    current_target = torch.clamp(current_target, min=0.0)
                    max_val = current_target.abs().max()
                    if max_val > 0:
                        sc = max_val / 15.0
                        current_target = ((current_target / sc) + 1e-5).round() * sc

        img_raw = current_target.view_as(template_img)
        img_max_abs = img_raw.abs().max()

        if is_first_layer:
            if img_max_abs > 0:
                return torch.clamp(img_raw / img_max_abs, INPUT_MIN, INPUT_MAX)
            return torch.clamp(img_raw, INPUT_MIN, INPUT_MAX)
        else:
            # For deep layers, prefer hard-clamping (the original behaviour)
            # because BN's affine mapping is sensitive to global rescaling.
            if img_max_abs > 50.0 * INPUT_MAX:
                p99 = torch.quantile(img_raw.abs().flatten(), 0.99).item()
                if p99 > INPUT_MAX:
                    img_raw = img_raw * (INPUT_MAX / max(p99, 1e-6))
            return torch.clamp(img_raw, INPUT_MIN, INPUT_MAX)

    def _score(img):
        model.set_mode('rram')
        with torch.no_grad():
            model(img)
            raw_int = target_layer.last_x_int[0, start:end].int()
            mag_int = torch.abs(raw_int).int()
            sign_int = torch.sign(raw_int)
            mc_best = 0
            for mask in [1, 2, 4, 8]:
                eff_bit = ((mag_int & mask).bool().float() * sign_int).int()
                mc = torch.sum(eff_bit == target_bin).item()
                if mc > mc_best: mc_best = mc
        return mc_best

    with torch.no_grad():
        for sf in scale_sweep:
            img_try = _do_inversion(sf)
            score = _score(img_try)
            if score > best_match_count:
                best_match_count = score
                best_img = img_try.clone()
                if best_match_count == 32:
                    break  # PASS, no need to keep sweeping
            # Keep model in 'ternary' for the next inversion's pinv/etc
            model.set_mode('ternary')

    model.set_mode('rram')
    if best_match_count == 32:
        return True, best_img, 32
    return False, best_img if best_img is not None else template_img.clone(), best_match_count
        
def _smooth_bit_on(abs_x, k, T):
    """Smooth approx of (|x|.int() & 2**k) > 0, valid on [0, 16)."""
    if k == 3:
        cs = [7.5]                                   # rise
    elif k == 2:
        cs = [3.5, 7.5, 11.5, 15.5]                  # rise, fall, rise, fall
    elif k == 1:
        cs = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
    else:  # k == 0
        cs = [c + 0.5 for c in range(16)]            # 0.5, 1.5, ..., 15.5
    p = torch.zeros_like(abs_x)
    for i, c in enumerate(cs):
        p = p + torch.sigmoid(T * (abs_x - c)) * (1 if i % 2 == 0 else -1)
    return p

def _cycle_loss(p_on, segment_target, target_tensor, temperature, gamma, anchor):
    """Mirror of bit3_loss but on an arbitrary smooth bit indicator p_on."""
    eps = 1e-6
    m_act = (target_tensor == 1) | (target_tensor == -1)
    m_zer = (target_tensor == 0)
    loss = torch.tensor(0.0, device=p_on.device)

    if m_act.any():
        p_a = p_on[:, m_act]
        loss = loss - torch.mean((1 - p_a).pow(gamma) * torch.log(p_a + eps))
        # Sign hinge: anchor>0 is the smallest |x| in this bit's "on" region
        signs = target_tensor[m_act].float().unsqueeze(0).expand_as(segment_target[:, m_act])
        loss = loss + torch.mean(torch.clamp(anchor - signs * segment_target[:, m_act], min=0.0))

    if m_zer.any():
        p_z = p_on[:, m_zer]
        loss = loss - torch.mean(p_z.pow(gamma) * torch.log(1 - p_z + eps))

    return loss

def multibit_loss(segment_target, target_tensor, temperature=2.0, gamma=2.0, soft_min_tau=0.5):
    abs_x = torch.abs(segment_target)
    anchors = {3: 8.5, 2: 4.5, 1: 2.5, 0: 1.5}    # smallest |x| in each bit's first "on" interval, +0.5 margin
    losses = []
    for k in (3, 2, 1, 0):
        p = _smooth_bit_on(abs_x, k, temperature).clamp(0.0, 1.0)
        losses.append(_cycle_loss(p, segment_target, target_tensor, temperature, gamma, anchors[k]))
    L = torch.stack(losses)                       # [4]
    return -soft_min_tau * torch.logsumexp(-L / soft_min_tau, dim=0)

def _bit_transition_points(k, max_value=32.0):
    """Return an alternating list of (threshold, sign) pairs describing
    the rising/falling edges of bit `k` of `|x|.int()` over `[0, max_value)`.
    """
    period = 2 ** (k + 1)
    half   = 2 ** k
    pts = []
    j = 0
    while True:
        rise = j * period + half - 0.5
        if rise >= max_value:
            break
        pts.append((rise, +1.0))
        fall = (j + 1) * period - 0.5
        if fall < max_value:
            pts.append((fall, -1.0))
        else:
            break
        j += 1
    return pts

# =====================================================================
# 2.  Smooth differentiable bit indicator
# =====================================================================
def _smooth_bit_on(abs_x, k, temperature, max_value=32.0):
    """Smooth approximation of `(|x|.int() & (1 << k)) > 0`.
    """
    pts = _bit_transition_points(k, max_value)
    p = torch.zeros_like(abs_x)
    for centre, sign in pts:
        p = p + sign * torch.sigmoid(temperature * (abs_x - centre))
    return p.clamp(0.0, 1.0)
 
 
# =====================================================================
# 3.  Per-cycle bit-matching loss
# =====================================================================
def _cycle_loss(p_on, segment_target, target_tensor,
                gamma=2.0, sign_anchor=8.5):
    """Mirror of bit3_loss but on an arbitrary smooth bit indicator.
    """
    eps = 1e-6
    m_active = (target_tensor == 1) | (target_tensor == -1)
    m_zero   = (target_tensor == 0)
 
    loss = torch.tensor(0.0, device=p_on.device)
 
    if m_active.any():
        p_a = p_on[:, m_active]
        focal = (1.0 - p_a).pow(gamma)
        loss = loss - torch.mean(focal * torch.log(p_a + eps))
 
        # Sign hinge: push signed value past +/- sign_anchor
        signs_wanted = target_tensor[m_active].float() \
                                          .unsqueeze(0) \
                                          .expand_as(segment_target[:, m_active])
        loss = loss + torch.mean(
            torch.clamp(sign_anchor - signs_wanted * segment_target[:, m_active], min=0.0)
        )
 
    if m_zero.any():
        p_z = p_on[:, m_zero]
        focal = p_z.pow(gamma)
        loss = loss - torch.mean(focal * torch.log(1.0 - p_z + eps))
 
    return loss
 
# 4.  Soft-min-over-cycles aggregator (drop-in replacement for bit3_loss)
def multibit_loss(segment_target, target_tensor,
                  temperature=2.0, gamma=2.0, soft_min_tau=0.5,
                  max_value=32.0):
    """Loss that drives any of the four DAC bit cycles to match the target.
    """
    abs_x = torch.abs(segment_target)
 
    # Smallest |x| that turns each bit on, +0.5 margin -> sign-hinge anchor
    anchors = {3: 8.5, 2: 4.5, 1: 2.5, 0: 1.5}
 
    losses = []
    for k in (3, 2, 1, 0):
        p = _smooth_bit_on(abs_x, k, temperature, max_value=max_value)
        losses.append(
            _cycle_loss(p, segment_target, target_tensor,
                        gamma=gamma, sign_anchor=anchors[k])
        )
 
    L = torch.stack(losses)                              # shape [4]
    return -soft_min_tau * torch.logsumexp(-L / soft_min_tau, dim=0)

def bit3_loss(segment_target, target_tensor, temperature=2.0, gamma=2.0):
    """
    Focal, temperature-annealed loss targeting bit-3 of |x_int|.
    """
    abs_x = torch.abs(segment_target)
    
    # Smooth probability that bit 3 is on (boundary at 7.5)
    p_bit_on = torch.sigmoid(temperature * (abs_x - 7.5))
    
    mask_active = (target_tensor == 1) | (target_tensor == -1)  # bit should be ON
    mask_zero   = (target_tensor == 0)                           # bit should be OFF
    
    eps = 1e-6
    
    # 1. Active Bits Loss (Targeting ON)
    if mask_active.any():
        p_active = p_bit_on[:, mask_active]
        # Focal weighting: push harder on bits that are failing (p_active is low)
        focal_weight_on = (1.0 - p_active) ** gamma
        loss_on = -torch.mean(focal_weight_on * torch.log(p_active + eps))
        
        # Hinge Fix: Push safely past 7.5 to an 8.5 margin
        signs_wanted = target_tensor[mask_active].float().unsqueeze(0).expand_as(segment_target[:, mask_active])
        sign_loss = torch.mean(torch.clamp(8.5 - signs_wanted * segment_target[:, mask_active], min=0.0))
    else:
        loss_on, sign_loss = 0.0, 0.0
        
    # 2. Zero Bits Loss (Targeting OFF)
    if mask_zero.any():
        p_zero = p_bit_on[:, mask_zero]
        # Focal weighting: push harder on bits that are accidentally triggering (p_zero is high)
        focal_weight_off = p_zero ** gamma
        loss_off = -torch.mean(focal_weight_off * torch.log(1.0 - p_zero + eps))
    else:
        loss_off = 0.0
        
    return loss_on + loss_off + sign_loss

def run_gradient_optimization(model, target_layer_name, target_in,
                              target_idx, case_desc,
                              best_seed_img, best_match_count):
    """Standalone Phase 3: High-Intensity Gradient Fine-Tuning.
    """
    global TARGET_CASE_REGISTRY
 
    best_seed_img = best_seed_img.to(DEVICE)
 
    target_layer = None
    for name, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)) and m.name == target_layer_name:
            target_layer = m
            break
 
    target_rows = getattr(target_layer, 'active_defect_rows', {0})
    row_idx = list(target_rows)[0]
    start = row_idx * 32
 
    target_in_tensor_full = torch.tensor(target_in, dtype=torch.int32, device=DEVICE)

    # Tier 1.1 -- ReLU feasibility skip: ReLU-fed layers cannot produce negative inputs.
    RELU_FED_LAYERS = {"fc2", "fc3", "fc4"}
    _active_len_check = min(32, len(target_in))
    if target_layer_name in RELU_FED_LAYERS and any(v < 0 for v in target_in[:_active_len_check]):
        TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = "INFEASIBLE_RELU"
        print(f"      -> Result: SKIP (negative target at ReLU-fed layer)")
        return

    # Tier 1.2 -- DISABLED: fixed-scale override caused search/eval mismatch for fc4
    # (and broke fc3). Search now runs under the same dynamic scale as validation/eval.
    _orig_scale_fn = target_layer.get_dynamic_scale
    _USE_FIXED_SCALE = False

    # Tier 3.1 -- pre-fc4 leaky-ReLU override active only when targeting fc4.
    _USE_PRE_FC4_LEAKY = (target_layer_name == "fc4")
    _pre_fc4_slope = [0.0]  # mutable container so we can update during the loop

    # Tier 3.1 -- pre-fc2 leaky-ReLU override active only when targeting fc2.
    # (Parallels fc4: helps gradients flow back through the bn1+ReLU before fc2.)
    _USE_PRE_FC2_LEAKY = (target_layer_name == "fc2")
    _pre_fc2_slope = [0.0]

    original_relu = F.relu

    def _rram_match_batched(img_batch):
        F.relu = original_relu
        model.set_mode('rram')
        # Tier 3.1 -- score under hard ReLU; restore leaky override after.
        _saved_slope_fc4 = GRADOPT_PRE_LAYER_LEAKY.pop("fc4", None)
        _saved_slope_fc2 = GRADOPT_PRE_LAYER_LEAKY.pop("fc2", None)
        try:
            with torch.no_grad():
                model(img_batch) # Forward pass the WHOLE batch at once
                
                full_dim = target_layer.last_x_int.shape[1]
                end = min(start + 32, full_dim)
                active_len = end - start

                ri = target_layer.last_x_int[:, start:end].int()
                rm = torch.abs(ri)
                sign_ri = torch.sign(ri)
                
                best_mc = torch.zeros(img_batch.size(0), dtype=torch.int32, device=DEVICE)
                
                for mask in (1, 2, 4, 8):
                    rb = ((rm & mask).bool().float() * sign_ri).int()
                    # Compare across the batch (dim=1 is the spatial/feature dim)
                    mc = (rb == target_in_tensor_full[:active_len].unsqueeze(0)).sum(dim=1)
                    best_mc = torch.maximum(best_mc, mc.to(torch.int32))
                    
                return best_mc # Returns a tensor of scores, e.g., [32, 14, 30, 25...]
        finally:
            if _saved_slope_fc4 is not None:
                GRADOPT_PRE_LAYER_LEAKY["fc4"] = _saved_slope_fc4
            if _saved_slope_fc2 is not None:
                GRADOPT_PRE_LAYER_LEAKY["fc2"] = _saved_slope_fc2

    # Standalone validator -- batched scores aren't reproducible (batch-wide get_dynamic_scale).
    def _rram_match_single(img):
        # Force batch=1 so scale doesn't depend on prior batch context.
        if img.dim() == best_seed_img.dim() and img.size(0) != 1:
            img = img[:1]
        F.relu = original_relu
        model.set_mode('rram')
        # Tier 3.1 -- score under hard ReLU; restore leaky override after.
        _saved_slope_fc4 = GRADOPT_PRE_LAYER_LEAKY.pop("fc4", None)
        _saved_slope_fc2 = GRADOPT_PRE_LAYER_LEAKY.pop("fc2", None)
        try:
            with torch.no_grad():
                model(img)
                full_dim = target_layer.last_x_int.shape[1]
                end_s = min(start + 32, full_dim)
                active_len_s = end_s - start
                ri = target_layer.last_x_int[:, start:end_s].int()
                rm = torch.abs(ri); sign_ri = torch.sign(ri)
                best = 0
                for mask in (1, 2, 4, 8):
                    rb = ((rm & mask).bool().float() * sign_ri).int()
                    mc = (rb == target_in_tensor_full[:active_len_s].unsqueeze(0)).sum(dim=1)
                    m = int(mc.max().item())
                    if m > best: best = m
                return best
        finally:
            if _saved_slope_fc4 is not None:
                GRADOPT_PRE_LAYER_LEAKY["fc4"] = _saved_slope_fc4
            if _saved_slope_fc2 is not None:
                GRADOPT_PRE_LAYER_LEAKY["fc2"] = _saved_slope_fc2

    # Mutable leaky slope for annealing the surrogate ReLU
    leaky_slope = [0.15]
    F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=leaky_slope[0], inplace=inplace)
 
    model.set_mode('ternary')
 
    try:
        model(best_seed_img)
        full_dim = target_layer.last_x_int.shape[1]
 
        end = min(start + 32, full_dim)
        active_len = end - start
        if active_len <= 0:
            return
 
        # Adaptive difficulty: vectors with many ones are harder to achieve
        ones_count = sum(1 for v in target_in[:active_len] if v != 0)
        if ones_count > 20:
            batch_size, max_noise, num_of_steps = 64, 1.5, 20000
        elif ones_count > 12:
            batch_size, max_noise, num_of_steps = 32, 1.0, 15000
        else:
            batch_size, max_noise, num_of_steps = 16, 0.5, 10000
 
        rep_shape  = [batch_size] + [1] * (best_seed_img.dim() - 1)
        view_shape = [-1]         + [1] * (best_seed_img.dim() - 1)
 
        img_batch = best_seed_img.repeat(*rep_shape)
        noise_scales = torch.linspace(0.0, max_noise, batch_size, device=DEVICE).view(*view_shape)
        img_batch = torch.clamp(img_batch + torch.randn_like(img_batch) * noise_scales,
                                INPUT_MIN, INPUT_MAX)
        img_batch.requires_grad_(True)
 
        # ---- LR halved (was 0.08); multi-cycle gradient is denser ------------
        optimizer = torch.optim.AdamW([img_batch], lr=0.04, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2)
 
        target_tensor_for_mask = torch.tensor(target_in[:active_len], device=DEVICE)
        target_bin             = torch.tensor(target_in[:active_len], dtype=torch.int32, device=DEVICE)
 
        best_img = best_seed_img.detach().clone()
        oob_mask = torch.ones(full_dim, dtype=torch.bool, device=DEVICE)
        oob_mask[start:end] = False
        global_best_match = best_match_count
 
        rram_best_match = best_match_count
        rram_best_img   = best_seed_img.detach().clone()
        RRAM_EVAL_INTERVAL = 250
 
        for step in range(num_of_steps):
            # ---- Annealing schedules -------------------------------------
            progress = min(step / (num_of_steps * 0.8), 1.0)
            # Tier 2.2 -- DISABLED: a non-zero leak floor causes the same search/eval
            # disconnect that hurt fc3. All layers now use the original linear decay.
            leaky_slope[0]   = 0.15 * (1.0 - progress)
            current_temp         = 0.5 + 4.5 * progress      # 0.5 -> 5.0
            current_soft_min_tau = 1.0 - 0.9 * progress      # 1.0 -> 0.1   NEW

            # Tier 3.1 -- pre-fc4 leaky slope (only active when targeting fc4); decays to 0.
            if _USE_PRE_FC4_LEAKY:
                _pre_fc4_slope[0] = 0.30 * (1.0 - progress)
                GRADOPT_PRE_LAYER_LEAKY["fc4"] = _pre_fc4_slope[0]

            # Tier 3.1 -- pre-fc2 leaky slope (only active when targeting fc2); decays to 0.
            if _USE_PRE_FC2_LEAKY:
                _pre_fc2_slope[0] = 0.30 * (1.0 - progress)
                GRADOPT_PRE_LAYER_LEAKY["fc2"] = _pre_fc2_slope[0]
 
            optimizer.zero_grad()
            model(img_batch)
 
            x_int = target_layer.last_x_int
            full_dim = x_int.shape[1]
            end = min(start + 32, full_dim)
            active_len = end - start
            if x_int is None or active_len <= 0:
                break
 
            segment_target_full    = x_int[:, start:end]
            target_tensor_for_mask = torch.tensor(target_in[:active_len], device=DEVICE)
            target_bin             = torch.tensor(target_in[:active_len], dtype=torch.int32, device=DEVICE)
 
            # Spatial isolation
            B_size = img_batch.size(0)
            N_size = x_int.size(0)
            L_size = N_size // B_size
 
            if L_size > 1:  # CNN layer with multiple patches
                seg_reshaped = segment_target_full.view(B_size, L_size, -1)
 
                with torch.no_grad():
                    ri = seg_reshaped.int()
                    rm = torch.abs(ri)
                    sign_ri = torch.sign(ri)
                    best_per_patch = torch.zeros(B_size, L_size,
                                                 device=DEVICE, dtype=torch.int32)
                    for mask in (1, 2, 4, 8):
                        rb = ((rm & mask).bool().float() * sign_ri).int()
                        mc = (rb == target_bin.unsqueeze(0).unsqueeze(0)).sum(dim=2)
                        best_per_patch = torch.maximum(best_per_patch, mc.to(torch.int32))
                    best_patch_indices = best_per_patch.argmax(dim=1)
 
                batch_indices = torch.arange(B_size, device=DEVICE)
                segment_target = seg_reshaped[batch_indices, best_patch_indices, :]
            else:
                segment_target = segment_target_full
 
            # ==============================================================
            # Loss computation
            # ==============================================================
            loss_bit_match = multibit_loss(
                segment_target, target_tensor_for_mask,
                temperature=current_temp, gamma=2.0,
                soft_min_tau=current_soft_min_tau,
            )
 
            # Smooth L1 (Huber) for OOB padding to prevent masking
            loss_oob = F.smooth_l1_loss(
                x_int[:, oob_mask], torch.zeros_like(x_int[:, oob_mask])
            )
 
            # Target-conditioned variance
            loss_variance = torch.tensor(0.0, device=DEVICE)
            _mask_ones  = (target_tensor_for_mask == 1) | (target_tensor_for_mask == -1)
            _mask_zeros = (target_tensor_for_mask == 0)
            if _mask_ones.sum() > 1:
                loss_variance = loss_variance + torch.var(
                    segment_target[:, _mask_ones], dim=1
                ).mean()
            if _mask_zeros.sum() > 1:
                loss_variance = loss_variance + torch.var(
                    segment_target[:, _mask_zeros], dim=1
                ).mean()
 
            # Silence penalty: keep |x_int| comfortably below ADC_MAX (127)
            loss_silence = torch.mean(torch.relu(torch.abs(x_int) - 80.0))
 
            img_reg = torch.norm(img_batch)

            # Tier 1.3 -- Penalize pre-quant lanes that overshoot integer-1 window (FC4 only).
            # Threshold is derived from the actual dynamic scale, not a hardcoded constant.
            loss_pre_excess = torch.tensor(0.0, device=DEVICE)
            if target_layer_name == "fc4":
                _pre = getattr(target_layer, 'last_x_pre_quant', None)
                if _pre is not None and _pre.dim() == 2:
                    pre_window = _pre[:, start:end]
                    _scale_now = target_layer.get_dynamic_scale(_pre)
                    _scale_val = float(_scale_now.max().item()) if _scale_now.numel() > 0 else (1.0 / 15.0)
                    excess = F.relu(pre_window.abs() - 1.5 * _scale_val)
                    loss_pre_excess = excess.pow(2).mean()

            # Tier 1.4 -- Global lane-sum constraint for high-density targets (FC4 only).
            loss_sum = torch.tensor(0.0, device=DEVICE)
            if target_layer_name == "fc4":
                _nz_target = int((target_tensor_for_mask != 0).sum().item())
                if _nz_target >= 24:
                    _target_sum = float(target_tensor_for_mask.float().sum().item())
                    _x_q_window = segment_target.float()
                    loss_sum = (_x_q_window.sum(dim=1) - _target_sum).pow(2).mean()

            loss = (loss_bit_match
                    + 0.05 * loss_oob
                    + 0.01 * loss_variance
                    + 0.01 * loss_silence
                    + 1e-5 * img_reg
                    + 0.05 * loss_pre_excess
                    + 0.02 * loss_sum)
 
            # ==============================================================
            # Global-best tracking across all 4 cycles
            # ==============================================================
            with torch.no_grad():
                raw_int = segment_target.int()
                mag_int = torch.abs(raw_int)
                sign_int = torch.sign(raw_int)
 
                best_matches = torch.zeros(segment_target.size(0), device=DEVICE)
                for mask in (1, 2, 4, 8):
                    eff_bit = ((mag_int & mask).bool().float() * sign_int).int()
                    m = (eff_bit == target_bin.unsqueeze(0)).sum(dim=1).float()
                    best_matches = torch.maximum(best_matches, m)
                max_match_in_batch = best_matches.max().item()
 
                if max_match_in_batch > global_best_match:
                    global_best_match = int(max_match_in_batch)
                    best_patch_idx = int(best_matches.argmax().item())
                    imgs_per_batch = img_batch.size(0)
                    patches_per_img = segment_target.size(0) // imgs_per_batch
                    best_img_idx = best_patch_idx // patches_per_img
                    best_img = img_batch[best_img_idx:best_img_idx + 1].detach().clone()
 
            # ---- Periodic real-RRAM validation / early stopping --------
            if (step + 1) % RRAM_EVAL_INTERVAL == 0 or max_match_in_batch >= 30:
                rms = _rram_match_batched(img_batch.detach())
                # Standalone-validate top-K candidates; batched scores are unreliable.
                K_validate = min(4, img_batch.size(0))
                top_vals, top_idx = torch.topk(rms, K_validate)
                for k in range(K_validate):
                    cand = img_batch[top_idx[k]:top_idx[k]+1].detach().clone()
                    true_score = _rram_match_single(cand)
                    if true_score > rram_best_match:
                        rram_best_match = true_score
                        rram_best_img = cand
                    if true_score == 32:
                        break
                if rram_best_match == 32:
                    F.relu = original_relu
                    break

                F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=leaky_slope[0], inplace=inplace)
                model.set_mode('ternary')
 
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                img_batch.data = img_batch.data.clamp(INPUT_MIN, INPUT_MAX)
 
        # ===================================================================
        # Step E. Stochastic RRAM polish (single-image-validated)
        # ===================================================================
        if rram_best_match < 32:
            # Validate the seed standalone first.
            polish_seed = rram_best_img if rram_best_match >= global_best_match else best_img
            seed_true = _rram_match_single(polish_seed)
            if seed_true > rram_best_match:
                rram_best_match = seed_true
                rram_best_img = polish_seed.clone()

            BATCH_POLISH = 250 # Do 250 at a time
            for polish_round in range(5000 // BATCH_POLISH): # Now only runs 20 times!
                if rram_best_match == 32: break
                
                decay = max(0.02, 0.3 * (1.0 - (polish_round * BATCH_POLISH) / 5000))
                
                # Create a batch of repeated best images and add noise
                perturbed_batch = rram_best_img.repeat(BATCH_POLISH, 1, 1, 1) 
                perturbed_batch += torch.randn_like(perturbed_batch) * decay
                perturbed_batch = torch.clamp(perturbed_batch, INPUT_MIN, INPUT_MAX)
                
                # Cheap batched filter.
                rms = _rram_match_batched(perturbed_batch)
                # Standalone-validate top-K (batched scores don't reproduce).
                K_validate = 4
                top_vals, top_idx = torch.topk(rms, min(K_validate, rms.size(0)))
                for k in range(top_vals.size(0)):
                    cand = perturbed_batch[top_idx[k]:top_idx[k]+1].detach().clone()
                    true_score = _rram_match_single(cand)
                    if true_score > rram_best_match:
                        rram_best_match = true_score
                        rram_best_img = cand
                    if true_score == 32:
                        break
 
        # Step F. Per-bit targeted refinement for near-misses 
        if 30 <= rram_best_match < 32:
            F.relu = original_relu
            GRADOPT_PRE_LAYER_LEAKY.pop("fc4", None)  # Tier 3.1 -- refinement uses hard ReLU.
            GRADOPT_PRE_LAYER_LEAKY.pop("fc2", None)  # Tier 3.1 -- refinement uses hard ReLU.
            model.set_mode('rram')
            with torch.no_grad():
                model(rram_best_img)
                ri = target_layer.last_x_int[:, start:end].int()
                rm_abs = torch.abs(ri)
                rb = ((rm_abs & 8).bool().float() * torch.sign(ri)).int()
                bit_correct = (rb == target_bin.unsqueeze(0)).float()
                best_patch_idx = bit_correct.sum(dim=1).argmax()
                failing_mask = (bit_correct[best_patch_idx] == 0)
                num_failing = int(failing_mask.sum().item())
 
            if 0 < num_failing <= 4:
                bit_weights = torch.ones(32, device=DEVICE)
                bit_weights[failing_mask] = 10.0
 
                refine_slope = [0.05]
                F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=refine_slope[0], inplace=inplace)
                model.set_mode('ternary')
 
                img_refine = rram_best_img.clone()
                refine_batch = 8
                rep_r  = [refine_batch] + [1] * (img_refine.dim() - 1)
                view_r = [-1]           + [1] * (img_refine.dim() - 1)
                img_r = img_refine.repeat(*rep_r)
                noise_r = torch.randn_like(img_r) * torch.linspace(
                    0.0, 0.15, refine_batch, device=DEVICE).view(*view_r)
                img_r = torch.clamp(img_r + noise_r, INPUT_MIN, INPUT_MAX)
                img_r.requires_grad_(True)
 
                opt_r   = torch.optim.Adam([img_r], lr=0.005)
                sched_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=5000)
 
                for r_step in range(5000):
                    opt_r.zero_grad()
                    model(img_r)
                    x_r = target_layer.last_x_int
                    if x_r is None or end > x_r.shape[1]:
                        break
                    seg_r = x_r[:, start:end]
 
                    abs_x_r = torch.abs(seg_r)
                    p_on = torch.sigmoid(5.0 * (abs_x_r - 7.5))
 
                    m_active = (target_tensor_for_mask == 1) | (target_tensor_for_mask == -1)
                    m_zero   = (target_tensor_for_mask == 0)
                    eps = 1e-6
                    loss_r = torch.tensor(0.0, device=DEVICE)
 
                    if m_active.any():
                        p_a = p_on[:, m_active]
                        w_a = bit_weights[m_active].unsqueeze(0)
                        loss_r = loss_r - torch.mean(w_a * torch.log(p_a + eps))
                        signs_w = target_tensor_for_mask[m_active].float() \
                                                                  .unsqueeze(0) \
                                                                  .expand_as(seg_r[:, m_active])
                        hinge = torch.clamp(8.5 - signs_w * seg_r[:, m_active], min=0.0)
                        loss_r = loss_r + torch.mean(w_a * hinge)
                    if m_zero.any():
                        p_z = p_on[:, m_zero]
                        w_z = bit_weights[m_zero].unsqueeze(0)
                        loss_r = loss_r - torch.mean(w_z * torch.log(1.0 - p_z + eps))
 
                    loss_r = loss_r + 0.05 * F.smooth_l1_loss(
                        x_r[:, oob_mask], torch.zeros_like(x_r[:, oob_mask])
                    )
 
                    loss_r.backward()
                    opt_r.step()
                    sched_r.step()
                    with torch.no_grad():
                        img_r.data = img_r.data.clamp(INPUT_MIN, INPUT_MAX)
 
                    if (r_step + 1) % 200 == 0:
                        F.relu = original_relu
                        model.set_mode('rram')
                        with torch.no_grad():
                            for bi in range(img_r.size(0)):
                                cand_r = img_r[bi:bi + 1].detach().clone()
                                # Always single-image-validated
                                true_score = _rram_match_single(cand_r)
                                if true_score > rram_best_match:
                                    rram_best_match = true_score
                                    rram_best_img = cand_r
                                if true_score == 32:
                                    break
                        if rram_best_match == 32:
                            break
                        F.relu = lambda x, inplace=False: F.leaky_relu(x, negative_slope=refine_slope[0], inplace=inplace)
                        model.set_mode('ternary')
 
        best_img = rram_best_img
 
    finally:
        F.relu = original_relu
        target_layer.get_dynamic_scale = _orig_scale_fn  # Tier 1.2 -- restore original scale fn.
        GRADOPT_PRE_LAYER_LEAKY.pop("fc4", None)  # Tier 3.1 -- clear leaky override.
        GRADOPT_PRE_LAYER_LEAKY.pop("fc2", None)  # Tier 3.1 -- clear leaky override.
 
    # Final reporting -- best_img is standalone-validated by construction.
    model.set_mode('rram')
    final_outcome = "FAIL"
    best_synth_match = 0
    best_eff_bit = None  # Tier 4.1 -- track mismatched lanes for FAIL diagnostics.
 
    with torch.no_grad():
        model(best_img)
        raw_int_all = target_layer.last_x_int[:, start:end].int()
        sign = torch.sign(raw_int_all)
        mag_int = torch.abs(raw_int_all).int()
 
        for mask in (1, 2, 4, 8):
            eff_bit = ((mag_int & mask).bool().float() * sign).int()
            matches = (eff_bit == target_bin.unsqueeze(0)).sum(dim=1)
            best_patch_match = matches.max().item()
            if best_patch_match > best_synth_match:
                best_synth_match = best_patch_match
                best_eff_bit = eff_bit[matches.argmax().item()]
            if best_patch_match == 32:
                final_outcome = "PASS"

    # With validation in place these should agree; warn if state mutation broke that.
    if int(rram_best_match) != int(best_synth_match):
        print(f"      -> [WARN] rram_best_match ({rram_best_match}) != final ({best_synth_match}); "
              f"check for model state mutation between validation and final eval.")
 
    if final_outcome == "FAIL":
        print(f"      -> Result: FAIL (GradOpt) - Best Match: {best_synth_match}/32 "
              f"(ternary-opt: {global_best_match}/32, rram+relu: {rram_best_match}/32)")
        # Tier 4.1 -- log which lanes mismatched (target vs got) for diagnosis.
        if best_eff_bit is not None:
            mismatches = [(int(i), int(target_bin[i].item()), int(best_eff_bit[i].item()))
                          for i in (best_eff_bit != target_bin).nonzero().flatten().tolist()]
            print(f"         Mismatched lanes (idx, want, got): {mismatches}")
    else:
        print(f"      -> Result: PASS (GradOpt)")
 
    TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = f"{final_outcome} (GradOpt)"
    string_id = REVERSE_MAPPING.get(target_idx, "Unknown")
    base_fname = os.path.join(
        LOG_DIR, f"synth_{final_outcome}_GradOpt_{target_layer_name}_String{string_id}_Shift{target_idx}"
    )
    save_target_data(best_img, base_fname)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0; correct = 0; total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

def evaluate_detailed(model, train_loader, test_loader, desc, test_only=False):
    print(f"   Evaluating {desc}...")
    model.eval()
    def run_pass(loader, set_name):
        correct = 0; total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        return correct, total
        
    acc_train = 0.0
    if not test_only:
        c1, t1 = run_pass(train_loader, "Train")
        acc_train = 100. * c1 / t1 if t1 > 0 else 0.0
        
    c2, t2 = run_pass(test_loader, "Test")
    acc_test  = 100. * c2 / t2 if t2 > 0 else 0.0
    
    if not test_only:
        acc_full  = 100. * (c1 + c2) / (t1 + t2) if (t1 + t2) > 0 else 0.0
        print(f"      -> Train: {acc_train:.2f}% | Test: {acc_test:.2f}% | Full: {acc_full:.2f}%")
    else:
        acc_full = 0.0
        print(f"      -> Test: {acc_test:.2f}%")
        
    return acc_train, acc_test, acc_full

def clean_filename(s):
    return s.replace(" ", "_").replace("|", "").replace(":", "_").replace("&", "_").replace("(", "").replace(")", "").replace(",", "_")

def run_masking_analysis(model, test_loader, experiment_desc):
    clean_tag = clean_filename(experiment_desc)
    fname = os.path.join(LOG_DIR, f"masking_{clean_tag}.csv")
    
    # NEW: Dynamically cap at 10,000 or the total length of the test dataset, whichever is smaller.
    test_dataset_size = len(test_loader.dataset)
    dynamic_limit = min(test_dataset_size, 10000, ANALYSIS_IMAGE_LIMIT)
    
    print(f"\n   [MASKING] Running Trace for {dynamic_limit} Images from Test Set -> Saving to {fname}")
    model.eval()
    
    processed_count = 0
    masked_count = 0
    diverged_count = 0
    
    fault_backups = {}
    for name, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)):
            fault_backups[name] = m.adc_offset 
            m.adc_offset = 0
    
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "Image_Idx", "Clean_Pred", "Faulty_Pred", 
            "ADC_Clean_Scalar", "ADC_Faulty_Scalar", 
            "Masking_Mechanism", "Masking_Location",
            "PreMask_Clean_Vec", "PreMask_Faulty_Vec",
            "PostMask_Clean_Vec", "PostMask_Faulty_Vec",
            "Final_Softmax_Clean", "Final_Softmax_Faulty"
        ]
        writer.writerow(header)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if processed_count >= dynamic_limit: break
                
                data = data.to(DEVICE)
                for i in range(data.size(0)):
                    if processed_count >= dynamic_limit: break
                    
                    img = data[i:i+1]
                    ANALYZER.start_capture()
                    clean_out = model(img)
                    clean_pred = clean_out.argmax(dim=1).item()
                    final_clean_probs = F.softmax(clean_out, dim=1).cpu().numpy().tolist()
                    
                    for name, m in model.named_modules():
                        if isinstance(m, (RRAMConv2d, RRAMLinear)): m.adc_offset = fault_backups[name]
                    
                    ANALYZER.start_compare()
                    faulty_out = model(img)
                    faulty_pred = faulty_out.argmax(dim=1).item()
                    ANALYZER.stop()
                    
                    final_faulty_probs = F.softmax(faulty_out, dim=1).cpu().numpy().tolist()
                    
                    for name, m in model.named_modules():
                        if isinstance(m, (RRAMConv2d, RRAMLinear)): m.adc_offset = 0

                    if ANALYZER.start_point:
                        diverged_count += 1
                        adc_c = ANALYZER.initial_fault_val_clean
                        adc_f = ANALYZER.initial_fault_val_faulty
                        
                        pre_c = ANALYZER.mask_input_clean if ANALYZER.mask_input_clean else "N/A"
                        pre_f = ANALYZER.mask_input_faulty if ANALYZER.mask_input_faulty else "N/A"
                        post_c = ANALYZER.mask_output_clean if ANALYZER.mask_output_clean else "N/A"
                        post_f = ANALYZER.mask_output_faulty if ANALYZER.mask_output_faulty else "N/A"
                        
                        if clean_pred == faulty_pred:
                            masked_count += 1
                            if ANALYZER.end_point:
                                m_loc = ANALYZER.end_point
                                mech = ANALYZER.mechanism
                            else:
                                m_loc = "Softmax/Argmax"
                                mech = "Output Tolerance"
                                pre_c = str(final_clean_probs)
                                pre_f = str(final_faulty_probs)
                        else:
                            m_loc = "None"
                            mech = "None"

                        writer.writerow([
                            processed_count, clean_pred, faulty_pred,
                            adc_c, adc_f, mech, m_loc,
                            pre_c, pre_f, post_c, post_f,
                            str(final_clean_probs), str(final_faulty_probs)
                        ])
                    processed_count += 1

    for name, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)): m.adc_offset = fault_backups[name]
    print(f"      [MASKING] Done. {masked_count}/{diverged_count} faults masked.")

def get_model():
    if DATASET == 'KWS': 
        if MODEL_ARCH == 'PureLinear':
            if KWS_FEATURE_TYPE == 'TINYSNS':
                in_dim = 784 # <-- Updated from 256 to 784 (16ch x 49fr)
            elif KWS_FEATURE_TYPE == 'MFCC':
                in_dim = 1010
            else:
                in_dim = 16000
            # Explicitly set hidden_dim=256 for KWS to match combined_v2 architecture
            return PureLinearMLP(input_dim=in_dim, num_classes=KWS_NUM_CLASSES, hidden_dim=256).to(DEVICE)
            
        elif MODEL_ARCH == 'BinaryCNN':
            if KWS_FEATURE_TYPE == 'TINYSNS':
                return TinySNS_BinaryCNN(num_classes=KWS_NUM_CLASSES).to(DEVICE)
            else:
                raise ValueError("BinaryCNN currently expects TINYSNS features.")
                
        elif KWS_FEATURE_TYPE == 'MFCC':
            return MFCC_CNNKWS(num_classes=KWS_NUM_CLASSES).to(DEVICE)
        else:
            return RawAudioCNNKWS(num_classes=KWS_NUM_CLASSES).to(DEVICE)            
    # MNIST handling logic (Unchanged)
    elif MODEL_ARCH == 'LeNet5': 
        return LeNet5().to(DEVICE)
    elif MODEL_ARCH == 'PureLinear': 
        return PureLinearMLP(input_dim=784, num_classes=10).to(DEVICE)
    else: 
        return SimpleMLP().to(DEVICE)

pattern_0 = np.array([
    -37.0, -36.0, -34.0, -23.0, -22.0, -21.0, -19.0, -18.0, -17.0, -16.0,
    -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0,
    -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 13.0, 15.0, 202.0, 216.0, 217.0, 218.0
])

pattern_1 = np.array([
    -80.0, -79.0, -78.0, -40.0, -33.0, -32.0, -31.0, -29.0, -27.0, -25.0,
    -24.0, -23.0, -21.0, -20.0, -19.0, -18.0, -17.0, -15.0, -14.0, -13.0,
    -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
    -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 159.0, 173.0, 174.0, 175.0
])

pattern_2 = np.array([
    -80.0, -79.0, -78.0, -40.0, -33.0, -32.0, -31.0, -29.0, -27.0, -25.0,
    -24.0, -23.0, -21.0, -20.0, -19.0, -18.0, -17.0, -15.0, -14.0, -13.0,
    -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
    -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 11.0, 12.0, 13.0, 159.0, 173.0, 174.0, 175.0
])

pattern_3 = np.array([
    -91.0, -90.0, -89.0, -48.0, -40.0, -38.0, -37.0, -36.0, -35.0, -34.0,
    -33.0, -32.0, -31.0, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, -21.0,
    -18.0, -17.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0,
    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0,
    148.0, 162.0, 163.0, 164.0
])

pattern_4 = np.array([
    -117.0, -116.0, -115.0, -74.0, -73.0, -69.0, -68.0, -67.0, -66.0, -62.0,
    -55.0, -54.0, -53.0, -51.0, -50.0, -49.0, -48.0, -46.0, -44.0, -41.0,
    -40.0, -39.0, -37.0, -35.0, -33.0, -31.0, -30.0, -25.0, -24.0, -23.0,
    -22.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0, -12.0, -11.0, -10.0,
    -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0,
    2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
    14.0, 15.0, 17.0, 122.0, 136.0, 137.0, 138.0
])

pattern_5 = np.array([
    -191.0, -190.0, -189.0, -130.0, -115.0, -114.0, -113.0, -97.0, -93.0, -81.0,
    -80.0, -79.0, -62.0, -50.0, -48.0, -47.0, -42.0, -41.0, -40.0, -39.0,
    -38.0, -37.0, -36.0, -35.0, -34.0, -32.0, -31.0, -30.0, -29.0, -28.0,
    -27.0, -26.0, -25.0, -24.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0,
    -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -9.0, -8.0, -7.0, -6.0,
    -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 49.0, 62.0, 63.0
])

pattern_6 = np.array([
    -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0,
    3.0, 5.0, 230.0, 244.0, 245.0, 246.0
])

pattern_7 = np.array([
    -104.0, -103.0, -102.0, -58.0, -53.0, -50.0, -47.0, -46.0, -45.0, -43.0,
    -41.0, -40.0, -34.0, -33.0, -32.0, -31.0, -29.0, -26.0, -24.0, -23.0,
    -20.0, -19.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0,
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0,
    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0,
    14.0, 15.0, 28.0, 135.0, 149.0, 150.0, 151.0
])

pattern_8 = np.array([
    -76, -20, -13, -4, -3.0, -2, -1, 1, 2, 3,
    4.0, 5, 6.0, 7, 9, 10, 12, 13, 14, 17,
    20.0, 23, 29.0, 38.0, 39.0, 40.0, 60, 120, 127, 192.0,
    213.0, 226.0, 230.0, 233.0, 235.0, 236.0, 250.0, 251.0, 252.0
])

pattern_9 = np.array([
    -114, -64.0, -60.0, -59.0, -56, -45.0, -44.0, -42.0, -40.0, -38.0,
    -27.0, -26.0, -25.0, -24.0, -21.0, -20.0, -17, -15.0, -14, -13.0,
    -11.0, -8, -7, -6, -5, -4.0, -3, -2.0, -1.0, 1.0,
    2, 3.0, 4, 5.0, 6, 7, 8, 9.0, 11.0, 12.0,
    17, 36, 43.0, 61, 161.0, 164.0, 168.0, 171.0, 185.0, 186.0,
    187.0
])

pattern_10 = np.array([
    -108.0, -104.0, -92.0, -90.0, -87.0, -86.0, -81, -72.0, -69.0, -67.0,
    -66.0, -65.0, -62.0, -60.0, -57.0, -54.0, -51.0, -48.0, -36.0, -31.0,
    -28.0, -23.0, -20, -19.0, -17, -16.0, -12.0, -9.0, -7, -6.0,
    -5.0, -4, -3.0, -2, -1, 1, 2, 3, 4.0, 5,
    6, 7.0, 8, 9.0, 18, 33.0, 41, 93, 108, 119.0,
    123.0, 127.0, 130.0, 144.0, 145.0
])

PATTERNS = {
    0: pattern_0, 1: pattern_1, 2: pattern_2, 3: pattern_3, 4: pattern_4,
    5: pattern_5, 6: pattern_6, 7: pattern_7, 8: pattern_8, 9: pattern_9,
    10: pattern_10
}

adc_offsets_32_2 = np.array([-190,-15,-5,-2,2,63,150,245])

adc_offsets_w_r = np.array([
    -191., -190., -189., -130., -115., -114., -113., -108., -104.,
    -103., -102.,  -97.,  -93.,  -92.,  -90.,  -87.,  -86.,  -85.,
     -81.,  -80.,  -79.,  -72.,  -69.,  -67.,  -66.,  -65.,  -64.,
     -62.,  -60.,  -59.,  -58.,  -57.,  -56.,  -54.,  -53.,  -51.,
     -50.,  -48.,  -47.,  -46.,  -45.,  -44.,  -43.,  -42.,  -41.,
     -40.,  -39.,  -38.,  -37.,  -36.,  -35.,  -34.,  -33.,  -32.,
     -31.,  -30.,  -29.,  -28.,  -27.,  -26.,  -25.,  -24.,  -23.,
     -22.,  -21.,  -20.,  -19.,  -18.,  -17.,  -16.,  -15.,  -14.,
     -13.,  -12.,  -11.,  -10.,   -9.,   -8.,   -7.,   -6.,   -5.,
      -4.,   -3.,   -2.,   -1.,    0.,    1.,    2.,    3.,    4.,
       5.,    6.,    7.,    8.,    9.,   10.,   11.,   12.,   13.,
      14.,   15.,   17.,   20.,   23.,   25.,   28.,   29.,   33.,
      38.,   39.,   40.,   43.,   49.,   51.,   62.,   63.,   70.,
      71.,  108.,  115.,  119.,  123.,  127.,  130.,  135.,  144.,
     145.,  146.,  147.,  149.,  150.,  151.,  155.,  156.,  161.,
     164.,  167.,  168.,  170.,  171.,  185.,  186.,  187.,  191.,
     192.,  213.,  226.,  230.,  233.,  235.,  236.,  244.,  245.,
     246.,  250.,  251.,  252.
])

# ADC Offset values from 'merged_venn_final_r.txt'
adc_offsets_r = np.array([
    -191., -190., -189., -130., -115., -114., -113., -108., -104.,
    -103., -102.,  -97.,  -93.,  -92.,  -91.,  -90.,  -87.,  -86.,
     -83.,  -81.,  -80.,  -79.,  -72.,  -69.,  -67.,  -66.,  -65.,
     -64.,  -62.,  -61.,  -60.,  -59.,  -58.,  -57.,  -54.,  -53.,
     -51.,  -50.,  -48.,  -47.,  -46.,  -45.,  -44.,  -43.,  -42.,
     -41.,  -40.,  -39.,  -38.,  -37.,  -36.,  -35.,  -34.,  -33.,
     -32.,  -31.,  -30.,  -29.,  -28.,  -27.,  -26.,  -25.,  -24.,
     -23.,  -22.,  -21.,  -20.,  -19.,  -18.,  -17.,  -16.,  -15.,
     -14.,  -13.,  -12.,  -11.,  -10.,   -9.,   -8.,   -7.,   -6.,
      -5.,   -4.,   -3.,   -2.,   -1.,    0.,    1.,    2.,    3.,
       4.,    5.,    6.,    7.,    8.,    9.,   10.,   11.,   12.,
      13.,   14.,   15.,   16.,   20.,   22.,   23.,   25.,   28.,
      29.,   33.,   34.,   38.,   39.,   40.,   41.,   43.,   47.,
      49.,   53.,   62.,   63.,   70.,  101.,  108.,  110.,  117.,
     119.,  123.,  127.,  130.,  135.,  142.,  144.,  145.,  146.,
     149.,  150.,  151.,  155.,  161.,  164.,  168.,  171.,  185.,
     186.,  187.,  192.,  213.,  226.,  230.,  233.,  235.,  236.,
     244.,  245.,  246.,  250.,  251.,  252.
])

adc_offsets_32_2 = np.array([-190,-15,-5,-2,2,63,150,245])

adc_offsets_32_2 = adc_offsets_r

# LIF spike-count offset sweep tables
lif_spike_offsets = np.array([-30, -15, -10, -5, -2, 2, 5, 10, 15, 30])
LIF_PATTERNS = {pid: lif_spike_offsets for pid in PATTERNS}

# adc_offsets_8_2 = np.array([
#     -199, -198, -197, -137, -131, -130, -129, -128, -122, -121,
#     -120, -119, -110, -108, -103, -100,  -99,  -98,  -96,  -86,
#      -85,  -84,  -83,  -82,  -81,  -80,  -76,  -73,  -72,  -67,
#      -66,  -65,  -64,  -63,  -62,  -59,  -58,  -57,  -56,  -55,
#      -54,  -53,  -52,  -51,  -50,  -49,  -48,  -47,  -45,  -44,
#      -43,  -42,  -40,  -39,  -38,  -37,  -36,  -34,  -33,  -32,
#      -31,  -30,  -29,  -28,  -27,  -26,  -25,  -24,  -23,  -22,
#      -21,  -20,  -19,  -18,  -17,  -16,  -15,  -14,  -13,  -12,
#      -11,  -10,   -9,   -8,   -7,   -6,   -5,   -4,   -3,   -2,
#       -1,    0,    1,    2,    3,    4,    5,    6,    7,    8,
#        9,   10,   11,   12,   13,   14,   15,   16,   19,   21,
#       25,   26,   28,   29,   30,   31,   32,   34,   36,   37,
#       41,   42,   43,   44,   45,   46,   48,   53,   54,   55,
#       56,   76,   82,   84,   85,   86,   87,   88,   91,   93,
#       94,   97,  101,  104,  105,  106,  109,  118,  122,  123,
#      126,  129,  130,  131,  132,  133,  134,  135,  149,  174,
#      188,  189,  190,  214,  239,  253,  254,  255
# ])

# Helper: expand one layer name across all ADC offsets
def sweep(layer, adc_offset_input):
    return [[( layer, int(o))] for o in adc_offset_input]

def train_kws_pipeline(model, train_loader, val_loader):
    """Two-phase KWS training: FP warmup -> Ternary with AdamW, Grad Clip, Cosine LR, Label Smoothing."""
    # Inverse-frequency class weights to balance the GSC "unknown" class
    labels = torch.tensor(train_loader.dataset.file_labels)
    n_classes = labels.max().item() + 1
    counts = torch.bincount(labels, minlength=n_classes).float().clamp(min=1.0)
    class_weights = (counts.sum() / (n_classes * counts)).to(DEVICE)
    
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Phase 1: FP-warmup
    print("      [Warmup] Full-precision (AdamW, LR=1e-3)")
    model.set_mode('fp32')
    opt_fp = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_fp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fp, T_max=EPOCHS)
    
    for ep in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            opt_fp.zero_grad()
            loss = crit(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Grad clip
            opt_fp.step()
        sched_fp.step()
        
    # Phase 2: Ternary Phase
    print("      [Ternary] Quantized (AdamW, LR=5e-4)")
    model.set_mode('ternary')
    opt_t = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # Lower LR for ternary
    sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=EPOCHS)
    
    best_acc, patience_cnt = 0.0, 0
    best_state = None
    
    for ep in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            opt_t.zero_grad()
            loss = crit(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_t.step()
            
            # TWN weight clamp post-optimizer step stabilizes the STE active region
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, (RRAMConv2d, RRAMLinear)):
                        m.layer.weight.clamp_(-1.0, 1.0)
                        
        sched_t.step()
        _, val_acc, _ = evaluate_detailed(model, train_loader, val_loader, f"Ternary Ep {ep+1}", test_only=True)
        
        # Early stopping on val accuracy
        if val_acc > best_acc:
            best_acc, patience_cnt = val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= 5: 
                print(f"      -> Early stopping at epoch {ep+1}")
                break
                
    if best_state: model.load_state_dict(best_state)
    return evaluate_detailed(model, train_loader, val_loader, "Ternary Final")

def main():
    global INPUT_TARGETS 
    global WEIGHT_TARGETS
    global REVERSE_MAPPING
    global ATPG_STRING_MAPPING
    
    print("\n[System] Parsing ATPG strings into paired Input/Weight vectors...")
    INPUT_TARGETS = []
    WEIGHT_TARGETS = []
    ATPG_STRING_MAPPING = {}
    REVERSE_MAPPING = {}  
    
    # We will track uniqueness by the combined Input+Weight tuple
    flat_target_pool = [] 
    
    for s_idx, atpg_str in enumerate(ATPG_STRINGS[0]):
        print(f"   -> Processing ATPG String {s_idx}: {atpg_str}")
        
        parsed_inps, parsed_wgts = parse_multiple_atpg_pairs([[atpg_str]]) 
        shifted_pairs = generate_shifted_pairs(parsed_inps, parsed_wgts, enable_shifts=USE_SHIFTED_TARGETS)
        flat_inps, flat_wgts = flatten_and_check_global_pair_duplicates(shifted_pairs)
        
        target_indices = []
        for v_idx in range(len(flat_inps)):
            combined_tuple = (tuple(flat_inps[v_idx]), tuple(flat_wgts[v_idx]))
            
            found_idx = -1
            for i, existing_pair in enumerate(flat_target_pool):
                if existing_pair == combined_tuple:
                    found_idx = i
                    break
                    
            if found_idx == -1:
                flat_target_pool.append(combined_tuple)
                INPUT_TARGETS.append(flat_inps[v_idx])
                WEIGHT_TARGETS.append(flat_wgts[v_idx])
                found_idx = len(flat_target_pool) - 1
                
            if found_idx not in target_indices:
                target_indices.append(found_idx)
                
        ATPG_STRING_MAPPING[atpg_str] = target_indices
        print(f"      Mapped to {len(target_indices)} unique binary pairs.")

    # INPUT_TARGETS = flat_target_pool
    print(f"         -> Total unique global binary targets to simulate: {len(INPUT_TARGETS)}\n")
    print("target indices for each ATPG string:",INPUT_TARGETS)
    
    # global ATPG_STRING_MAPPING
    atpg_keys = list(ATPG_STRING_MAPPING.keys()) # Defined before it's used
    
    # 1. Map each target index to a list of ALL matching string indices
    for s_idx, atpg_str in enumerate(atpg_keys):
        for t_idx in ATPG_STRING_MAPPING[atpg_str]:
            if t_idx not in REVERSE_MAPPING:
                REVERSE_MAPPING[t_idx] = []
            REVERSE_MAPPING[t_idx].append(str(s_idx))

    # 2. Convert the lists into joined strings (e.g., "0_11_13") 
    # so the rest of the script's filename generators work untouched.
    for t_idx in REVERSE_MAPPING:
        REVERSE_MAPPING[t_idx] = "_".join(REVERSE_MAPPING[t_idx])

    # INPUT_TARGETS = [
    # [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # print("final input targets are => ",INPUT_TARGETS) 
    print(f"         -> Successfully generated {len(INPUT_TARGETS)} {type(INPUT_TARGETS)} unique 32-element targets.\n")
    # ---------------------------------------
        
    model_name_tag = MODEL_ARCH if DATASET != 'KWS' else ('MFCC_CNN' if KWS_FEATURE_TYPE == 'MFCC' else 'RawAudioCNN')
    print(f"--- RRAM SIMULATION: {model_name_tag} ---")
    print(f"--- Config: Layers={MLP_NUM_LAYERS}, BN={USE_BATCH_NORM}, FullVal={VALIDATE_ON_FULL_DATASET}, Faults={RUN_FAULT_INJECTION} ---")
    target_kw = "yes" # Set the same keyword as tinysns
    
    if DATASET == 'KWS':
        print("\n[System] Loading SpeechCommands Dataset...")
        data_dir = r'C:\Users\manos\Downloads'
        
        txt_path = os.path.join(data_dir, "testing_list.txt")
        if not os.path.exists(data_dir) or not os.path.exists(txt_path):
            print("   -> Missing core dataset files. Downloading SpeechCommands v0.02 (This might take a moment)...")
            os.makedirs(data_dir, exist_ok=True)
            tar_path = os.path.join(data_dir, "speech_commands_v0.02.tar.gz")
            urllib.request.urlretrieve("http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", tar_path)
            print("   -> Extracting (Please do not interrupt)...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            os.remove(tar_path)
            print("   -> Extraction complete!")

        train_set = GSCDataset(data_dir, target_keyword=target_kw, split='train', num_classes=KWS_NUM_CLASSES)
        test_set = GSCDataset(data_dir, target_keyword=target_kw, split='test', num_classes=KWS_NUM_CLASSES)
        
        print(f"   -> Found {len(train_set)} train files, {len(test_set)} test files.")
        
        collate_fn = KWSCollate(feature_type=KWS_FEATURE_TYPE)
        train_bs, test_bs = 256, 1024
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs, shuffle=True, collate_fn=collate_fn, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False, collate_fn=collate_fn, drop_last=False)
    else:
        if MODEL_ARCH == 'LeNet5':
            transform = transforms.Compose([
                transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) 
            ])
            train_bs, test_bs = 1024, 1024
        else:
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) 
            ])
            train_bs, test_bs = 256, 1024

        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False)
    
    # print("\n[Phase 1] Training Logic (Baseline, Pure FP32)")
    # # 1. Initialize model and set to standard full-precision mode
    # model_fp32 = get_model()
    # # Assuming your model has a set_mode toggle. If FP32 is the default, 
    # # you might not even need this next line!
    # model_fp32.set_mode('fp32') 
    
    # # 2. Setup standard Adam optimizer and Cross Entropy Loss
    # opt_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)
    # crit_fp32 = nn.CrossEntropyLoss()
    
    # # 3. Standard FP32 Training Loop
    # for epoch in range(EPOCHS):
    #     loss, acc = train_one_epoch(model_fp32, train_loader, opt_fp32, crit_fp32)
    #     print(f"   Epoch {epoch+1}: Train Acc {acc:.2f}%")
        
    # # 4. Evaluate using the exact same metrics
    # acc_fp32_tr, acc_fp32_te, acc_fp32_full = evaluate_detailed(
    #     model_fp32, 
    #     train_loader, 
    #     test_loader, 
    #     "FP32 Baseline"
    # )

    print("\n[Phase 2] Training Logic (Dynamic Scale, Pure Ternary)")
    model_tern = get_model()
    
    if DATASET == 'KWS':
        # Use the two-phase pipeline specifically for speech tasks
        acc_tern_tr, acc_tern_te, acc_tern_full = train_kws_pipeline(model_tern, train_loader, test_loader)
    else:
        # Preserve original MNIST execution flow
        model_tern.set_mode('ternary')
        opt = optim.Adam(model_tern.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            loss, acc = train_one_epoch(model_tern, train_loader, opt, crit)
            print(f"   Epoch {epoch+1}: Train Acc {acc:.2f}%")
        acc_tern_tr, acc_tern_te, acc_tern_full = evaluate_detailed(model_tern, train_loader, test_loader, "Ternary Software")

    print("\n[Phase 3] RRAM Mapped Baseline (Dynamic Scale)")
    model_rram = model_tern; model_rram.set_mode('rram')
    acc_rram_tr, acc_rram_te, acc_rram_full = evaluate_detailed(model_rram, train_loader, test_loader, "RRAM Clean")
    
    log_mapped_weights(model_rram)

    # ==========================================
    # Branch: LIF readout vs ADC readout
    # ==========================================
    is_lif = (READOUT_MODE == 'LIF'
              and DATASET == 'KWS'
              and MODEL_ARCH == 'PureLinear'
              and KWS_FEATURE_TYPE == 'TINYSNS')

    if is_lif:
        #  LIF PATH 
        print("\n" + "=" * 70)
        print("  LIF READOUT MODE ? all remaining phases use SNN spike-count domain")
        print("=" * 70)

        # Phase 3-LIF: clean LIF evaluation
        acc_lif_clean = evaluate_lif(
            model_rram, test_loader, LIF_THRESHOLDS,
            LIF_TIME_STEPS, LIF_LEAK, n_max=2000, desc="LIF Clean")

        # Phase 3.5-LIF: target generation
        if RUN_TARGET_GENERATION:
            check_and_generate_all_16_cases_lif(model_rram, train_loader, test_loader)
        else:
            print("\n[Phase 3.5-LIF] SKIPPED (Target Generation)")

        # Phase 4-LIF: fault injection + masking
        fault_results = []
        aggregated_results = {}  # populated only on SYNTHETIC eval path
        if RUN_FAULT_INJECTION:
            print(f"\n[Phase 4-LIF] Fault Injection Sensitivity Analysis (Mode: {EVAL_MODE})")

            # Sweep spike offsets per hidden layer (parity with the ADC sweep over adc_offsets_32_2).
            LIF_EXPERIMENTS = [
                *sweep("fc1", lif_spike_offsets),
                *sweep("fc2", lif_spike_offsets),
                *sweep("fc3", lif_spike_offsets),
            ]

            # --- SYNTHETIC EVALUATION BRANCH (LIF analog of the ADC version) ---
            if EVAL_MODE == 'SYNTHETIC':
                print(f"   -> Loading Synthetic LIF images from: {SYNTHETIC_LOG_DIR}")
                if not os.path.exists(SYNTHETIC_LOG_DIR):
                    print("   -> [Error] Synthetic log directory not found.")
                else:
                    import re
                    experiment_groups = {}
                    pass_files = {}
                    fail_counts = {}

                    for file_name in os.listdir(SYNTHETIC_LOG_DIR):
                        # Only consume LIF-prefixed synthetic files
                        if not file_name.startswith('lif_'):
                            continue
                        if file_name.endswith('.csv'):
                            match = re.search(r'(PASS|FAIL)_[A-Za-z]+_(.+?)_String([0-9_]+)_Shift(\d+)', file_name)
                            if match:
                                status     = match.group(1)
                                layer_name = match.group(2)
                                string_ids = [int(sid) for sid in match.group(3).split('_')]
                                t_idx      = int(match.group(4))

                                for s_id in string_ids:
                                    if s_id in LIF_PATTERNS:
                                        key = (layer_name, s_id, t_idx)
                                        if status == 'FAIL':
                                            fail_counts[key] = fail_counts.get(key, 0) + 1
                                        elif status == 'PASS':
                                            if key not in pass_files: pass_files[key] = []
                                            pass_files[key].append(os.path.join(SYNTHETIC_LOG_DIR, file_name))

                    for key, p_files in pass_files.items():
                        if fail_counts.get(key, 0) == 0:
                            experiment_groups[key] = p_files

                    if not experiment_groups:
                        print("   -> [Error] No valid combinations met the strictly zero-FAIL criteria.")
                    else:
                        for (layer_name, pattern_id, t_idx), file_paths in experiment_groups.items():
                            print(f"\n   -> Running tasks for Layer: {layer_name} | Shift ID: {t_idx} | Files: {len(file_paths)}")

                            synth_dataset = SyntheticPassDataset(file_paths=file_paths)
                            synth_loader  = torch.utils.data.DataLoader(synth_dataset, batch_size=256, shuffle=False)

                            # --- INJECT UNIQUE TARGET WEIGHTS BEFORE TESTING ---
                            model_rram.restore_all_pristine_weights()
                            model_rram.inject_target_weights(layer_name, WEIGHT_TARGETS[t_idx])

                            offsets = LIF_PATTERNS[pattern_id]
                            for offset in offsets:
                                case_config = [(layer_name, int(offset))]
                                case_desc   = f"LIF SPIKE OFFSET | {layer_name}:{int(offset)} (Pattern {pattern_id})"

                                defect_acc, mismatch_rate, total_imgs = evaluate_synthetic_faults_lif(
                                    model_rram, synth_loader, case_config,
                                    thresholds=LIF_THRESHOLDS,
                                    time_steps=LIF_TIME_STEPS,
                                    leak=LIF_LEAK)
                                fault_results.append((case_desc, 0.0, defect_acc, mismatch_rate))

                                agg_key = (layer_name, int(offset))
                                if agg_key not in aggregated_results:
                                    aggregated_results[agg_key] = {'matches': 0, 'mismatches': 0, 'total': 0}

                                aggregated_results[agg_key]['matches']    += round((defect_acc   / 100.0) * total_imgs)
                                aggregated_results[agg_key]['mismatches'] += round((mismatch_rate / 100.0) * total_imgs)
                                aggregated_results[agg_key]['total']      += total_imgs

                                # Masking analysis with the same spike fault
                                spike_faults = {ln: int(off) for ln, off in case_config}
                                run_lif_masking_analysis(model_rram, synth_loader, case_desc, spike_faults)

                        model_rram.reset_all_faults()
                        model_rram.restore_all_pristine_weights()

            # --- STANDARD DATASET EVALUATION BRANCH ---
            else:
                for i, case_config in enumerate(LIF_EXPERIMENTS):
                    model_rram.reset_all_faults()
                    spike_faults = {}
                    parts        = []
                    for layer_name, offset in case_config:
                        spike_faults[layer_name] = int(offset)
                        parts.append(f"{layer_name}:{int(offset):+d}")
                    case_desc = "LIF SPIKE OFFSET | " + " & ".join(parts)
                    print(f"--- LIF Exp {i+1}: {case_desc} ---")

                    acc = evaluate_lif(
                        model_rram, test_loader, LIF_THRESHOLDS,
                        LIF_TIME_STEPS, LIF_LEAK,
                        spike_faults=spike_faults,
                        n_max=2000, desc=f"LIF Fault {i+1}")
                    fault_results.append((case_desc, 0.0, acc, 0.0))

                    run_lif_masking_analysis(model_rram, test_loader, case_desc, spike_faults)
        else:
            print("\n[Phase 4-LIF] SKIPPED (Fault Injection)")

        # Summary
        print("\n" + "=" * 85)
        if EVAL_MODE == 'SYNTHETIC' and RUN_FAULT_INJECTION and aggregated_results:
            print(f"FINAL AGGREGATED SUMMARY (LIF READOUT): {model_name_tag} (Synthetic PASS Fault Resilience)")
            print("=" * 85)
            print(f"{'LAYER:OFFSET':<20} | {'OVERALL DEFECT ACC (Match)':<28} | {'OVERALL MISMATCH':<20} | {'TOTAL TESTED'}")
            print("-" * 90)
            for (layer, offset) in sorted(aggregated_results.keys(), key=lambda x: (x[0], float(x[1]))):
                data = aggregated_results[(layer, offset)]
                tot  = data['total']
                if tot > 0:
                    overall_defect_acc = 100.0 * data['matches'] / tot
                    overall_mismatch   = 100.0 * data['mismatches'] / tot
                else:
                    overall_defect_acc = 0.0
                    overall_mismatch   = 0.0
                row_desc = f"{layer}:{offset}"
                print(f"{row_desc:<20} | {overall_defect_acc:>15.2f}%{' ':>12} | {overall_mismatch:>14.2f}%{' ':>5} | {tot:>10}")
        else:
            print(f"FINAL SUMMARY (LIF READOUT): {model_name_tag}")
            print("=" * 85)
            print(f"{'EXPERIMENT':<50} | {'TEST':<10}")
            print("-" * 65)
            print(f"{'2. Ternary (Soft)':<50} | {acc_tern_te:.2f}%")
            print(f"{'3. RRAM ADC (Clean)':<50} | {acc_rram_te:.2f}%")
            print(f"{'3. RRAM LIF (Clean)':<50} | {acc_lif_clean:.2f}%")
            print("-" * 65)
            for desc, _, te, _ in fault_results:
                print(f"{desc:<50} | {te:.2f}%")

    else:
        # ==============================================================
        #  ADC PATH: original code, completely unchanged
        # ==============================================================
        if RUN_TARGET_GENERATION: check_and_generate_all_16_cases(model_rram, train_loader, test_loader)
        else: print("\n[Phase 3.5] SKIPPED (Target Generation)")
        
        fault_results = [] 
        if RUN_FAULT_INJECTION:
            print(f"\n[Phase 4] Fault Injection Sensitivity Analysis (Mode: {EVAL_MODE})")
            EXPERIMENTS = []
            
            if DATASET == 'KWS':
                if MODEL_ARCH == 'PureLinear':
                    EXPERIMENTS = [
                         [("fc1", 2)], 
                         [("fc2", -2)], [("fc2", 2)], [("fc2", 5)], [("fc2", 10)], 
                         [("fc3", 2)], [("fc4", 2)]
                     ]
                elif MODEL_ARCH == 'BinaryCNN': 
                    EXPERIMENTS = [
                        [("conv1", 2)], [("conv2", 2)], 
                        [("fc1", -2)], [("fc1", 2)], [("fc1", 5)],
                        [("fc2", 2)]
                    ]
                else: 
                    EXPERIMENTS = [
                        [("conv1", 2)], [("conv2", 2)], 
                        [("fc1", -2)], [("fc1", 2)], [("fc1", 5)],
                        [("fc2", 2)]
                    ]
            elif MODEL_ARCH == 'LeNet5':
                #  EXPERIMENTS = [
                #     [("conv1", 2)],          
                #     *sweep("conv2",adc_offsets_32_2),          
                #     [("fc1",   2)],          
                #     *sweep("fc2",adc_offsets_32_2),            
                #     [("fc3",   2)],          
                # ]
                EXPERIMENTS = [
                    *sweep("conv1",adc_offsets_32_2),         
                    *sweep("conv2",adc_offsets_32_2),          
                    *sweep("fc1",adc_offsets_32_2),          
                    *sweep("fc2",adc_offsets_32_2),            
                    *sweep("fc3",adc_offsets_32_2),        
                ]
            elif MODEL_ARCH == 'PureLinear':
                #  EXPERIMENTS = [
                #     [("fc1", 2)],            
                #     *sweep("fc2", adc_offsets_32_2),            
                #     [("fc3", 2)],            
                #     [("fc4", 2)],            
                # ]
                EXPERIMENTS = [
                    *sweep("fc1", adc_offsets_32_2),            
                    *sweep("fc2", adc_offsets_32_2),            
                    *sweep("fc3", adc_offsets_32_2),          
                    *sweep("fc4", adc_offsets_32_2),           
                ]
            else:
                if MLP_NUM_LAYERS == 3: EXPERIMENTS = [[("fc1", 2)], [("fc2", -2)], [("fc2", 2)], [("fc2", 5)], [("fc2", 10)], [("fc3", 2)]]
                elif MLP_NUM_LAYERS == 4: EXPERIMENTS = [[("fc1", 2)], [("fc2", -2)], [("fc2", 2)], [("fc2", 5)], [("fc2", 10)], [("fc3", 2)], [("fc4", 2)]]

            # --- SYNTHETIC EVALUATION BRANCH ---
            if EVAL_MODE == 'SYNTHETIC':
                print(f"   -> Loading Synthetic images from: {SYNTHETIC_LOG_DIR}")
                
                if not os.path.exists(SYNTHETIC_LOG_DIR):
                    print("   -> [Error] Synthetic log directory not found.")
                else:
                    import re
                    experiment_groups = {}
                    pass_files = {}
                    fail_counts = {}
                    
                    for file_name in os.listdir(SYNTHETIC_LOG_DIR):
                        if file_name.endswith('.csv'):
                            # Capture specific shift index (t_idx) so weights can be injected perfectly
                            match = re.search(r'(PASS|FAIL)_[A-Za-z]+_(.+?)_String([0-9_]+)_Shift(\d+)', file_name)
                            if match:
                                status = match.group(1)
                                layer_name = match.group(2)
                                string_ids = [int(sid) for sid in match.group(3).split('_')]
                                t_idx = int(match.group(4)) # Extracted Target Index
                                
                                for s_id in string_ids:
                                    if s_id in PATTERNS:
                                        # Key now isolates by specific Shift index!
                                        key = (layer_name, s_id, t_idx) 
                                        
                                        if status == 'FAIL': fail_counts[key] = fail_counts.get(key, 0) + 1
                                        elif status == 'PASS':
                                            if key not in pass_files: pass_files[key] = []
                                            pass_files[key].append(os.path.join(SYNTHETIC_LOG_DIR, file_name))
                                            
                    for key, p_files in pass_files.items():
                        layer_name, pattern_id, t_idx = key
                        if fail_counts.get(key, 0) == 0:
                            experiment_groups[key] = p_files
                                        
                    if not experiment_groups:
                        print("   -> [Error] No valid combinations met the strictly zero-FAIL criteria.")
                    else:
                        aggregated_results = {}
                        
                        # Process by exact Layer + Pattern ID + Target Shift
                        for (layer_name, pattern_id, t_idx), file_paths in experiment_groups.items():
                            print(f"\n   -> Running tasks for Layer: {layer_name} | Shift ID: {t_idx} | Files: {len(file_paths)}")
                            
                            synth_dataset = SyntheticPassDataset(file_paths=file_paths)
                            synth_loader = torch.utils.data.DataLoader(synth_dataset, batch_size=256, shuffle=False)
                            
                            # --- INJECT UNIQUE TARGET WEIGHTS BEFORE TESTING ---
                            model_rram.restore_all_pristine_weights()
                            model_rram.inject_target_weights(layer_name, WEIGHT_TARGETS[t_idx])
                            
                            offsets = PATTERNS[pattern_id]
                            for offset in offsets:
                                case_config = [(layer_name, offset)]
                                case_desc = f"OFFSET | {layer_name}:{offset} (Pattern {pattern_id})"
                                
                                defect_acc, mismatch_rate, total_imgs = evaluate_synthetic_faults(model_rram, synth_loader, case_config)
                                fault_results.append((case_desc, 0.0, defect_acc, mismatch_rate))
                                
                                agg_key = (layer_name, offset) # Aggregate purely by offset
                                if agg_key not in aggregated_results:
                                    aggregated_results[agg_key] = {'matches': 0, 'mismatches': 0, 'total': 0}
                                
                                aggregated_results[agg_key]['matches'] += round((defect_acc / 100.0) * total_imgs)
                                aggregated_results[agg_key]['mismatches'] += round((mismatch_rate / 100.0) * total_imgs)
                                aggregated_results[agg_key]['total'] += total_imgs
                                
                                for l_name, param in case_config:
                                    model_rram.configure_faults('offset', (param,), layer_name=l_name)
                                run_masking_analysis(model_rram, synth_loader, case_desc)
                        model_rram.reset_all_faults()
                        model_rram.restore_all_pristine_weights()

            # --- STANDARD DATASET EVALUATION BRANCH ---
            else:
                for i, case_config in enumerate(EXPERIMENTS):
                    model_rram.reset_all_faults()
                    case_desc_parts = []
                    for layer_name, param in case_config:
                        vals = (param,); desc_val = str(param)
                        model_rram.configure_faults('offset', vals, layer_name=layer_name)
                        case_desc_parts.append(f"{layer_name}:{desc_val}")
                    case_desc = f"OFFSET | " + " & ".join(case_desc_parts)
                    print(f"--- Exp {i+1}: {case_desc} ---")
                    
                    use_test_only = not VALIDATE_ON_FULL_DATASET
                    tr, te, full = evaluate_detailed(model_rram, train_loader, test_loader, case_desc, test_only=use_test_only)
                    fault_results.append((case_desc, tr, te, full))
                    
                    run_masking_analysis(model_rram, test_loader, case_desc)
        else:
            print(f"\n[Phase 4] SKIPPED (Fault Injection)")

        print("\n=========================================================================================")
        print("\n=========================================================================================")
        if EVAL_MODE == 'SYNTHETIC':
            print(f"FINAL AGGREGATED SUMMARY: {model_name_tag} (Synthetic PASS Fault Resilience)")
            print("=========================================================================================")
            print(f"{'LAYER:OFFSET':<20} | {'OVERALL DEFECT ACC (Match)':<28} | {'OVERALL MISMATCH':<20} | {'TOTAL TESTED'}")
            print("-" * 90)
            
            # Sort by layer name, then numerically by offset value
            for (layer, offset) in sorted(aggregated_results.keys(), key=lambda x: (x[0], float(x[1]))):
                data = aggregated_results[(layer, offset)]
                tot = data['total']
                if tot > 0:
                    overall_defect_acc = 100.0 * data['matches'] / tot
                    overall_mismatch = 100.0 * data['mismatches'] / tot
                else:
                    overall_defect_acc = 0.0
                    overall_mismatch = 0.0
                    
                row_desc = f"{layer}:{offset}"
                # Format exactly as requested: layer:offset | x% | y%
                print(f"{row_desc:<20} | {overall_defect_acc:>15.2f}%{' ':>12} | {overall_mismatch:>14.2f}%{' ':>5} | {tot:>10}")
        else:
            print(f"FINAL SUMMARY: {model_name_tag} (Baseline vs Faults)")
            print("=========================================================================================")
            print(f"{'EXPERIMENT':<45} | {'TRAIN':<10} | {'TEST':<10} | {'FULL':<10}")
            print("-" * 85)
            print(f"{'2. Ternary (Soft)':<45} | {acc_tern_tr:.2f}%     | {acc_tern_te:.2f}%     | {acc_tern_full:.2f}%")
            print(f"{'3. RRAM (Clean)':<45} | {acc_rram_tr:.2f}%     | {acc_rram_te:.2f}%     | {acc_rram_full:.2f}%")
            print("-" * 85)
            for desc, tr, te, full in fault_results:
                tr_str = f"{tr:.2f}%" if tr > 0 else "N/A"
                full_str = f"{full:.2f}%" if full > 0 else "N/A"
                print(f"{desc:<45} | {tr_str:<10} | {te:.2f}%     | {full_str:<10}")

if __name__ == "__main__":
    main()