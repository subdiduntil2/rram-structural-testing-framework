import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DEVICE, DATASET, MODEL_ARCH, KWS_FEATURE_TYPE, KWS_NUM_CLASSES,
    MLP_NUM_LAYERS, USE_BATCH_NORM, LOG_DIR,
)
from simulator_core import ANALYZER
from mapping import RRAMConv2d, RRAMLinear


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
