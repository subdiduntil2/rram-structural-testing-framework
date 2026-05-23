import torch
import torch.nn as nn
import torch.nn.functional as F

from simulator_core import (
    SimulatorConfig,
    ANALYZER,
    TernaryWeightFn,
    InputQuantizerFn,
    LeakyClampFn,
    OutputClampFn,
)


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
