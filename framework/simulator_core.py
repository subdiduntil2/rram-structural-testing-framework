import torch


class SimulatorConfig:
    XB_SIZE = 32
    DAC_MIN = -15
    DAC_MAX = 15
    ADC_MIN = -128
    ADC_MAX = 127

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
