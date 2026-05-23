import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from config import DEVICE, LOG_DIR, INPUT_MIN, INPUT_MAX, MODEL_ARCH
from simulator_core import SimulatorConfig, TernaryWeightFn
from mapping import RRAMConv2d, RRAMLinear
from models import GRADOPT_PRE_LAYER_LEAKY
from data import save_target_data


def check_and_generate_all_16_cases(model, train_loader, test_loader):
    print("\n[Phase 3.5] Target Generation Pipeline (Det -> MNIST -> GradOpt)")
    model.eval()
    
    target_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)) and m.tile_map is not None:
            target_layers.append(m.name)
            for t_idx in range(len(config.INPUT_TARGETS)):
                config.TARGET_CASE_REGISTRY[(m.name, t_idx)] = "MISS"

    hits = {(layer_name, t_idx): False for layer_name in target_layers for t_idx in range(len(config.INPUT_TARGETS))}
    closest_matches = {(layer_name, t_idx): {'count': -1, 'img': None} for layer_name in target_layers for t_idx in range(len(config.INPUT_TARGETS))}
    
    # Grab a single template image for shaping the Deterministic Inversion
    template_img, _ = next(iter(test_loader))
    template_img = template_img[0:1].to(DEVICE) 

    # ==============================================================
    # STEP 1: DETERMINISTIC INVERSION (Math First)
    # ==============================================================
    print("   [STEP 1] Running Deterministic Inversion...")
    for layer_name in target_layers:
        for t_idx, t_in_list in enumerate(config.INPUT_TARGETS):
            
            # --- INJECT UNIQUE TARGET WEIGHTS ---
            model.restore_all_pristine_weights()
            model.inject_target_weights(layer_name, config.WEIGHT_TARGETS[t_idx])
            
            success, det_img, match_count = attempt_deterministic_inversion(model, layer_name, t_in_list, template_img)
            
            if success:
                hits[(layer_name, t_idx)] = True
                config.TARGET_CASE_REGISTRY[(layer_name, t_idx)] = "PASS (Det)"
                string_id = config.REVERSE_MAPPING.get(t_idx, "Unknown")
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
            for t_idx, t_in_list in enumerate(config.INPUT_TARGETS):
                if hits[(layer_name, t_idx)]: continue 
                
                # --- INJECT UNIQUE TARGET WEIGHTS ---
                model.restore_all_pristine_weights()
                model.inject_target_weights(layer_name, config.WEIGHT_TARGETS[t_idx])
                
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
                                    config.TARGET_CASE_REGISTRY[(layer_name, t_idx)] = "PASS (MNIST)"
                                    img_tensor = data[img_idx].clone()
                                    
                                    string_id = config.REVERSE_MAPPING.get(t_idx, "Unknown")
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
        for t_idx, t_in_list in enumerate(config.INPUT_TARGETS):
            if not hits[(layer_name, t_idx)]:
                best_count = closest_matches[(layer_name, t_idx)]['count']
                best_img = closest_matches[(layer_name, t_idx)]['img']
                
                if best_img is None:
                    config.TARGET_CASE_REGISTRY[(layer_name, t_idx)] = "SKIPPED (No Seed)"
                    continue
                
                # --- INJECT UNIQUE TARGET WEIGHTS ---
                model.restore_all_pristine_weights()
                model.inject_target_weights(layer_name, config.WEIGHT_TARGETS[t_idx])
                
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
    atpg_keys = list(config.ATPG_STRING_MAPPING.keys())
    target_headers = " | ".join([f"Str {i}".center(10) for i in range(len(atpg_keys))])
    print(f"{'Layer':<10} | {target_headers}")
    print("-" * (13 + 13 * len(atpg_keys)))
    
    layers = sorted(list(set(k[0] for k in config.TARGET_CASE_REGISTRY.keys())))
    for layer in layers:
        row_str = f"{layer:<10}"
        for s_idx, atpg_str in enumerate(atpg_keys):
            mapped_indices = config.ATPG_STRING_MAPPING[atpg_str]
            
            # Fetch the status for all binary vectors tied to this string
            statuses = [config.TARGET_CASE_REGISTRY.get((layer, t_idx), "MISS") for t_idx in mapped_indices]
            
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
        config.TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = "INFEASIBLE_RELU"
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
 
    config.TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = f"{final_outcome} (GradOpt)"
    string_id = config.REVERSE_MAPPING.get(target_idx, "Unknown")
    base_fname = os.path.join(
        LOG_DIR, f"synth_{final_outcome}_GradOpt_{target_layer_name}_String{string_id}_Shift{target_idx}"
    )
    save_target_data(best_img, base_fname)
