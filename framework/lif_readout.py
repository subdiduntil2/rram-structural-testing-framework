import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from config import (
    DEVICE, LOG_DIR, INPUT_MIN, INPUT_MAX, MODEL_ARCH, ANALYSIS_IMAGE_LIMIT,
    LIF_THRESHOLDS, LIF_TIME_STEPS, LIF_LEAK, LIF_SPIKE_BINARIZE_THRESH,
)
from simulator_core import SimulatorConfig, TernaryWeightFn
from mapping import RRAMConv2d, RRAMLinear
from models import GRADOPT_PRE_LAYER_LEAKY
from data import save_target_data
from target_gen import multibit_loss


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


# ---- LIF target generation (Det -> Dataset -> GradOpt) ---------------------

def check_and_generate_all_16_cases_lif(model, train_loader, test_loader):
    """Spike-count-domain target generation.
    """
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
        for ti in range(len(config.INPUT_TARGETS)):
            config.TARGET_CASE_REGISTRY[(ln, ti)] = "MISS"

    hits    = {(ln, ti): False for ln in target_layer_names for ti in range(len(config.INPUT_TARGETS))}
    closest = {(ln, ti): {'count': -1, 'img': None}
               for ln in target_layer_names for ti in range(len(config.INPUT_TARGETS))}

    template_img, _ = next(iter(test_loader))
    template_img = template_img[0:1].to(DEVICE)

    # === STEP 1  Deterministic inversion ===
    print("   [STEP 1-LIF] Deterministic pseudo-inverse ...")
    for li, ln in enumerate(target_layer_names):
        for ti, t_in in enumerate(config.INPUT_TARGETS):
            if hits[(ln, ti)]: continue
            # --- INJECT UNIQUE TARGET WEIGHTS (parity with ADC pipeline) ---
            model.restore_all_pristine_weights()
            model.inject_target_weights(ln, config.WEIGHT_TARGETS[ti])
            ok, img, mc = _det_inv_lif(model, rram_layers, th_list, li, t_in,
                                       template_img, T, bth)
            if ok:
                hits[(ln, ti)] = True
                config.TARGET_CASE_REGISTRY[(ln, ti)] = "PASS (Det-LIF)"
                string_id = config.REVERSE_MAPPING.get(ti, "Unknown")
                save_target_data(img, os.path.join(LOG_DIR, f"lif_PASS_Det_{ln}_String{string_id}_Shift{ti}"))
            elif img is not None and mc > closest[(ln, ti)]['count']:
                closest[(ln, ti)] = {'count': mc, 'img': img.cpu()}

    # === STEP 2  Dataset search ===
    print("\n   [STEP 2-LIF] Dataset spike-count scan ...")
    with torch.no_grad():
        for li, ln in enumerate(target_layer_names):
            for ti, t_in in enumerate(config.INPUT_TARGETS):
                if hits[(ln, ti)]: continue

                # --- INJECT UNIQUE TARGET WEIGHTS (parity with ADC pipeline) ---
                model.restore_all_pristine_weights()
                model.inject_target_weights(ln, config.WEIGHT_TARGETS[ti])
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
                                config.TARGET_CASE_REGISTRY[(ln, ti)] = "PASS (Dataset-LIF)"
                                string_id = config.REVERSE_MAPPING.get(ti, "Unknown")
                                save_target_data(data[ii], os.path.join(LOG_DIR, f"lif_PASS_Dataset_{ln}_String{string_id}_Shift{ti}"))
                                done = True
                                break
                            elif mc > closest[(ln, ti)]['count']:
                                closest[(ln, ti)] = {'count': mc,
                                                     'img': data[ii:ii+1].cpu()}

    # === STEP 3  Gradient optimisation ===
    print("\n   [STEP 3-LIF] Gradient optimisation for remaining MISSES ...")
    for li, ln in enumerate(target_layer_names):
        for ti, t_in in enumerate(config.INPUT_TARGETS):
            if hits[(ln, ti)]: continue
            seed = closest[(ln, ti)]['img']
            cnt  = closest[(ln, ti)]['count']
            if seed is None:
                config.TARGET_CASE_REGISTRY[(ln, ti)] = "SKIPPED (No Seed)"
                continue
            # --- INJECT UNIQUE TARGET WEIGHTS (parity with ADC pipeline) ---
            model.restore_all_pristine_weights()
            model.inject_target_weights(ln, config.WEIGHT_TARGETS[ti])
            _grad_opt_lif(model, rram_layers, th_list, li, t_in, ti, seed, cnt, T, bth)

    # Restore pristine weights so that subsequent fault-injection phases
    model.restore_all_pristine_weights()
    print("   [System] Restored pristine hardware/software weights for normal operations.")

    # === summary table ===
    print("\n" + "=" * 80)
    print("LIF TARGET GENERATION SUMMARY")
    print("=" * 80)
    hdr = " | ".join([f"Tgt {i}".center(15) for i in range(len(config.INPUT_TARGETS))])
    print(f"{'Layer':<10} | {hdr}")
    print("-" * (13 + 18 * len(config.INPUT_TARGETS)))
    for ln in sorted(set(k[0] for k in config.TARGET_CASE_REGISTRY)):
        row = f"{ln:<10}"
        for ti in range(len(config.INPUT_TARGETS)):
            row += f" | {config.TARGET_CASE_REGISTRY.get((ln,ti),'N/A'):^15}"
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
    target_layer_name = rram_layers[target_li].name
    seed_img = seed_img.to(DEVICE)
 
    target_layer = None
    for _, m in model.named_modules():
        if isinstance(m, (RRAMConv2d, RRAMLinear)) and m.name == target_layer_name:
            target_layer = m
            break
    if target_layer is None:
        config.TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = "SKIPPED"
        return
 
    target_rows = getattr(target_layer, 'active_defect_rows', {0})
    row_idx = list(target_rows)[0]
    start = row_idx * 32
    target_in_tensor_full = torch.tensor(target_in, dtype=torch.int32, device=DEVICE)

    # Tier 1.1 -- ReLU-fed layers cannot accept negative inputs.
    RELU_FED_LAYERS = {"fc2", "fc3", "fc4"}
    _active_len_check = min(32, len(target_in))
    if target_layer_name in RELU_FED_LAYERS and any(v < 0 for v in target_in[:_active_len_check]):
        config.TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = "INFEASIBLE_RELU"
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

    config.TARGET_CASE_REGISTRY[(target_layer_name, target_idx)] = f"{final_outcome} (GradOpt-LIF)"
    string_id = config.REVERSE_MAPPING.get(target_idx, "Unknown")
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
