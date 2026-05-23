#!/usr/bin/env python3
"""Ma et al., "Efficient Low Cost Alternative Testing of Analog Crossbar Arrays for DNNs", IEEE ITC 2022. DOI: 10.1109/ITC50671.2022.00060"""

# ---------- 0. Imports ----------
import argparse
import os
import sys
import time
import math
import json
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# sklearn -- core ML kernels demanded by the paper
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats as scipy_stats

# Optional MARS (py-earth); otherwise SplineTransformer+Ridge (sklearn>=1.0),
# else GradientBoostingRegressor as last resort. See MARSlikeRegressor.
HAVE_EARTH = False
try:
    from pyearth import Earth                         # noqa: F401
    HAVE_EARTH = True
except Exception:
    try:
        from sklearn_contrib_py_earth import Earth    # noqa: F401  (community fork)
        HAVE_EARTH = True
    except Exception:
        HAVE_EARTH = False

HAVE_SPLINE = False
try:
    from sklearn.preprocessing import SplineTransformer  # noqa: F401
    HAVE_SPLINE = True
except Exception:
    HAVE_SPLINE = False

# ---------- 1. Configuration (paper-driven hardware constants + emulator settings) ----------

# --- HfO_x RRAM device ranges from Section IV-A of the paper -----------------
RLRS = 50e3      # Low Resistance State [Ohm]
RHRS = 1e6       # High Resistance State [Ohm]
G_MAX = 1.0 / RLRS
G_MIN = 1.0 / RHRS

# --- Variability statistics (Section IV-A, eqn. (1)) -------------------------
DEFAULT_ALPHA      = 0.50
DEFAULT_PSYS_SIG   = 0.05    # systematic (slow, wafer-/die-level)
DEFAULT_PRAND_SIG  = 0.10    # random (cell-to-cell)

# --- Stuck-at faults (Section VI-B) ------------------------------------------
DEFAULT_FAULT_RATE   = 0.10   # 10 % of memristors in faulty devices
DEFAULT_FAULTY_DEV   = 0.10   # 10 % of DUTs are faulty
DEFAULT_SA0_FRACTION = 0.50   # SA0 vs SA1 ratio

# --- ADC offset faults (itc26-style: scalar offset on victim channel [0,0,0] per layer) ---
DEFAULT_ADC_OFFSET_LOW    = -64.0  # uniform draw range for the per-layer offset value
DEFAULT_ADC_OFFSET_HIGH   =  64.0
DEFAULT_ADC_LAYER_RATE    = 0.5    # P(a layer in a faulty DUT gets an ADC offset)
DEFAULT_ADC_FAULTY_DEV    = 0.10   # fraction of DUTs that exhibit any ADC offset fault

# --- Pass/fail/fuzzy thresholds (Section VI) ---------------------------------
DEFAULT_ATH        = 0.85
DEFAULT_PRI_LOW    = 0.75
DEFAULT_PRI_HIGH   = 1.00
DEFAULT_K_CONF     = 2.0
DEFAULT_T_CONF     = 0.95     # 95% Student's-t CI

# --- Crossbar tiling / DAC / ADC (kept identical to itc26_6_5_26.py) ---------
class SimulatorConfig:
    XB_SIZE = 32     # 32x32 crossbar tile
    DAC_MIN = -15    # 4-bit signed
    DAC_MAX =  15
    ADC_MIN = -128   # 8-bit signed clamp
    ADC_MAX =  127

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Ternary primitives + DAC / ADC clamps (from itc26_6_5_26.py) ----------

class TernaryWeightFn(torch.autograd.Function):
    """Symmetric ternary quantizer: w -> sign(w) * 1{|w| > 0.7 * mean|w|}."""
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        delta = 0.7 * w.abs().mean()
        out = torch.zeros_like(w)
        out[w >  delta] =  1.0
        out[w < -delta] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # straight-through estimator
        return grad_out


class InputQuantizerFn(torch.autograd.Function):
    """Scale + round + clamp inputs to the DAC's signed integer range."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        x_int = (x / alpha).round().clamp(SimulatorConfig.DAC_MIN,
                                          SimulatorConfig.DAC_MAX)
        return x_int

    @staticmethod
    def backward(ctx, grad_out):
        # straight-through w.r.t. input, no grad to alpha (detached scale)
        return grad_out.clone(), None


class LeakyClampFn(torch.autograd.Function):
    """ADC clamp with a small leak past saturation -- helps gradient flow."""
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return torch.clamp(x, min=lo, max=hi)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        g = grad_out.clone()
        oob = (x < ctx.lo) | (x > ctx.hi)
        g[oob] *= 0.1
        return g, None, None


class OutputClampFn(torch.autograd.Function):
    """Final per-layer accumulator clamp, sized to # of row tiles."""
    @staticmethod
    def forward(ctx, x, num_row_blocks):
        hi =  1905.0 * num_row_blocks
        lo = -1920.0 * num_row_blocks
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return torch.clamp(x, min=lo, max=hi)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        g = grad_out.clone()
        oob = (x < ctx.lo) | (x > ctx.hi)
        g[oob] *= 0.1
        return g, None


class TileMap:
    """Pads the ternarized weight matrix and reshapes into 32x32 tiles."""
    def __init__(self, weight_matrix: torch.Tensor):
        self.device = weight_matrix.device
        out_ch, in_ch = weight_matrix.shape
        # ternarize using the same delta = 0.7 * mean(|W|) rule
        delta = 0.7 * weight_matrix.abs().mean()
        w_int = torch.zeros_like(weight_matrix)
        w_int[weight_matrix >  delta] =  1.0
        w_int[weight_matrix < -delta] = -1.0

        xb = SimulatorConfig.XB_SIZE
        pad_in  = (xb - (in_ch  % xb)) % xb
        pad_out = (xb - (out_ch % xb)) % xb
        w_padded = F.pad(w_int, (0, pad_in, 0, pad_out), value=0)
        new_out, new_in = w_padded.shape

        num_row_blocks = new_in  // xb
        num_col_blocks = new_out // xb
        w_blocked = w_padded.view(num_col_blocks, xb, num_row_blocks, xb)
        # (col_block, row_block, xb_row, xb_col) -> flat list of tiles
        self.crossbar_array = (w_blocked.permute(0, 2, 3, 1)
                                       .contiguous()
                                       .view(-1, xb, xb))
        self.num_row_blocks = num_row_blocks
        self.num_col_blocks = num_col_blocks
        self.pad_in         = pad_in
        self.logical_shape  = (out_ch, in_ch)


class RRAMLayerBase(nn.Module):
    """Common machinery for RRAM-mapped linear / conv layers."""
    def __init__(self, name: str = "layer"):
        super().__init__()
        self.name = name
        self.mode = 'fp32'

        # tiles
        self.tile_map        = None
        self.pristine_tiles  = None     # never modified after mapping
        self.clean_tiles     = None     # working "ideal" copy of pristine
        self.effective_tiles = None     # what VMM actually reads

        # itc26-style ADC offset on victim channel [col=0,row=0,ch=0]; 0 = no fault
        self.adc_offset = 0.0

        # for reference / future analyses (not used by the paper pipeline)
        self.last_x_int = None

    # --- mapping --------------------------------------------------------
    def map_to_hardware(self, w: torch.Tensor):
        self.tile_map        = TileMap(w)
        self.pristine_tiles  = self.tile_map.crossbar_array.clone()
        self.clean_tiles     = self.pristine_tiles.clone()
        self.effective_tiles = self.clean_tiles.clone()

    def restore_pristine_weights(self):
        if self.pristine_tiles is not None:
            self.clean_tiles     = self.pristine_tiles.clone()
            self.effective_tiles = self.clean_tiles.clone()

    def reset_to_clean(self):
        """Discard variability + all faults: weights -> clean, adc_offset -> 0."""
        if self.clean_tiles is not None:
            self.effective_tiles = self.clean_tiles.clone()
        self.adc_offset = 0.0

    # --- DAC scale helper -----------------------------------------------
    def get_dynamic_scale(self, x):
        max_val = x.abs().max()
        if max_val == 0:
            return torch.tensor(1.0, device=x.device)
        return (max_val / SimulatorConfig.DAC_MAX).detach()

    # --- bit-serial VMM through the tiled crossbar ----------------------
    def rram_forward_matmul(self, x_int, bias_int=None):
        """x_int [B, in_features] in [DAC_MIN, DAC_MAX] -> [B, out_features] float."""
        B   = x_int.size(0)
        sign = torch.sign(x_int)
        sign[sign == 0] = 1.0
        mag_int = torch.abs(x_int).int()
        masks   = [1, 2, 4, 8]   # 4-bit magnitude

        num_row = self.tile_map.num_row_blocks
        num_col = self.tile_map.num_col_blocks
        w_grid  = self.effective_tiles.view(num_col, num_row,
                                            SimulatorConfig.XB_SIZE,
                                            SimulatorConfig.XB_SIZE)
        out_acc = torch.zeros((B, num_col, num_row, SimulatorConfig.XB_SIZE),
                              device=x_int.device)

        for bit_pos, m in enumerate(masks):
            bit   = (mag_int & m).bool().float()
            x_bit = bit * sign
            x_pad = F.pad(x_bit, (0, self.tile_map.pad_in), value=0)
            x_blk = x_pad.view(B, num_row, SimulatorConfig.XB_SIZE)

            # einsum: per-tile partial dot products -> [B, col_block, row_block, xb_col]
            partial = torch.einsum('bkr, jkrc -> bjkc', x_blk, w_grid)

            # itc26-style ADC offset on victim channel [col=0, row=0, ch=0] with wrap-around
            if self.adc_offset != 0.0 and num_col > 0 and num_row > 0:
                partial[:, 0, 0, 0] = partial[:, 0, 0, 0] + self.adc_offset
                partial = torch.where(partial > SimulatorConfig.ADC_MAX,
                                      partial - SimulatorConfig.ADC_MAX, partial)
                partial = torch.where(partial < SimulatorConfig.ADC_MIN,
                                      partial - SimulatorConfig.ADC_MIN, partial)

            adc_out = LeakyClampFn.apply(partial,
                                         SimulatorConfig.ADC_MIN,
                                         SimulatorConfig.ADC_MAX)
            out_acc += adc_out * (1 << bit_pos)

        # sum across row blocks -> [B, num_col, xb_col]
        layer_out = out_acc.sum(dim=2).view(B, -1)
        # crop padded outputs
        out_cropped = layer_out[:, :self.tile_map.logical_shape[0]]
        if bias_int is not None:
            out_cropped = out_cropped + bias_int.unsqueeze(0)
        return out_cropped

class RRAMConv2d(RRAMLayerBase):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, name="conv"):
        super().__init__(name)
        self.layer = nn.Conv2d(in_c, out_c, k, stride=stride,
                               padding=padding, bias=False)

    def forward(self, x):
        if self.mode == 'fp32':
            return self.layer(x)

        kernel_elems = (self.layer.in_channels
                        * self.layer.kernel_size[0]
                        * self.layer.kernel_size[1])
        num_row_blocks = ((kernel_elems + SimulatorConfig.XB_SIZE - 1)
                          // SimulatorConfig.XB_SIZE)

        w_eff = TernaryWeightFn.apply(self.layer.weight)
        scale = self.get_dynamic_scale(x)

        if self.mode == 'ternary':
            # training-time path with gradient flow + bit-serial hard path
            x_q = InputQuantizerFn.apply(x, scale)
            with torch.no_grad():
                sign = torch.sign(x_q); sign[sign == 0] = 1.0
                mag_int = torch.abs(x_q).int()
                hw_acc = 0
                for bit_pos, m in enumerate([1, 2, 4, 8]):
                    bit   = (mag_int & m).bool().float()
                    x_bit = bit * sign
                    raw = F.conv2d(x_bit, w_eff,
                                   stride=self.layer.stride,
                                   padding=self.layer.padding)
                    clamped = torch.clamp(raw,
                                          SimulatorConfig.ADC_MIN,
                                          SimulatorConfig.ADC_MAX)
                    hw_acc = hw_acc + clamped * (1 << bit_pos)
                if self.layer.bias is not None:
                    hw_acc = hw_acc + self.layer.bias.view(1, -1, 1, 1)
            soft = F.conv2d(x_q, w_eff, bias=self.layer.bias,
                            stride=self.layer.stride,
                            padding=self.layer.padding)
            out = soft + (hw_acc - soft).detach()
            return OutputClampFn.apply(out, num_row_blocks)

        # ---- mode == 'rram' --------------------------------------------
        if self.tile_map is None:
            self.map_to_hardware(self.layer.weight.view(self.layer.out_channels, -1))

        x_int = InputQuantizerFn.apply(x, scale)
        x_uf  = F.unfold(x_int,
                         kernel_size=self.layer.kernel_size,
                         stride=self.layer.stride,
                         padding=self.layer.padding)
        B, C_flat, L = x_uf.shape
        x_flat = x_uf.transpose(1, 2).reshape(B * L, C_flat)
        out_flat = self.rram_forward_matmul(x_flat, bias_int=None)
        out_int  = out_flat.view(B, L, -1).transpose(1, 2)

        h_out = ((x.shape[2] + 2 * self.layer.padding[0]
                 - self.layer.kernel_size[0]) // self.layer.stride[0] + 1)
        w_out = ((x.shape[3] + 2 * self.layer.padding[1]
                 - self.layer.kernel_size[1]) // self.layer.stride[1] + 1)
        out_float = out_int.view(B, -1, h_out, w_out).float()
        if self.layer.bias is not None:
            out_float = out_float + self.layer.bias.view(1, -1, 1, 1)
        return OutputClampFn.apply(out_float, num_row_blocks)

class RRAMLinear(RRAMLayerBase):
    def __init__(self, in_f, out_f, name="fc", is_output=False):
        super().__init__(name)
        self.is_output = is_output
        self.layer = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x):
        if self.mode == 'fp32':
            return self.layer(x)

        num_row_blocks = ((self.layer.in_features + SimulatorConfig.XB_SIZE - 1)
                          // SimulatorConfig.XB_SIZE)
        w_eff = TernaryWeightFn.apply(self.layer.weight)
        scale = self.get_dynamic_scale(x)

        if self.mode == 'ternary':
            x_q = InputQuantizerFn.apply(x, scale)
            with torch.no_grad():
                sign = torch.sign(x_q); sign[sign == 0] = 1.0
                mag_int = torch.abs(x_q).int()
                hw_acc = torch.zeros((x.size(0), self.layer.out_features),
                                     device=x.device)
                for bit_pos, m in enumerate([1, 2, 4, 8]):
                    bit   = (mag_int & m).bool().float()
                    x_bit = bit * sign
                    raw = F.linear(x_bit, w_eff)
                    hw_acc += torch.clamp(raw,
                                          SimulatorConfig.ADC_MIN,
                                          SimulatorConfig.ADC_MAX) * (1 << bit_pos)
                if self.layer.bias is not None:
                    hw_acc += self.layer.bias
            soft = F.linear(x_q, w_eff, self.layer.bias)
            out = soft + (hw_acc - soft).detach()
            return OutputClampFn.apply(out, num_row_blocks)

        # ---- mode == 'rram' --------------------------------------------
        if self.tile_map is None:
            self.map_to_hardware(self.layer.weight)

        x_int = InputQuantizerFn.apply(x, scale)
        out   = self.rram_forward_matmul(x_int, bias_int=None).float()
        if self.layer.bias is not None:
            out = out + self.layer.bias
        return OutputClampFn.apply(out, num_row_blocks)

# ---------- 4. ModelWrapper + the three MNIST models  (LeNet5 / SimpleMLP / PureLinear) ----------
class ModelWrapper(nn.Module):
    """Convenience helpers to flip every RRAM layer between fp32 / ternary / rram."""
    def set_mode(self, mode: str):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)):
                m.mode = mode
                if mode == 'rram' and m.tile_map is None:
                    if isinstance(m, RRAMConv2d):
                        w = m.layer.weight.view(m.layer.out_channels, -1)
                    else:
                        w = m.layer.weight
                    m.map_to_hardware(w)

    def reset_all_to_clean(self):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)):
                m.reset_to_clean()

    def restore_all_pristine(self):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)):
                m.restore_pristine_weights()

    def rram_layers(self):
        for m in self.modules():
            if isinstance(m, (RRAMConv2d, RRAMLinear)):
                yield m

class LeNet5(ModelWrapper):
    """Classic LeNet-5 on 32x32 MNIST inputs (Resize -> 32x32)."""
    def __init__(self):
        super().__init__()
        self.conv1 = RRAMConv2d(1,  6, 5, name="conv1")
        self.pool  = nn.AvgPool2d(2, 2)
        self.conv2 = RRAMConv2d(6, 16, 5, name="conv2")
        self.fc1   = RRAMLinear(400, 120, name="fc1")
        self.fc2   = RRAMLinear(120,  84, name="fc2")
        self.fc3   = RRAMLinear( 84,  10, name="fc3", is_output=True)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = x.reshape(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleMLP(ModelWrapper):
    """3-hidden-layer MLP on flat MNIST."""
    def __init__(self, num_layers=3, hidden=700):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        in_dim = 784
        for i in range(num_layers - 1):
            layers.append(RRAMLinear(in_dim, hidden, name=f"fc{i+1}"))
            in_dim = hidden
        layers.append(RRAMLinear(hidden, 10, name=f"fc{num_layers}",
                                 is_output=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

class PureLinearMLP(ModelWrapper):
    """4-FC pure-linear network (no conv / no BN here -- mirrors itc26 default)."""
    def __init__(self, input_dim=784, num_classes=10, hidden_dim=128):
        super().__init__()
        self.fc1 = RRAMLinear(input_dim, 256,        name="fc1")
        self.fc2 = RRAMLinear(256,       hidden_dim, name="fc2")
        self.fc3 = RRAMLinear(hidden_dim, hidden_dim,name="fc3")
        self.fc4 = RRAMLinear(hidden_dim, num_classes, name="fc4",
                              is_output=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def build_model(arch: str) -> ModelWrapper:
    a = arch.lower()
    if a == 'lenet5':       return LeNet5().to(DEVICE)
    if a == 'simplemlp':    return SimpleMLP().to(DEVICE)
    if a == 'purelinear':   return PureLinearMLP().to(DEVICE)
    raise ValueError(f"Unknown --arch '{arch}'")


# ---------- 5. Data loaders  (MNIST, with the LeNet5 32x32 convention) ----------

def make_mnist_loaders(arch: str, batch_size=256, data_root='./data',
                      test_size=None):
    if arch.lower() == 'lenet5':
        tfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    train_set = datasets.MNIST(data_root, train=True,  download=True, transform=tfm)
    test_set  = datasets.MNIST(data_root, train=False, download=True, transform=tfm)

    if test_size is not None and test_size < len(test_set):
        # Deterministic: just take the first `test_size` images so that
        # benchmark / DUT comparisons reference identical inputs.
        test_set = Subset(test_set, list(range(test_size)))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=512,
                              shuffle=False, num_workers=0)
    return train_loader, test_loader, test_set


# ---------- 6. Train the base ternary network (one-time, cacheable) ----------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += out.argmax(1).eq(target).sum().item()
        total   += target.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_top1(model, loader) -> float:
    model.eval()
    correct, total = 0, 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        out = model(data)
        correct += out.argmax(1).eq(target).sum().item()
        total   += target.size(0)
    return correct / max(1, total)


def train_or_load_base_model(arch: str, train_loader, test_loader,
                             epochs=5, lr=1e-3, ckpt_path=None,
                             verbose=True):
    model = build_model(arch)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        if verbose:
            print(f"[train] Loading cached model: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(sd)
    else:
        if verbose:
            print(f"[train] Training base ternary {arch} for {epochs} epochs"
                  f" on {DEVICE} ...")
        model.set_mode('ternary')
        optim_ = optim.Adam(model.parameters(), lr=lr)
        crit   = nn.CrossEntropyLoss()
        for ep in range(epochs):
            l, a = train_one_epoch(model, train_loader, optim_, crit)
            if verbose:
                print(f"   epoch {ep+1}/{epochs}: loss={l:.4f} train_acc={a:.2f}%")
        if ckpt_path is not None:
            os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            if verbose:
                print(f"[train] Saved checkpoint -> {ckpt_path}")

    # snapshot the ideal RRAM behaviour
    model.set_mode('rram')              # this triggers tile_map creation
    if verbose:
        acc = evaluate_top1(model, test_loader)
        print(f"[train] Ideal RRAM accuracy on test set: {acc*100:.2f}%")
    return model


# ---------- 7. Paper Section IV-A:  RRAM variability modeling at the *tile* level ----------

def _gen_perturbation_matrix(shape, alpha, psys_sigma, prand_sigma, rng,
                             psys_scalar=None):
    """pij = alpha*psys + (1-alpha)*prand (Eq.1). psys: one scalar per DUT
    (die-level, supplied via psys_scalar); prand: i.i.d. per memristor."""
    if psys_scalar is None:
        psys_scalar = float(rng.normal(loc=1.0, scale=psys_sigma))
    prand = rng.normal(loc=1.0, scale=prand_sigma, size=shape).astype(np.float32)
    return (alpha * psys_scalar + (1.0 - alpha) * prand).astype(np.float32)


def apply_variability_to_model(model: ModelWrapper,
                               alpha=DEFAULT_ALPHA,
                               psys_sigma=DEFAULT_PSYS_SIG,
                               prand_sigma=DEFAULT_PRAND_SIG,
                               rng: np.random.Generator = None):
    """One DUT's process perturbation: psys drawn ONCE per DUT (die-level,
    shared across all layers), prand i.i.d. per cell. Wni = Wi (-) P (Sec IV-A)."""
    if rng is None:
        rng = np.random.default_rng()

    # ONE die-level systematic draw per DUT
    psys_die = float(rng.normal(loc=1.0, scale=psys_sigma))

    for m in model.rram_layers():
        if m.clean_tiles is None:
            continue
        shape = tuple(m.clean_tiles.shape)
        P_np = _gen_perturbation_matrix(shape, alpha,
                                        psys_sigma, prand_sigma,
                                        rng, psys_scalar=psys_die)
        P_t  = torch.from_numpy(P_np).to(m.clean_tiles.device)
        m.effective_tiles = m.clean_tiles * P_t

# ---------- 8. Paper Section VI-B:  stuck-at fault injection at the memristor level ----------
def inject_stuck_at_faults_to_model(model: ModelWrapper,
                                    fault_rate=DEFAULT_FAULT_RATE,
                                    sa0_fraction=DEFAULT_SA0_FRACTION,
                                    rng: np.random.Generator = None):
    """In-place: add stuck-at faults to the *current* effective_tiles.
    """
    if rng is None:
        rng = np.random.default_rng()
    for m in model.rram_layers():
        if m.effective_tiles is None:
            continue
        with torch.no_grad():
            t = m.effective_tiles
            n = t.numel()
            num_faults = int(np.floor(fault_rate * n))
            if num_faults <= 0:
                continue
            idx = rng.choice(n, size=num_faults, replace=False)
            # Decide SA0 vs SA1 per-fault
            sa0_mask = rng.random(num_faults) < sa0_fraction
            # SA0 -> +/- 1 with random sign; SA1 -> 0
            new_vals = np.where(sa0_mask,
                                np.where(rng.random(num_faults) < 0.5, 1.0, -1.0),
                                0.0).astype(np.float32)
            flat = t.view(-1)
            flat[torch.from_numpy(idx).to(flat.device).long()] = \
                torch.from_numpy(new_vals).to(flat.device)
            m.effective_tiles = flat.view(t.shape)


# ---------- 8b. ADC offset fault injection (itc26-style, single victim channel per layer) ----------

def inject_adc_offset_faults_to_model(model: ModelWrapper,
                                      layer_fault_rate=DEFAULT_ADC_LAYER_RATE,
                                      offset_low=DEFAULT_ADC_OFFSET_LOW,
                                      offset_high=DEFAULT_ADC_OFFSET_HIGH,
                                      rng: np.random.Generator = None) -> int:
    """For each RRAM layer, w.p. layer_fault_rate set adc_offset ~ U(low, high).
    Returns the number of layers that received an offset (>=1 == DUT is faulty)."""
    if rng is None:
        rng = np.random.default_rng()
    n_hit = 0
    for m in model.rram_layers():
        if rng.random() < layer_fault_rate:
            m.adc_offset = float(rng.uniform(offset_low, offset_high))
            n_hit += 1
        else:
            m.adc_offset = 0.0
    return n_hit


# ---------- 9. Inference helpers:  per-image classification matrix + signature vectors ----------

@torch.no_grad()
def device_predict_all(model: ModelWrapper, X: torch.Tensor,
                       batch_size=256) -> np.ndarray:
    """Return final-layer logits [N, num_classes] for every image in X."""
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size].to(DEVICE)
        outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def predict_correctness_row(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Binary row vector for the classification matrix (1=correct)."""
    return (logits.argmax(axis=1) == y).astype(np.uint8)


def signature_from_logits(logits_compact: np.ndarray) -> np.ndarray:
    """Concatenate final-layer responses across the compact subset.
    """
    return logits_compact.reshape(-1).astype(np.float32)


# ---------- 10. Paper Section IV-B:  image down-selection via agglomerative clustering ----------

def select_compact_test_subset(C: np.ndarray, num_clusters: int,
                               rng: np.random.Generator = None) -> list:
    """C [N_devices, M_images] in {0,1} -> up to num_clusters representative image indices."""
    if rng is None:
        rng = np.random.default_rng()

    M = C.shape[1]
    if num_clusters >= M:
        return list(range(M))
    if num_clusters <= 0:
        return []

    image_vectors = C.T  # shape (M, N_devices) -- each row is one image
    jitter = rng.normal(scale=1e-3, size=image_vectors.shape).astype(np.float32)
    feats  = image_vectors.astype(np.float32) + jitter

    clust = AgglomerativeClustering(n_clusters=num_clusters, linkage='average')
    labels = clust.fit_predict(feats)

    selected = []
    for lab in np.unique(labels):
        members = np.where(labels == lab)[0]
        # Pick the member closest to the cluster mean (most "central")
        if len(members) == 1:
            selected.append(int(members[0]))
        else:
            center = feats[members].mean(axis=0, keepdims=True)
            d2 = ((feats[members] - center) ** 2).sum(axis=1)
            selected.append(int(members[d2.argmin()]))
    return sorted(selected)


# ---------- 11. Paper Section IV-C:  Elliptic Envelope outlier detector ----------

class OutlierDetector:
    """Wraps sklearn's EllipticEnvelope with a PCA pre-step for robustness.
    """
    def __init__(self, contamination=0.10, max_dim=64, random_state=0):
        self.contamination = contamination
        self.max_dim       = max_dim
        self.random_state  = random_state
        self.scaler        = StandardScaler()
        self.proj_basis    = None   # (n_features, k)
        self.ee            = None
        self._fit_n        = 0

    def _project(self, X):
        Xs = self.scaler.transform(X)
        if self.proj_basis is None:
            return Xs
        return Xs @ self.proj_basis

    def fit(self, X):
        n, d = X.shape
        Xs = self.scaler.fit_transform(X)
        # cap target dim well below n_samples so EE's MCD has room to breathe
        target_dim = min(self.max_dim, max(2, n // 4), d)
        if target_dim < d:
            # randomized PCA via SVD of the centered standardized data
            # (sklearn's TruncatedSVD would also work; we keep deps minimal)
            u, s, vt = np.linalg.svd(Xs, full_matrices=False)
            self.proj_basis = vt[:target_dim].T  # (d, k)
            Xp = Xs @ self.proj_basis
        else:
            self.proj_basis = None
            Xp = Xs

        self.ee = EllipticEnvelope(contamination=self.contamination,
                                   support_fraction=None,
                                   random_state=self.random_state)
        try:
            self.ee.fit(Xp)
        except Exception as e:
            # MCD can fail on ill-conditioned data; fall back to a simpler
            # Mahalanobis-style detector with empirical covariance.
            print(f"[ee] EllipticEnvelope failed ({e}); falling back to "
                  f"shrunk covariance.")
            from sklearn.covariance import ShrunkCovariance
            cov = ShrunkCovariance().fit(Xp)
            self._fallback_cov = cov
            self.ee = None
        self._fit_n = n
        return self

    def predict(self, X):
        """+1 for inliers, -1 for outliers (matches sklearn's convention)."""
        Xp = self._project(X)
        if self.ee is not None:
            return self.ee.predict(Xp)
        # fallback: threshold Mahalanobis distance at the chi^2 quantile
        m = self._fallback_cov.mahalanobis(Xp - self._fallback_cov.location_)
        thresh = scipy_stats.chi2.ppf(1 - self.contamination, df=Xp.shape[1])
        return np.where(m <= thresh, 1, -1)


# ---------- 12. Paper Section IV-D:  MARS regressor + Student's-t pass/fail/fuzzy ----------

class MARSlikeRegressor:
    """MARS regressor (pyearth) if available, else SplineTransformer+Ridge,
    else GradientBoostingRegressor. Defaults tuned for high-D signatures."""
    def __init__(self, random_state=0,
                 spline_n_knots=3, spline_degree=2, ridge_alpha=1.0):
        if HAVE_EARTH:
            from pyearth import Earth
            self.kind = 'mars'
            self.model = Earth(max_terms=200, max_degree=2,
                               feature_importance_type='gcv')
        elif HAVE_SPLINE:
            # Piecewise-quadratic spline basis (close cousin of MARS hinges) + Ridge.
            from sklearn.preprocessing import SplineTransformer
            self.kind  = 'spline+ridge'
            self.spline_n_knots = spline_n_knots
            self.spline_degree  = spline_degree
            self.ridge_alpha    = ridge_alpha
            self._random_state  = random_state
            self.model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                SplineTransformer(n_knots=spline_n_knots,
                                  degree=spline_degree,
                                  include_bias=False),
                Ridge(alpha=ridge_alpha, random_state=random_state),
            )
        else:
            # Last-resort tree-ensemble fallback.
            self.kind  = 'gbr'
            self.model = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=random_state)

    def fit(self, X, y):
        # Guard against tiny training sets that would crash SplineTransformer.
        if self.kind == 'spline+ridge' and X.shape[0] < (self.spline_n_knots + 2):
            self.kind  = 'gbr'
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=self._random_state)
        try:
            self.model.fit(X, y)
        except Exception as e:
            # Degenerate spline expansion -> fall back to plain Ridge.
            if self.kind == 'spline+ridge':
                print(f"[regressor] SplineTransformer failed ({e}); "
                      f"falling back to StandardScaler + Ridge.")
                self.kind  = 'ridge'
                self.model = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    Ridge(alpha=self.ridge_alpha,
                          random_state=self._random_state),
                )
                self.model.fit(X, y)
            else:
                raise
        return self

    def predict(self, X):
        return self.model.predict(X)


def _interval_sigmas(y_true, y_pred, lo, hi, n_bins=8):
    """Stratified residual-sigma over the PRI (Section IV-D, eqn. for E_j)."""
    edges = np.linspace(lo, hi, n_bins + 1)
    sig_bins = np.zeros(n_bins, dtype=np.float32)
    for j in range(n_bins):
        mask = (y_pred >= edges[j]) & (y_pred < edges[j+1])
        if mask.sum() >= 4:
            sig_bins[j] = float(np.std(y_true[mask] - y_pred[mask], ddof=1))
        else:
            sig_bins[j] = float('nan')
    # Fill NaN bins with the closest valid neighbour, else the global sigma.
    global_sig = float(np.std(y_true - y_pred, ddof=1))
    for j in range(n_bins):
        if not np.isfinite(sig_bins[j]):
            sig_bins[j] = global_sig
    return edges, sig_bins, global_sig


def _sigma_at(pred, edges, sig_bins, global_sig):
    if pred <= edges[0]:    return float(sig_bins[0])
    if pred >= edges[-1]:   return float(sig_bins[-1])
    j = int(np.searchsorted(edges, pred) - 1)
    j = max(0, min(j, len(sig_bins) - 1))
    return float(sig_bins[j])


def classify_pass_fail_fuzzy(mu, sigma_at_mu, ath, k_conf=DEFAULT_K_CONF,
                             t_conf=DEFAULT_T_CONF, df=None):
    """Pass / fail / fuzzy decision per Section IV-D.
    """
    if df is not None and df > 1:
        k = float(scipy_stats.t.ppf(1 - (1 - t_conf) / 2.0, df=df))
    else:
        k = float(k_conf)
    half = k * sigma_at_mu
    if mu - half >= ath:
        return 'pass'
    if mu + half <= ath:
        return 'fail'
    return 'fuzzy'


# ---------- 13. Orchestration helpers:  building DUTs, classification matrix, signatures ----------

def _stack_dataset_to_tensor(dataset, n_max=None):
    """Stack a torchvision Subset/MNIST into one (X, y) tensor pair."""
    if n_max is None:
        n_max = len(dataset)
    n = min(n_max, len(dataset))
    Xs, ys = [], []
    for i in range(n):
        x, y = dataset[i]
        Xs.append(x.unsqueeze(0))
        ys.append(int(y))
    X = torch.cat(Xs, dim=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def materialize_dut(model, alpha, psys_sigma, prand_sigma,
                    inject_faults, fault_rate, sa0_fraction,
                    inject_adc_faults=False, adc_layer_rate=DEFAULT_ADC_LAYER_RATE,
                    adc_offset_low=DEFAULT_ADC_OFFSET_LOW,
                    adc_offset_high=DEFAULT_ADC_OFFSET_HIGH,
                    rng=None):
    """Reset -> variability -> optional SA0/SA1 -> optional ADC offsets (itc26-style)."""
    model.reset_all_to_clean()
    apply_variability_to_model(model, alpha=alpha,
                               psys_sigma=psys_sigma,
                               prand_sigma=prand_sigma,
                               rng=rng)
    if inject_faults:
        inject_stuck_at_faults_to_model(model,
                                        fault_rate=fault_rate,
                                        sa0_fraction=sa0_fraction,
                                        rng=rng)
    if inject_adc_faults:
        inject_adc_offset_faults_to_model(model,
                                          layer_fault_rate=adc_layer_rate,
                                          offset_low=adc_offset_low,
                                          offset_high=adc_offset_high,
                                          rng=rng)


def build_classification_matrix(model, X, y, n_devices,
                                alpha, psys_sigma, prand_sigma,
                                inject_faults, faulty_dev_fraction,
                                fault_rate, sa0_fraction,
                                rng, batch_size=256, verbose=True,
                                inject_adc_faults=False,
                                adc_faulty_dev_fraction=DEFAULT_ADC_FAULTY_DEV,
                                adc_layer_rate=DEFAULT_ADC_LAYER_RATE,
                                adc_offset_low=DEFAULT_ADC_OFFSET_LOW,
                                adc_offset_high=DEFAULT_ADC_OFFSET_HIGH):
    """Run `n_devices` Monte-Carlo DUTs over the full image set X."""
    M = X.shape[0]
    n_classes = None
    C   = np.zeros((n_devices, M), dtype=np.uint8)
    acc = np.zeros((n_devices,),   dtype=np.float32)
    all_logits = None
    is_faulty  = np.zeros((n_devices,), dtype=bool)   # any-fault flag (SA or ADC)

    t0 = time.time()
    for d in range(n_devices):
        # Per-DUT independent decisions for stuck-at and ADC offset faults
        dev_sa_faulty  = (inject_faults
                          and rng.random() < faulty_dev_fraction)
        dev_adc_faulty = (inject_adc_faults
                          and rng.random() < adc_faulty_dev_fraction)
        is_faulty[d] = bool(dev_sa_faulty or dev_adc_faulty)
        materialize_dut(model,
                        alpha=alpha,
                        psys_sigma=psys_sigma,
                        prand_sigma=prand_sigma,
                        inject_faults=dev_sa_faulty,
                        fault_rate=fault_rate,
                        sa0_fraction=sa0_fraction,
                        inject_adc_faults=dev_adc_faulty,
                        adc_layer_rate=adc_layer_rate,
                        adc_offset_low=adc_offset_low,
                        adc_offset_high=adc_offset_high,
                        rng=rng)
        logits = device_predict_all(model, X, batch_size=batch_size)
        if all_logits is None:
            n_classes = logits.shape[1]
            all_logits = np.zeros((n_devices, M, n_classes), dtype=np.float32)
        all_logits[d] = logits
        row = predict_correctness_row(logits, y)
        C[d] = row
        acc[d] = row.mean()
        if verbose and ((d + 1) % max(1, n_devices // 10) == 0
                        or d == 0 or d == n_devices - 1):
            dt = time.time() - t0
            tag = []
            if dev_sa_faulty:  tag.append('SA')
            if dev_adc_faulty: tag.append('ADC')
            tag_str = '+'.join(tag) if tag else 'clean'
            print(f"   [bench] device {d+1}/{n_devices} | "
                  f"acc={acc[d]*100:5.2f}%  "
                  f"fault={tag_str}  elapsed={dt:.1f}s")
    # restore so subsequent code starts clean
    model.reset_all_to_clean()
    return C, acc, all_logits, is_faulty


def signatures_from_logits(all_logits: np.ndarray,
                           compact_idx: list) -> np.ndarray:
    """all_logits: (D, M, K)  ->  signatures: (D, len(compact_idx) * K)."""
    sub = all_logits[:, compact_idx, :]              # (D, S, K)
    return sub.reshape(sub.shape[0], -1).astype(np.float32)


# ---------- 14. Single-shot test  (Section VI-A:  Compact Test Image Subset Analysis) ----------

def run_compact_subset_analysis(model, X, y, args, rng,
                                results_csv=None, verbose=True):
    """Sweep compact subset size; track prediction-error SD and test speedup (Fig.4)."""
    M = X.shape[0]

    # ---- Phase A: build the benchmark classification matrix once.
    if verbose:
        print(f"\n[Fig.4] Building benchmark classification matrix "
              f"with {args.bench_devices} DUTs over {M} images ...")
    C_b, acc_b, log_b, faulty_b = build_classification_matrix(
        model, X, y, args.bench_devices,
        alpha=args.alpha,
        psys_sigma=args.psys_sigma, prand_sigma=args.prand_sigma,
        inject_faults=args.inject_faults,
        faulty_dev_fraction=args.faulty_dev_fraction,
        fault_rate=args.fault_rate, sa0_fraction=args.sa0_fraction,
        rng=rng, batch_size=args.batch_size, verbose=verbose,
        inject_adc_faults=args.inject_adc_faults,
        adc_faulty_dev_fraction=args.adc_faulty_dev_fraction,
        adc_layer_rate=args.adc_layer_rate,
        adc_offset_low=args.adc_offset_low,
        adc_offset_high=args.adc_offset_high)

    # ---- Phase B: build a separate validation pool of DUTs.
    if verbose:
        print(f"[Fig.4] Building DUT pool with {args.device_count} DUTs ...")
    C_d, acc_d, log_d, faulty_d = build_classification_matrix(
        model, X, y, args.device_count,
        alpha=args.alpha,
        psys_sigma=args.psys_sigma, prand_sigma=args.prand_sigma,
        inject_faults=args.inject_faults,
        faulty_dev_fraction=args.faulty_dev_fraction,
        fault_rate=args.fault_rate, sa0_fraction=args.sa0_fraction,
        rng=rng, batch_size=args.batch_size, verbose=verbose,
        inject_adc_faults=args.inject_adc_faults,
        adc_faulty_dev_fraction=args.adc_faulty_dev_fraction,
        adc_layer_rate=args.adc_layer_rate,
        adc_offset_low=args.adc_offset_low,
        adc_offset_high=args.adc_offset_high)

    # ---- Phase C: sweep compact-subset sizes.
    pcts = list(args.subset_sweep)
    rows = []
    if verbose:
        print(f"\n[Fig.4] Sweeping compact subset percentages: {pcts}")
        print(f"   {'pct':>6s} {'|S|':>5s} "
              f"{'sigma':>8s} {'rho':>6s} {'speedup':>8s} {'fpr':>6s}")

    for pct in pcts:
        S = max(1, int(round(pct / 100.0 * M)))
        sel = select_compact_test_subset(C_b, num_clusters=S, rng=rng)
        S_actual = len(sel)

        # Signatures from the benchmark matrix (training set for the regressor)
        Xb = signatures_from_logits(log_b, sel)
        yb = acc_b
        # Signatures from the DUT pool (the "validation" of regressor quality)
        Xd = signatures_from_logits(log_d, sel)
        yd = acc_d

        # Train regressor (paper's 80/20 split is internal
        reg = MARSlikeRegressor(random_state=args.seed).fit(Xb, yb)
        pred_d = reg.predict(Xd)
        residuals = yd - pred_d
        sigma = float(np.std(residuals, ddof=1))
        # Pearson correlation -- the paper reports rho ~ 0.95
        if pred_d.std() > 0 and yd.std() > 0:
            rho = float(np.corrcoef(pred_d, yd)[0, 1])
        else:
            rho = float('nan')

        # Stratified sigma over the PRI for proper fuzzy bounds
        edges, sig_bins, sig_global = _interval_sigmas(yd, pred_d,
                                                       args.pri_low,
                                                       args.pri_high,
                                                       n_bins=8)

        # Pass / fail / fuzzy classification
        n_pass = n_fail = n_fuzzy = 0
        n_outliers = 0
        false_pos = 0
        # Outlier detector trained on bench signatures (contamination = empirical fault rate)
        any_fault = args.inject_faults or args.inject_adc_faults
        det = OutlierDetector(contamination=max(0.01,
                              float(faulty_b.mean()) if any_fault else 0.05))
        det.fit(Xb)
        out_flags = det.predict(Xd)  # +1 inlier, -1 outlier

        for i in range(len(yd)):
            if out_flags[i] == -1:
                n_outliers += 1
                # outliers go to standard testing -> NOT a regressor decision,
                # so they don't count toward "false positives among classified"
                continue
            mu = float(pred_d[i])
            s_at = _sigma_at(mu, edges, sig_bins, sig_global)
            cls = classify_pass_fail_fuzzy(mu, s_at, args.ath,
                                           k_conf=args.k_conf,
                                           t_conf=args.t_conf,
                                           df=len(yb) - 1)
            if cls == 'pass':
                n_pass += 1
                if yd[i] < args.ath:
                    false_pos += 1
            elif cls == 'fail':
                n_fail += 1
            else:
                n_fuzzy += 1

        # ---- Test speedup: NST / NAT --------------------------------------
        # Standard: every device takes the full M images.
        NST = args.device_count * M
        # Alternate: every device takes |compact| images. Outliers and fuzzy
        # devices are then re-tested with the full M images on top.
        n_re = n_outliers + n_fuzzy
        NAT = args.device_count * S_actual + n_re * M
        speedup = NST / max(1, NAT)

        # Test quality = false positives / total classified pass+fail
        n_classified = n_pass + n_fail
        fpr = (false_pos / max(1, n_classified))

        rows.append(dict(
            pct=pct, S=S_actual, sigma=sigma, rho=rho,
            speedup=speedup, fpr=fpr,
            pass_=n_pass, fail=n_fail, fuzzy=n_fuzzy, outlier=n_outliers,
            sigma_global=sig_global,
        ))
        if verbose:
            print(f"   {pct:6.2f} {S_actual:5d} "
                  f"{sigma:8.4f} {rho:6.3f} {speedup:7.2f}x "
                  f"{fpr*100:5.2f}%")

    # ---- Optional CSV
    if results_csv is not None:
        import csv
        os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
        with open(results_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        if verbose:
            print(f"\n[Fig.4] Wrote sweep results -> {results_csv}")

    return rows, (C_b, acc_b, log_b, faulty_b)


# ---------- 15. ML kernel retraining loop  (Section VI-B:  Fig. 5) ----------

def run_retraining_pipeline(model, X, y, args, rng,
                            seed_bench=None,
                            results_csv=None, verbose=True):
    """ML kernel retraining loop (Section VI-B, Fig.5)."""
    M = X.shape[0]

    # ---- Compact subset is fixed at args.compact_pct (paper uses 3%)
    S = max(1, int(round(args.compact_pct / 100.0 * M)))

    # ---- Either reuse a pre-built benchmark or build it fresh
    if seed_bench is not None:
        C_b, acc_b, log_b, faulty_b = seed_bench
    else:
        if verbose:
            print(f"[Fig.5] Building seed benchmark: {args.bench_devices} DUTs"
                  f" over {M} images ...")
        C_b, acc_b, log_b, faulty_b = build_classification_matrix(
            model, X, y, args.bench_devices,
            alpha=args.alpha,
            psys_sigma=args.psys_sigma, prand_sigma=args.prand_sigma,
            inject_faults=args.inject_faults,
            faulty_dev_fraction=args.faulty_dev_fraction,
            fault_rate=args.fault_rate, sa0_fraction=args.sa0_fraction,
            rng=rng, batch_size=args.batch_size, verbose=verbose,
            inject_adc_faults=args.inject_adc_faults,
            adc_faulty_dev_fraction=args.adc_faulty_dev_fraction,
            adc_layer_rate=args.adc_layer_rate,
            adc_offset_low=args.adc_offset_low,
            adc_offset_high=args.adc_offset_high)

    # Pick the compact subset once (the paper does NOT re-cluster per
    # retrain step; the EE + regressor are what get retrained).
    sel = select_compact_test_subset(C_b, num_clusters=S, rng=rng)
    if verbose:
        print(f"[Fig.5] Compact subset size: {len(sel)} ({args.compact_pct}%)")

    Xb = signatures_from_logits(log_b, sel)
    yb = acc_b

    # contamination = expected outlier rate, summed across both fault types
    cont = 0.05
    if args.inject_faults:     cont = max(cont, DEFAULT_FAULTY_DEV)
    if args.inject_adc_faults: cont = max(cont, args.adc_faulty_dev_fraction)

    det = OutlierDetector(contamination=max(0.01, cont)).fit(Xb)
    reg = MARSlikeRegressor(random_state=args.seed).fit(Xb, yb)

    # initial residual stats (used later for per-bin sigma)
    pred_b = reg.predict(Xb)
    edges, sig_bins, sig_global = _interval_sigmas(
        yb, pred_b, args.pri_low, args.pri_high, n_bins=8)

    # ---- Retraining loop ----------------------------------------------------
    Xtrain = Xb.copy()
    ytrain = yb.copy()

    rows = []
    seen_dut = 0
    if verbose:
        header = (f"{'iter':>4s} {'cum':>5s} {'speedup':>8s} {'fpr':>6s} "
                  f"{'pass':>5s} {'fail':>5s} {'fuzzy':>5s} {'out':>5s} "
                  f"{'sigma':>8s} {'rho':>6s} {'|train|':>8s}")
        print(f"\n[Fig.5] Retraining loop: {args.retrain_total} DUTs total, "
              f"batch={args.retrain_batch}\n   {header}")

    n_batches = math.ceil(args.retrain_total / args.retrain_batch)
    for it in range(n_batches):
        bs = min(args.retrain_batch, args.retrain_total - seen_dut)
        # Materialize this batch of DUTs and run them through the test flow.
        batch_C, batch_acc, batch_logits, batch_faulty = build_classification_matrix(
            model, X, y, bs,
            alpha=args.alpha,
            psys_sigma=args.psys_sigma, prand_sigma=args.prand_sigma,
            inject_faults=args.inject_faults,
            faulty_dev_fraction=args.faulty_dev_fraction,
            fault_rate=args.fault_rate, sa0_fraction=args.sa0_fraction,
            rng=rng, batch_size=args.batch_size, verbose=False,
            inject_adc_faults=args.inject_adc_faults,
            adc_faulty_dev_fraction=args.adc_faulty_dev_fraction,
            adc_layer_rate=args.adc_layer_rate,
            adc_offset_low=args.adc_offset_low,
            adc_offset_high=args.adc_offset_high)

        # Signatures + regressor predictions
        Xbatch = signatures_from_logits(batch_logits, sel)
        ybatch = batch_acc
        out_flags = det.predict(Xbatch)
        pred = reg.predict(Xbatch)
        if pred.std() > 0 and ybatch.std() > 0:
            rho = float(np.corrcoef(pred, ybatch)[0, 1])
        else:
            rho = float('nan')

        n_pass = n_fail = n_fuzzy = n_outliers = false_pos = 0
        per_dut_class = []
        for i in range(bs):
            if out_flags[i] == -1:
                n_outliers += 1
                per_dut_class.append('outlier')
                continue
            mu = float(pred[i])
            s_at = _sigma_at(mu, edges, sig_bins, sig_global)
            cls = classify_pass_fail_fuzzy(mu, s_at, args.ath,
                                           k_conf=args.k_conf,
                                           t_conf=args.t_conf,
                                           df=len(ytrain) - 1)
            per_dut_class.append(cls)
            if cls == 'pass':
                n_pass += 1
                if ybatch[i] < args.ath:
                    false_pos += 1
            elif cls == 'fail':
                n_fail += 1
            else:
                n_fuzzy += 1

        # --- Speedup --------------------------------------------------------
        NST = bs * M
        n_re = n_outliers + n_fuzzy
        NAT = bs * len(sel) + n_re * M
        speedup = NST / max(1, NAT)

        # --- Test quality ---------------------------------------------------
        n_classified = n_pass + n_fail
        fpr = false_pos / max(1, n_classified)

        # Standard testing for outliers + fuzzies tells us their actual
        # accuracy.  Inliers within the PRI get added to the training set.
        for i in range(bs):
            inside_pri = (args.pri_low <= ybatch[i] <= args.pri_high)
            cls = per_dut_class[i]
            if cls == 'outlier':
                continue                       # discarded from training set
            if not inside_pri:
                continue                       # below-cutoff / above-PRI ignored
            Xtrain = np.vstack([Xtrain, Xbatch[i:i+1]])
            ytrain = np.concatenate([ytrain, ybatch[i:i+1]])

        # --- Refit the ML kernels for the next batch -----------------------
        det = OutlierDetector(contamination=max(0.01, cont)).fit(Xtrain)
        reg = MARSlikeRegressor(random_state=args.seed).fit(Xtrain, ytrain)
        pred_t = reg.predict(Xtrain)
        sigma_g = float(np.std(ytrain - pred_t, ddof=1))
        edges, sig_bins, sig_global = _interval_sigmas(
            ytrain, pred_t, args.pri_low, args.pri_high, n_bins=8)

        seen_dut += bs
        rows.append(dict(
            iter=it+1, seen=seen_dut, speedup=speedup, fpr=fpr,
            pass_=n_pass, fail=n_fail, fuzzy=n_fuzzy, outlier=n_outliers,
            sigma=sigma_g, rho=rho, train_size=len(ytrain),
        ))
        if verbose:
            print(f"   {it+1:4d} {seen_dut:5d} {speedup:7.2f}x "
                  f"{fpr*100:5.2f}% {n_pass:5d} {n_fail:5d} "
                  f"{n_fuzzy:5d} {n_outliers:5d} "
                  f"{sigma_g:8.4f} {rho:6.3f} {len(ytrain):8d}")

    if results_csv is not None:
        import csv
        os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
        with open(results_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        if verbose:
            print(f"\n[Fig.5] Wrote retraining trace -> {results_csv}")

    return rows


# ---------- 16. CLI / main ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Low-cost alternative testing of analog crossbar arrays "
                    "(Ma et al., ITC 2022) -- with itc26 ternary RRAM emulator.")

    # --- model / data
    p.add_argument('--arch', type=str, default='PureLinear',
                   choices=['LeNet5', 'SimpleMLP', 'PureLinear'],
                   help='MNIST architecture from itc26_6_5_26.py.')
    p.add_argument('--epochs', type=int, default=3,
                   help='Ternary training epochs (skipped if checkpoint exists).')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    p.add_argument('--no_cache', action='store_true',
                   help='Force retraining even if a checkpoint exists.')

    # --- variability statistics (Section IV-A)
    p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                   help='Systematic / random mix (Eq. 1).')
    p.add_argument('--psys_sigma',  type=float, default=DEFAULT_PSYS_SIG)
    p.add_argument('--prand_sigma', type=float, default=DEFAULT_PRAND_SIG)

    # --- stuck-at faults (Section VI-B)
    p.add_argument('--inject_faults', action='store_true',
                   help='Enable stuck-at fault injection on a subset of DUTs.')
    p.add_argument('--fault_rate', type=float, default=DEFAULT_FAULT_RATE,
                   help='Fraction of memristors stuck within a faulty DUT.')
    p.add_argument('--faulty_dev_fraction', type=float, default=DEFAULT_FAULTY_DEV,
                   help='Fraction of DUTs that get faults at all.')
    p.add_argument('--sa0_fraction', type=float, default=DEFAULT_SA0_FRACTION,
                   help='Within stuck cells: SA0 vs SA1 ratio.')

    # --- ADC offset faults (itc26-style, single victim channel per affected layer)
    p.add_argument('--inject_adc_faults', action='store_true',
                   help='Enable itc26-style ADC offset injection (victim channel [0,0,0] per layer).')
    p.add_argument('--adc_faulty_dev_fraction', type=float, default=DEFAULT_ADC_FAULTY_DEV,
                   help='Fraction of DUTs that exhibit any ADC offset fault.')
    p.add_argument('--adc_layer_rate', type=float, default=DEFAULT_ADC_LAYER_RATE,
                   help='Per-layer probability of ADC offset within a faulty DUT.')
    p.add_argument('--adc_offset_low', type=float, default=DEFAULT_ADC_OFFSET_LOW,
                   help='Lower bound of uniform draw for the ADC offset value.')
    p.add_argument('--adc_offset_high', type=float, default=DEFAULT_ADC_OFFSET_HIGH,
                   help='Upper bound of uniform draw for the ADC offset value.')

    # --- DUT pool sizing
    p.add_argument('--device_count',   type=int, default=200,
                   help='Held-out DUT pool for the Fig. 4 analysis.')
    p.add_argument('--bench_devices',  type=int, default=200,
                   help='Devices used to build the classification matrix.')
    p.add_argument('--test_size',      type=int, default=2000,
                   help='Number of MNIST test images to use as the universe '
                        '(paper uses 10000).')

    # --- compact subset / regressor / classifier
    p.add_argument('--compact_pct', type=float, default=3.0,
                   help='Compact-subset size as a percent of test set.')
    p.add_argument('--subset_sweep', type=float, nargs='+',
                   default=[0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0],
                   help='Percentages to sweep in the Fig. 4 analysis.')
    p.add_argument('--ath',      type=float, default=DEFAULT_ATH,
                   help='Accuracy threshold a_th (Section II).')
    p.add_argument('--pri_low',  type=float, default=DEFAULT_PRI_LOW)
    p.add_argument('--pri_high', type=float, default=DEFAULT_PRI_HIGH)
    p.add_argument('--k_conf',   type=float, default=DEFAULT_K_CONF,
                   help='k in mu +/- k*sigma (Section IV-D).')
    p.add_argument('--t_conf',   type=float, default=DEFAULT_T_CONF,
                   help='Student-t coverage (defaults to 95%).')

    # --- ML kernel retraining (Section VI-B)
    p.add_argument('--retrain_total',  type=int, default=600,
                   help='Total DUTs processed under retraining (paper: 6000).')
    p.add_argument('--retrain_batch',  type=int, default=100,
                   help='DUTs per retraining iteration.')
    p.add_argument('--skip_retrain',   action='store_true',
                   help='Run only the Fig. 4 sweep, not the Fig. 5 retraining.')
    p.add_argument('--skip_sweep',     action='store_true',
                   help='Run only the Fig. 5 retraining, not the Fig. 4 sweep.')

    # --- bookkeeping
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, default='./results',
                   help='Where CSV traces are written.')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if verbose:
        if HAVE_EARTH:
            backend = 'pyearth (MARS)'
        elif HAVE_SPLINE:
            backend = 'SplineTransformer+Ridge (MARS-like)'
        else:
            backend = 'GradientBoostingRegressor (last-resort fallback)'
        print(f"[main] device = {DEVICE}, arch = {args.arch}, "
              f"regressor backend = {backend}")

    # ---- 1. Train (or load) the base ternary model -------------------------
    train_loader, test_loader, test_ds = make_mnist_loaders(
        args.arch, batch_size=args.batch_size,
        data_root=args.data_root, test_size=args.test_size)

    ckpt = None if args.no_cache else os.path.join(
        args.ckpt_dir, f"{args.arch.lower()}_ternary.pt")
    model = train_or_load_base_model(
        args.arch, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr,
        ckpt_path=ckpt, verbose=verbose)

    # ---- 2. Build the test image tensor (we want one big tensor X) ---------
    X, y = _stack_dataset_to_tensor(test_ds, n_max=args.test_size)
    if verbose:
        print(f"[main] X test shape = {tuple(X.shape)}")

    # ---- 3. Fig. 4: Compact Test Image Subset Analysis ---------------------
    seed_bench = None
    if not args.skip_sweep:
        sweep_csv = os.path.join(args.out_dir,
                                 f"sweep_{args.arch.lower()}.csv")
        rows, seed_bench = run_compact_subset_analysis(
            model, X, y, args, rng,
            results_csv=sweep_csv, verbose=verbose)
        if verbose:
            best = max(rows, key=lambda r: r['speedup'])
            print(f"\n[Fig.4] Best speedup = {best['speedup']:.2f}x at "
                  f"pct={best['pct']}%, sigma={best['sigma']:.4f}, "
                  f"rho={best['rho']:.3f}")

    # ---- 4. Fig. 5: ML kernel retraining loop ------------------------------
    if not args.skip_retrain:
        fault_tag = ''
        if args.inject_faults:     fault_tag += '_sa'
        if args.inject_adc_faults: fault_tag += '_adc'
        retrain_csv = os.path.join(args.out_dir,
                                   f"retrain_{args.arch.lower()}{fault_tag}.csv")
        rows = run_retraining_pipeline(
            model, X, y, args, rng,
            seed_bench=seed_bench,
            results_csv=retrain_csv, verbose=verbose)
        if verbose and rows:
            last = rows[-1]
            print(f"\n[Fig.5] Final-batch speedup = {last['speedup']:.2f}x, "
                  f"FPR = {last['fpr']*100:.2f}%, "
                  f"|train| = {last['train_size']}")

    if verbose:
        print("\n[main] Done.")

if __name__ == '__main__':
    main()