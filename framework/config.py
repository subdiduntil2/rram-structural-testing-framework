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
WEIGHT_TARGETS = []
REVERSE_MAPPING = {}
ATPG_STRING_MAPPING = {}
