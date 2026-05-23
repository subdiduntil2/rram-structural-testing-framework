import os
import shutil
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import config
from config import (
    DEVICE, DATASET, MODEL_ARCH, MLP_NUM_LAYERS, USE_BATCH_NORM,
    VALIDATE_ON_FULL_DATASET, RUN_FAULT_INJECTION, RUN_TARGET_GENERATION,
    EVAL_MODE, READOUT_MODE, KWS_FEATURE_TYPE, KWS_NUM_CLASSES,
    SYNTHETIC_LOG_DIR, LOG_DIR, FRESH_LOGS,
    USE_SHIFTED_TARGETS, ATPG_STRINGS,
    LIF_THRESHOLDS, LIF_TIME_STEPS, LIF_LEAK, EPOCHS,
)
from models import get_model, log_mapped_weights
from data import GSCDataset, KWSCollate, SyntheticPassDataset
from atpg import (
    parse_multiple_atpg_pairs, generate_shifted_pairs,
    flatten_and_check_global_pair_duplicates, sweep,
    PATTERNS, LIF_PATTERNS, adc_offsets_32_2, lif_spike_offsets,
)
from fault_injection import (
    train_kws_pipeline, train_one_epoch, evaluate_detailed,
    evaluate_synthetic_faults, run_masking_analysis,
)
from target_gen import check_and_generate_all_16_cases
from lif_readout import (
    check_and_generate_all_16_cases_lif, evaluate_lif,
    evaluate_synthetic_faults_lif, run_lif_masking_analysis,
)


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    print("device is => ", DEVICE)

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

    print("\n[System] Parsing ATPG strings into paired Input/Weight vectors...")
    config.INPUT_TARGETS.clear()
    config.WEIGHT_TARGETS.clear()
    config.ATPG_STRING_MAPPING.clear()
    config.REVERSE_MAPPING.clear()
    
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
                config.INPUT_TARGETS.append(flat_inps[v_idx])
                config.WEIGHT_TARGETS.append(flat_wgts[v_idx])
                found_idx = len(flat_target_pool) - 1
                
            if found_idx not in target_indices:
                target_indices.append(found_idx)
                
        config.ATPG_STRING_MAPPING[atpg_str] = target_indices
        print(f"      Mapped to {len(target_indices)} unique binary pairs.")

    # INPUT_TARGETS = flat_target_pool
    print(f"         -> Total unique global binary targets to simulate: {len(config.INPUT_TARGETS)}\n")
    print("target indices for each ATPG string:",config.INPUT_TARGETS)
    
    # global ATPG_STRING_MAPPING
    atpg_keys = list(config.ATPG_STRING_MAPPING.keys()) # Defined before it's used
    
    # 1. Map each target index to a list of ALL matching string indices
    for s_idx, atpg_str in enumerate(atpg_keys):
        for t_idx in config.ATPG_STRING_MAPPING[atpg_str]:
            if t_idx not in config.REVERSE_MAPPING:
                config.REVERSE_MAPPING[t_idx] = []
            config.REVERSE_MAPPING[t_idx].append(str(s_idx))

    # 2. Convert the lists into joined strings (e.g., "0_11_13") 
    # so the rest of the script's filename generators work untouched.
    for t_idx in config.REVERSE_MAPPING:
        config.REVERSE_MAPPING[t_idx] = "_".join(config.REVERSE_MAPPING[t_idx])

    # INPUT_TARGETS = [
    # [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # print("final input targets are => ",INPUT_TARGETS) 
    print(f"         -> Successfully generated {len(config.INPUT_TARGETS)} {type(config.INPUT_TARGETS)} unique 32-element targets.\n")
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
                            model_rram.inject_target_weights(layer_name, config.WEIGHT_TARGETS[t_idx])

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
                            model_rram.inject_target_weights(layer_name, config.WEIGHT_TARGETS[t_idx])
                            
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
