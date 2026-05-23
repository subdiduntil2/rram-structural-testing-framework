import os
import csv
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import DEVICE, EPOCHS, LOG_DIR, ANALYSIS_IMAGE_LIMIT
from simulator_core import ANALYZER
from mapping import RRAMConv2d, RRAMLinear


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
