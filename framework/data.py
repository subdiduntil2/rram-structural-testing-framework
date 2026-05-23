import os
import wave
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from config import (
    DATASET, KWS_FEATURE_TYPE,
    INPUT_MIN, INPUT_MAX,
)


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
