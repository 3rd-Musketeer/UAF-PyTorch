import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.fft import fft, rfft
from utils.augmentations import *
from scipy import signal as scisig
from torch.utils.data import Dataset
from tqdm import tqdm


def generate_augment_pairs(sigs, labs, subs, config):
    wl = config.window_length
    step = config.window_step
    threshold = config.threshold
    classes = config.classes
    if config.pass_band:
        sos = scisig.iirfilter(4, config.pass_band, btype="lowpass", ftype='butter', output='sos', fs=config.sampling_freq)
        filter = lambda sig: scisig.sosfilt(sos, sig, axis=-1)

    pairs = []
    mapping = {}
    for i, c in enumerate(classes):
        mapping[c] = i
    cl = []
    for sig, lab, sub in tqdm(zip(sigs, labs, subs), desc="Generating pairs"):
        assert sig.ndim == 2, f"(C, T) {sig.shape}"
        assert lab.ndim == 1, f"(T) {lab.shape}"
        if config.pass_band:
            sig = filter(sig)
        lf, rg = 0, wl
        while rg < sig.shape[-1]:
            y = lab[lf:rg]
            if max(np.bincount(y)) / len(y) > threshold:
                y = np.argmax(np.bincount(y))
                if y in classes:
                    y = mapping[y]
                    x = sig[:, lf:rg]

                    weak_aug = scaling(jitter(x))
                    strong_aug = jitter(permute(x))

                    x = torch.tensor(x, dtype=torch.float32)
                    weak_aug = torch.tensor(weak_aug, dtype=torch.float32)
                    strong_aug = torch.tensor(strong_aug, dtype=torch.float32)
                    y = torch.tensor(y, dtype=torch.int64)
                    cl.append(y)
                    pairs.append((x, weak_aug, strong_aug, sub, y))
            lf += step
            rg += step
    counts = np.bincount(cl)
    mean_counts = np.mean(counts)
    print("Dataset class count:", counts, "Sum: ", len(cl))
    if (mean_counts < counts).any:
        new_pairs = []
        cc = []
        print("Re-balancing classes")
        for c in range(len(counts)):
            tmp = []
            for ins in pairs:
                if ins[-1] == c:
                    tmp.append(ins)
            assert counts[c] == len(tmp)
            if counts[c] > 2 * mean_counts:
                num_samples = int(mean_counts ** 2 / counts[c])
                idx = np.random.choice(len(tmp), num_samples)
                for i in idx:
                    new_pairs.append(pairs[i])
                cc.append(num_samples)
            else:
                new_pairs += tmp
                cc.append(len(tmp))
        pairs = new_pairs
        print("Re-balanced class counts:", cc, "Sum:", sum(cc))
    return pairs


def preprocess(data_dir, config):
    data_dict = torch.load(data_dir)
    augmented_pairs = generate_augment_pairs(
        data_dict["sig"],
        data_dict["lab"],
        data_dict["sub"],
        config,
    )
    return augmented_pairs


class TSTCCDataset(Dataset):
    def __init__(self, data_dir, config):
        self.dataset = preprocess(data_dir, config)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
