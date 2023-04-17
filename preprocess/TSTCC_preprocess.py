import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.fft import fft, rfft
from utils.augmentations import *
from scipy import signal as scisig
from torch.utils.data import Dataset
from tqdm import tqdm


def generate_augment_pairs(sigs, labs, config):
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
    for sig, lab in tqdm(zip(sigs, labs), desc="Generating pairs"):
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

                    pairs.append((x, weak_aug, strong_aug, y))
            lf += step
            rg += step

    return pairs


def preprocess(data_dir, config):
    data_dict = torch.load(data_dir)
    augmented_pairs = generate_augment_pairs(data_dict["sig"], data_dict["lab"], config)
    return augmented_pairs


class TSTCCDataset(Dataset):
    def __init__(self, data_dir, config, mode=None, sampler=None):
        self.data = preprocess(data_dir, config)
        if sampler:
            self.sampled_set, self.remaining_set = sampler(self.data)
        self.mode = None

    def __getitem__(self, index):
        if self.mode == "train":
            return self.sampled_set[index]
        elif self.mode == "test":
            return self.remaining_set[index]
        else:
            return self.data[index]

    def __len__(self):
        if self.mode == "train":
            return len(self.sampled_set)
        elif self.mode == "test":
            return len(self.remaining_set)
        else:
            return len(self.data)

    def train(self):
        self.mode = "train"

    def test(self):
        self.mode = "test"

    def all(self):
        self.mode = "all"
