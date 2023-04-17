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

    sos = scisig.iirfilter(4, config.pass_band, btype="lowpass", ftype='butter', output='sos', fs=config.sampling_freq)
    filter = lambda sig: scisig.sosfilt(sos, sig, axis=-1)

    pairs = []
    mapping = {}
    for i, c in enumerate(classes):
        mapping[c] = i
    for sig, lab in tqdm(zip(sigs, labs), desc="Generating pairs"):
        assert sig.ndim == 2, "(C, T)"
        assert lab.ndim == 1, "(T)"
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


def preprocess(data_dict, config):
    augmented_pairs = generate_augment_pairs(data_dict["signal"], data_dict["label"], config)
    return augmented_pairs
