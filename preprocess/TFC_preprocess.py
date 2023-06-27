import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.fft import fft, rfft
from utils.augmentations import *
from scipy import signal as scisig
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


class AugmentBank:
    def __init__(self, config):
        self.fns = [jitter, scaling, permute]
        if "N" in config.augmentation:
            self.fns.append(neighboring_segment)
        if "P" in config.augmentation:
            self.fns.append(permute_channels)
        self.wl = config.window_length

    def __call__(self, x_t1, x_t2):
        fn = np.random.choice(self.fns)
        if fn is neighboring_segment:
            return neighboring_segment(x_t2, self.wl)
        else:
            return fn(x_t1)


def generate_augment_pairs(sigs, labs, subs, config):
    wl = config.window_length
    seg = config.window_length + config.window_padding
    step = config.window_step
    threshold = config.threshold
    classes = config.classes
    if config.pass_band:
        sos = scisig.iirfilter(4, config.pass_band, btype="lowpass", ftype='butter', output='sos', fs=config.sampling_freq)
        filter = lambda sig: scisig.sosfilt(sos, sig, axis=-1)
    time_augment = AugmentBank(config)
    pairs = []
    mapping = {}
    for i, c in enumerate(classes):
        mapping[c] = i
    cl = []
    for sig, lab, sub in tqdm(zip(sigs, labs, subs), desc="Generating pairs"):
        assert sig.ndim == 2, "(C, T)"
        assert lab.ndim == 1, "(T)"
        if config.pass_band:
            sig = filter(sig)
        lf, rg1, rg2 = 0, wl, seg
        while rg2 < sig.shape[-1]:
            y = lab[lf:rg1]
            if max(np.bincount(y)) / len(y) > threshold:
                y = np.argmax(np.bincount(y))
                if y in classes:
                    y = mapping[y]
                    x_t1 = sig[:, lf:rg1]
                    x_t2 = sig[:, lf:rg2]

                    # spectrum = np.fft.rfft(x_t1, axis=-1) / wl

                    x_f = np.abs(fft(x_t1, axis=-1, norm="ortho"))
                    # magnitude = np.abs(spectrum)[..., :-1]
                    # phase = np.abs(np.angle(spectrum)[..., :-1])
                    # plt.subplot(2, 1, 1)
                    # plt.plot(magnitude[0, ...])
                    # plt.subplot(2, 1, 2)
                    # plt.plot(phase[0, ...])
                    # plt.show()

                    # x_f = np.concatenate(
                    #     [magnitude, phase],
                    #     axis=-1,
                    # )

                    aug_t = time_augment(x_t1, x_t2)
                    aug_f = frequency_masking(x_f)

                    x_t1 = torch.tensor(x_t1, dtype=torch.float32)
                    x_f = torch.tensor(x_f, dtype=torch.float32)
                    aug_t = torch.tensor(aug_t, dtype=torch.float32)
                    aug_f = torch.tensor(aug_f, dtype=torch.float32)
                    y = torch.tensor(y, dtype=torch.int64)

                    cl.append(y)
                    pairs.append((x_t1, x_f, aug_t, aug_f, sub, y))
            lf += step
            rg1 += step
            rg2 += step
    print("Dataset Class count:", np.bincount(cl), "Sum: ", len(cl))
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


class TFCDataset(Dataset):
    def __init__(self, data_dir=None, config=None, dataset=None):
        if dataset is None:
            self.dataset = preprocess(data_dir, config)
        else:
            self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def split(self):
        Xs = []
        ys = []
        for item in self.dataset:
            Xs.append(item[0].unsqueeze(0))
            ys.append(item[-1].unsqueeze(0))
        return torch.concatenate(Xs).numpy(), torch.concatenate(ys).numpy()
