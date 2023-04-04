import torch
from numpy.fft import fft
from utils.augmentations import *
from scipy import signal as scisig
from torch.utils.data import Dataset
from tqdm import tqdm


class AugmentBank:
    def __init__(self, wl):
        self.fns = [jitter, scaling, permute]
        self.wl = wl

    def __call__(self, x_t1, x_t2):
        fn = np.random.choice(self.fns)
        if fn is neighboring_segment:
            return neighboring_segment(x_t2, self.wl)
        else:
            return fn(x_t1)


def generate_augment_pairs(sigs, labs, config):
    wl = config.window_length
    seg = config.window_length + config.window_padding
    step = config.window_step
    threshold = config.threshold
    classes = config.classes

    sos = scisig.iirfilter(4, config.pass_band, btype="lowpass", ftype='butter', output='sos', fs=config.sampling_freq)
    filter = lambda sig: scisig.sosfilt(sos, sig, axis=-1)
    time_augment = AugmentBank(wl)

    pairs = []
    mapping = {}
    for i, c in enumerate(classes):
        mapping[c] = i
    for sig, lab in tqdm(zip(sigs, labs), desc="Generating pairs"):
        assert sig.ndim == 2, "(C, T)"
        assert lab.ndim == 1, "(T)"
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
                    x_f = np.abs(fft(x_t1, axis=-1))

                    aug_t = time_augment(x_t1, x_t2)
                    aug_f = frequency_masking(x_f)

                    x_t1 = torch.tensor(x_t1, dtype=torch.float32)
                    x_f = torch.tensor(x_f, dtype=torch.float32)
                    aug_t = torch.tensor(aug_t, dtype=torch.float32)
                    aug_f = torch.tensor(aug_f, dtype=torch.float32)
                    y = torch.tensor(y, dtype=torch.int64)

                    pairs.append((x_t1, x_f, aug_t, aug_f, y))
            lf += step
            rg1 += step
            rg2 += step

    return pairs


def preprocess(data_dir, config):
    data_dict = torch.load(data_dir)
    augmented_pairs = generate_augment_pairs(data_dict["sig"], data_dict["lab"], config)
    return augmented_pairs


class TFCDataset(Dataset):
    def __init__(self, data_dir, config):
        self.augmented_pairs = preprocess(data_dir, config)

    def __getitem__(self, index):
        return self.augmented_pairs[index]

    def __len__(self):
        return len(self.augmented_pairs)
