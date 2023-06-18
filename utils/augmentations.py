import numpy as np
from configs.TFC_configs import Configs

# Temporal data format (Batch, Channel, Time)
config = Configs().dataset_config


def reshape(fn):
    def wrapper(signals, *args, **kwargs):
        assert isinstance(signals, np.ndarray)
        prev_shape = signals.shape
        assert signals.ndim in [2, 3], "Input data should be either (B, T), (C, T), (B, C, T)"
        if signals.ndim == 2:
            signals = signals[np.newaxis, ...]
        return fn(signals, *args, **kwargs).reshape(*prev_shape[:-1], -1)

    return wrapper


# Time Series Augmentation

@reshape
def jitter(signals, ratio=config.jitter_ratio):
    # step-wise
    scale = np.broadcast_to(ratio * np.abs(np.amax(signals, axis=-1, keepdims=True)), signals.shape)
    noise = np.random.normal(loc=0, scale=scale, size=signals.shape)
    return signals + noise


@reshape
def scaling(signals, ratio=config.scaling_ratio):
    # instance-wise
    factor = np.broadcast_to(np.random.normal(loc=1, scale=ratio, size=(signals.shape[0], 1, 1)), signals.shape)
    return signals * factor


@reshape
def permute_channels(signals):
    # return np.apply_along_axis(np.random.permutation, axis=1, arr=signals)
    # two_idx = np.random.choice(signals.shape[1], size=2)
    # tmp = signals[:, two_idx[0], :]
    # signals[:, two_idx[0], :] = signals[:, two_idx[1], :]
    # signals[:, two_idx[1], :] = tmp
    # shift_idx = np.random.randint(signals.shape[1], size=1)[0]
    shift_idx = signals.shape[1] // 2
    lf = signals[:, :shift_idx, :]
    rg = signals[:, shift_idx:, :]
    return np.concatenate([lf, rg], axis=1)


@reshape
def neighboring_segment(signals, window_length):
    span = signals.shape[-1]
    lfs = np.random.randint(int(span - window_length + 1), size=signals.shape[0])
    return np.stack([sig[..., lf:lf + window_length] for lf, sig in zip(lfs, signals)])


@reshape
def permute(signals, num_seg=config.num_permute):
    # instance-wise
    span = signals.shape[-1]
    num_seg = np.random.randint(2, min(num_seg, span) + 1, size=signals.shape[0])
    ret = []
    for sig, num in zip(signals, num_seg):
        small_seg_span = span // num
        long_seg_num = span % small_seg_span
        sub_spans = np.cumsum(
            ([small_seg_span] * (num - long_seg_num) +
             [small_seg_span + 1] * long_seg_num)[:-1]
        )
        split = np.split(sig, sub_spans, axis=-1)
        np.random.shuffle(split)
        ret.append(np.concatenate(split, axis=-1))
    return np.stack(ret)


@reshape
def reverse(signals):
    return signals[..., ::-1].copy()


# Frequency Spectrum Augmentation

@reshape
def frequency_masking(spectrums, ratio=config.frequency_masking_ratio, damp=config.frequency_masking_damp):
    # step-wise
    remove_mask = np.random.rand(*spectrums.shape) >= ratio
    add_mask = np.random.rand(*spectrums.shape) <= ratio
    return spectrums * remove_mask + add_mask * damp * np.amax(spectrums, axis=-1, keepdims=True)


if __name__ == "__main__":
    @reshape
    def fn(x):
        print(x.shape)
        return x


    sig = np.random.rand(8, 500)
    print((1, *sig.shape))
    ret = fn(sig)
