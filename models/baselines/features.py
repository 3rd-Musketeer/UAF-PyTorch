import numpy as np
import torch


def MAV(lst):
    return np.mean(np.abs(lst), keepdims=True)


def MAV_pt(lst):
    lst = torch.from_numpy(lst)
    return (torch.mean(torch.abs(lst), dim=-1, keepdim=True)).numpy()


def RMS(lst):
    return np.sqrt(np.mean(np.square(lst), keepdims=True))


def RMS_pt(lst):
    lst = torch.from_numpy(lst)
    return (torch.sqrt(torch.mean(torch.square(lst), dim=-1, keepdim=True))).numpy()


def SSC(lst):
    lst = np.pad(lst, (1, 1), 'constant', constant_values=0)
    res = []
    for i in range(1, len(lst) - 1):
        res.append(int((lst[i] - lst[i - 1]) * (lst[i] - lst[i + 1]) >= 0))
    return np.sum(res, keepdims=True)


def SSC_pt(lst):
    lst = np.pad(lst, (1, 1), 'constant', constant_values=0)
    res = []
    for i in range(1, len(lst) - 1):
        res.append(int((lst[i] - lst[i - 1]) * (lst[i] - lst[i + 1]) >= 0))
    return np.sum(res, keepdims=True)


def WL(lst):
    lst = np.pad(lst, (1, 0), 'constant', constant_values=0)
    return np.sum(np.abs(np.diff(lst)), keepdims=True)


def WL_pt(lst):
    lst = np.pad(lst, (1, 0), 'constant', constant_values=0)
    lst = torch.from_numpy(lst)
    return torch.sum(torch.abs(torch.diff(lst)), dim=-1, keepdim=True)


def HP(lst):
    Ahp = np.var(lst)

    # mob = lambda lst: np.sqrt(var(np.gradient(lst)) / var(lst))
    def mob(lst):
        denom = np.var(lst)
        numer = np.sqrt(np.var(np.gradient(lst)))
        if denom == 0 or numer == 0:
            return 1
        else:
            return numer / denom

    Mhp = mob(lst)
    Chp = mob(np.gradient(lst)) / Mhp
    return np.array([Ahp, Mhp, Chp])
