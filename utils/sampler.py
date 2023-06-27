import torch
import numpy as np


def split_dataset(dataset, num_samples, num_classes=None, shuffle=True):
    all_labels = []
    all_subs = []
    for idx in range(len(dataset)):
        batch = dataset[idx]
        all_labels.append(batch[-1].item())
        all_subs.append(batch[-2])
    if num_classes is None:
        num_classes = len(np.unique(all_labels))
    else:
        num_classes = num_classes
    map_sub = {}
    for i, sub in enumerate(np.unique(all_subs)):
        map_sub[sub] = i
    num_sub = len(map_sub)
    cl_pos = [[[] for _ in range(num_sub)] for _ in range(num_classes)]
    for i, (y, sub) in enumerate(zip(all_labels, all_subs)):
        cl_pos[y][map_sub[sub]].append(i)
    idx = []
    idx_inv = []
    cl_cnt = []
    num_sampled_data = 0
    for cl in range(num_classes):
        cnt = np.sum([len(cl_pos[cl][s]) for s in range(num_sub)])
        cl_cnt.append(cnt)
        if isinstance(num_samples, int):
            num = num_samples
        elif isinstance(num_samples, float):
            num = int(num_samples * cnt)
            if int(cnt * num_samples) <= 0:
                raise RuntimeWarning(f"num_samples of class {cl} is zero!")
        else:
            raise ValueError(f"Invalid input value type {type(num_samples)}")
        num_sample_sub = int(np.round(num / num_sub))
        for s in range(num_sub):
            if shuffle:
                np.random.shuffle(cl_pos[cl][s])
            lf = np.random.randint(0, max(len(cl_pos[cl][s]) - num_sample_sub, 1), 1)[0]
            rg = lf + num_sample_sub
            idx += cl_pos[cl][s][lf:rg]
            idx_inv += cl_pos[cl][s][:lf] + cl_pos[cl][s][rg:]
            num_sampled_data += num_sample_sub
    print("Class count: ", cl_cnt, "Sum: ", np.sum(cl_cnt), "Sampled data: ", num_sampled_data)
    return [dataset[i] for i in idx], [dataset[i] for i in idx_inv]


class PerClassSampler:
    def __init__(self, num_samples, num_classes=None):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __call__(self, dataset):
        all_labels = []
        for idx in range(len(dataset)):
            batch = dataset[idx]
            all_labels.append(batch[-1].item())
        if self.num_classes is None:
            num_classes = len(torch.unique(all_labels))
        else:
            num_classes = self.num_classes
        cl_pos = [[] for _ in range(num_classes)]
        for i, y in enumerate(all_labels):
            cl_pos[y].append(i)
        idx = []
        idx_inv = []
        for cl in range(num_classes):
            np.random.shuffle(cl_pos[cl])
            if isinstance(self.num_samples, int):
                lf = np.random.randint(0, len(cl_pos[cl]) - self.num_samples, 1)[0]
                rg = lf + self.num_samples
                idx += cl_pos[cl][lf:rg]
                idx_inv += cl_pos[cl][:lf] + cl_pos[cl][rg:]
            elif isinstance(self.num_samples, float):
                num = int(self.num_samples * len(cl_pos[cl]))
                lf = np.random.randint(0, len(cl_pos[cl]) - num, 1)[0]
                rg = lf + num
                idx += cl_pos[cl][lf:rg]
                idx_inv += cl_pos[cl][:lf] + cl_pos[cl][rg:]
                if int(len(cl_pos[cl]) * self.num_samples) <= 0:
                    raise RuntimeWarning(f"num_samples of class {cl} is zero!")
            else:
                raise ValueError(f"Invalid input value type {type(self.num_samples)}")
        return [dataset[i] for i in idx], [dataset[i] for i in idx_inv]
