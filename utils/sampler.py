import torch
import numpy as np


def split_list(list_, partition):
    num = len(list_)
    num_splits = [0]
    for num_i in partition:
        if isinstance(num_i, float):
            num_splits.append(int(num * num_i))
        elif isinstance(num_i, int):
            num_splits.append(num_i)
        else:
            raise ValueError(f"Invalid partition number {num_i}. It should be either float or int!")
    total_num = sum(num_splits)
    step_splits = np.cumsum(num_splits)
    if total_num > num:
        raise RuntimeError(f"No enough instances to partition")
    splits = []
    for lf, rg in zip(step_splits[:-1], step_splits[1:]):
        splits.append(list_[lf:rg])
    item_pnt = total_num
    while item_pnt < num:
        group_pnt = 0
        while item_pnt < num and group_pnt < len(splits):
            splits[group_pnt].append(list_[item_pnt])
            item_pnt += 1
            group_pnt += 1
    return splits


def partition_subject_wise_dataset(dataset, partition):
    splits = split_list(dataset, partition)
    for i, group in enumerate(splits):
        tmp = []
        for item in group:
            tmp += item
    return splits


def per_class_divide(dataset, partition, num_classes=None):
    all_labels = []
    for idx in range(len(dataset)):
        batch = dataset[idx]
        all_labels.append(batch[-1].item())
    if num_classes is None:
        num_classes = len(torch.unique(all_labels))
    else:
        num_classes = num_classes
    cl_pos = [[] for _ in range(num_classes)]
    for i, y in enumerate(all_labels):
        cl_pos[y].append(i)
    idx = []
    idx_inv = []
    for cl in range(num_classes):
        np.random.shuffle(cl_pos[cl])
        if isinstance(num_samples, int):
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
