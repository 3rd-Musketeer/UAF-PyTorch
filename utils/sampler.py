import torch
import numpy as np


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
                lf = np.random.randint(0, len(cl_pos[cl])-self.num_samples, 1)[0]
                rg = lf + self.num_samples
                idx += cl_pos[cl][lf:rg]
                idx_inv += cl_pos[cl][:lf] + cl_pos[cl][rg:]
            elif isinstance(self.num_samples, float):
                num = int(self.num_samples * len(cl_pos[cl]))
                lf = np.random.randint(0, len(cl_pos[cl])-num, 1)[0]
                rg = lf + num
                idx += cl_pos[cl][lf:rg]
                idx_inv += cl_pos[cl][:lf] + cl_pos[cl][rg:]
                if int(len(cl_pos[cl]) * self.num_samples) <= 0:
                    raise RuntimeWarning(f"num_samples of class {cl} is zero!")
            else:
                raise ValueError(f"Invalid input value type {type(self.num_samples)}")
        return [dataset[i] for i in idx], [dataset[i] for i in idx_inv]
