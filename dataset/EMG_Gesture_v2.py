import os
import lightning.pytorch as pl
from configs.TSTCC_configs import EMGGestureConfig
from utils.download import download
import zipfile
import pandas as pd
import numpy as np
from torch.utils.data import random_split
import torch
import json


def raw_to_np(path, save_dir):
    dirs = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    def dir2filedir(dir):
        files = []
        for p in dir:
            for f in os.listdir(p):
                fd = os.path.join(p, f)
                if os.path.isfile(fd) and '.txt' in f:
                    files.append(fd)
        return files

    data = extract_raw_data(dir2filedir(dirs))
    torch.save(data, save_dir)
    return data


def extract_raw_data(file_paths):
    data = []
    for fp in file_paths:
        doc = pd.read_csv(fp, sep='\t').values.swapaxes(0, 1)
        signal = np.array(doc[1: 9]).astype('float32')
        label = np.array(doc[9]).astype('uint8')
        data.append(dict(signal=signal, label=label))
    return data


class EMGGestureDataModule:
    def __init__(
            self,
            config: EMGGestureConfig,
    ):
        super(EMGGestureDataModule, self).__init__()
        self.config = config
        self.data = None

    def prepare_data(self, save_dir=None):
        # download dataset
        file_name = os.path.split(self.config.url)[1]
        dir_name = file_name.split(".")[0]
        target_dir = os.path.join(self.config.save_dir, dir_name)
        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)
            file_dir = download(self.config.url, self.config.save_dir)
            if not os.path.exists(os.path.join(self.config.save_dir, dir_name)):
                with zipfile.ZipFile(file_dir, mode="r") as file:
                    file.extractall(self.config.save_dir)
        if save_dir is None:
            save_dir = os.path.join(self.config.save_dir, "data.pt")
        if not os.path.exists(save_dir):
            self.data = raw_to_np(target_dir, save_dir)
        else:
            self.data = torch.load(save_dir)

    def preprocess_data(self, preprocess_fn):
        subject_wise_dataset = []
        for subject in self.data:
            subject_wise_dataset.append(preprocess_fn(subject))
        return subject_wise_dataset