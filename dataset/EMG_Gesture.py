import os
import lightning.pytorch as pl
from configs.TFC_configs import EMGGestureConfig
from utils.download import download
import zipfile
import pandas as pd
import numpy as np
from preprocess.TFC_preprocess import TFCDataset
from torch.utils.data import DataLoader
import torch


def get_file_dirs(path, partition):
    dirs = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    np.random.shuffle(dirs)
    train_size = int(len(dirs) * partition[0])
    val_size = int(len(dirs) * partition[1])
    train_path = dirs[: train_size]
    val_path = dirs[train_size: train_size + val_size]
    test_path = dirs[train_size + val_size:]
    print("Number of patients:\ntrain: {}\neval: {}\ntest: {}".format(len(train_path), len(val_path), len(test_path)))

    def path2filedir(path):
        files = []
        for p in path:
            for f in os.listdir(p):
                fd = os.path.join(p, f)
                if os.path.isfile(fd) and '.txt' in f:
                    files.append(fd)
        return files

    train_files = path2filedir(train_path)
    val_files = path2filedir(val_path)
    test_files = path2filedir(test_path)

    return train_files, val_files, test_files


def extract_raw_data(file_paths):
    sig_sequences = []
    lab_sequences = []
    for fp in file_paths:
        doc = pd.read_csv(fp, sep='\t').values.swapaxes(0, 1)
        signal = np.array(doc[1: 9]).astype('float32')
        label = np.array(doc[9]).astype('uint8')
        sig_sequences.append(signal)
        lab_sequences.append(label)
    return sig_sequences, lab_sequences


class EMGGestureDataModule(pl.LightningDataModule):
    def __init__(
            self,
            config: EMGGestureConfig,
    ):
        super(EMGGestureDataModule, self).__init__()
        self.config = config

    def prepare_data(self) -> None:
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
        # split dataset
        train_files, val_files, test_files = get_file_dirs(target_dir, self.config.partition)
        # preprocess dataset
        if not os.path.exists(path := os.path.join(self.config.save_dir, "train_set.pt")):
            train_sig, train_lab = extract_raw_data(train_files)
            torch.save({"sig": train_sig, "lab": train_lab}, path)
        if not os.path.exists(path := os.path.join(self.config.save_dir, "val_set.pt")):
            val_sig, val_lab = extract_raw_data(val_files)
            torch.save({"sig": val_sig, "lab": val_lab}, path)
        if not os.path.exists(path := os.path.join(self.config.save_dir, "test_set.pt")):
            test_sig, test_lab = extract_raw_data(test_files)
            torch.save({"sig": test_sig, "lab": test_lab}, path)

    def setup(self, stage):
        if stage == "fit":
            self.train_set = TFCDataset(
                os.path.join(self.config.save_dir, "train_set.pt"),
                self.config,
            )
            self.val_set = TFCDataset(
                os.path.join(self.config.save_dir, "val_set.pt"),
                self.config,
            )
        elif stage == "test" or stage == "predict":
            self.test_set = TFCDataset(
                os.path.join(self.config.save_dir, "test_set.pt"),
                self.config,
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.config.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size)
