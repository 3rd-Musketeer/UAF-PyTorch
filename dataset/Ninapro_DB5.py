import os
import lightning.pytorch as pl
from configs.TSTCC_configs import NinaproDB5Config
from utils.download import download
import zipfile
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
import torch
import json


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
            sub_number = os.path.split(p)[-1][1:]
            for f in os.listdir(p):
                fd = os.path.join(p, f)
                if os.path.isfile(fd) and '.mat' in f and 'E2' in f:
                    files.append((fd, int(sub_number)))
        return files

    train_files = path2filedir(train_path)
    val_files = path2filedir(val_path)
    test_files = path2filedir(test_path)

    return train_files, val_files, test_files


def extract_raw_data(file_paths):
    sig_sequences = []
    lab_sequences = []
    sub_sequences = []
    for fp, sub in file_paths:
        matfile = scio.loadmat(fp)
        signal = np.array(matfile["emg"][..., :8].swapaxes(0, 1), dtype="float32")
        label = np.array(matfile["restimulus"], dtype="uint8").squeeze(-1)
        sig_sequences.append(signal)
        lab_sequences.append(label)
        sub_sequences.append(sub)
    return sig_sequences, lab_sequences, sub_sequences


class NinaproDB5DataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_type,
            config: NinaproDB5Config,
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.config = config
        self.similarity_check = ["partition", "sampling_freq", "pass_band"]

    def prepare_data(self, auto_download_and_zip=False) -> None:
        # download dataset
        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)
            if auto_download_and_zip:
                for url in self.config.urls:
                    file_name = os.path.split(url)[-1]
                    dir_name = os.path.join(self.config.save_dir, file_name)
                    file_dir = download(url, self.config.save_dir)
                    if not os.path.exists(os.path.join(self.config.save_dir, dir_name)):
                        with zipfile.ZipFile(file_dir, mode="r") as file:
                            file.extractall(self.config.save_dir)

        dataset_config = {
            itm: self.config.__getattribute__(itm) for itm in self.similarity_check
        }
        similarity_flag = True

        # split dataset
        if os.path.exists(jsfile_path := os.path.join(self.config.save_dir, "dataset_config.json")):
            with open(jsfile_path, "r") as jsfile:
                prev_dataset_config = json.load(jsfile)
            for itm in self.similarity_check:
                if prev_dataset_config[itm] != dataset_config[itm]:
                    similarity_flag = False
                    break
        else:
            similarity_flag = False

        if not similarity_flag:
            train_files, val_files, test_files = get_file_dirs(self.config.save_dir, self.config.partition)
            print("Generating cache dataset")
            with open(jsfile_path, "w") as jsfile:
                json.dump(dataset_config, jsfile)
            # preprocess dataset
            # if not os.path.exists(path := os.path.join(self.config.save_dir, "train_set.pt")):
            path = os.path.join(self.config.save_dir, "train_set.pt")
            train_sig, train_lab, train_sub = extract_raw_data(train_files)
            torch.save({"sig": train_sig, "lab": train_lab, "sub": train_sub}, path)
            # if not os.path.exists(path := os.path.join(self.config.save_dir, "val_set.pt")):
            path = os.path.join(self.config.save_dir, "val_set.pt")
            val_sig, val_lab, val_sub = extract_raw_data(val_files)
            torch.save({"sig": val_sig, "lab": val_lab, "sub": val_sub}, path)
            # if not os.path.exists(path := os.path.join(self.config.save_dir, "test_set.pt")):
            path = os.path.join(self.config.save_dir, "test_set.pt")
            test_sig, test_lab, test_sub = extract_raw_data(test_files)
            torch.save({"sig": test_sig, "lab": test_lab, "sub": test_sub}, path)

    def pretrain_dataset(self):
        return self.dataset_type(
            os.path.join(self.config.save_dir, "train_set.pt"),
            self.config,
        )

    def finetune_dataset(self):
        return self.dataset_type(
            os.path.join(self.config.save_dir, "test_set.pt"),
            self.config,
        )

# def test():
#     from configs.TSTCC_configs import NinaproDB5Config
#     from preprocess.TSTCC_preprocess import TSTCCDataset
#     config = NinaproDB5Config()
#     ninapro_db5_dataset = NinaproDB5DataModule(config=config, dataset_type=TSTCCDataset)
#     ninapro_db5_dataset.prepare_data()
#     pretrain_dataset = ninapro_db5_dataset.pretrain_dataset()
#     finetune_dataset = ninapro_db5_dataset.finetune_dataset()