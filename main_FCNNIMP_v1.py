import os
import shutil
import sys
from models.baselines.ml_baselines import get_baseline_performance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from dataset.EMG_Gesture_v2 import EMGGestureDataModule
from dataset.Ninapro_DB5 import NinaproDB5DataModule
from models.FCNNIMP import LitFCNNIMP
from configs.FCNNIMP_configs import Configs
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from lightning.pytorch import seed_everything
from shutil import copyfile
from preprocess.TSTCC_preprocess import TSTCCDataset
from utils.sampler import split_dataset
from utils.terminal_logger import TerminalLogger
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--per_class_samples", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Initialization
config_dir = "configs/FCNNIMP_configs.py"
preprocess_dir = "preprocess/TSTCC_preprocess.py"
# config_dir = r"test_run/version_23/TFC_configs.py"
import_path = ".".join(config_dir.split(".")[0].split("/"))
print(f"from {import_path} import Configs")
exec(f"from {import_path} import Configs")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision('medium')

configs = Configs()
configs.training_config.per_class_samples = args.per_class_samples
configs.training_config.seed = args.seed
configs.training_config.version = f"samples_{configs.training_config.per_class_samples}_pe_{configs.training_config.pretrain_epoch}_fe_{configs.training_config.finetune_epoch}_seed_{configs.training_config.seed}"


log_dir = os.path.join(
    configs.training_config.log_save_dir,
    configs.training_config.experiment_name,
    configs.training_config.version,
    "log"
)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

seed_everything(configs.training_config.seed)

for fn in configs.training_config.bag_of_metrics.values():
    fn.to(device)

# dataset = EMGGestureDataModule(
#     dataset_type=TSTCCDataset,
#     config=configs.dataset_config,
# )
dataset = EMGGestureDataModule(
    dataset_type=TSTCCDataset,
    config=configs.dataset_config,
)
dataset.prepare_data()

pretrain_dataset = dataset.pretrain_dataset()
finetune_dataset = dataset.finetune_dataset()

finetune_train, val_and_test = split_dataset(
    dataset=finetune_dataset,
    num_samples=configs.training_config.per_class_samples,
    shuffle=True,
)

finetune_val, finetune_test = split_dataset(
    dataset=val_and_test,
    num_samples=0.5,
    shuffle=True,
)

lit_FCNNIMP = LitFCNNIMP(configs)

logger = TensorBoardLogger(
    save_dir=configs.training_config.log_save_dir,
    name=configs.training_config.experiment_name,
    version=configs.training_config.version,
)

if "pretrain" in configs.training_config.mode:
    lit_FCNNIMP.pretrain()
    pretrain_loop = pl.Trainer(
        deterministic=False,
        max_epochs=configs.training_config.pretrain_epoch,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=1,
    )

    pretrain_loop.fit(
        model=lit_FCNNIMP,
        train_dataloaders=DataLoader(
            dataset=pretrain_dataset,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
            pin_memory=True,

        ),
    )

if "finetune" in configs.training_config.mode:
    lit_FCNNIMP.finetune()
    finetune_loop = pl.Trainer(
        deterministic=False,
        max_epochs=configs.training_config.finetune_epoch,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    finetune_loop.fit(
        model=lit_FCNNIMP,
        train_dataloaders=DataLoader(
            dataset=finetune_train,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
            pin_memory=True,
        ),
        val_dataloaders=DataLoader(
            dataset=finetune_val,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
    )

    finetune_loop.test(
        model=lit_FCNNIMP,
        dataloaders=DataLoader(
            dataset=finetune_test,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
            pin_memory=True,
        ),
    )

# for fn in configs.training_config.bag_of_metrics.values():
#     fn.to("cpu")

# baseline_model = [
#     DecisionTreeClassifier(),
#     KNeighborsClassifier(),
#     AdaBoostClassifier(),
# ]
#
# get_baseline_performance(
#     models=baseline_model,
#     train_data=finetune_train,
#     test_data=finetune_test,
#     metrics=configs.training_config.bag_of_metrics,
# )

config_save_dir = os.path.join(logger.log_dir, config_dir.split("/")[1])
print(f"Saving config file at {config_save_dir}")
print(copyfile(config_dir, config_save_dir))
preprocess_save_dir = os.path.join(logger.log_dir, preprocess_dir.split("/")[1])
print(f"Saving preprocess file at {preprocess_save_dir}")
print(copyfile(preprocess_dir, preprocess_save_dir))

# input("Press any key to end the process")
# log_save_dir = "log.txt"
# print(f"Saving log file at {log_save_dir}")
# if os.path.exists(os.path.join(logger.log_dir, "log.txt")):
#     os.remove(os.path.join(logger.log_dir, "log.txt"))
# print(copyfile(log_save_dir, os.path.join(logger.log_dir, "log.txt")))
# err_log_save_dir = "err_log.txt"
# print(f"Saving err log file at {err_log_save_dir}")
# if os.path.exists(os.path.join(logger.log_dir, "err_log.txt")):
#     os.remove(os.path.join(logger.log_dir, "err_log.txt"))
# print(copyfile(log_save_dir, os.path.join(logger.log_dir, "err_log.txt")))
