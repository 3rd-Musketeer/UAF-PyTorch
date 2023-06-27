import os
import shutil
from dataset.EMG_Gesture_v2 import EMGGestureDataModule
from dataset.Ninapro_DB5 import NinaproDB5DataModule
from models.baselines.InceptionTime import LitInceptionTime
from models.baselines.FCNN import  LitFCNN
from configs.TSTCC_configs import Configs
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from lightning.pytorch import seed_everything
from shutil import copyfile
from preprocess.TSTCC_preprocess import TSTCCDataset
from preprocess.TFC_preprocess import TFCDataset
from utils.sampler import split_dataset
from torch.utils.data import DataLoader
import argparse
import sys
from utils.terminal_logger import TerminalLogger

parser = argparse.ArgumentParser()
parser.add_argument("--per_class_samples", type=int, default=100)
parser.add_argument("--config_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="NINA", help="EMG or NINA")
parser.add_argument("--pretrain_model_dir", type=str, default=None)
parser.add_argument("--model", type=str, default="FCNN", help="InceptionTime, FCNN")
parser.add_argument("--log_save_dir", type=str, default="run1")
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--pretrain_epoch", type=int, default=None)
parser.add_argument("--finetune_epoch", type=int, default=None)
args = parser.parse_args()

if args.experiment_name is None:
    args.experiment_name = args.model

# Initialization
if args.config_dir is None:
    config_dir = f"configs/FullSup_configs.py"
else:
    config_dir = args.config_dir

preprocess_dir = f"preprocess/TSTCC_preprocess.py"
# config_dir = r"test_run/version_23/TFC_configs.py"
import_path = ".".join(config_dir.split(".")[0].split("/"))
print(f"from {import_path} import Configs")
exec(f"from {import_path} import Configs")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision('medium')

configs = Configs(args.dataset)
for k, v in args.__dict__.items():
    if v is not None:
        configs.training_config.__setattr__(k, v)
# configs.training_config.per_class_samples = args.per_class_samples
# configs.training_config.seed = args.seed
configs.training_config.version = f"{args.dataset}_" \
                                  f"samples_{configs.training_config.per_class_samples}_" \
                                  f"pe_{configs.training_config.pretrain_epoch}_" \
                                  f"fe_{configs.training_config.finetune_epoch}_" \
                                  f"seed_{configs.training_config.seed}"

log_dir = os.path.join(
    configs.training_config.log_save_dir,
    configs.training_config.experiment_name,
    configs.training_config.version,
)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

sys.stdout = TerminalLogger(os.path.join(log_dir, "log.txt"), sys.stdout)
sys.stderr = TerminalLogger(os.path.join(log_dir, "err_log.txt"), sys.stderr)

seed_everything(configs.training_config.seed)

for fn in configs.training_config.bag_of_metrics.values():
    fn.to(device)

if args.dataset == "EMG":
    DataModule = EMGGestureDataModule
elif args.dataset == "NINA":
    DataModule = NinaproDB5DataModule
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

dataset = DataModule(
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

if args.model == "FCNN":
    lit_model = LitFCNN(configs)
elif args.model == "InceptionTime":
    lit_model = LitInceptionTime(configs)
else:
    raise ValueError("Model error!")

logger = TensorBoardLogger(
    save_dir=configs.training_config.log_save_dir,
    name=configs.training_config.experiment_name,
    version=configs.training_config.version,
)

if args.pretrain_model_dir:
    lit_model.load_from_checkpoint(args.pretrain_model_dir)

pretrain_loop = pl.Trainer(
    deterministic=False,
    max_epochs=configs.training_config.pretrain_epoch,
    precision="16-mixed",
    logger=logger,
    log_every_n_steps=1,
    enable_checkpointing=False,
)

pretrain_loop.fit(
    model=lit_model,
    train_dataloaders=DataLoader(
        dataset=pretrain_dataset,
        batch_size=configs.dataset_config.batch_size,
        shuffle=True,
    ),
)

finetune_loop = pl.Trainer(
    deterministic=False,
    max_epochs=configs.training_config.finetune_epoch,
    precision="16-mixed",
    logger=logger,
    log_every_n_steps=1,
    enable_checkpointing=False,
)

finetune_loop.fit(
    model=lit_model,
    train_dataloaders=DataLoader(
        dataset=finetune_train,
        batch_size=configs.dataset_config.batch_size,
        shuffle=True,
    ),
)

if args.finetune_epoch == 0:
    pretrain_loop.test(
        model=lit_model,
        dataloaders=DataLoader(
            dataset=finetune_test,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
        ),
    )
else:
    finetune_loop.test(
        model=lit_model,
        dataloaders=DataLoader(
            dataset=finetune_test,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
        ),
    )

config_save_dir = os.path.join(logger.log_dir, config_dir.split("/")[1])
print(f"Saving config file at {config_save_dir}")
print(copyfile(config_dir, config_save_dir))
preprocess_save_dir = os.path.join(logger.log_dir, preprocess_dir.split("/")[1])
print(f"Saving preprocess file at {preprocess_save_dir}")
print(copyfile(preprocess_dir, preprocess_save_dir))

sys.stderr.close()
sys.stdout.close()
