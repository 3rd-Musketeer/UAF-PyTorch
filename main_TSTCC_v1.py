import os
from models.baselines.ML_baselines import get_baseline_performance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from dataset.EMG_Gesture_v2 import EMGGestureDataModule
from models.TSTCC.lit_model import LitTSTCC
from configs.TSTCC_configs import Configs
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from lightning.pytorch import seed_everything
from shutil import copyfile
from preprocess.TSTCC_preprocess import TSTCCDataset
from utils.sampler import split_dataset
from torch.utils.data import DataLoader

# Initialization
config_dir = "configs/TSTCC_configs.py"
preprocess_dir = "preprocess/TSTCC_preprocess.py"
# config_dir = r"test_run/version_23/TFC_configs.py"
import_path = ".".join(config_dir.split(".")[0].split("/"))
print(f"from {import_path} import Configs")
exec(f"from {import_path} import Configs")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision('medium')
configs = Configs()
seed_everything(configs.training_config.seed)

for fn in configs.training_config.bag_of_metrics.values():
    fn.to(device)

emg_gesture_dataset = EMGGestureDataModule(
    dataset_type=TSTCCDataset,
    config=configs.dataset_config,
)
emg_gesture_dataset.prepare_data()

pretrain_dataset = emg_gesture_dataset.pretrain_dataset()
finetune_dataset = emg_gesture_dataset.finetune_dataset()

finetune_train, val_and_test = split_dataset(
    dataset=finetune_dataset,
    num_samples=0.5,
    shuffle=True,
)

finetune_val, finetune_test = split_dataset(
    dataset=val_and_test,
    num_samples=0.2,
    shuffle=True,
)

lit_TSTCC = LitTSTCC(configs)

logger = TensorBoardLogger(
    save_dir=configs.training_config.log_save_dir,
    name=configs.training_config.experiment_name,
)

if "pretrain" in configs.training_config.mode:
    lit_TSTCC.pretrain()
    pretrain_loop = pl.Trainer(
        deterministic=False,
        max_epochs=configs.training_config.pretrain_epoch,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=1,
    )

    pretrain_loop.fit(
        model=lit_TSTCC,
        train_dataloaders=DataLoader(
            dataset=pretrain_dataset,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
        ),
    )

if "freeze" in configs.training_config.mode:
    lit_TSTCC.freeze_encoder()

if "finetune" in configs.training_config.mode:
    lit_TSTCC.finetune()
    finetune_loop = pl.Trainer(
        deterministic=False,
        max_epochs=configs.training_config.finetune_epoch,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    finetune_loop.fit(
        model=lit_TSTCC,
        train_dataloaders=DataLoader(
            dataset=finetune_train,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
        ),
        val_dataloaders=DataLoader(
            dataset=finetune_val,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
        )
    )

    finetune_loop.test(
        model=lit_TSTCC,
        dataloaders=DataLoader(
            dataset=finetune_test,
            batch_size=configs.dataset_config.batch_size,
            shuffle=True,
        ),
    )

baseline_model = DecisionTreeClassifier()

get_baseline_performance(
    model=baseline_model,
    train_data=finetune_train,
    test_data=finetune_test,
)

config_save_dir = os.path.join(logger.log_dir, config_dir.split("/")[1])
print(f"Saving config file at {config_save_dir}")
print(copyfile(config_dir, config_save_dir))
preprocess_save_dir = os.path.join(logger.log_dir, preprocess_dir.split("/")[1])
print(f"Saving preprocess file at {preprocess_save_dir}")
print(copyfile(preprocess_dir, preprocess_save_dir))