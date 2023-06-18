import os
from models.baselines.ml_baselines import get_baseline_performance
from sklearn.ensemble import RandomForestClassifier
from dataset.EMG_Gesture_v1 import EMGGestureDataModule
from models.TS2Vec.lit_model import LitTS2Vec
from configs.TS2Vec_configs import Configs
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from lightning.pytorch import seed_everything
from shutil import copyfile
from preprocess.TS2Vec_preprocess import TS2VecDataset
from utils.sampler import PerClassSampler

# Initialization
config_dir = "configs/TS2Vec_configs.py"
preprocess_dir = "preprocess/TS2Vec_preprocess.py"
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

sampler = PerClassSampler(
    num_samples=configs.training_config.per_class_samples,
    num_classes=configs.dataset_config.num_classes
)
emg_gesture_dataset = EMGGestureDataModule(
    dataset_type=TS2VecDataset,
    config=configs.dataset_config,
    sampler=sampler,
)
emg_gesture_dataset.prepare_data()

lit_TS2Vec = LitTS2Vec(configs)

logger = TensorBoardLogger(
    save_dir=configs.training_config.log_save_dir,
    name=configs.training_config.experiment_name,
)

if "pretrain" in configs.training_config.mode:
    lit_TS2Vec.pretrain()
    pretrain_loop = pl.Trainer(
        deterministic=False,
        max_epochs=configs.training_config.pretrain_epoch,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=1,
    )

    emg_gesture_dataset.setup("pretrain")

    pretrain_loop.fit(
        model=lit_TS2Vec,
        train_dataloaders=emg_gesture_dataset.dataloader(),
    )

if "finetune" in configs.training_config.mode:
    lit_TS2Vec.finetune()
    finetune_loop = pl.Trainer(
        deterministic=False,
        max_epochs=configs.training_config.finetune_epoch,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )

    emg_gesture_dataset.setup("finetune_train")

    finetune_loop.fit(
        model=lit_TS2Vec,
        train_dataloaders=emg_gesture_dataset.dataloader(),
    )

    emg_gesture_dataset.setup("finetune_test")

    finetune_loop.test(
        model=lit_TS2Vec,
        dataloaders=emg_gesture_dataset.dataloader(),
    )

baseline_model = RandomForestClassifier(n_estimators=20, max_depth=30)

finetune_dataset = emg_gesture_dataset.current_set
train_data = finetune_dataset.sampled_set
test_data = finetune_dataset.remaining_set

get_baseline_performance(
    model=baseline_model,
    train_data=train_data,
    test_data=test_data,
)

config_save_dir = os.path.join(logger.log_dir, config_dir.split("/")[1])
print(f"Saving config file at {config_save_dir}")
print(copyfile(config_dir, config_save_dir))
preprocess_save_dir = os.path.join(logger.log_dir, preprocess_dir.split("/")[1])
print(f"Saving preprocess file at {preprocess_save_dir}")
print(copyfile(preprocess_dir, preprocess_save_dir))