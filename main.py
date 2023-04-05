import os
from models.ML_baselines import get_baseline_performance
from sklearn.ensemble import RandomForestClassifier
from dataset.EMG_Gesture import EMGGestureDataModule
from models.TFC import LitTFCEncoder, LitTFC
from configs.TFC_configs import Configs
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from lightning.pytorch import seed_everything
from shutil import copyfile

# Initialization
config_dir = "configs/TFC_configs.py"
preprocess_dir = "preprocess/TFC_preprocess.py"
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


emg_gesture_dataset = EMGGestureDataModule(configs.dataset_config)
emg_gesture_dataset.prepare_data()

lit_TFC_encoder = LitTFCEncoder(configs)

logger = TensorBoardLogger(
    save_dir="",
    name="test_run"
)

pretrain_loop = pl.Trainer(
    deterministic=False,
    max_epochs=configs.training_config.pretrain_epoch,
    precision="16-mixed",
    logger=logger,
    log_every_n_steps=1,
)

emg_gesture_dataset.setup("fit")

pretrain_loop.fit(
    model=lit_TFC_encoder,
    train_dataloaders=emg_gesture_dataset.train_dataloader(),
    val_dataloaders=emg_gesture_dataset.val_dataloader(),
)

save_dir = os.path.join(logger.log_dir, config_dir.split("/")[1])
print(f"Saving config file at {save_dir}")
print(copyfile(config_dir, save_dir))
print(f"Saving preprocess file at {save_dir}")
print(copyfile(preprocess_dir, save_dir))

ckpt_path = os.path.join(logger.log_dir, "checkpoints")
if os.path.exists(ckpt_path):
    pretrained_model_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
else:
    pretrained_model_path = None

lit_TFC = LitTFC(pretrained_model_path, configs)

finetune_loop = pl.Trainer(
    deterministic=False,
    max_epochs=configs.training_config.finetune_epoch,
    precision="16-mixed",
    logger=logger,
    log_every_n_steps=1,
    enable_checkpointing=False
)

finetune_loop.fit(
    model=lit_TFC,
    train_dataloaders=emg_gesture_dataset.val_dataloader(),
)

emg_gesture_dataset.setup("test")

finetune_loop.test(
    model=lit_TFC,
    dataloaders=emg_gesture_dataset.test_dataloader(),
)

baseline_model = RandomForestClassifier(n_estimators=20, max_depth=30)
get_baseline_performance(
    model=baseline_model,
    train_loader=emg_gesture_dataset.train_dataloader(),
    test_loader=emg_gesture_dataset.test_dataloader(),
)
