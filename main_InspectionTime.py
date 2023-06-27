import os
import shutil
from dataset.EMG_Gesture_v2 import EMGGestureDataModule
from configs.FCNN_configs import Configs
import torch
from lightning.pytorch import seed_everything
from preprocess.TSTCC_preprocess import TSTCCDataset
from utils.sampler import split_dataset
from tsai.all import *
from dataset.Ninapro_DB5 import NinaproDB5DataModule

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--per_class_samples", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

# Initialization
config_dir = "configs/FCNN_configs.py"
preprocess_dir = "preprocess/TSTCC_preprocess.py"
# config_dir = r"test_run/version_23/TFC_configs.py"
import_path = ".".join(config_dir.split(".")[0].split("/"))
print(f"from {import_path} import Configs")
exec(f"from {import_path} import Configs")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision('medium')

configs = Configs()
if args.per_class_samples:
    configs.training_config.per_class_samples = args.per_class_samples
if args.seed:
    configs.training_config.seed = args.seed
configs.training_config.version = f"samples_{configs.training_config.per_class_samples}_" \
                                  f"ep_{configs.training_config.epoch}_" \
                                  f"seed_{configs.training_config.seed}"

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

dataset = NinaproDB5DataModule(
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

X_train, y_train = pretrain_dataset.split()
X_test, y_test = TSTCCDataset(dataset=finetune_test).split()
print(X_train.shape, y_train.shape, np.unique(y_train))
print(X_test.shape, y_test.shape, np.unique(y_test))
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
print(len(splits))
tfms = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=128)

# dsid = 'NATOPS'
# X, y, splits = get_UCR_data(dsid, return_split=False)
# tfms = [None, [Categorize()]]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)

learn.lr_find()
learn.fit_one_cycle(100)

logits, labels = learn.get_preds(dl=dls.valid)
metrics = {}
for name, fn in configs.training_config.bag_of_metrics.items():
    metrics[f"{name}"] = fn(logits, labels)
print(metrics)


