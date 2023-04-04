from preprocess.TFC_preprocess import TFCDataset
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC


class Configs:
    def __init__(self):
        # preprocess configs
        self.dataset_config = EMGGestureConfig()
        self.model_config = ModelConfig(self.dataset_config)
        self.training_config = TrainingConfig(self.dataset_config)


class EMGGestureConfig:
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip"
        self.save_dir = "dataset/EMGGesture"

        self.data_class = TFCDataset
        self.batch_size = 128
        self.partition = (0.5, 0.2, 0.3)

        self.sampling_freq = 1000
        self.pass_band = 200
        self.classes = (1, 2, 3, 4, 5, 6)
        self.window_length = 512
        self.window_padding = 32
        self.threshold = 0
        self.channels = 8
        self.num_classes = len(self.classes)


class ModelConfig:
    def __init__(self, dataset_config: EMGGestureConfig):
        # (B, C, T)
        self.span = dataset_config.window_length  # keeping up with window length
        self.in_channels = dataset_config.channels
        self.num_classes = len(dataset_config.classes)

        self.projector_kernel_size = 1
        self.projector_hidden = [256, 128]
        self.projector_bias = False
        self.projector_dropout = 0

        self.transformer_mlp_dim = 2 * self.span
        self.transformer_n_head = 2 if self.span / 2 != 128 else 4
        self.transformer_dropout = 0.1
        self.transformer_num_layers = 2

        self.classifier_in_channels = self.projector_hidden[-1] * 2
        self.classifier_hidden = [128, self.num_classes]
        self.classifier_dropout = 0.1

        self.loss_temperature = 0.2
        self.loss_margin = 1
        self.loss_weight = 0.5


class TrainingConfig:
    def __init__(self, config):
        self.bag_of_metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                num_classes=config.num_classes,
                average="macro",
            ),
            "f1": F1Score(
                task="multiclass",
                num_classes=config.num_classes,
                average="macro",
            ),
            "precision": Precision(
                task="multiclass",
                num_classes=config.num_classes,
                average="macro",
            ),
            "recall": Recall(
                task="multiclass",
                num_classes=config.num_classes,
                average="macro",
            ),
            "auroc": AUROC(
                task="multiclass",
                num_classes=config.num_classes,
                average="macro",
            ),
        }
        self.seed = 114514
        self.encoder_plr = 1e-3
        self.encoder_flr = 1e-5
        self.classifier_lr = 1e-3
        self.encoder_weight_decay = 3e-5
        self.classifier_weight_decay = 0
        self.classifier_lrs_factor = 0.1
        self.classifier_lrs_cooldown = 10
        self.classifier_lrs_minlr = self.classifier_lr * 1e-3
