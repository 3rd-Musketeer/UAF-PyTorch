from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC


class Configs:
    def __init__(self, dataset="EMG"):
        # preprocess configs
        if dataset == "EMG":
            self.dataset_config = EMGGestureConfig()
        elif dataset == "NINA":
            self.dataset_config = NinaproDB5Config()
        self.model_config = ModelConfig(self.dataset_config)
        self.training_config = TrainingConfig(self.dataset_config)


class EMGGestureConfig:
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip"
        self.save_dir = "dataset/EMGGesture"

        self.batch_size = 256
        self.partition = [0.8, 0., 0.2]

        self.sampling_freq = 1000
        self.pass_band = 200
        self.classes = [1, 2, 3, 4, 5, 6]
        self.window_length = 256
        self.window_padding = 32
        self.window_step = 64
        self.threshold = 0
        self.channels = 8
        self.num_classes = len(self.classes)

        self.jitter_ratio = 0.1
        self.scaling_ratio = 0.1
        self.num_permute = 8


class NinaproDB5Config:
    def __init__(self):
        self.url = [
            "http://ninapro.hevs.ch/download/file/fid/457",
            "http://ninapro.hevs.ch/download/file/fid/458",
            "http://ninapro.hevs.ch/download/file/fid/459",
            "http://ninapro.hevs.ch/download/file/fid/467",
            "http://ninapro.hevs.ch/download/file/fid/461",
            "http://ninapro.hevs.ch/download/file/fid/462",
            "http://ninapro.hevs.ch/download/file/fid/463",
            "http://ninapro.hevs.ch/download/file/fid/464",
            "http://ninapro.hevs.ch/download/file/fid/465",
            "http://ninapro.hevs.ch/download/file/fid/466",
        ]
        self.save_dir = "dataset/Ninapro_DB5"

        self.batch_size = 256
        self.partition = [0.6, 0, 0.4]

        self.sampling_freq = 200
        self.pass_band = None
        self.classes = [0, 6, 13, 14, 15, 16]
        self.window_length = 512
        self.window_padding = 32
        self.window_step = 64
        self.threshold = 0
        self.channels = 8
        self.num_classes = len(self.classes)

        self.jitter_ratio = 0.1
        self.scaling_ratio = 0.1
        self.num_permute = 8
        self.frequency_masking_ratio = 0.01
        self.frequency_masking_damp = 0.5


class ModelConfig:
    def __init__(self, dataset_config):
        # (B, C, T)
        self.span = dataset_config.window_length  # keeping up with window length

        self.input_channels = dataset_config.channels
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = dataset_config.num_classes
        self.dropout = 0.35
        self.conv_output_dim = self.span // 8
        self.feature_len = 128

        self.hidden_dim = 100
        self.timesteps = self.conv_output_dim // 4

        self.loss_temperature = 0.2

        self.classifier_hidden = [512, self.num_classes]
        self.classifier_dropout = 0.15


class TrainingConfig:
    def __init__(self, config):
        self.bag_of_metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                num_classes=config.num_classes,
                average="micro",
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
        self.log_save_dir = "run1"
        self.experiment_name = "TSTCC"

        self.mode = "pretrain_finetune"

        self.seed = 42
        self.pretrain_epoch = 100
        self.finetune_epoch = 100

        self.lr = 3e-4

        self.classifier_lr = 1e-4
        self.classifier_weight_decay = 3e-3

        self.per_class_samples = 100

        self.version = f"samples_{self.per_class_samples}_pe_{self.pretrain_epoch}_fe_{self.finetune_epoch}_seed_{self.seed}"

