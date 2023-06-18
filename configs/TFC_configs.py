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
        self.frequency_masking_ratio = 0.01
        self.frequency_masking_damp = 0.5


class ModelConfig:
    def __init__(self, dataset_config: EMGGestureConfig):
        # (B, C, T)
        self.span = dataset_config.window_length  # keeping up with window length
        self.in_channels = dataset_config.channels
        self.num_classes = len(dataset_config.classes)

        self.projector_kernel_size = 3
        self.projector_hidden = [256, 128]
        self.projector_bias = False
        self.projector_dropout = 0.1

        self.transformer_mlp_dim = 2 * self.span
        self.transformer_n_head = 2 if self.span / 2 != 128 else 4
        self.transformer_dropout = 0.1
        self.transformer_num_layers = 2

        self.classifier_in_channels = self.projector_hidden[-1] * 2
        self.classifier_hidden = [512, self.num_classes]
        self.classifier_dropout = 0.1

        self.loss_temperature = 0.2
        self.loss_margin = 1
        self.loss_weight = 0.2


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
        self.log_save_dir = "FYP"
        self.experiment_name = "TFC/Random Init"

        self.mode = "finetune"

        self.seed = 114514
        self.pretrain_epoch = 0
        self.finetune_epoch = 100

        self.encoder_plr = 1e-4
        self.encoder_flr = 1e-5
        self.classifier_lr = 1e-3
        self.encoder_weight_decay = 3e-5
        self.classifier_weight_decay = 1e-5
        self.classifier_lrs_factor = 0.1
        self.classifier_lrs_cooldown = 5
        self.classifier_lrs_patience = 10
        self.classifier_lrs_minlr = self.classifier_lr * 1e-3

        self.per_class_samples = 100

        self.version = f"samples_{self.per_class_samples}_pe_{self.pretrain_epoch}_fe_{self.finetune_epoch}_seed_{self.seed}"
