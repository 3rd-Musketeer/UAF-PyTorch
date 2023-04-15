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
        self.partition = [0.4, 0.4, 0.2]

        self.sampling_freq = 1000
        self.pass_band = 200
        self.classes = [1, 2, 3, 4, 5, 6]
        self.window_length = 512
        self.window_padding = 32
        self.window_step = 256
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

        self.max_train_length = 3000
        self.repr_dims = 320
        self.temporal_unit = 0
        self.hidden_dims = 64
        self.encoder_depth = 10
        self.encoding_window = "full_series"


class TrainingConfig:
    def __init__(self, config):
        self.bag_of_metrics = {
            # "accuracy": Accuracy(
            #     task="multiclass",
            #     num_classes=config.num_classes,
            #     average="macro",
            # ),
            "f1": F1Score(
                task="multiclass",
                num_classes=config.num_classes,
                average="macro",
            ),
            # "precision": Precision(
            #     task="multiclass",
            #     num_classes=config.num_classes,
            #     average="macro",
            # ),
            # "recall": Recall(
            #     task="multiclass",
            #     num_classes=config.num_classes,
            #     average="macro",
            # ),
            # "auroc": AUROC(
            #     task="multiclass",
            #     num_classes=config.num_classes,
            #     average="macro",
            # ),
        }
        self.log_save_dir = "log"
        self.experiment_name = "test_phase"

        self.seed = 315
        self.pretrain_epoch = 5
        self.finetune_epoch = 70

        self.encoder_plr = 1e-4
        self.encoder_flr = 1e-5
        self.classifier_lr = 1e-3
        self.encoder_weight_decay = 3e-5
        self.classifier_weight_decay = 1e-5
        self.classifier_lrs_factor = 0.1
        self.classifier_lrs_cooldown = 5
        self.classifier_lrs_patience = 10
        self.classifier_lrs_minlr = self.classifier_lr * 1e-3