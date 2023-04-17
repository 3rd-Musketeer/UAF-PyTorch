from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC


class Configs:
    def __init__(self):
        # preprocess configs
        self.dataset_config = EMGGestureConfig()
        self.model_config = ModelConfig(self.dataset_config)
        self.training_config = TrainingConfig(self.dataset_config)


class EMGGestureConfig:
    def __init__(self):
        self.urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip"]
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
        self.loss_weight = 0.8


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
