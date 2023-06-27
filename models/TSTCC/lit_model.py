import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from utils.loss import NTXentLoss
from models.TSTCC.model import base_Model
from models.TSTCC.TC import TC
from torchvision.ops.misc import MLP

########################################################################################
import configs.TSTCC_configs


class LitTSTCC(pl.LightningModule):
    def __init__(
            self,
            config: configs.TSTCC_configs.Configs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.model = base_Model(config.model_config)
        self.temporal_contr_model = TC(config.model_config)
        self.classifier = MLP(
            in_channels=config.model_config.feature_len,
            hidden_channels=config.model_config.classifier_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=True,
            dropout=config.model_config.classifier_dropout,
        )
        self.contrast_loss = NTXentLoss(config.model_config.loss_temperature)
        self.loss = nn.CrossEntropyLoss()
        self.mode = None
        self.config = config

    def pretrain(self):
        self.mode = "pretrain"

    def finetune(self):
        self.mode = "finetune"

    def random_init_encoder(self):
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def set_grad(self, del_list=("logits"), requires_grad=False):
        model_dict = self.model.state_dict().copy()
        for param_key in model_dict.keys():
            for del_key in del_list:
                if del_key in param_key:
                    del model_dict[param_key]
        for k, v in self.model.parameters():
            if k in model_dict:
                v.requires_grad = requires_grad

    def freeze_encoder(self):
        self.set_grad(del_list=("logits"), requires_grad=False)

    def unfreeze_encoder(self):
        self.set_grad(del_list=("logits"), requires_grad=True)

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training_config.lr,
            weight_decay=3e-4,
        )
        temporal_contr_optimizer = torch.optim.Adam(
            self.temporal_contr_model.parameters(),
            lr=self.config.training_config.lr,
            weight_decay=3e-4,
        )

        classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.config.training_config.classifier_lr,
            weight_decay=self.config.training_config.classifier_weight_decay,
        )

        return [model_optimizer, temporal_contr_optimizer, classifier_optimizer]

    def manual_optimize(self, loss, mode):
        model_optimizer, temporal_contr_optimizer, classifier_optimizer = self.optimizers()

        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        if mode == "pretrain":
            temporal_contr_optimizer.zero_grad()

        self.manual_backward(loss)

        model_optimizer.step()
        if mode == "finetune":
            classifier_optimizer.step()
        if mode == "pretrain":
            temporal_contr_optimizer.step()

    def train_loop(self, batch, idx, mode):
        x, weak_aug, strong_aug, _, y = batch
        if mode == "pretrain":
            predictions1, features1 = self.model(weak_aug)
            predictions2, features2 = self.model(strong_aug)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)

            temp_cont_loss1, temp_cont_lstm_feat1 = self.temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = self.temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1
            zjs = temp_cont_lstm_feat2
        else:
            output = self.model(x)

        # compute loss
        if mode == "pretrain":
            lambda1 = 1
            lambda2 = 0.7
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + self.contrast_loss(zis, zjs) * lambda2
        elif mode == "finetune":  # supervised training or fine-tuning
            features, _ = output
            # target = F.one_hot(y, num_classes=self.config.dataset_config.num_classes)
            predictions = self.classifier(features)
            loss = self.loss(predictions, y)

            metrics = {}
            for name, fn in self.config.training_config.bag_of_metrics.items():
                metrics[f"finetune_{name}"] = fn(predictions, y)
            self.log_dict(metrics, on_epoch=True, prog_bar=True)
        else:
            raise ValueError(f"Unknown mode \"{mode}\"")

        self.log(f"{mode}_loss", loss, prog_bar=True, on_epoch=True)
        self.manual_optimize(loss, self.mode)

        return loss

    def predict_loop(self, batch, idx):
        x, _, _, _, y = batch
        features, _ = self.model(x)
        predictions = self.classifier(features)
        # target = F.one_hot(y, num_classes=self.config.dataset_config.num_classes)
        loss = self.loss(predictions, y)

        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"test_{name}"] = fn(predictions, y)
        self.log_dict(metrics)

        return loss

    def training_step(self, batch, idx):
        return self.train_loop(batch, idx, mode=self.mode)

    def validation_step(self, batch, idx):
        return self.predict_loop(batch, idx)

    def test_step(self, batch, idx):
        return self.predict_loop(batch, idx)

    def get_features(self, x):
        features, _ = self.model(x)
        return features
