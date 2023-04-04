import os

from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.ops import MLP
from configs.TFC_configs import Configs
import lightning.pytorch as pl
from utils.loss import TFCLoss

"""Two contrastive encoders"""


class TFCEncoder(nn.Module):
    def __init__(self, config: Configs):
        super(TFCEncoder, self).__init__()
        config = config.model_config
        encoder_layers_t = TransformerEncoderLayer(d_model=config.span,
                                                   nhead=config.transformer_n_head,
                                                   dim_feedforward=config.transformer_mlp_dim,
                                                   dropout=config.transformer_dropout,
                                                   batch_first=True,
                                                   )
        self.transformer_encoder_t = nn.Sequential(
            TransformerEncoder(encoder_layers_t, config.transformer_num_layers),
            nn.Conv1d(
                in_channels=config.in_channels,
                out_channels=1,
                kernel_size=config.projector_kernel_size,
                bias=False
            ),
        )
        self.projector_t = MLP(
            in_channels=config.span,
            hidden_channels=config.projector_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=config.projector_bias,
            dropout=config.projector_dropout,
        )

        encoder_layers_f = TransformerEncoderLayer(d_model=config.span,
                                                   nhead=config.transformer_n_head,
                                                   dim_feedforward=config.transformer_mlp_dim,
                                                   dropout=config.transformer_dropout,
                                                   batch_first=True,
                                                   )
        self.transformer_encoder_f = nn.Sequential(
            TransformerEncoder(encoder_layers_f, config.transformer_num_layers),
            nn.Conv1d(
                in_channels=config.in_channels,
                out_channels=1,
                kernel_size=config.projector_kernel_size,
                bias=False
            ),
        )
        self.projector_f = MLP(
            in_channels=config.span,
            hidden_channels=config.projector_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=config.projector_bias,
            dropout=config.projector_dropout,
        )

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        # input (B, C, T)
        # Transformer requires (B, T, C)
        """Time-based contrastive encoder"""
        h_t = self.transformer_encoder_t(x_in_t).squeeze(-2)
        # Now (B, F)
        """Cross-space projector"""
        z_t = self.projector_t(h_t)
        """Frequency-based contrastive encoder"""
        h_f = self.transformer_encoder_f(x_in_f).squeeze(-2)
        """Cross-space projector"""
        z_f = self.projector_f(h_f)
        return h_t, h_f, z_t, z_f


class LitTFCEncoder(pl.LightningModule):
    def __init__(self, config: Configs):
        super().__init__()
        self.model = TFCEncoder(config)
        self.loss = TFCLoss(
            weight=config.model_config.loss_weight,
            temperature=config.model_config.loss_temperature,
            margin=config.model_config.loss_margin,
        )
        self.config = config
        self.save_hyperparameters()

    def training_step(self, batch, idx):
        return self.pretrain_loop(batch, idx, "pretrain")

    def validation_step(self, batch, idx):
        return self.pretrain_loop(batch, idx, "preval")

    def pretrain_loop(self, batch, idx, mode):
        x_t, x_f, aug_t, aug_f, _ = batch
        x_ht, x_hf, x_zt, x_zf = self.model(x_t, x_f)
        aug_ht, aug_hf, aug_zt, aug_zf = self.model(aug_t, aug_f)

        loss = self.loss(
            x_repr=(x_ht, x_hf, x_zt, x_zf),
            aug_repr=(aug_ht, aug_hf, aug_zt, aug_zf),
        )

        self.log(f"{mode}_TFC_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training_config.encoder_plr,
            weight_decay=self.config.training_config.encoder_weight_decay,
        )
        return optimizer

    def forward(self, batch, idx):
        x_t, x_f, aug_t, aug_f, _ = batch
        x_ht, x_hf, x_zt, x_zf = self.model(x_t, x_f)
        aug_ht, aug_hf, aug_zt, aug_zf = self.model(aug_t, aug_f)

        loss = self.loss(
            x_repr=(x_ht, x_hf, x_zt, x_zf),
            aug_repr=(aug_ht, aug_hf, aug_zt, aug_zf),
        )

        feature = torch.concatenate([x_zt, x_zf], dim=-1)

        return loss, feature


class LitTFC(pl.LightningModule):
    def __init__(self, pretrained_encoder_path, config: Configs):
        super().__init__()
        self.automatic_optimization = False
        if pretrained_encoder_path:
            self.encoder = LitTFCEncoder.load_from_checkpoint(pretrained_encoder_path)
        else:
            self.encoder = LitTFCEncoder(config)
        self.classifier = MLP(
            in_channels=config.model_config.projector_hidden[-1] * 2,
            hidden_channels=config.model_config.classifier_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=True,
            dropout=config.model_config.classifier_dropout,
        )
        self.config = config
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def training_step(self, batch, idx):
        return self.finetune_loop(batch, idx, "finetune_train")

    def validation_step(self, batch, idx):
        return self.finetune_loop(batch, idx, "finetune_val")

    def test_step(self, batch, idx):
        return self.finetune_loop(batch, idx, "finetune_test")

    def finetune_loop(self, batch, idx, mode):
        # self.encoder.model.eval()
        labels = batch[-1]
        tfc_loss, features = self.encoder(batch, idx)
        self.log(f"{mode}_tfc_loss", tfc_loss)
        logits = self.classifier(features)
        preds = torch.softmax(logits, dim=-1)
        targets = nn.functional.one_hot(
            labels,
            num_classes=self.config.dataset_config.num_classes
        ).float()
        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"{mode}_{name}"] = fn(preds, labels)
        self.log_dict(metrics)
        ce_loss = self.loss(preds, targets)
        self.log(f"{mode}_ce_loss", ce_loss)

        loss = ce_loss + tfc_loss

        if "train" in mode:
            self.manual_optimize(loss)

        return loss

    def manual_optimize(self, loss):
        opts = self.optimizers()
        lrs = self.lr_schedulers()

        for opt in opts:
            opt.zero_grad()

        self.manual_backward(loss)

        for opt in opts:
            opt.step()

        lrs.step(loss)

    def configure_optimizers(self):
        optimizer_classifier = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.config.training_config.classifier_lr,
            weight_decay=self.config.training_config.encoder_weight_decay,
        )
        optimizer_encoder = torch.optim.Adam(
            self.encoder.model.parameters(),
            lr=self.config.training_config.encoder_flr,
            weight_decay=self.config.training_config.encoder_weight_decay,
        )

        scheduler_classifier = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_classifier,
            mode="min",
            factor=self.config.training_config.classifier_lrs_factor,
            cooldown=self.config.training_config.classifier_lrs_cooldown,
            min_lr=self.config.training_config.classifier_lrs_minlr,
        )

        return [optimizer_encoder, optimizer_classifier], [scheduler_classifier]
