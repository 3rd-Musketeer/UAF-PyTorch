import os
from torch import nn
import torch
import lightning.pytorch as pl
from utils.loss import TFCLoss
from models.TFC.TFCEncoder import TFCEncoder
from configs.TFC_configs import Configs
from torchvision.ops import MLP


class LitTFC(pl.LightningModule):
    def __init__(self, config: Configs):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = TFCEncoder(config)
        self.classifier = MLP(
            in_channels=config.model_config.projector_hidden[-1] * 2,
            hidden_channels=config.model_config.classifier_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.GELU,
            bias=True,
            dropout=config.model_config.classifier_dropout,
        )
        self.config = config
        self.loss = nn.CrossEntropyLoss()
        self.tfc_loss = TFCLoss(
            weight=config.model_config.loss_weight,
            temperature=config.model_config.loss_temperature,
            margin=config.model_config.loss_margin,
        )
        self.last_epoch = 0
        self.mode = "pretrain"
        self.freeze_enc = False
        self.save_hyperparameters()

    def pretrain(self):
        self.mode = "pretrain"

    def finetune(self):
        self.mode = "finetune"

    def configure_optimizers(self):
        optimizer_classifier = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.config.training_config.classifier_lr,
            weight_decay=self.config.training_config.classifier_weight_decay,
        )
        optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.config.training_config.encoder_plr,
            weight_decay=self.config.training_config.encoder_weight_decay,
        )

        scheduler_classifier = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_classifier,
            patience=self.config.training_config.classifier_lrs_patience,
            mode="min",
            factor=self.config.training_config.classifier_lrs_factor,
            cooldown=self.config.training_config.classifier_lrs_cooldown,
            min_lr=self.config.training_config.classifier_lrs_minlr,
        )

        scheduler_encoder = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer_encoder,
            step_size=self.config.training_config.finetune_epoch // 4,
            gamma=0.4,
        )

        self.metric_accummulator = []

        return [optimizer_encoder, optimizer_classifier], [scheduler_classifier, scheduler_encoder]

    def manual_optimize(self, loss, monitor):
        encoder_opt, classifier_opt = self.optimizers()
        classifier_lrs, encoder_lrs = self.lr_schedulers()

        encoder_opt.zero_grad()
        if self.mode == "finetune":
            classifier_opt.zero_grad()

        self.manual_backward(loss)

        if self.trainer.current_epoch < 1000 and not self.freeze_enc:
            encoder_opt.step()
        if self.mode == "finetune":
            classifier_opt.step()

        if self.trainer.current_epoch != self.last_epoch:
            self.last_epoch = self.trainer.current_epoch
            if self.mode == "finetune":
                classifier_lrs.step(torch.mean(torch.Tensor(self.metric_accummulator)))
                self.metric_accummulator = []
            encoder_lrs.step()
        elif self.mode == "finetune":
            self.metric_accummulator.append(monitor)
        else:
            pass

    def freeze_encoder(self):
        self.freeze_enc = True

    def unfreeze_encoder(self):
        self.freeze_enc = False

    def pretrain_loop(self, batch):
        x_t, x_f, aug_t, aug_f, _, _ = batch
        x_ht, x_hf, x_zt, x_zf = self.encoder(x_t, x_f)
        aug_ht, aug_hf, aug_zt, aug_zf = self.encoder(aug_t, aug_f)

        loss = self.tfc_loss(
            x_repr=(x_ht, x_hf, x_zt, x_zf),
            aug_repr=(aug_ht, aug_hf, aug_zt, aug_zf),
        )

        self.log(f"{self.mode}_TFC_loss", loss, on_epoch=True, prog_bar=True)

        self.manual_optimize(loss, 0)

        return loss

    def finetune_loop(self, batch, mode):
        # self.encoder.model.eval()
        x_t, x_f, aug_t, aug_f, _, labels = batch

        x_ht, x_hf, x_zt, x_zf = self.encoder(x_t, x_f)
        aug_ht, aug_hf, aug_zt, aug_zf = self.encoder(aug_t, aug_f)

        tfc_loss = self.tfc_loss(
            x_repr=(x_ht, x_hf, x_zt, x_zf),
            aug_repr=(aug_ht, aug_hf, aug_zt, aug_zf),
        )

        features = torch.cat((x_zt, x_zf), dim=-1)

        self.log(f"finetune_{mode}_tfc_loss", tfc_loss, on_epoch=True, prog_bar=True)

        logits = self.classifier(features)

        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"{self.mode}_{name}"] = fn(logits, labels)
        self.log_dict(metrics, on_epoch=True)
        ce_loss = self.loss(logits, labels)
        self.log(f"finetune_{mode}_ce_loss", ce_loss, on_epoch=True, prog_bar=True)

        loss = ce_loss + tfc_loss

        if mode == "train":
            self.manual_optimize(loss, ce_loss)
        return loss

    def training_step(self, batch, idx):
        if self.mode == "pretrain":
            return self.pretrain_loop(batch)
        elif self.mode == "finetune":
            return self.finetune_loop(batch, "train")
        else:
            raise ValueError(f"Unknown training mode {self.mode}")

    def validation_step(self, batch, idx):
        return self.finetune_loop(batch, "val")

    def test_step(self, batch, idx):
        return self.finetune_loop(batch, "test")

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         [
    #             {
    #                 "params": self.classifier.parameters(),
    #                 "weight_decay": self.config.training_config.encoder_weight_decay,
    #                 "lr": self.config.training_config.encoder_flr,
    #             },
    #             {
    #                 "params": self.encoder.model.parameters(),
    #                 "weight_decay": self.config.training_config.classifier_weight_decay,
    #                 "lr": self.config.training_config.classifier_lr,
    #             },
    #         ],
    #     )
    #
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer=optimizer,
    #         mode="min",
    #         factor=self.config.training_config.classifier_lrs_factor,
    #         cooldown=self.config.training_config.classifier_lrs_cooldown,
    #         min_lr=self.config.training_config.classifier_lrs_minlr,
    #     )
    #
    #     lr_scheduler_config = {
    #         # REQUIRED: The scheduler instance
    #         "scheduler": scheduler,
    #         # The unit of the scheduler's step size, could also be 'step'.
    #         # 'epoch' updates the scheduler on epoch end whereas 'step'
    #         # updates it after a optimizer update.
    #         "interval": "epoch",
    #         # How many epochs/steps should pass between calls to
    #         # `scheduler.step()`. 1 corresponds to updating the learning
    #         # rate after every epoch/step.
    #         "frequency": 1,
    #         # Metric to to monitor for schedulers like `ReduceLROnPlateau`
    #         "monitor": 'finetune_train_ce_loss',
    #         # If set to `True`, will enforce that the value specified 'monitor'
    #         # is available when the scheduler is updated, thus stopping
    #         # training if not found. If set to `False`, it will only produce a warning
    #         "strict": True,
    #         # If using the `LearningRateMonitor` callback to monitor the
    #         # learning rate progress, this keyword can be used to specify
    #         # a custom logged name
    #         "name": None,
    #     }
    #
    #     return [optimizer], [lr_scheduler_config]
