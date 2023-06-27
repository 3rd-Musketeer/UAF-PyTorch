import flit_core.buildapi
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision.ops.misc import MLP

import configs.FCNN_configs


class Encoder(nn.Module):
    def __init__(self, input_channels, out_channels=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=128,
                kernel_size=8,
                padding="same",
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=256,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=8,
                padding="same",
            ),
        )

    def forward(self, x):
        return self.model(x)


class pool_and_flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return x


class LitFCNNIMP(pl.LightningModule):
    def __init__(self, config: configs):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = Encoder(config.model_config.input_channels, config.model_config.feature_channels)
        self.decoder = Decoder(config.model_config.feature_channels, config.model_config.input_channels)
        self.model = nn.Sequential(
            self.encoder,
            nn.ReLU(),
            self.decoder,
        )
        self.classifier = MLP(
            in_channels=config.model_config.feature_channels * config.dataset_config.window_length,
            hidden_channels=config.model_config.classifier_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=True,
            dropout=config.model_config.classifier_dropout,
        )
        self.full_model = nn.Sequential(
            self.encoder,
            pool_and_flatten(),
            self.classifier,
        )
        self.pretrain_loss = nn.MSELoss()
        self.finetune_loss = nn.CrossEntropyLoss()
        self.mode = None
        self.cur_epoch = 0
        self.config = config

    def pretrain(self):
        self.mode = "pretrain"

    def finetune(self):
        self.mode = "finetune"

    def configure_optimizers(self):
        encoder_opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training_config.lr,
            weight_decay=0,
        )

        classifier_opt = torch.optim.Adam(
            self.full_model.parameters(),
            lr=self.config.training_config.classifier_lr,
            weight_decay=0,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            classifier_opt,
            gamma=.1,
            step_size=self.config.training_config.lr_step,
        )

        return [encoder_opt, classifier_opt], [scheduler]

    def freeze_encoder(self):
        pass

    def unfreeze_encoder(self):
        pass

    def manual_optimize(self, loss):
        encoder_opt, classifier_opt = self.optimizers()
        scheduler = self.lr_schedulers()
        encoder_opt.zero_grad()
        if self.mode == "pretrain":
            encoder_opt.zero_grad()
        elif self.mode == "finetune":
            classifier_opt.zero_grad()
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        self.manual_backward(loss)
        if self.mode == "pretrain":
            encoder_opt.step()
        elif self.mode == "finetune":
            classifier_opt.step()
            if self.trainer.current_epoch != self.cur_epoch:
                self.cur_epoch = self.trainer.current_epoch
                scheduler.step()
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def pretrain_loop(self, batch, idx):
        y = batch[0]
        zero_mask = (torch.randn(*y.shape) > self.config.model_config.masking_ratio).to(y.device)
        x = y * zero_mask
        x_hat = self.model(x)
        loss = self.pretrain_loss(x_hat, y)
        self.manual_optimize(loss)
        self.log(f"{self.mode}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def finetune_loop(self, batch, idx):
        x, y = batch[0], batch[-1]
        logits = self.full_model(x)
        loss = self.finetune_loss(logits, y)
        self.manual_optimize(loss)
        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"finetune_{name}"] = fn(logits, y)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.log(f"{self.mode}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def predict_loop(self, batch, idx):
        x, y = batch[0], batch[-1]
        logits = self.full_model(x)
        loss = self.finetune_loss(logits, y)
        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"finetune_{name}"] = fn(logits, y)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.log(f"{self.mode}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, idx):
        if self.mode == "pretrain":
            return self.pretrain_loop(batch, idx)
        elif self.mode == "finetune":
            return self.finetune_loop(batch, idx)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def validation_step(self, batch, idx):
        return self.predict_loop(batch, idx)

    def test_step(self, batch, idx):
        return self.predict_loop(batch, idx)
