import torch
import torch.nn as nn
import lightning.pytorch as pl


class FCNN(nn.Module):
    def __init__(
            self,
            input_channels,
            num_features,
    ):
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
                out_channels=128,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=num_features,
                kernel_size=1,
                padding="same",
            ),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits.reshape(logits.shape[0], -1)


class LitFCNN(pl.LightningModule):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.model = FCNN(
            config.model_config.input_channels,
            config.model_config.feature_len,
        )
        self.classifier = nn.Linear(config.model_config.feature_len, config.dataset_config.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.config = config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training_config.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.training_config.lr_step,
        )

        return [optimizer], [lr_scheduler]

    def train_loop(self, batch, mode):
        x, y = batch[0], batch[-1]
        features = self.model(x)
        logits = self.classifier(features)
        loss = self.loss(logits, y)
        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"{mode}_{name}"] = fn(logits, y)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, idx):
        return self.train_loop(batch, "train")

    def validation_step(self, batch, idx):
        return self.train_loop(batch, "val")

    def test_step(self, batch, idx):
        return self.train_loop(batch, "test")

    def get_features(self, x):
        return self.model(x)