from tsai.models.InceptionTime import InceptionTime
import torch
import torch.nn as nn
import lightning.pytorch as pl


class LitInceptionTime(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = InceptionTime(config.dataset_config.channels, config.model_config.feature_len)
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
