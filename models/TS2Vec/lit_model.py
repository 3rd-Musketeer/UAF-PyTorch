import lightning.pytorch as pl
from torchvision.ops import MLP
import configs.TS2Vec_configs
from models.TS2Vec.ts2vec import TS2Vec
import torch.nn as nn
import torch
import numpy as np
from models.TS2Vec.utils import take_per_row
from models.TS2Vec.losses import hierarchical_contrastive_loss


class Lit_TS2Vec(pl.LightningModule):
    def __init__(self, config: configs.TS2Vec_configs.Configs):
        super().__init__()

        self.automatic_optimization = False

        self.encoder = TS2Vec(
            input_dims=config.dataset_config.channels,
            output_dims=config.model_config.repr_dims,
            hidden_dims=config.model_config.hidden_dims,
            depth=config.model_config.encoder_depth,
        )
        self._net = self.encoder.get_net()
        self.net = self.encoder.net

        self.classifier = MLP(
            in_channels=config.model_config.projector_hidden[-1] * 2,
            hidden_channels=config.model_config.classifier_hidden,
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            bias=True,
            dropout=config.model_config.classifier_dropout,
        )

        self.mode = None
        self.config = config

        self.loss = nn.CrossEntropyLoss()

    def pretrain(self):
        self.mode = "pretrain"

    def finetune(self):
        self.mode = "finetune"

    def configure_optimizers(self):
        encoder_optimizer = torch.optim.AdamW(
            self.encoder.get_net().parameters(),
            lr=self.config.training_config.pretrain_lr,
        )

        classifier_optimizer = torch.optim.Adam(
            self.classifier.paramerers(),
            lr=self.config.training_config.classifier_lr,
        )

        return [encoder_optimizer, classifier_optimizer]

    def manual_optimize(self, loss):
        encoder_opt, classifier_opt = self.optimizers()
        # classifier_lrs, encoder_lrs = self.lr_schedulers()

        encoder_opt.zero_grad()
        classifier_opt.zero_grad()

        self.manual_backward(loss)

        encoder_opt.step()
        if self.mode == "finetune":
            classifier_opt.step()

    def pretrain_loop(self, batch):
        x = batch[0]

        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.config.model_config.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
        out1 = out1[:, -crop_l:]

        out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
        out2 = out2[:, :crop_l]

        loss = hierarchical_contrastive_loss(
            out1,
            out2,
            temporal_unit=self.temporal_unit
        )

        self.manual_optimize(loss, self.mode)

        self.net.update_parameters(self._net)

        return loss

    def finetune_loop(self, batch):
        x, y = batch
        out = self.encoder.eval_with_pooling(
            x,
            encoding_window=self.config.model_config.encoding_window,
        )
        repr = out.reshape(x.shape[0], -1)

        logits = self.classifier(repr)

        loss = self.loss(logits, y)

        self.manual_optimize(loss)

        return loss

    def training_step(self, batch, idx):
        if self.mode == "pretrain":
            return self.pretrain_loop(batch)
        elif self.mode == "finetune":
            return self.finetune_loop(batch)
        else:
            raise ValueError(f"Known training mode {self.mode}")

    def validation_step(self, batch, idx):
        pass

    def test_step(self, batch, idx):
        pass