import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from models.TimesNet.Embed import DataEmbedding
from models.TimesNet.Conv_Blocks import Inception_Block_V1
import lightning.pytorch as pl
from torchvision.ops import MLP
from models.baselines.FCNN import FCNN


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None


class test_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(
                config.enc_in * config.seq_len,
                1024,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(
                1024,
                config.num_class,
            ),
        )

    def forward(self, x):
        return self.model(x.reshape(x.shape[0], -1))


class LitTimesNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.model = Model(config.model_config)
        # self.model = FCNN(
        #     8, 6
        # )
        self.loss = nn.CrossEntropyLoss()
        self.config = config
        self.last_epoch=0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training_config.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.training_config.lr_step,
            gamma=0.5,
        )

        return [optimizer], [lr_scheduler]
        #return optimizer

    def manual_optimize(self, loss):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
        optimizer.step()
        if self.trainer.current_epoch != self.last_epoch:
            self.last_epoch = self.trainer.current_epoch
            scheduler.step()

    def train_loop(self, batch, mode):
        x, y = batch[0], batch[-1]
        B, C, T = x.shape
        x = x.swapaxes(1, 2)
        logits = self.model(x)
        loss = self.loss(logits, y)
        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"{mode}_{name}"] = fn(logits, y)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        self.manual_optimize(loss)
        return loss

    def training_step(self, batch, idx):
        return self.train_loop(batch, "train")

    def validation_step(self, batch, idx):
        return self.train_loop(batch, "val")

    def test_step(self, batch, idx):
        return self.train_loop(batch, "test")
