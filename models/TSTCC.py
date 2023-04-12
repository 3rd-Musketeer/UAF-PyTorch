import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import lightning.pytorch as pl
from utils.loss import NTXentLoss


########################################################################################
import configs.TSTCC_configs


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t


class base_Model(nn.Module):
    def __init__(self, configs: configs.TSTCC_configs.ModelConfig):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding="same"),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x


class TC(nn.Module):
    def __init__(self, configs: configs.TSTCC_configs.ModelConfig):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.hidden_dim, self.num_channels) for _ in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()

        self.projection_head = nn.Sequential(
            nn.Linear(configs.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.hidden_dim, depth=4,
                                               heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        device = features_aug1.device
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(
            device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)


class LitTSTCC(pl.LightningModule):
    def __init__(
            self,
            config: configs.TSTCC_configs.Configs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.model = base_Model(config.model_config)
        self.temporal_contr_model = TC(config.model_config)
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
            weight_decay=3e-4
        )

        return [model_optimizer, temporal_contr_optimizer]

    def manual_optimize(self, loss, mode):
        model_optimizer, temporal_contr_optimizer = self.optimizers()

        model_optimizer.zero_grad()
        if mode == "pretrain":
            temporal_contr_optimizer.zero_grad()

        self.manual_backward(loss)

        model_optimizer.step()
        if mode == "pretrain":
            temporal_contr_optimizer.step()

    def train_loop(self, batch, idx, mode):
        x, weak_aug, strong_aug, y = batch
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
            predictions, features = output
            target = F.one_hot(y, num_classes=self.config.dataset_config.num_classes)
            loss = self.loss(predictions, y)
        else:
            raise ValueError(f"Unknown mode \"{mode}\"")

        self.log("loss", loss, prog_bar=True, on_epoch=True)
        self.manual_optimize(loss, self.mode)

        return loss

    def predict_loop(self, batch, idx):
        x, _, _, y = batch
        predictions, features = self.model(x)
        target = F.one_hot(y, num_classes=self.config.dataset_config.num_classes)
        loss = self.loss(predictions, y)

        metrics = {}
        for name, fn in self.config.training_config.bag_of_metrics.items():
            metrics[f"test_{name}"] = fn(predictions, y)
        self.log_dict(metrics)

        return loss

    def training_step(self, batch, idx):
        self.train_loop(batch, idx, mode=self.mode)

    def validation_step(self, batch, idx):
        self.predict_loop(batch, idx)

    def test_step(self, batch, idx):
        self.predict_loop(batch, idx)

