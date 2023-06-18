import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import lightning.pytorch as pl

class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = 0
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None


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
