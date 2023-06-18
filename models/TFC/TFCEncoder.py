import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.ops import MLP
from configs.TFC_configs import Configs


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
                padding="same",
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
                padding="same",
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
