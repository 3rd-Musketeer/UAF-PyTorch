import lightning.pytorch as pl
from models.TS2Vec.encoder import TSEncoder
from models.TS2Vec.losses import hierarchical_contrastive_loss

class Lit_TS2Vec(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(
            self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

    def