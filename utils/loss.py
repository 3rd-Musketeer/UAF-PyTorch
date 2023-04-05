import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class NTXentLoss_poly(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat(
            (torch.ones(2 * self.batch_size, 1), torch.zeros(2 * self.batch_size, negatives.shape[-1])), dim=-1).to(
            self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label * torch.nn.functional.softmax(logits, dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1 / self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss


class NTXentLoss:
    """NTXentLoss as originally described in https://arxiv.org/abs/2002.05709.
    This loss is used in self-supervised learning setups and requires the two views of the input datapoint
    to be returned distinctly by Sender and Receiver.
    Note that this loss considers in-batch negatives and and negatives samples are taken within each agent
    datapoints i.e. each non-target element in sender_input and in receiver_input is considered a negative sample.
    >>> x_i = torch.eye(128)
    >>> x_j = torch.eye(128)
    >>> loss_fn = NTXentLoss()
    >>> loss, aux = loss_fn(None, x_i, None, x_j, None, None)
    >>> aux["acc"].mean().item()
    1.0
    >>> aux["acc"].shape
    torch.Size([256])
    >>> x_i = torch.eye(256)
    >>> x_j = torch.eye(128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    Traceback (most recent call last):
        ...
    RuntimeError: sender_output and receiver_output must be of the same shape, found ... instead
    >>> _ = torch.manual_seed(111)
    >>> x_i = torch.rand(128, 128)
    >>> x_j = torch.rand(128, 128)
    >>> loss, aux = NTXentLoss.ntxent_loss(x_i, x_j)
    >>> aux['acc'].mean().item() * 100  # chance level with a batch size of 128, 1/128 * 100 = 0.78125
    0.78125
    """

    def __init__(
            self,
            temperature: float = 1.0,
            similarity: str = "cosine",
    ):
        self.temperature = temperature

        similarities = {"cosine", "dot"}
        assert (
                similarity.lower() in similarities
        ), f"Cannot recognize similarity function {similarity}"
        self.similarity = similarity.lower()

    @staticmethod
    def ntxent_loss(
            sender_output: torch.Tensor,
            receiver_output: torch.Tensor,
            temperature: float = 1.0,
            similarity: str = "cosine",
    ):

        if sender_output.shape != receiver_output.shape:
            raise RuntimeError(
                f"sender_output and receiver_output must be of the same shape, "
                f"found {sender_output.shape} and {receiver_output.shape} instead"
            )
        batch_size = sender_output.shape[0]

        input = torch.cat((sender_output, receiver_output), dim=0)

        if similarity == "cosine":
            similarity_f = torch.nn.CosineSimilarity(dim=2)
            similarity_matrix = (
                    similarity_f(input.unsqueeze(1), input.unsqueeze(0)) / temperature
            )
        else:
            similarity_matrix = input @ input.t()

        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(batch_size * 2, 1)

        mask = torch.ones(batch_size * 2, batch_size * 2, dtype=torch.bool).fill_diagonal_(0)

        negative_samples = similarity_matrix[mask].reshape(batch_size * 2, -1)

        labels = torch.zeros(batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = F.cross_entropy(logits, labels, reduction="mean") / 2

        # acc = (torch.argmax(logits.detach(), dim=1) == labels).float().detach()
        return loss

    def __call__(self, sender_output, receiver_output):
        return self.ntxent_loss(
            sender_output,
            receiver_output,
            temperature=self.temperature,
            similarity=self.similarity,
        )


class TripletLoss(nn.Module):
    """
    Compute normal triplet loss or soft margin triplet loss given triplets
    """

    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


class TFCLoss(nn.Module):
    def __init__(self, weight=0.5, temperature=0.2, margin=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrast_loss = NTXentLoss(temperature)
        # self.contrast_loss = NTXentLoss_poly(
        #     "cuda",
        #     256,
        #     0.2,
        #     True
        # )
        self.triplet_loss = lambda ap, an: ap - an + margin
        self.weight = weight

    def forward(self, x_repr, aug_repr, mode="pretrain"):
        x_ht, x_hf, x_zt, x_zf = x_repr
        aug_ht, aug_hf, aug_zt, aug_zf = aug_repr

        sim_t = self.contrast_loss(x_ht, aug_ht)
        sim_f = self.contrast_loss(x_hf, aug_hf)
        sim_loss = (sim_t + sim_f)

        ap = self.contrast_loss(x_zt, x_zf)
        an1 = self.contrast_loss(x_zt, aug_zf)
        an2 = self.contrast_loss(aug_zt, x_zf)
        an3 = self.contrast_loss(aug_zt, aug_zf)
        tfc_loss = (self.triplet_loss(ap, an1) +
                    self.triplet_loss(ap, an2) +
                    self.triplet_loss(ap, an3))
        if mode == "pretrain":
            return self.weight * tfc_loss + (1 - self.weight) * sim_loss
        elif mode == "finetune":
            return self.weight * ap + (1 - self.weight) * sim_loss
        else:
            raise ValueError(f"Unknow mode {mode}")
