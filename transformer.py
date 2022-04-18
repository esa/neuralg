import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Needed for training and evaluation
from losses import *
from RandomMatrixDataSet import (
    get_sample,
    RandomMatrixDataSet,
    SingularvalueMatrix,
    EigenMatrix,
)
from train import train_on_batch, run_training
from evaluation import *
from plotting import plot_loss_logs, error_histogram, plot_mean_identity_approx
import math
from typing import Tuple
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_hid: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_seq_length: int = 128,
        k_size: int = (2, 2),
    ):

        super().__init__()

        d_model = math.prod(k_size)

        self.d_model = d_model
        self.n_heads = n_heads
        self.k_size = k_size

        # The position encoding is concatenated to the input
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        encoder_layers = TransformerEncoderLayer(
            d_model,
            n_heads,
            d_hid,
            dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layers, n_layers)

        # This is what I added in terms of decoder
        decoder_layers = TransformerDecoderLayer(
            d_model,
            n_heads,
            d_hid,
            dropout,
            batch_first=True,
        )
        self.decoder = TransformerDecoder(decoder_layers, n_layers)

        self.init_weights()

    def init_weights(self) -> None:
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        return

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, d_model]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """

        src_unfolded = F.unfold(src, kernel_size=self.k_size).permute(0, 2, 1)
        # print(f'==[ src unfolded: {src_unfolded}')
        # print(f"==[ src unfolded shape: {src_unfolded.shape}")

        # src_pos_encoded = self.pos_encoder(src_unfolded)
        src_pos_encoded = self.pos_encoder(src_unfolded)
        # print(f'==[ src_pos_encoded: {src_pos_encoded}')
        # print(f'==[ src_pos_encoded shape: {src_pos_encoded.shape}')

        mask = self.generate_square_subsequent_mask(
            src_unfolded.shape[1], src.shape[0] * self.n_heads
        )
        # print(f'==[ mask: {mask}')
        # print(f'==[ mask shape: {mask.shape}')

        # encoded_src = self.encoder(src_pos_encoded, mask)
        encoded_src = self.encoder(src_pos_encoded)
        # print(f'==[ encoded_src: {encoded_src}')
        # print(f'==[ encoded_src shape: {encoded_src.shape}')

        tgt_unfolded = F.unfold(tgt, kernel_size=self.k_size).permute(0, 2, 1)
        # print(f'==[ tgt unfolded: {tgt_unfolded}')
        # print(f'==[ tgt unfolded shape: {tgt_unfolded.shape}')

        tgt_pos_encoded = self.pos_encoder(tgt_unfolded)
        # print(f'==[ tgt_pos_encoded: {tgt_pos_encoded}')
        # print(f'==[ tgt_pos_encoded shape: {tgt_pos_encoded.shape}')

        decoded_tgt = self.decoder(tgt_pos_encoded, encoded_src, mask)
        # decoded_tgt = self.decoder(tgt_pos_encoded, encoded_src)
        # print(f'==[ decoded_tgt: {decoded_tgt}')
        # print(f"==[ decoded_tgt.shape: {decoded_tgt.shape}")

        with torch.no_grad():
            input_ones = torch.ones((1, *src.shape[1:]), dtype=src.dtype)
            divisor = F.fold(
                F.unfold(input_ones, kernel_size=self.k_size),
                kernel_size=self.k_size,
                output_size=tgt.shape[-2:],
            )

        tgt_folded = (
            F.fold(
                decoded_tgt.permute(0, 2, 1),
                kernel_size=self.k_size,
                output_size=tgt.shape[-2:],
            )
            / divisor
        )
        # print(f'==[ tgt_folded: {tgt_folded}')
        # print(f"==[ tgt_folded.shape: {tgt_folded.shape}")

        return tgt_folded

    def generate_square_subsequent_mask(self, sz: int, batch: int = 1) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask[None, :, :].repeat(batch, 1, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()

        # NOTE: We keep d_model as a parameter, but it is not actually used here.
        # In the original paper, they *add* the position encoding
        # to the token vector, but in this instance we concatenate
        # the two, so it is not necessary to know the dimensionality
        # of the input here.

        # Primes up to the position encoding dimensionality
        d_model = d_model // 2 if d_model >= 2 else 1

        prm = primes(d_model)
        # print(f'==[ prm:\n{prm}')

        # Frequencies
        freqs = torch.from_numpy(prm)[:, None].repeat(1, max_len)
        # print(f'==[ freqs:\n{freqs.shape}')

        # Lengths (up to max_len)
        lengths = torch.linspace(0, max_len - 1, max_len)[None, :].repeat(d_model, 1)
        # print(f'==[ lengths.shape:\n{lengths.shape}')

        # Imaginary unit
        i = complex(0, 1)

        # Instead of cos and sin, use the relation e^{iwk} = cos(wk) + i sin(wk).
        # Create a tensor of all (i*w*k) values.
        pe = torch.exp((i * lengths * freqs))

        # Use the real and imaginary parts of the computed values as the position encoding
        pe = torch.cat((pe.real[:, None], pe.imag[:, None]), 1)
        if d_model == 1:
            # Bug...?
            pe = pe.flatten(1)
        else:
            pe = pe.flatten(end_dim=1)
        pe = pe[None, :, :]

        # print(f"==[ pe: {pe}")
        # print(f"==[ pe shape: {pe.shape}")
        # print(f'==[ pe[0]: {pe[0]}')

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # print(f'==[ x:\n{x}')
        # print(f"==[ x shape:\n{x.shape}")

        # Slice of the positional encoding matrix e
        # equal to the length of the input (batched)
        pos_enc = self.pe[:, :, : x.shape[1]].permute(0, 2, 1).repeat(x.shape[0], 1, 1)
        # print(f"==[ pos_enc:\n{pos_enc}")
        # print(f"==[ pos_enc shape:\n{pos_enc.shape}")

        # Concatenate the input and the position encodings
        # x = torch.cat((x, pos_enc.repeat(x.shape[0],1,1)), 1).permute(0,2,1)
        # x += pos_enc
        # print(f'==[ x:\n{x}')
        # print(f"==[ x shape:\n{x.shape}")

        return x + pos_enc
