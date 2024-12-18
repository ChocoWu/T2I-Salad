# Implement TransformerEncoder that can consider timesteps as optional args for Diffusion.

import copy
import math
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _gelu2(x):
    return x * F.sigmoid(1.702 * x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu2":
        return _gelu2
    else:
        raise RuntimeError(
            "activation should be relu/gelu/gelu2, not {}".format(activation)
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _AdaNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: int):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 condition_seq_len,
                 n_embd,  # the embed dim
                 condition_embd,  # condition dim
                 n_head,  # the number of heads
                 seq_len=None,  # the max length of sequence
                 attn_pdrop=0.1,  # attention dropout prob
                 resid_pdrop=0.1,  # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)try:
        try:
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        except RuntimeError as e:
            print(e)
            print(x.size())
            print(encoder_output.size())
            print(self.n_head)
            exit(0)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        d_model=1024,
        nhead=16,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        # extension for diffusion
        diffusion_step: int = 100,
        timestep_type: str = None,
        text_condition: bool = False,
        text_condition_seq_len: int = 77,
        max_seq_length: int = 25,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        condition_dim: int = 512
    ) -> None:
        super().__init__()

        assert norm_first  # minGPT-based implementations are designed for prenorm only
        assert timestep_type in [
            None,
            "adalayernorm",
            "adainnorm",
            "adalayernorm_abs",
            "adainnorm_abs",
            "adalayernorm_mlp",
            "adainnorm_mlp",
        ]
        layer_norm_eps = 1e-5  # fixed

        self.norm_first = norm_first
        self.diffusion_step = diffusion_step
        self.timestep_type = timestep_type

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = torch.nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        if text_condition:
            self.cross_attn = CrossAttention(
                    condition_seq_len=text_condition_seq_len,
                    n_embd=d_model,
                    condition_embd=condition_dim,
                    n_head=nhead,
                    seq_len=max_seq_length,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
            )
        self.text_condition = text_condition
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        if timestep_type is not None:
            if "adalayernorm" in timestep_type:
                self.norm1 = AdaLayerNorm(d_model, diffusion_step, timestep_type)
            elif "adainnorm" in timestep_type:
                self.norm1 = AdaInsNorm(d_model, diffusion_step, timestep_type)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            timestep: Tensor = None,
            text: Tensor = None,
            text_mask: Tensor = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            if text is not None:
                x = x + self._ca_block(x, y=text, mask=src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            if text is not None:
                x = x + self._ca_block(x,  y=text, mask=src_mask)
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    # cross-attention block
    def _ca_block(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
        x, _ = self.cross_attn(x, encoder_output=y)
        return x


class TransformerEncoder(nn.Module):
    """
    Close to torch.nn.TransformerEncoder, but with timestep support for diffusion
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
        text: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,

    ) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                timestep=timestep,
                text=text,
                text_mask=text_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
