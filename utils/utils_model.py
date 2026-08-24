"""
Backbone MLP for tabular diffusion model (with time embedding).
- The key model class is `MLPDiffusionContinuous`.

Adapted from: https://github.com/yandex-research/tab-ddpm/blob/main/tab_ddpm/modules.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, List


ModuleType = Union[str, Callable[..., nn.Module]]


def make_block_mask(in_groups: torch.Tensor,
                    out_groups: torch.Tensor,
                    A: torch.Tensor) -> torch.Tensor:
    """
    Build a [out_dim, in_dim] mask where
      mask[p, q] = A[in_groups[q], out_groups[p]].

    in_groups  : LongTensor shape (in_dim,)
    out_groups : LongTensor shape (out_dim,)
    A          : Tensor shape (K, K), adjacency between groups
    """
    # in_groups[None, :] : shape (1, in_dim)
    # out_groups[:, None]: shape (out_dim, 1)
    # advanced‐index into A, broadcasts to (out_dim, in_dim)
    mask = A[in_groups[None, :], out_groups[:, None]]
    return mask.float()


class MaskedLinear(nn.Linear):
    """A Linear layer whose weight is elementwise-multiplied by a fixed mask each forward."""
    def __init__(self, in_features, out_features, mask: torch.Tensor, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        # mask should be same shape as weight: [out_features, in_features]
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)
    
def compute_all_groups(
    groups_0: torch.LongTensor,   # shape [d_in], values in {0,…,K-1}
    groups_out: torch.LongTensor, # shape [d_out], values in {0,…,K-1}
    hidden_dims: List[int],       # e.g. [512,256,256,128]
) -> List[torch.LongTensor]:
    """
    Returns [groups_0, groups_1, ..., groups_L, groups_out], where:
      - groups_i is a LongTensor of length hidden_dims[i-1] (or d_in for i=0)
        assigning each neuron to one of K groups.
      - groups_out is, by default, the same as groups_0 (so your MLP head
        reconnects to the original grouping).
    We assign group sizes proportionally:
        size_j = floor(layer_size * (n_j / n))
    and dump any remainder into the last group.
    """
    K = int(groups_0.max().item()) + 1           # number of groups
    n = len(groups_0)                            # total input dim
    # count how many input features in each group
    base_counts = [(groups_0 == j).sum().item() for j in range(K)]
    
    all_groups = [groups_0]
    
    for H in hidden_dims:
        # proportional allocation
        sizes = [(H * cnt) // n for cnt in base_counts]
        # fix rounding: assign leftover to last group
        rem = H - sum(sizes)
        sizes[-1] += rem
        
        # build the tensor of length H
        layer_groups = torch.tensor(
            [j for j, sz in enumerate(sizes) for _ in range(sz)],
            dtype=torch.long, device=groups_0.device
        )
        all_groups.append(layer_groups)
    
    # final output grouping: reconnect back to input groups by default
    all_groups.append(groups_out)
    
    return all_groups


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == "ReGLU"
            else GEGLU() if module_type == "GEGLU" else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
            in_groups: torch.LongTensor,
            out_groups: torch.LongTensor,
            A: torch.Tensor,
        ) -> None:
            super().__init__()
            mask = make_block_mask(in_groups, out_groups,A)

            self.linear = MaskedLinear(d_in, d_out, mask, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
        groups_0: torch.LongTensor,
        groups_out: torch.LongTensor,
        hidden_dims: List[int],
        A: torch.Tensor,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ["ReGLU", "GEGLU"]
        
        all_groups = compute_all_groups(groups_0,groups_out,hidden_dims)
        # after computing your all_groups list
        self.blocks = nn.ModuleList()
        for i,(d, dropout) in enumerate(zip(d_layers, dropouts)):
            in_groups  = all_groups[i]     # grouping for previous layer
            out_groups = all_groups[i+1]   # grouping for this block's outputs
            blk = MLP.Block(
                d_in   = all_groups[i].numel(),   # same as prev_dim
                d_out  = all_groups[i+1].numel(), # equals d
                bias   = True,
                activation = activation,
                dropout = dropout,
                in_groups = in_groups,
                out_groups = out_groups,
                A = A,
            )
            self.blocks.append(blk)

        # final head
        in_groups  = all_groups[-2]
        out_groups = all_groups[-1]
        mask = make_block_mask(in_groups, out_groups,A)
        self.head = MaskedLinear(in_groups.numel(), out_groups.numel(), mask, bias=True)


    @classmethod
    def make_baseline(
        cls: Type["MLP"],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
        groups_0: torch.LongTensor,
        groups_out: torch.LongTensor,
        hidden_dims: List[int],
        A: torch.Tensor,
    ) -> "MLP":
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                "if d_layers contains more than two elements, then"
                " all elements except for the first and the last ones must be equal."
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            # activation="ReLU",
            activation="SiLU",
            d_out=d_out,
            groups_0=groups_0,
            groups_out=groups_out,
            hidden_dims=hidden_dims,
            A = A,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x



class MLPDiffusionContinuous(nn.Module):
    """
    Masked MLP backbone with shared-per-group time embedding.

    Args:
        d_in:         # of input features (n1+…+nK)
        hidden_dims:  list of hidden layer sizes for the MLP
        dim_t:        total time-embedding dim (must be divisible by K)
        groups_0:     LongTensor of shape [d_in], values in {0,…,K-1}
        A:            Adjacency matrix for projection step (cross-group connections)
    """
    def __init__(
        self,
        d_in: int,
        hidden_dims: List[int],
        dim_t: int,
        groups_0: torch.LongTensor,
        A: torch.Tensor,
    ):
        super().__init__()
        # --- group bookkeeping ---
        self.register_buffer("groups_0", groups_0)
        self.K = int(groups_0.max().item()) + 1
        assert dim_t % self.K == 0, "dim_t must be divisible by K"
        self.dim_t_per   = dim_t // self.K
        self.dim_t_total = dim_t

        # --- masked projection x -> R^{K*dim_t_per} ---
        # time_groups[p] = which group the p-th embed coord belongs to
        time_groups = (
            torch.arange(self.K, device=groups_0.device)
            .repeat_interleave(self.dim_t_per)
        )  # shape [dim_t_total]

        # Use original adjacency matrix A for projection (allows cross-group connections)
        mask_proj = make_block_mask(self.groups_0, time_groups, A)
        self.proj = MaskedLinear(
            in_features  = d_in,
            out_features = self.dim_t_total,
            mask         = mask_proj,
            bias         = True,
        )

        # --- small time-MLP on a single dim_t_per chunk ---
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_t_per, self.dim_t_per),
            nn.SiLU(),
            nn.Linear(self.dim_t_per, self.dim_t_per),
        )

        # --- Create identity matrix for MLP (only within-group connections) ---
        A_identity = torch.eye(self.K, device=groups_0.device, dtype=A.dtype)

        # --- main MLP backbone: in=dim_t_total, out=d_in ---
        self.mlp = MLP.make_baseline(
            d_in      = self.dim_t_total,
            d_layers  = hidden_dims,
            dropout   = 0.0,
            d_out     = d_in,
            groups_0  = time_groups,
            groups_out = self.groups_0,
            hidden_dims = hidden_dims,
            A = A_identity,  # Use identity matrix - only within-group connections
        )

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        # 1) compute a single time embedding of size [B, dim_t_per]
        te = timestep_embedding(timesteps, self.dim_t_per)  # [B,dim_t_per]
        te = self.time_embed(te)                           # [B,dim_t_per]

        # 2) replicate it for each group -> [B, K, dim_t_per] -> [B, dim_t_total]
        te = te.unsqueeze(1).repeat(1, self.K, 1)
        te = te.view(x.shape[0], -1)                       # [B, dim_t_total]

        # 3) masked projection + shared-time add
        x = self.proj(x) + te                              # [B, dim_t_total]

        # 4) masked MLP to get back to [B, d_in]
        return self.mlp(x)
