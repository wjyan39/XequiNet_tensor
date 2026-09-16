"""
TorchScript-compatible building blocks for the XPaiNN-Orb / XPaiNN-Orb/X2
Hamiltonian models.

These classes mirror the eager modules in ``xequinet/nn`` with the *same
submodule names and parameter shapes*, so a JIT model composed of them can
load weights from a checkpoint trained with the original architecture via
``load_state_dict(strict=False)``.

Differences from the eager blocks:
    * Python-only constructs (``o3.Irreps.slices()``, ``Counter``, python
      dicts, ``next(self.parameters())``) are replaced by precomputed
      attributes / registered buffers.
    * PBC is not supported; only the non-PBC code path is implemented.
    * ``o3.wigner_3j`` tensors are precomputed at construction time.

The already TorchScript-compatible blocks (``EquivariantDot``,
``FullEdgeKernel``, ``FullEdgeSPH``) and the e3nn / torch layers are reused
from ``xequinet/nn``.
"""

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from e3nn import o3

from ..nn.o3layer import EquivariantDot, Int2c1eEmbedding, resolve_actfn
from ..nn.rbf import resolve_cutoff, resolve_rbf
from ..nn.tensorproduct import get_feasible_tp, prod

__all__ = [
    "JitXEmbedding",
    "JitNormGate",
    "JitSelfLayer",
    "JitPairLayer",
    "JitPairLayerSPH",
    "JitBlockNorm2d",
    "JitXMatTrans",
    "JitZMatTrans",
    "JitExpansion",
    "JitZExpansion",
    "JitMatriceOut",
    "JitZMatriceOut",
    "JitTwoBodyBlockPad",
    "JitMat2GraphLabel",
    "JitTwoBodyBlockMask",
    "JitBuildMatPerMole",
]


class JitXEmbedding(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.xpainn.XEmbedding``.
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        if embed_basis == "one-hot":
            self.node_embed = nn.Embedding(120, self.node_dim, padding_idx=0)
            self.int2c1e = nn.Identity()
            self.node_lin = nn.Identity()
            self.one_hot = True
        else:
            self.node_embed = nn.Identity()
            self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
            self.node_lin = nn.Linear(self.int2c1e.embed_dim, self.node_dim)
            nn.init.zeros_(self.node_lin.bias)
            self.one_hot = False
        self.sph_harm = o3.SphericalHarmonics(
            self.edge_irreps, normalize=True, normalization="component"
        )
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vec = pos[edge_index[0]] - pos[edge_index[1]] - shifts
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        if self.one_hot:
            x_scalar = self.node_embed(at_no)
        else:
            x = self.int2c1e(at_no)
            x_scalar = self.node_lin(x)
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)
        rsh = self.sph_harm(vec[:, [1, 2, 0]])
        return x_scalar, rbf, fcut, rsh


class JitNormGate(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.o3layer.NormGate``.
    """

    def __init__(self, irreps: o3.Irreps, actfn: str = "silu"):
        super().__init__()
        self.irreps = irreps if isinstance(irreps, o3.Irreps) else o3.Irreps(irreps)
        self.stop0 = int(self.irreps.slices()[0].stop)
        self.norm = o3.Norm(self.irreps)
        self.activation = resolve_actfn(actfn)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irreps:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(
            self.irreps[1:], o3.Irreps(f"{num_mul_wo_0}x0e")
        )
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul),
            self.activation,
            nn.Linear(num_mul, num_mul),
        )
        self.num_mul = num_mul
        self.num_mul_wo_0 = num_mul_wo_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm(x)[:, self.stop0:]
        f0 = torch.cat([x[:, :self.stop0], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(x[:, self.stop0:], gates[:, self.stop0:])
        return torch.cat([gates[:, :self.stop0], gated], dim=-1)


class JitSelfLayer(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.matlayer.SelfLayer``.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        actfn: str = "silu",
    ):
        super().__init__()
        self.irreps_in = irreps_in if isinstance(irreps_in, o3.Irreps) else o3.Irreps(irreps_in)
        self.irreps_hidden = irreps_hidden if isinstance(irreps_hidden, o3.Irreps) else o3.Irreps(irreps_hidden)
        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_in, self.irreps_in, self.irreps_hidden, "uuu",
        )
        self.tp_node = o3.TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_tp_out,
            instructions,
            shared_weights=True,
            internal_weights=True,
        )
        self.linear_node_l = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_in, biases=True)
        self.linear_node_r = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_in, biases=True)
        self.linear_node_p = o3.Linear(irreps_in=self.irreps_tp_out, irreps_out=self.irreps_hidden, biases=False)
        self.normgate_l = JitNormGate(self.irreps_in, actfn)
        self.normgate_r = JitNormGate(self.irreps_in, actfn)
        self.normgate_p = JitNormGate(self.irreps_tp_out, actfn)

    def forward(
        self,
        x: torch.Tensor,
        old_fii: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xl = self.normgate_l(x)
        xl = self.linear_node_l(xl)
        xr = self.normgate_r(x)
        xr = self.linear_node_r(xr)
        xtp = self.tp_node(xl, xr)
        xtp = self.normgate_p(xtp)
        fii = self.linear_node_p(xtp)
        if old_fii is not None:
            fii = fii + old_fii
        return fii


class JitPairLayer(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.matlayer.PairLayer``.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        edge_dim: int = 20,
        actfn: str = "silu",
        rbf_bias: bool = True,
    ):
        super().__init__()
        self.irreps_in = irreps_in if isinstance(irreps_in, o3.Irreps) else o3.Irreps(irreps_in)
        self.irreps_hidden = irreps_hidden if isinstance(irreps_hidden, o3.Irreps) else o3.Irreps(irreps_hidden)
        self._edge_dim = edge_dim
        self.actfn = resolve_actfn(actfn)
        self.stop0 = int(self.irreps_in.slices()[0].stop)

        self.linear_node_pre = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_in, biases=True)
        self.inner_dot = EquivariantDot(self.irreps_in)
        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_in, self.irreps_in, self.irreps_hidden, tp_mode="uuu",
        )
        self.tp_node_pair = o3.TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.linear_edge_scalar = nn.Sequential(
            nn.Linear(self.irreps_in[0][0] + self.irreps_in.num_irreps, 128, bias=True),
            self.actfn,
            nn.Linear(128, self.tp_node_pair.weight_numel, bias=True),
        )
        self.linear_edge_rbf = nn.Sequential(
            nn.Linear(self._edge_dim, 128, bias=rbf_bias),
            self.actfn,
            nn.Linear(128, self.tp_node_pair.weight_numel, bias=rbf_bias),
        )
        self.linear_node_post = o3.Linear(irreps_in=self.irreps_tp_out, irreps_out=self.irreps_hidden, biases=False)
        self.normgate_pre = JitNormGate(self.irreps_in, actfn)
        self.normgate_post = JitNormGate(self.irreps_tp_out, actfn)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        old_fij: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_feat = x
        s0 = self.inner_dot(node_feat[edge_index[0], :], node_feat[edge_index[1], :])[:, self.stop0:]
        s0 = torch.cat(
            [
                x[edge_index[0]][:, :self.stop0],
                x[edge_index[1]][:, :self.stop0],
                s0,
            ],
            dim=-1,
        )
        tp_weights = self.linear_edge_scalar(s0) * self.linear_edge_rbf(edge_attr)
        x_prime = self.normgate_pre(self.linear_node_pre(x))
        fij = self.tp_node_pair(x_prime[edge_index[0], :], x_prime[edge_index[1], :], tp_weights)
        fij = self.normgate_post(fij)
        fij = self.linear_node_post(fij)
        if old_fij is not None:
            fij = fij + old_fij
        return fij


class JitPairLayerSPH(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.matlayer.PairLayerSPH``.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        max_l: int = 4,
        edge_dim: int = 20,
        actfn: str = "silu",
        rbf_bias: bool = True,
    ):
        super().__init__()
        self.irreps_in = irreps_in if isinstance(irreps_in, o3.Irreps) else o3.Irreps(irreps_in)
        self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        self.irreps_hidden = irreps_hidden if isinstance(irreps_hidden, o3.Irreps) else o3.Irreps(irreps_hidden)
        self._edge_dim = edge_dim
        self.actfn = resolve_actfn(actfn)
        self.stop0 = int(self.irreps_in.slices()[0].stop)

        self.linear_node_pre = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_in, biases=True)
        self.inner_dot = EquivariantDot(self.irreps_in)
        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_in, self.irreps_rshs, self.irreps_hidden, tp_mode="uvu",
        )
        self.pair_rsh_conv = o3.TensorProduct(
            self.irreps_in,
            self.irreps_rshs,
            self.irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.linear_edge_scalar = nn.Sequential(
            nn.Linear(self.irreps_in[0][0] + self.irreps_in.num_irreps, 128, bias=True),
            self.actfn,
            nn.Linear(128, self.pair_rsh_conv.weight_numel, bias=True),
        )
        self.linear_edge_rbf = nn.Sequential(
            nn.Linear(self._edge_dim, 128, bias=rbf_bias),
            self.actfn,
            nn.Linear(128, self.pair_rsh_conv.weight_numel, bias=rbf_bias),
        )
        self.linear_node_post = o3.Linear(irreps_in=self.irreps_tp_out, irreps_out=self.irreps_hidden, biases=False)
        self.normgate_pre = JitNormGate(self.irreps_in, actfn)
        self.normgate_post = JitNormGate(self.irreps_tp_out, actfn)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_rshs: torch.Tensor,
        edge_index: torch.LongTensor,
        old_fij: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_feat = x
        s0 = self.inner_dot(node_feat[edge_index[0], :], node_feat[edge_index[1], :])[:, self.stop0:]
        s0 = torch.cat(
            [
                x[edge_index[0]][:, :self.stop0],
                x[edge_index[1]][:, :self.stop0],
                s0,
            ],
            dim=-1,
        )
        tp_weights = self.linear_edge_scalar(s0) * self.linear_edge_rbf(edge_attr)
        x_prime = self.normgate_pre(self.linear_node_pre(x))
        x_ij_prime = 0.5 * (x_prime[edge_index[0], :] + x_prime[edge_index[1], :])
        fij = self.pair_rsh_conv(x_ij_prime, edge_rshs, tp_weights)
        fij = self.normgate_post(fij)
        fij = self.linear_node_post(fij)
        if old_fij is not None:
            fij = fij + old_fij
        return fij


class JitBlockNorm2d(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.xpainnorb.BlockNorm2d``.
    """

    def __init__(self, irreps_out: Union[str, o3.Irreps, Iterable]):
        super().__init__()
        self.irreps = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        irreps_layout_1d = []
        cur_offset = 0
        for mul, ir in self.irreps:
            irreps_layout_1d.extend(
                (torch.arange(mul, dtype=torch.int32).repeat(ir.dim) + cur_offset).tolist()
            )
            cur_offset += mul
        irreps_layout_1d = torch.tensor(irreps_layout_1d).long()
        idx_ten_0 = (
            irreps_layout_1d
            .unsqueeze(1)
            .expand(self.irreps.dim, self.irreps.dim)
        )
        idx_ten_1 = (
            irreps_layout_1d
            .unsqueeze(0)
            .expand(self.irreps.dim, self.irreps.dim)
        )
        idx_ten_flat = (idx_ten_0 * self.irreps.num_irreps + idx_ten_1).flatten()
        self.register_buffer("idx_ten_flat", idx_ten_flat)
        self.num_irreps_sq = int(self.irreps.num_irreps ** 2)

    def forward(self, feat_ten: torch.Tensor) -> torch.Tensor:
        out_shape = torch.jit.annotate(List[int], [])
        for s in feat_ten.shape[:-2]:
            out_shape.append(s)
        out_shape.append(self.num_irreps_sq)
        sum_sqr = torch.zeros(
            out_shape,
            device=feat_ten.device,
            dtype=feat_ten.dtype,
        )
        sum_sqr = sum_sqr.index_add_(
            dim=-1, index=self.idx_ten_flat, source=feat_ten.flatten(-2, -1).pow(2)
        )
        return torch.sqrt(sum_sqr)


class JitXMatTrans(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.xpainnorb.XMatTrans``.
    """

    def __init__(
        self,
        irreps_node: Union[str, o3.Irreps, Iterable] = "128x0e+128x1o+128x2e+128x3o+128x4e",
        hidden_dim: int = 64,
        max_l: int = 4,
        edge_dim: int = 20,
        actfn: str = "silu",
        rbf_bias: bool = True,
    ):
        super().__init__()
        self.irreps_in = irreps_node if isinstance(irreps_node, o3.Irreps) else o3.Irreps(irreps_node)
        assert 2 * self.irreps_in.lmax >= max_l
        self.irreps_hidden_base = o3.Irreps(
            [(hidden_dim, (l, (-1) ** l)) for l in range(self.irreps_in.lmax + 1)]
        )
        irreps_hidden = [(hidden_dim, (0, 1))]
        for l in range(2, 2 * max_l + 1):
            irreps_hidden.append((hidden_dim, (l // 2, (-1) ** l)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        if self.irreps_in == self.irreps_hidden_base:
            self.pretrans = False
        else:
            self.pretrans = True
            self.node_pre_trans = o3.Linear(self.irreps_in, self.irreps_hidden_base, biases=False)
        self.node_self_layer = JitSelfLayer(self.irreps_hidden_base, self.irreps_hidden, actfn)
        self.node_pair_layer = JitPairLayer(self.irreps_hidden_base, self.irreps_hidden, edge_dim, actfn, rbf_bias)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        fii: Optional[torch.Tensor] = None,
        fij: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pretrans:
            node_feat = self.node_pre_trans(node_feat)
        fii = self.node_self_layer(node_feat, fii)
        fij = self.node_pair_layer(node_feat, edge_attr, edge_index, fij)
        return fii, fij


class JitZMatTrans(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.xpainnorb.ZMatTrans``.
    """

    def __init__(
        self,
        irreps_node: Union[str, o3.Irreps, Iterable] = "128x0e+128x1o+128x2e+128x3o+128x4e",
        hidden_dim: int = 64,
        max_l: int = 4,
        edge_dim: int = 20,
        actfn: str = "silu",
        rbf_bias: bool = True,
    ):
        super().__init__()
        self.irreps_in = irreps_node if isinstance(irreps_node, o3.Irreps) else o3.Irreps(irreps_node)
        assert 2 * self.irreps_in.lmax >= max_l
        self.irreps_hidden_base = o3.Irreps(
            [(hidden_dim, (l, (-1) ** l)) for l in range(self.irreps_in.lmax + 1)]
        )
        irreps_hidden = [(hidden_dim, (0, 1))]
        for l in range(2, 2 * max_l + 1):
            irreps_hidden.append((hidden_dim, (l // 2, (-1) ** l)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        if self.irreps_in == self.irreps_hidden_base:
            self.pretrans = False
        else:
            self.pretrans = True
            self.node_pre_trans = o3.Linear(self.irreps_in, self.irreps_hidden_base, biases=False)
        self.node_self_layer = JitSelfLayer(self.irreps_hidden_base, self.irreps_hidden, actfn)
        self.node_pair_layer = JitPairLayerSPH(
            self.irreps_hidden_base, self.irreps_hidden, self.irreps_in.lmax, edge_dim, actfn, rbf_bias
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_rshs: torch.Tensor,
        edge_index: torch.LongTensor,
        fii: Optional[torch.Tensor] = None,
        fij: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pretrans:
            node_feat = self.node_pre_trans(node_feat)
        fii = self.node_self_layer(node_feat, fii)
        fij = self.node_pair_layer(node_feat, edge_attr, edge_rshs, edge_index, fij)
        return fii, fij


class JitExpansion(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.matlayer.Expansion``.

    The static instruction table and the ``o3.wigner_3j`` tensors are
    precomputed at construction time and stored as typed attributes.
    """

    def __init__(
        self,
        irreps_block: o3.Irreps,
        irreps_out: o3.Irreps,
        node_dim: int,
        actfn: str = "silu",
        pair_out: bool = False,
    ):
        super().__init__()
        self.irreps_block = irreps_block if isinstance(irreps_block, o3.Irreps) else o3.Irreps(irreps_block)
        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.pair_out = pair_out
        self.actfn = resolve_actfn(actfn)

        instructions = self.get_expansion_path(self.irreps_block, self.irreps_out, self.irreps_out)
        self.num_path_weight = sum(prod(ins[-1]) for ins in instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in instructions if ins[0] == 0])
        self.num_weights = self.num_path_weight + self.num_bias

        self.num_ins = len(instructions)
        self.ins_i = [int(ins[0]) for ins in instructions]
        self.ins_j = [int(ins[1]) for ins in instructions]
        self.ins_k = [int(ins[2]) for ins in instructions]
        self.ins_size = [int(prod(ins[-1])) for ins in instructions]
        self.ins_bias_size = [int(prod(ins[-1][1:])) for ins in instructions]
        self.ins_num_in = [int(ins[-1][0]) for ins in instructions]
        self.ins_num_out1 = [int(ins[-1][1]) for ins in instructions]
        self.ins_num_out2 = [int(ins[-1][2]) for ins in instructions]
        self.ins_w3j = []
        for ins in instructions:
            l_in = self.irreps_block[ins[0]].ir.l
            l_out1 = self.irreps_out[ins[1]].ir.l
            l_out2 = self.irreps_out[ins[2]].ir.l
            self.ins_w3j.append(o3.wigner_3j(l_out1, l_out2, l_in))

        block_starts = [0]
        block_muls, block_dims = [], []
        for mul, ir in self.irreps_block:
            block_starts.append(block_starts[-1] + mul * ir.dim)
            block_muls.append(int(mul))
            block_dims.append(int(ir.dim))
        self.block_starts = block_starts
        self.block_muls = block_muls
        self.block_dims = block_dims
        self.num_block = len(self.irreps_block)
        self.num_out = len(self.irreps_out)
        self.out_dims = [int(mul * ir.dim) for mul, ir in self.irreps_out]

        if pair_out:
            self.lin_weight = nn.Sequential(
                nn.Linear(2 * node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_path_weight, bias=True),
            )
            self.lin_bias = nn.Sequential(
                nn.Linear(2 * node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_bias, bias=True),
            )
        else:
            self.lin_weight = nn.Sequential(
                nn.Linear(node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_path_weight, bias=True),
            )
            self.lin_bias = nn.Sequential(
                nn.Linear(node_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_bias, bias=True),
            )

    def get_expansion_path(
        self,
        irrep_in: o3.Irreps,
        irrep_out_1: o3.Irreps,
        irrep_out_2: o3.Irreps,
    ) -> List:
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    def forward(self, feat: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        x_in = feat
        batch_num = x_in.shape[0]
        x_in_s = torch.jit.annotate(List[torch.Tensor], [])
        for b in range(self.num_block):
            x_in_s.append(
                x_in[:, self.block_starts[b]:self.block_starts[b + 1]].reshape(
                    batch_num, self.block_muls[b], self.block_dims[b]
                )
            )
        weights = self.lin_weight(node_emb)
        bias_weights = self.lin_bias(node_emb)

        num_out = self.num_out
        outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [])
        for _ in range(num_out * num_out):
            outputs.append(None)
        flat_weight_index = 0
        bias_weight_index = 0
        for t in range(self.num_ins):
            i = self.ins_i[t]
            j = self.ins_j[t]
            k = self.ins_k[t]
            x1 = x_in_s[i]
            w3j = self.ins_w3j[t].to(x1.dtype).to(x1.device)
            weight = weights[:, flat_weight_index:flat_weight_index + self.ins_size[t]].reshape(
                [-1, self.ins_num_in[t], self.ins_num_out1[t], self.ins_num_out2[t]]
            )
            result = torch.einsum("bwuv, bwk -> buvk", weight, x1)
            if i == 0:
                bias_weight = bias_weights[:, bias_weight_index:bias_weight_index + self.ins_bias_size[t]].reshape(
                    [-1, self.ins_num_out1[t], self.ins_num_out2[t]]
                )
                bias_weight_index += self.ins_bias_size[t]
                result = result + bias_weight.unsqueeze(-1)
            result = torch.einsum("ijk, buvk -> buivj", w3j, result) / self.block_muls[i]
            flat_weight_index += self.ins_size[t]
            result = result.reshape(batch_num, self.out_dims[j], self.out_dims[k])
            idx = j * num_out + k
            prev = outputs[idx]
            if prev is None:
                outputs[idx] = result
            else:
                outputs[idx] = prev + result

        rows = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(num_out):
            blocks = torch.jit.annotate(List[torch.Tensor], [])
            for j in range(num_out):
                val = outputs[i * num_out + j]
                if val is None:
                    blocks.append(
                        torch.zeros(
                            batch_num,
                            self.out_dims[i],
                            self.out_dims[j],
                            device=x_in.device,
                            dtype=x_in.dtype,
                        )
                    )
                else:
                    blocks.append(val)
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output


class JitZExpansion(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.matlayer.ZExpansion``.

    Only the ``shared_weights=False`` path is implemented (as used by
    ``ZMatriceOut``).
    """

    def __init__(
        self,
        irreps_block: o3.Irreps,
        irreps_out: o3.Irreps,
        embed_dim: int,
        actfn: str = "silu",
        pair_out: bool = False,
        shared_weights: bool = False,
    ):
        super().__init__()
        assert not shared_weights, "JitZExpansion only supports shared_weights=False"
        self.irreps_block = irreps_block if isinstance(irreps_block, o3.Irreps) else o3.Irreps(irreps_block)
        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.pair_out = pair_out
        self.actfn = resolve_actfn(actfn)
        self.shared_weights = False

        instructions = self.get_expansion_path(
            self.irreps_block, self.irreps_out, self.irreps_out, trainable=True
        )
        self.num_path_weight = sum(prod(ins[-1]) for ins in instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in instructions if ins[0] == 0])
        self.num_weights = self.num_path_weight + self.num_bias

        self.num_ins = len(instructions)
        self.ins_i = [int(ins[0]) for ins in instructions]
        self.ins_j = [int(ins[1]) for ins in instructions]
        self.ins_k = [int(ins[2]) for ins in instructions]
        self.ins_size = [int(prod(ins[-1])) for ins in instructions]
        self.ins_bias_size = [int(prod(ins[-1][1:])) for ins in instructions]
        self.ins_num_in = [int(ins[-1][0]) for ins in instructions]
        self.ins_num_out1 = [int(ins[-1][1]) for ins in instructions]
        self.ins_num_out2 = [int(ins[-1][2]) for ins in instructions]
        self.ins_w3j = []
        for ins in instructions:
            l_in = self.irreps_block[ins[0]].ir.l
            l_out1 = self.irreps_out[ins[1]].ir.l
            l_out2 = self.irreps_out[ins[2]].ir.l
            self.ins_w3j.append(o3.wigner_3j(l_out1, l_out2, l_in))

        block_starts = [0]
        block_muls, block_dims = [], []
        for mul, ir in self.irreps_block:
            block_starts.append(block_starts[-1] + mul * ir.dim)
            block_muls.append(int(mul))
            block_dims.append(int(ir.dim))
        self.block_starts = block_starts
        self.block_muls = block_muls
        self.block_dims = block_dims
        self.num_block = len(self.irreps_block)
        self.num_out = len(self.irreps_out)
        self.out_dims = [int(mul * ir.dim) for mul, ir in self.irreps_out]

        if pair_out:
            self.lin_weight = nn.Linear(embed_dim, self.num_path_weight, bias=False)
            self.lin_bias = nn.Identity()
        else:
            self.lin_weight = nn.Sequential(
                nn.Linear(embed_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_path_weight, bias=True),
            )
            self.lin_bias = nn.Sequential(
                nn.Linear(embed_dim, 64, bias=True),
                self.actfn,
                nn.Linear(64, self.num_bias, bias=True),
            )

    def get_expansion_path(
        self,
        irrep_in: o3.Irreps,
        irrep_out_1: o3.Irreps,
        irrep_out_2: o3.Irreps,
        trainable: bool,
    ) -> List:
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, trainable, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    def forward(self, feat: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        x_in = feat
        batch_num = x_in.shape[0]
        x_in_s = torch.jit.annotate(List[torch.Tensor], [])
        for b in range(self.num_block):
            x_in_s.append(
                x_in[:, self.block_starts[b]:self.block_starts[b + 1]].reshape(
                    batch_num, self.block_muls[b], self.block_dims[b]
                )
            )
        weights = self.lin_weight(feat_emb)
        if self.pair_out:
            bias_weights = feat_emb
        else:
            bias_weights = self.lin_bias(feat_emb)

        num_out = self.num_out
        outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [])
        for _ in range(num_out * num_out):
            outputs.append(None)
        flat_weight_index = 0
        bias_weight_index = 0
        for t in range(self.num_ins):
            i = self.ins_i[t]
            j = self.ins_j[t]
            k = self.ins_k[t]
            x1 = x_in_s[i]
            w3j = self.ins_w3j[t].to(x1.dtype).to(x1.device)
            weight = weights[:, flat_weight_index:flat_weight_index + self.ins_size[t]].reshape(
                [-1, self.ins_num_in[t], self.ins_num_out1[t], self.ins_num_out2[t]]
            )
            result = torch.einsum("bwuv, bwk -> buvk", weight, x1)
            if i == 0 and not self.pair_out:
                bias_weight = bias_weights[:, bias_weight_index:bias_weight_index + self.ins_bias_size[t]].reshape(
                    [-1, self.ins_num_out1[t], self.ins_num_out2[t]]
                )
                bias_weight_index += self.ins_bias_size[t]
                result = result + bias_weight.unsqueeze(-1)
            result = torch.einsum("ijk, buvk -> buivj", w3j, result) / self.block_muls[i]
            flat_weight_index += self.ins_size[t]
            result = result.reshape(batch_num, self.out_dims[j], self.out_dims[k])
            idx = j * num_out + k
            prev = outputs[idx]
            if prev is None:
                outputs[idx] = result
            else:
                outputs[idx] = prev + result

        rows = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(num_out):
            blocks = torch.jit.annotate(List[torch.Tensor], [])
            for j in range(num_out):
                val = outputs[i * num_out + j]
                if val is None:
                    blocks.append(
                        torch.zeros(
                            batch_num,
                            self.out_dims[i],
                            self.out_dims[j],
                            device=x_in.device,
                            dtype=x_in.dtype,
                        )
                    )
                else:
                    blocks.append(val)
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output


class JitMatriceOut(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.matlayer.MatriceOut``.

    PBC is not supported; the non-PBC code path is always used.
    """

    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps, Iterable] = "3x0e + 2x1o + 1x2e",
        node_dim: int = 128,
        hidden_dim: int = 64,
        block_dim: int = 32,
        max_l: int = 4,
        actfn: str = "silu",
        symmetrize: bool = True,
    ):
        super().__init__()
        irreps_hidden_base = [(hidden_dim, (0, 1))]
        irreps_block = [(block_dim, (0, 1))]
        for l in range(2, 2 * max_l + 1):
            irreps_hidden_base.append((hidden_dim, (l // 2, (-1) ** l)))
            irreps_block.append((block_dim, (l // 2, (-1) ** l)))
        self.irreps_hidden_base = o3.Irreps(irreps_hidden_base)
        self.irreps_block = o3.Irreps(irreps_block)
        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.symmetrize = symmetrize
        self.linear_out_ii = o3.Linear(self.irreps_hidden_base, self.irreps_block, biases=False)
        self.linear_out_ij = o3.Linear(self.irreps_hidden_base, self.irreps_block, biases=False)
        self.expand_ii = JitExpansion(self.irreps_block, self.irreps_out, node_dim, actfn, pair_out=False)
        self.expand_ij = JitExpansion(self.irreps_block, self.irreps_out, node_dim, actfn, pair_out=True)

    def forward(
        self,
        fii: torch.Tensor,
        fij: torch.Tensor,
        node_embed: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_index_cluster: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embed_pair = torch.cat(
            [node_embed[edge_index[0], :], node_embed[edge_index[1], :]], dim=-1
        )
        fii = self.linear_out_ii(fii)
        fij = self.linear_out_ij(fij)
        diagnol_block = self.expand_ii(fii, node_embed)
        off_diagnol_pair = self.expand_ij(fij, node_embed_pair)
        off_diagnol_block = off_diagnol_pair
        if self.symmetrize:
            diagnol_block = 0.5 * (diagnol_block + diagnol_block.transpose(-1, -2))
            _, src_indices = torch.sort(edge_index_cluster[1], stable=True)
            off_diagnol_block_trans = off_diagnol_block.transpose(-1, -2)[src_indices]
            off_diagnol_block = 0.5 * (off_diagnol_block + off_diagnol_block_trans)
        return diagnol_block, off_diagnol_block


class JitZMatriceOut(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.nn.matlayer.ZMatriceOut``.

    PBC is not supported; the non-PBC code path is always used.
    """

    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps, Iterable] = "3x0e + 2x1o + 1x2e",
        node_dim: int = 128,
        edge_dim: int = 36,
        hidden_dim: int = 64,
        block_dim: int = 32,
        max_l: int = 4,
        actfn: str = "silu",
        symmetrize: bool = True,
    ):
        super().__init__()
        irreps_hidden = [(hidden_dim, (0, 1))]
        irreps_block = [(block_dim, (0, 1))]
        for l in range(2, 2 * max_l + 1):
            irreps_hidden.append((hidden_dim, (l // 2, (-1) ** l)))
            irreps_block.append((block_dim, (l // 2, (-1) ** l)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_block = o3.Irreps(irreps_block)
        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.symmetrize = symmetrize
        self.linear_out_ii = o3.Linear(self.irreps_hidden, self.irreps_block, biases=False)
        self.linear_out_ij = o3.Linear(self.irreps_hidden, self.irreps_block, biases=False)
        self.linear_edge_attr = nn.Linear(edge_dim, 2 * node_dim, bias=False)
        self.linear_edge_pair = nn.Sequential(
            nn.Linear(2 * node_dim, 64, bias=True),
            resolve_actfn(actfn),
            nn.Linear(64, 2 * node_dim, bias=True),
        )
        self.expand_ii = JitZExpansion(self.irreps_block, self.irreps_out, node_dim, actfn, pair_out=False)
        self.expand_ij = JitZExpansion(self.irreps_block, self.irreps_out, node_dim * 2, actfn, pair_out=True)

    def forward(
        self,
        fii: torch.Tensor,
        fij: torch.Tensor,
        node_embed: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_index_cluster: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fii = self.linear_out_ii(fii)
        fij = self.linear_out_ij(fij)
        diagnol_block = self.expand_ii(fii, node_embed)
        node_pair_embed = torch.cat(
            [node_embed[edge_index[0], :], node_embed[edge_index[1], :]], dim=-1
        )
        edge_embed = self.linear_edge_attr(edge_attr) * self.linear_edge_pair(node_pair_embed)
        off_diagnol_pair = self.expand_ij(fij, edge_embed)
        off_diagnol_block = off_diagnol_pair
        if self.symmetrize:
            diagnol_block = 0.5 * (diagnol_block + diagnol_block.transpose(-1, -2))
            _, src_indices = torch.sort(edge_index_cluster[1], stable=True)
            off_diagnol_block_trans = off_diagnol_block.transpose(-1, -2)[src_indices]
            off_diagnol_block = 0.5 * (off_diagnol_block + off_diagnol_block_trans)
        return diagnol_block, off_diagnol_block


class JitTwoBodyBlockPad(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.utils.qc_matrice_graph.TwoBodyBlockPad``
    for ``rep_dims=(0, 1)``.

    The per-element rep map is stored as dense element-indexed buffers
    (``repid_table`` + ``valid_table`` + ``count_table``) instead of a python
    ``dict`` of variable-length tensors.
    """

    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps, Iterable],
        rep_dims: Tuple[int, int] = (0, 1),
        possible_elements: list = None,
        basisname: str = "def2svp",
        map_type: str = "pyscf",
    ):
        super().__init__()
        from ..utils.qc import ELEMENTS_DICT, ELEMENTS_LIST, get_l_from_basis
        from ..utils.qc_matrice_graph import resolve_m_idx_type

        assert tuple(rep_dims) == (0, 1), "JitTwoBodyBlockPad only supports rep_dims=(0, 1)"
        self.rep_dims = (0, 1)
        self.irreps = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        num_reps_1d = 0
        num_channels = torch.LongTensor([0 for _ in range(self.irreps.lmax + 1)])
        dim_per_l = torch.LongTensor([2 * l + 1 for l in range(self.irreps.lmax + 1)])
        for mul, ir in self.irreps:
            num_reps_1d += mul * ir.dim
            num_channels[ir.l] = mul
        self.num_reps_1d = num_reps_1d
        offset_per_l = torch.cumsum(num_channels * dim_per_l, dim=0)
        offset_per_l = torch.concat((torch.LongTensor([0]), offset_per_l))

        m_idx_map = resolve_m_idx_type(map_type)
        rows = {}
        max_nrep = 1
        for ele in possible_elements:
            z = ELEMENTS_DICT[ele] if isinstance(ele, str) else ele
            l_list = get_l_from_basis(basisname, z)
            repid = []
            offset, cur_l = 0, l_list[0]
            for l in l_list:
                if l > cur_l:
                    offset = 0
                    cur_l = l
                repid.append(torch.LongTensor(m_idx_map[l]) + offset + offset_per_l[l])
                offset += 2 * l + 1
            row = torch.cat(repid)
            rows[z] = row
            max_nrep = max(max_nrep, row.numel())

        repid_table = torch.zeros(len(ELEMENTS_LIST), max_nrep, dtype=torch.long)
        valid_table = torch.zeros(len(ELEMENTS_LIST), max_nrep, dtype=torch.bool)
        count_table = torch.zeros(len(ELEMENTS_LIST), dtype=torch.long)
        for z, row in rows.items():
            repid_table[z, :row.numel()] = row
            valid_table[z, :row.numel()] = True
            count_table[z] = row.numel()
        self.register_buffer("repid_table", repid_table)
        self.register_buffer("valid_table", valid_table)
        self.register_buffer("count_table", count_table)

    def forward(self, at_no: torch.LongTensor, feat_ten: torch.Tensor) -> torch.Tensor:
        natms = at_no.shape[0]
        rows = self.repid_table[at_no]
        valid = self.valid_table[at_no]
        dst_rep_ids_1d = rows[valid]
        dst_offsets_1d = (
            torch.arange(natms, dtype=torch.long, device=feat_ten.device)
            .unsqueeze(1)
            .expand(natms, rows.shape[1])[valid]
        )
        dst_flat_ids_1d = dst_offsets_1d * self.num_reps_1d + dst_rep_ids_1d
        n1d = self.num_reps_1d * natms
        out = torch.zeros(n1d * n1d, dtype=feat_ten.dtype, device=feat_ten.device)
        dst_flat_ids_2d = (
            dst_flat_ids_1d.unsqueeze(1) * n1d + dst_flat_ids_1d.unsqueeze(0)
        ).view(-1)
        out.index_add_(0, dst_flat_ids_2d, feat_ten.flatten())
        out = out.view(natms, self.num_reps_1d, natms, self.num_reps_1d).transpose(1, 2)
        return out


class JitMat2GraphLabel(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.utils.qc_matrice_graph.Mat2GraphLabel``.

    Converts a QC overlap matrix in the atomic-orbital basis into the
    atom-wise (diagonal) and atom-pair (off-diagonal) 2-body feature tensors
    used by the XPaiNN-Orb/X2 model.
    """

    def __init__(
        self,
        target_irreps: Union[str, o3.Irreps, Iterable],
        possible_elements: list,
        basisname: str = "def2tzvp",
        map_type: str = "pyscf",
    ):
        super().__init__()
        self.twobodypad = JitTwoBodyBlockPad(
            target_irreps,
            rep_dims=(0, 1),
            possible_elements=possible_elements,
            basisname=basisname,
            map_type=map_type,
        )

    def forward(
        self,
        at_no: torch.LongTensor,
        feat_matrice: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = at_no.shape[0]
        folded_X = self.twobodypad(at_no, feat_matrice)
        diagnol_mask = torch.eye(num_nodes, dtype=torch.bool, device=feat_matrice.device)
        node_label = folded_X[diagnol_mask, :, :]
        edge_label = folded_X[edge_index[0], edge_index[1]]
        return node_label, edge_label


class JitTwoBodyBlockMask(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.utils.qc_matrice_graph.TwoBodyBlockMask``.

    The per-element rep mask is stored as a single element-indexed buffer
    (atomic number -> mask row) instead of a python ``dict``.
    """

    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps, Iterable],
        possible_elements: list,
        basisname: str = "def2svp",
    ):
        super().__init__()
        from ..utils.qc import ELEMENTS_DICT, ELEMENTS_LIST, get_l_from_basis

        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.num_channels_1d = self.irreps_out.num_irreps
        num_reps_1d = 0
        num_channels = torch.LongTensor([0 for _ in range(self.irreps_out.lmax + 1)])
        dim_per_l = torch.LongTensor([2 * l + 1 for l in range(self.irreps_out.lmax + 1)])
        for mul, ir in self.irreps_out:
            num_reps_1d += mul * ir.dim
            num_channels[ir.l] = mul
        self.num_reps_1d = num_reps_1d
        offset_per_l = torch.cumsum(num_channels * dim_per_l, dim=0)
        offset_per_l = torch.concat((torch.LongTensor([0]), offset_per_l))

        mask_table = torch.zeros(len(ELEMENTS_LIST), num_reps_1d, dtype=torch.int32)
        for ele in possible_elements:
            if isinstance(ele, str):
                ele = ELEMENTS_DICT[ele]
            l_list = get_l_from_basis(basisname, ele)
            offset, cur_l = 0, l_list[0]
            for l in l_list:
                if l > cur_l:
                    offset = 0
                    cur_l = l
                pos_idx = torch.arange(2 * l + 1) + offset + offset_per_l[l]
                mask_table[ele, pos_idx] = 1
                offset += 2 * l + 1
        self.register_buffer("mask_table", mask_table)

    def forward(
        self,
        at_no: torch.LongTensor,
        edge_index: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dst_rep_1d_mask = self.mask_table[at_no]
        dst_rep_node_mask = dst_rep_1d_mask.unsqueeze(1) * dst_rep_1d_mask.unsqueeze(2)
        dst_rep_ket_mask_1d = dst_rep_1d_mask[edge_index[1]].view(
            edge_index.shape[1], 1, self.num_reps_1d
        )
        dst_rep_bra_mask_1d = dst_rep_1d_mask[edge_index[0]].view(
            edge_index.shape[1], self.num_reps_1d, 1
        )
        dst_rep_edge_mask = dst_rep_bra_mask_1d * dst_rep_ket_mask_1d
        return dst_rep_node_mask.to(torch.bool), dst_rep_edge_mask.to(torch.bool)


class JitBuildMatPerMole(nn.Module):
    """
    TorchScript-compatible mirror of ``xequinet.utils.qc_matrice_graph.BuildMatPerMole``.

    The per-element number of basis functions is stored as an element-indexed
    buffer instead of a python ``dict``, and ``torch.index_select`` (patched
    globally by ``torch_geometric``) is replaced by advanced indexing.
    """

    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps, Iterable],
        possible_elements: List[str],
        basisname: str = "def2-svp",
        map_type: str = "pyscf",
    ):
        super().__init__()
        from ..utils.qc import ELEMENTS_DICT, ELEMENTS_LIST, get_l_from_basis
        from ..utils.qc_matrice_graph import resolve_m_idx_type

        self.irreps_out = irreps_out if isinstance(irreps_out, o3.Irreps) else o3.Irreps(irreps_out)
        self.num_channels_1d = self.irreps_out.num_irreps
        out_repid_map = torch.zeros(self.irreps_out.dim, dtype=torch.long)
        m_idx_map = resolve_m_idx_type(map_type)
        src_id, offset_per_l = 0, [0]
        for mul, ir in self.irreps_out:
            cur_num_reps = mul * ir.dim
            offset_per_l.append(cur_num_reps + offset_per_l[-1])
            offset = 0
            for dst_n_current in range(mul):
                for m_qc in range(2 * ir.l + 1):
                    m_out = m_idx_map[ir.l][m_qc]
                    out_repid_map[src_id] = offset_per_l[ir.l] + offset + m_out
                    src_id += 1
                offset += 2 * ir.l + 1
        self.register_buffer("out_repid_map", out_repid_map)

        num_basis_table = torch.zeros(len(ELEMENTS_LIST), dtype=torch.long)
        for ele in possible_elements:
            if isinstance(ele, str):
                ele = ELEMENTS_DICT[ele]
            l_list = get_l_from_basis(basisname, ele)
            num_basis_table[ele] = sum(2 * l + 1 for l in l_list)
        self.register_buffer("num_basis_table", num_basis_table)

        self.num_reps_1d = self.irreps_out.dim

    def forward(
        self,
        res_node: torch.Tensor,
        res_edge: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        at_no: torch.LongTensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        out_repid_map = self.out_repid_map
        node_ten_tmp = res_node[:, :, out_repid_map]
        edge_ten_tmp = res_edge[:, :, out_repid_map]
        node_ten = node_ten_tmp[:, out_repid_map, :]
        edge_ten = edge_ten_tmp[:, out_repid_map, :]

        natms = at_no.shape[0]
        res = torch.zeros(
            natms, natms, self.num_reps_1d, self.num_reps_1d,
            device=res_node.device, dtype=res_node.dtype,
        )
        res_mask = torch.zeros(
            natms, natms, self.num_reps_1d, self.num_reps_1d,
            device=res_node.device,
        ).to(torch.bool)
        diagnol_mask = torch.eye(natms, device=res.device).to(torch.bool)
        res[diagnol_mask, :, :] = node_ten
        res_mask[diagnol_mask, :, :] = node_mask
        res[edge_index[0], edge_index[1], :, :] = edge_ten
        res_mask[edge_index[0], edge_index[1], :, :] = edge_mask
        res = res.transpose(1, 2)
        res_mask = res_mask.transpose(1, 2)
        tot_num_basis = self.num_basis_table[at_no].sum()
        return res[res_mask].reshape(tot_num_basis, tot_num_basis)
