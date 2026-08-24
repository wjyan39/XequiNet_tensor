from typing import Union, Tuple, Iterable 

import torch 
import torch.nn as nn 
from e3nn import o3 

from .rbf import resolve_rbf, resolve_cutoff
from .matlayer import SelfLayer, PairLayer, PairLayerSPH


class FullEdgeKernel(nn.Module):
    """
    RBF embedding layer for fully connected graph.
    """
    def __init__(
        self,
        rbf_kernel: str = "gaussian",
        num_basis: int = 32,
        cutoff: float = 8.0,
        cutoff_fn: str = "cosine",
    ):
        super().__init__()
        self._num_rbf = num_basis
        self._rcut = cutoff 
        self.rbf_kernel = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)
    
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        # pos = pos[:, [1, 2, 0]] # (x, y, z) to (y, z, x)
        vec = pos[edge_index[0]] - pos[edge_index[1]] - shifts 
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True) 
        rbfs = self.rbf_kernel(dist)
        fcut = self.cutoff_fn(dist)
        return rbfs * fcut


class FullEdgeSPH(nn.Module):
    """
    Spherical Harmonics embedding layer for fully connected graph.
    """
    def __init__(
        self,
        irreps_node: Union[str, o3.Irreps, Iterable] = "64x0e + 64x1o + 64x2e + 64x3o + 64x4e",
    ):
        super().__init__()
        irreps_node = irreps_node if isinstance(irreps_node, o3.Irreps) else o3.Irreps(irreps_node)
        max_l = irreps_node.lmax
        self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        self.sph_harm = o3.SphericalHarmonics(self.irreps_rshs, normalize=True, normalization="component")
    
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
        shifts: torch.Tensor,
    ):
        # pos = pos[:, [1, 2, 0]]
        vec = pos[edge_index[0]] - pos[edge_index[1]] - shifts
        vec = vec[:, [1, 2, 0]]
        rsh = self.sph_harm(vec)
        return rsh 
    
    def __repr__(self):
        return f"FullEdgeSPH(l_max = {self.irreps_rshs.lmax})"


class BlockNorm2d(nn.Module):
    r"""
    Class for sub-block-wise normalization of the 2-body feature tensor.
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
            .expand(
                self.irreps.dim, self.irreps.dim
            )
        )
        idx_ten_1 = (
            irreps_layout_1d
            .unsqueeze(0)
            .expand(
                self.irreps.dim, self.irreps.dim
            )
        )
        idx_ten_flat = (
            idx_ten_0 * self.irreps.num_irreps + idx_ten_1
        ).flatten()
        self.register_buffer("idx_ten_flat", idx_ten_flat)
    
    def forward(self, feat_ten: torch.Tensor) -> torch.Tensor:
        out_shape = list(feat_ten.shape[:-2]) + [self.irreps.num_irreps**2]
        sum_sqr = torch.zeros(
            out_shape,
            device=feat_ten.device,
            dtype=feat_ten.dtype,
        )
        sum_sqr = sum_sqr.index_add_(
            dim=-1, index=self.idx_ten_flat, source=feat_ten.flatten(-2, -1).pow(2)
        )
        return torch.sqrt(sum_sqr)

    def __repr__(self):
        return f"BlockNorm2d({self.irreps}) with {self.irreps.num_irreps**2} channels"


class XMatTrans(nn.Module):
    """
    Read out transform for XPaiNN-Orb from XPaiNN layer output.
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
        self.irreps_hidden_base = o3.Irreps([(hidden_dim, (l, (-1)**l)) for l in range(self.irreps_in.lmax + 1)]) 
        irreps_hidden = [(hidden_dim, (0, 1))]
        for l in range(2, 2*max_l+1):
            irreps_hidden.append((hidden_dim, (l//2, (-1)**l)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        # Building block 
        if self.irreps_in == self.irreps_hidden_base:
            self.pretrans = False 
        else:
            self.pretrans = True
            self.node_pre_trans = o3.Linear(self.irreps_in, self.irreps_hidden_base, biases=False)
        self.node_self_layer = SelfLayer(self.irreps_hidden_base, self.irreps_hidden, actfn)
        self.node_pair_layer = PairLayer(self.irreps_hidden_base, self.irreps_hidden, edge_dim, actfn, rbf_bias)
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        fii: Union[torch.Tensor, None],
        fij: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pretrans:
            node_feat = self.node_pre_trans(node_feat)
        fii = self.node_self_layer(node_feat, fii)
        fij = self.node_pair_layer(node_feat, edge_attr, edge_index, fij)
        return fii, fij


class ZMatTrans(nn.Module):
    """
    Read out transform for XPaiNNorb from XPaiNN layer output.
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
        self.irreps_hidden_base = o3.Irreps([(hidden_dim, (l, (-1)**l)) for l in range(self.irreps_in.lmax + 1)]) 
        irreps_hidden = [(hidden_dim, (0, 1))]
        for l in range(2, 2*max_l+1):
            irreps_hidden.append((hidden_dim, (l//2, (-1)**l)))
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        # Building block 
        if self.irreps_in == self.irreps_hidden_base:
            self.pretrans = False 
        else:
            self.pretrans = True
            self.node_pre_trans = o3.Linear(self.irreps_in, self.irreps_hidden_base, biases=False)
        self.node_self_layer = SelfLayer(self.irreps_hidden_base, self.irreps_hidden, actfn)
        self.node_pair_layer = PairLayerSPH(self.irreps_hidden_base, self.irreps_hidden, self.irreps_in.lmax, edge_dim, actfn, rbf_bias)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_rshs: torch.Tensor,
        edge_index: torch.LongTensor,
        fii: Union[torch.Tensor, None],
        fij: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pretrans:
            node_feat = self.node_pre_trans(node_feat)
        fii = self.node_self_layer(node_feat, fii)
        fij = self.node_pair_layer(node_feat, edge_attr, edge_rshs, edge_index, fij)
        return fii, fij



