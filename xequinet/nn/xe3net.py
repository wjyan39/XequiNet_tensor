"""
Simple Equivariant GCN for test purpose. 
"""

from typing import Union, Tuple, Iterable 

import torch 
import torch.nn as nn 
from e3nn import o3 
from torch_scatter import scatter 

from .tensorproduct import get_feasible_tp 
from .o3layer import resolve_actfn, Gate, Int2c1eEmbedding
from .rbf import resolve_rbf, resolve_cutoff 


class XE3embedding(nn.Module):
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
        """
        Args:
            `embed_dim`: Embedding dimension. (default: 16s + 8p + 4d = 28)
            `node_dim`: Node dimension.
            `edge_irreps`: Edge irreps.
            `num_basis`: Number of the radial basis functions.
            `rbf_kernel`: Radial basis function type.
            `cutoff`: Cutoff distance for the neighbor atoms.
            `cutoff_fn`: Cutoff function type.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.edge_num_irreps = self.edge_irreps.num_irreps
        # self.embedding = nn.Embedding(100, self.node_dim)
        self.int2c1e = Int2c1eEmbedding(embed_basis, aux_basis)
        self.node_lin = nn.Linear(self.int2c1e.embed_dim, self.node_dim)
        nn.init.zeros_(self.node_lin.bias)
        max_l = self.edge_irreps.lmax
        self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        self.sph_harm = o3.SphericalHarmonics(self.irreps_rshs, normalize=True, normalization="component")
        self.rbf = resolve_rbf(rbf_kernel, num_basis, cutoff)
        self.cutoff_fn = resolve_cutoff(cutoff_fn, cutoff)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            `x`: Atomic features.
            `pos`: Atomic coordinates.
            `edge_index`: Edge index.
        Returns:
            `x_scalar`: Scalar features.
            `rbf`: Values under radial basis functions.
            `fcut`: Values under cutoff function.
            `rsh`: Real spherical harmonics.
        """
        # calculate distance and relative position
        pos = pos[:, [1, 2, 0]]  # [x, y, z] -> [y, z, x]
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        # node linear
        # x_scalar = self.embedding(at_no)
        x = self.int2c1e(at_no)
        x_scalar = self.node_lin(x)
        # calculate radial basis function
        rbf = self.rbf(dist)
        fcut = self.cutoff_fn(dist)
        # calculate spherical harmonics
        rsh = self.sph_harm(vec)  # unit vector, normalized by component
        return x_scalar, rbf, fcut, rsh


class GraphConv(nn.Module):
    def __init__(
        self,
        irreps_node_in: Union[str, o3.Irreps, Iterable],
        irreps_node_out: Union[str, o3.Irreps, Iterable],
        num_basis: int = 16, 
        actfn: str = "silu",
        use_gate_activation:bool = True,
    ):
        super().__init__() 
        self.irreps_node_in = irreps_node_in if isinstance(irreps_node_in, o3.Irreps) else o3.Irreps(irreps_node_in)
        self.irreps_node_out = irreps_node_out if isinstance(irreps_node_out, o3.Irreps) else o3.Irreps(irreps_node_out)
        max_l = self.irreps_node_out.lmax
        self.irreps_rshs = o3.Irreps.spherical_harmonics(max_l)
        self.actfn = resolve_actfn(actfn)
        self.use_gate_activation = use_gate_activation and self.irreps_node_in == self.irreps_node_out

        self.irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_node_in, self.irreps_rshs, self.irreps_node_out, tp_mode="uvu"
        )
        self.tp = o3.TensorProduct(
            self.irreps_node_in,
            self.irreps_rshs,
            self.irreps_tp_out,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.mlp_edge_rbf = nn.Sequential(
            nn.Linear(num_basis, 128, bias=True),
            self.actfn, 
            nn.Linear(128, self.tp.weight_numel, bias=True) 
        )
        self.lin_node_pre = o3.Linear(self.irreps_node_in, self.irreps_node_in, biases=True)
        
        if self.use_gate_activation:
            self.gate_act = Gate(self.irreps_tp_out)
        
        self.lin_node_post = o3.Linear(self.irreps_tp_out, self.irreps_node_out)
    
    def forward(
        self, 
        node_feat:torch.Tensor, 
        edge_attr:torch.Tensor, 
        edge_rshs:torch.Tensor, 
        edge_index:torch.LongTensor
    ):
        x = self.lin_node_pre(node_feat) 
        x_j = x[edge_index[1], :] 
        msg_j = self.tp(x_j, edge_rshs, self.mlp_edge_rbf(edge_attr)) 
        accu_msg = scatter(msg_j, edge_index[0], dim=0, dim_size=node_feat.size(0)) 
        if self.use_gate_activation:
            out = x + self.lin_node_post(self.gate_act(accu_msg))
        else:
            out = self.lin_node_post(accu_msg)
        return out 


