from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
from torch_cluster import radius_graph

from ..nn import resolve_actfn
from ..nn.xpainn import XPainnMessage, XPainnUpdate
from ..nn.xpainnorb import FullEdgeKernel, FullEdgeSPH
from .jit_blocks import (
    JitBlockNorm2d,
    JitBuildMatPerMole,
    JitMat2GraphLabel,
    JitMatriceOut,
    JitTwoBodyBlockMask,
    JitXEmbedding,
    JitXMatTrans,
    JitZMatTrans,
    JitZMatriceOut,
)
from ..utils import NetConfig, unit_conversion, get_default_unit, get_atomic_energy


class JitGradOut(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 64,
        actfn: str = "silu",
    ):
        """
        Args:
            `node_dim`: Dimension of node feature.
            `hidden_dim`: Dimension of hidden layer.
            `actfn`: Activation function.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            resolve_actfn(actfn),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        x_scalar: torch.Tensor,
        coord: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            `x_scalar`: Scalar features.
            `at_no`: Atomic numbers.
            `coord`: Atomic coordinates.
        Returns:
            `energy`: Total energy.
            `grads`: Gradients.
        """
        atom_out = self.out_mlp(x_scalar)
        energy = atom_out.sum()
        grad = torch.autograd.grad(
            [energy,],
            [coord,],
            retain_graph=True,
            create_graph=True,
        )
        nuc_grad = grad[0]
        if nuc_grad is None:  # condition needed to unwrap optional for torchscript
            raise RuntimeError("Gradient is None")
        return energy, nuc_grad


class JitPaiNN(nn.Module):
    """
    XPaiNN model for JIT script. This model does not consider batch.
    """
    def __init__(self, config: NetConfig):
        super().__init__()
        self.embed = JitXEmbedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList([
            XPainnMessage(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                num_basis=config.num_basis,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.update = nn.ModuleList([
            XPainnUpdate(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.out = JitGradOut(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            actfn=config.activation,
        )
        self.cutoff = config.cutoff
        self.max_edges = config.max_edges
        self.prop_unit, self.len_unit = get_default_unit()
        atom_sp = get_atomic_energy(config.atom_ref) - get_atomic_energy(config.batom_ref)
        self.register_buffer("atom_sp", atom_sp)
        self.len_unit_conv = unit_conversion("Bohr", self.len_unit)
        self.prop_unit_conv = unit_conversion(self.prop_unit, "AU")
        self.grad_unit_conv = unit_conversion(f"{self.prop_unit}/{self.len_unit}", "AU")

    def forward(
        self,
        at_no: torch.LongTensor,
        coord: torch.Tensor,
        charge: int,
        spin: int,
    ) -> Dict[str, torch.Tensor]:
        coord.requires_grad_(True)
        coord = coord * self.len_unit_conv
        edge_index = radius_graph(coord, r=self.cutoff, max_num_neighbors=self.max_edges)
        shifts = torch.zeros((edge_index.shape[1], 3), device=coord.device)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, coord, edge_index, shifts)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        energy, nuc_grad = self.out(x_scalar, coord)
        energy = (energy.double() + self.atom_sp[at_no].sum()) * self.prop_unit_conv
        nuc_grad = nuc_grad.double() * self.grad_unit_conv
        result = {"energy": energy, "gradient": nuc_grad}
        return result


class JitEleEmbedding(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.sqrt_dim = math.sqrt(node_dim)
        self.q_linear = nn.Linear(node_dim, node_dim)
        nn.init.zeros_(self.q_linear.bias)
        self.k_linear = nn.Linear(1, node_dim)
        nn.init.zeros_(self.k_linear.bias)
        self.v_linear = nn.Linear(1, node_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        ele: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `x`: Node features.
            `ele`: Electronic features.
        Returns:
            Atomic features.
        """
        q = self.q_linear(x)
        k = self.k_linear(ele)
        v = self.v_linear(ele)
        dot = torch.sum(q * k, dim=1, keepdim=True) / self.sqrt_dim
        attn = torch.softmax(dot, dim=0)
        return attn * v


class JitPaiNNEle(JitPaiNN):
    """XPaiNN-ele model for JIT script. This model does not consider batch."""
    def __init__(self, config: NetConfig):
        super().__init__(config)
        self.charge_ebd = nn.ModuleList([
            JitEleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])
        self.spin_ebd = nn.ModuleList([
            JitEleEmbedding(node_dim=config.node_dim)
            for _ in range(config.action_blocks)
        ])

    def forward(
        self,
        at_no: torch.LongTensor,
        coord: torch.Tensor,
        charge: int,
        spin: int,
    ) -> Dict[str, torch.Tensor]:
        coord.requires_grad_(True)
        charge_t = torch.tensor([[charge]], dtype=coord.dtype, device=coord.device)
        spin_t = torch.tensor([[spin]], dtype=coord.dtype, device=coord.device)
        coord = coord * self.len_unit_conv
        edge_index = radius_graph(coord, r=self.cutoff, max_num_neighbors=self.max_edges)
        shifts = torch.zeros((edge_index.shape[1], 3), device=coord.device)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, coord, edge_index, shifts)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for ce, se, msg, upd in zip(self.charge_ebd, self.spin_ebd, self.message, self.update):
            x_scalar = x_scalar + ce(x_scalar, charge_t) + se(x_scalar, spin_t)
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        energy, nuc_grad = self.out(x_scalar, coord)
        energy = (energy.double() + self.atom_sp[at_no].sum()) * self.prop_unit_conv
        nuc_grad = nuc_grad.double() * self.grad_unit_conv
        result = {"energy": energy, "gradient": nuc_grad}
        return result


class JitXPaiNNOrb(nn.Module):
    """
    XPaiNN-Orb model for JIT script, assembling the full Hamiltonian matrix
    for a single molecule. This model does not consider batch nor PBC.

    The input coordinates are expected in ``config.default_length_unit`` and
    the returned ``"Fock"`` matrix is in Hartree.
    """

    def __init__(self, config: NetConfig):
        super().__init__()
        assert config.num_mat_conv <= config.action_blocks
        self.begin_read_idx = config.action_blocks - config.num_mat_conv
        self.pbc = False
        self.cutoff = config.cutoff
        self.max_edges = config.max_edges
        self.full_edge_index = config.full_edge_index
        self.embed = JitXEmbedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList([
            XPainnMessage(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                num_basis=config.num_basis,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.update = nn.ModuleList([
            XPainnUpdate(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.edge_full_embed = FullEdgeKernel(
            rbf_kernel=config.pair_rbf_kernel,
            num_basis=config.pair_num_basis,
            cutoff=config.pair_cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.mat_transform = nn.ModuleList([
            JitXMatTrans(
                irreps_node=config.edge_irreps,
                hidden_dim=config.mat_hidden_dim,
                max_l=config.max_l,
                edge_dim=config.pair_num_basis,
                actfn=config.activation,
                rbf_bias=True,
            )
            for _ in range(config.num_mat_conv)
        ])
        self.output = JitMatriceOut(
            irreps_out=config.irreps_out,
            node_dim=config.node_dim,
            hidden_dim=config.mat_hidden_dim,
            block_dim=config.mat_block_dim,
            max_l=config.max_l,
            actfn=config.activation,
            symmetrize=config.symmetrize,
        )
        self.mask_generator = JitTwoBodyBlockMask(
            irreps_out=config.irreps_out,
            possible_elements=config.possible_elements,
            basisname=config.target_basisname,
        )
        self.matrix_builder = JitBuildMatPerMole(
            irreps_out=config.irreps_out,
            possible_elements=config.possible_elements,
            basisname=config.target_basisname,
            map_type=config.m_idx_map_type,
        )
        self.prop_unit_conv = unit_conversion(config.default_property_unit, "Hartree")

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        edge_index = radius_graph(pos, r=self.cutoff, max_num_neighbors=self.max_edges)
        natms = at_no.shape[0]
        if self.full_edge_index:
            fc_edge_index = edge_index
        else:
            diag_mask = torch.eye(natms, dtype=torch.bool, device=pos.device)
            fc_edge_index = torch.nonzero(torch.logical_not(diag_mask)).T.long()
        shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device, dtype=pos.dtype)
        fc_shifts = torch.zeros((fc_edge_index.shape[1], 3), device=pos.device, dtype=pos.dtype)
        node_mask, edge_mask = self.mask_generator(at_no, fc_edge_index)

        x_scalar, rbf, fcut, rshs = self.embed(at_no, pos, edge_index, shifts)
        x_spherical = torch.zeros(
            (x_scalar.shape[0], rshs.shape[1]), device=x_scalar.device, dtype=x_scalar.dtype
        )
        full_rbfs = self.edge_full_embed(pos, fc_edge_index, fc_shifts)

        node_sph_ten: Optional[torch.Tensor] = None
        edge_sph_ten: Optional[torch.Tensor] = None
        for idx, (msg, upd) in enumerate(zip(self.message, self.update)):
            x_scalar, x_spherical = msg(x_scalar, x_spherical, rbf, fcut, rshs, edge_index)
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
            if idx >= self.begin_read_idx:
                for mt_idx, mt in enumerate(self.mat_transform):
                    if mt_idx == idx - self.begin_read_idx:
                        node_sph_ten, edge_sph_ten = mt(
                            x_spherical, full_rbfs, fc_edge_index, node_sph_ten, edge_sph_ten
                        )
        if node_sph_ten is None or edge_sph_ten is None:
            raise RuntimeError("No matrix transformation block was applied")
        node_blocks, edge_blocks = self.output(
            node_sph_ten, edge_sph_ten, x_scalar, fc_edge_index, fc_edge_index
        )
        fock = self.matrix_builder(
            node_blocks, edge_blocks, node_mask, edge_mask, at_no, fc_edge_index
        )
        return {"Fock": fock * self.prop_unit_conv}


class JitXPaiNN3Orb(nn.Module):
    """
    XPaiNN-Orb/X2 model for JIT script, assembling the full Hamiltonian matrix
    for a single molecule. This model does not consider batch nor PBC.

    ``ovlp`` is the overlap matrix in the atomic-orbital basis of
    ``config.target_basisname`` (shape ``(nbasis, nbasis)``); the X2 pair
    features (``fc_edge_attr``) are generated internally from it. The overlap
    is folded in ``float64`` and then cast to the model dtype. The input
    coordinates are expected in ``config.default_length_unit`` and the returned
    ``"Fock"`` matrix is in Hartree.
    """

    def __init__(self, config: NetConfig):
        super().__init__()
        assert config.num_mat_conv <= config.action_blocks
        self.begin_read_idx = config.action_blocks - config.num_mat_conv
        self.pbc = False
        self.cutoff = config.cutoff
        self.max_edges = config.max_edges
        self.full_edge_index = config.full_edge_index
        self.embed = JitXEmbedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList([
            XPainnMessage(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                num_basis=config.num_basis,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.update = nn.ModuleList([
            XPainnUpdate(
                node_dim=config.node_dim,
                edge_irreps=config.edge_irreps,
                actfn=config.activation,
                norm_type=config.norm_type,
            )
            for _ in range(config.action_blocks)
        ])
        self.edge_full_embed = JitBlockNorm2d(irreps_out=config.irreps_out)
        self.edge_full_rshs = FullEdgeSPH(irreps_node=config.edge_irreps)
        self.edge_attr_generator = JitMat2GraphLabel(
            target_irreps=config.irreps_out,
            possible_elements=config.possible_elements,
            basisname=config.target_basisname,
            map_type=config.m_idx_map_type,
        )
        self.mat_transform = nn.ModuleList([
            JitZMatTrans(
                irreps_node=config.edge_irreps,
                hidden_dim=config.mat_hidden_dim,
                max_l=config.max_l,
                edge_dim=config.pair_num_basis,
                actfn=config.activation,
                rbf_bias=False,
            )
            for _ in range(config.num_mat_conv)
        ])
        self.output = JitZMatriceOut(
            irreps_out=config.irreps_out,
            node_dim=config.node_dim,
            edge_dim=config.pair_num_basis,
            hidden_dim=config.mat_hidden_dim,
            block_dim=config.mat_block_dim,
            max_l=config.max_l,
            actfn=config.activation,
            symmetrize=config.symmetrize,
        )
        self.mask_generator = JitTwoBodyBlockMask(
            irreps_out=config.irreps_out,
            possible_elements=config.possible_elements,
            basisname=config.target_basisname,
        )
        self.matrix_builder = JitBuildMatPerMole(
            irreps_out=config.irreps_out,
            possible_elements=config.possible_elements,
            basisname=config.target_basisname,
            map_type=config.m_idx_map_type,
        )
        self.prop_unit_conv = unit_conversion(config.default_property_unit, "Hartree")

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        ovlp: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        edge_index = radius_graph(pos, r=self.cutoff, max_num_neighbors=self.max_edges)
        natms = at_no.shape[0]
        if self.full_edge_index:
            fc_edge_index = edge_index
        else:
            diag_mask = torch.eye(natms, dtype=torch.bool, device=pos.device)
            fc_edge_index = torch.nonzero(torch.logical_not(diag_mask)).T.long()
        shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device, dtype=pos.dtype)
        fc_shifts = torch.zeros((fc_edge_index.shape[1], 3), device=pos.device, dtype=pos.dtype)
        node_mask, edge_mask = self.mask_generator(at_no, fc_edge_index)

        x_scalar, rbfs, fcut, rshs = self.embed(at_no, pos, edge_index, shifts)
        x_spherical = torch.zeros(
            (x_scalar.shape[0], rshs.shape[1]), device=x_scalar.device, dtype=x_scalar.dtype
        )
        _, fc_edge_attr = self.edge_attr_generator(at_no, ovlp.to(torch.float64), fc_edge_index)
        fc_edge_attr = fc_edge_attr.to(x_scalar.dtype)
        full_rbfs = self.edge_full_embed(fc_edge_attr)
        full_rshs = self.edge_full_rshs(pos, fc_edge_index, fc_shifts)

        node_sph_ten: Optional[torch.Tensor] = None
        edge_sph_ten: Optional[torch.Tensor] = None
        for idx, (msg, upd) in enumerate(zip(self.message, self.update)):
            x_scalar, x_spherical = msg(x_scalar, x_spherical, rbfs, fcut, rshs, edge_index)
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
            if idx >= self.begin_read_idx:
                for mt_idx, mt in enumerate(self.mat_transform):
                    if mt_idx == idx - self.begin_read_idx:
                        node_sph_ten, edge_sph_ten = mt(
                            x_spherical, full_rbfs, full_rshs, fc_edge_index,
                            node_sph_ten, edge_sph_ten,
                        )
        if node_sph_ten is None or edge_sph_ten is None:
            raise RuntimeError("No matrix transformation block was applied")
        node_blocks, edge_blocks = self.output(
            node_sph_ten, edge_sph_ten, x_scalar, full_rbfs, fc_edge_index, fc_edge_index
        )
        fock = self.matrix_builder(
            node_blocks, edge_blocks, node_mask, edge_mask, at_no, fc_edge_index
        )
        return {"Fock": fock * self.prop_unit_conv}


def resolve_jit_model(config: NetConfig) -> nn.Module:
    version = config.version.lower()
    if version == "xpainn":
        return JitPaiNN(config)
    elif version == "xpainn-ele":
        return JitPaiNNEle(config)
    elif version == "xpainn-mat":
        return JitXPaiNNOrb(config)
    elif version == "xpainn3-mat":
        return JitXPaiNN3Orb(config)
    else:
        raise NotImplementedError(f"Unsupported model {config.version}")
    