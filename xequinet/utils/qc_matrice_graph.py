from typing import List, Tuple, Union, Iterable, Dict

import numpy as np 
import torch 
from e3nn.o3 import Irreps
from scipy.linalg import eigh

from pyscf import gto 
from .qc import ELEMENTS_DICT, get_l_from_basis


m_idx_map = {
    0: [0],
    1: [2, 0, 1],  # (x, y, z) to (y, z, x)
    2: [0, 1, 2, 3, 4], # (xy, yz, z2, xz, x2y2)
    3: [0, 1, 2, 3, 4, 5, 6], # (-3, -2, -1, 0, 1, 2, 3)
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8], # (-4, -3, -2, -1, 0, 1, 2, 3, 4)
}

m_idx_map_orca = {
    0: [0],
    1: [1, 2, 0],  # (z, x, y) to (y, z, x)
    2: [2, 3, 1, 4, 0], # (z^2, xz, yz, x2y2, xy) 
    3: [3, 4, 2, 5, 1, 6, 0], # (0, 1, -1, 2, -2, 3, -3)
    4: [4, 5, 3, 6, 2, 7, 1, 8, 0], # (0, 1, -1, 2, -2, 3, -3, 4, -4)
}

m_idx_map_origin = {
    0: [0],
    1: [0, 1, 2],  # (y, z, x)
    2: [0, 1, 2, 3, 4], # (xy, yz, z2, xz, x2y2)
    3: [0, 1, 2, 3, 4, 5, 6], # (-3, -2, -1, 0, 1, 2, 3)
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8], # (-4, -3, -2, -1, 0, 1, 2, 3, 4)
}


def resolve_m_idx_type(map_type:str="pyscf") -> Dict[int, List[int]]:
    map_type = map_type.lower()
    if map_type in ["orca"]:
        return m_idx_map_orca 
    elif map_type in ["pyscf", "gaussian", "rest"]:
        return m_idx_map
    else:
        return m_idx_map_origin


class TwoBodyBlockPad:
    r"""
    Class for converting 2-body feature (rank-2 tensor) from QC calculation to e3nn format 2d layout.
    The returned 2d SO(3) tensor is of symmetric layout along the rep_dims. 
    """
    def __init__(
        self,
        irreps_out: Union[str, Irreps, Iterable],
        rep_dims: Tuple[int, int],
        possible_elements: list,
        basisname: str = "def2svp",
        m_idx_map: Dict[int, List[int]] = m_idx_map
    ):
        """
        Args:
            `irreps_out`: e3nn.o3.Irreps, layout of the output padded 2d tensor.
            `rep_dims`: original two dimensions of the two-body feature tensor.
            `possible_elements`: list of exisisting elements in the dataset.
            `basisname`: name of the target basis set.
            `m_idx_map`: mapping from output m index from the QC program to torch_gauge's m index. 
        """
        self.irreps = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out)
        self.rep_dims = rep_dims
        self._generate_buffers(possible_elements, basisname, m_idx_map) 

    def _generate_buffers(self, possible_elements, basisname, m_idx_map):
        self.out_repid_map = {}
        self.num_channels_1d = self.irreps.num_irreps 
        num_reps_1d = 0
        num_channels = torch.LongTensor([0 for _ in range(self.irreps.lmax+1)]) 
        dim_per_l = torch.LongTensor([2*l + 1 for l in range(self.irreps.lmax+1)])
        for mul, ir in self.irreps:
            cur_num_reps = mul * ir.dim 
            num_reps_1d += cur_num_reps
            num_channels[ir.l] = mul 
        self.num_reps_1d = num_reps_1d
        offset_per_l = torch.cumsum(num_channels * dim_per_l, dim=0)
        offset_per_l = torch.concat((torch.LongTensor([0]), offset_per_l))
        for ele in possible_elements:
            if isinstance(ele, str):
                ele = ELEMENTS_DICT[ele]
            repid_map = []
            l_list = get_l_from_basis(basisname, ele)
            offset, cur_l = 0, l_list[0]
            for l in l_list:
                if l > cur_l:  # new shell 
                    offset = 0
                    cur_l = l 
                repid_map.append(torch.LongTensor(m_idx_map[l]) + offset + offset_per_l[l])
                offset += 2 * l + 1
            self.out_repid_map[ele] = torch.cat(repid_map)
    
    def __call__(self, at_no: torch.LongTensor, feat_ten: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `at_no`: atoms in the current molecule.
            `feat_ten`: input 2-body feature tensor. 
        Return:
            torch.Tensor: padded tensor of shape (natm, natm, Irrep, Irrep), natms refers to number of atoms in the system.
                As input matrice is a fully connected graph by abstract.
        """
        natms = len(at_no)
        # generate dst idx for padding
        dst_rep_ids_1d = torch.cat([self.out_repid_map[ele.item()] for ele in at_no]) 
        dst_offsets_1d = torch.arange(
            natms, dtype=torch.long, device=feat_ten.device
        ).repeat_interleave(
            torch.tensor([len(self.out_repid_map[ele.item()]) for ele in at_no], dtype=torch.long, device=feat_ten.device) 
        )
        dst_flat_ids_1d = dst_offsets_1d * self.num_reps_1d + dst_rep_ids_1d 
        # prepare for output 
        sp_2body_feat_ten_flat = torch.zeros(
            *feat_ten.shape[: self.rep_dims[0]], (self.num_reps_1d * natms) ** 2, *feat_ten.shape[self.rep_dims[1] + 1 :],
            dtype = feat_ten.dtype,
            device = feat_ten.device
        )
        # scatter interleaved reps to the padded 2d layout
        dst_flat_ids_2d = (
            dst_flat_ids_1d.unsqueeze(1) * (self.num_reps_1d * natms) + dst_flat_ids_1d.unsqueeze(0)
        ).view(-1) 
        sp_2body_feat_ten_flat.index_add_(self.rep_dims[0], dst_flat_ids_2d, feat_ten.flatten(*self.rep_dims)) 
        sp_2body_feat_ten = (
            sp_2body_feat_ten_flat.view(
                *feat_ten.shape[:self.rep_dims[0]], 
                natms, 
                self.num_reps_1d, 
                natms, 
                self.num_reps_1d,
                *feat_ten.shape[self.rep_dims[1]+1 :] 
            ).transpose(self.rep_dims[1], self.rep_dims[1] + 1)
            .contiguous()
        )
        return sp_2body_feat_ten


class TwoBodyBlockMask:
    r'''
    module to generate mask for two body feature Irrep padded tensor for batched training.
    '''
    def __init__(
        self, 
        irreps_out: Union[str, Irreps, Iterable], 
        possible_elements: list, 
        basisname: str = "def2svp"
    ):
        """
        Args:
            `out_irreps`: e3nn layout of the 2d tensor being masked.
            `possible_elements`: list of existing elements in the dataset.
            `basisname`: target basis set. 
        """
        self.irreps_out = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out)
        self.num_channels_1d = self.irreps_out.num_irreps
        # num_reps_1d, offset_per_l = 0, [0]
        num_reps_1d = 0
        num_channels = torch.LongTensor([0 for _ in range(self.irreps_out.lmax+1)]) 
        dim_per_l = torch.LongTensor([2*l + 1 for l in range(self.irreps_out.lmax+1)])
        for mul, ir in self.irreps_out:
            cur_num_reps = mul * ir.dim
            num_reps_1d += cur_num_reps
            num_channels[ir.l] = mul 
        self.num_reps_1d = num_reps_1d
        offset_per_l = torch.cumsum(num_channels * dim_per_l, dim=0)
        offset_per_l = torch.concat((torch.LongTensor([0]), offset_per_l))
        self.out_repid_mask = {}

        for ele in possible_elements:
            if isinstance(ele, str):
                ele = ELEMENTS_DICT[ele]
            l_list = get_l_from_basis(basisname, ele)
            offset, cur_l = 0, l_list[0]
            ele_repid_mask = torch.zeros(self.num_reps_1d, dtype=torch.int32)
            for l in l_list:
                if l > cur_l:   # new shell 
                    offset = 0 
                    cur_l = l 
                pos_idx = torch.arange(2*l+1) + offset + offset_per_l[l]
                ele_repid_mask[pos_idx] = 1
                offset += 2 * l + 1
            self.out_repid_mask[ele] = ele_repid_mask

    def __call__(self, at_no: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Args:
            `at_no`: list of atomic numbers in the current batch.
            `edge_index`: PyG edge index of shape (2, num_edge).
        Return:
            mask for node and edge tensor 
        """
        device = at_no.device
        dst_rep_1d_mask = torch.stack([self.out_repid_mask[ele.item()] for ele in at_no]).to(device)
        # Diagnol mask for node 
        dst_rep_node_mask = dst_rep_1d_mask.unsqueeze(1) * dst_rep_1d_mask.unsqueeze(2) 
        # Off-diagnol mask for edge 
        dst_rep_ket_mask_1d = torch.index_select(dst_rep_1d_mask, dim=0, index=edge_index[1]).view(edge_index.shape[1], 1, self.num_reps_1d)
        dst_rep_bra_mask_1d = torch.index_select(dst_rep_1d_mask, dim=0, index=edge_index[0]).view(edge_index.shape[1], self.num_reps_1d, 1) 
        dst_rep_edge_mask = (dst_rep_bra_mask_1d * dst_rep_ket_mask_1d)
        return dst_rep_node_mask.bool(), dst_rep_edge_mask.bool()


class Mat2GraphLabel:
    r"""
    module to convert QC matrice to PyG data label for each atom (node) and atom pair (edge of complete graph).
    """
    def __init__(
        self, 
        target_irreps: Union[str, Irreps, Iterable], 
        possible_elements: list, 
        basisname: str = "def2tzvp",
        map_type: str = "pyscf",
    ):
        """
        Args:
            `target_irreps`: e3nn layout of the 2d tensor being masked.
            `possible_elements`: list of existing elements in the dataset.
            `basisname`: target basis set. 
        """
        target_irreps = target_irreps if isinstance(target_irreps, Irreps) else Irreps(target_irreps)
        self.twobodypad = TwoBodyBlockPad(
            target_irreps,
            rep_dims=(0, 1),
            possible_elements=possible_elements,
            basisname=basisname,
            m_idx_map=resolve_m_idx_type(map_type),
        )

    def __call__(self, data, feat_matrice: torch.Tensor, at_no: torch.LongTensor = None, edge_index: torch.Tensor = None):
        """
        Args:
            `data`: PyG data object.
            `feat_matrice`: input 2-body feature tensor.
            `at_no`: atomic numbers of the current molecule.
            `edge_index`: PyG edge index of shape (2, num_edge).
        Return:
            node_label: torch.Tensor of shape (natm, Irrep, Irrep)
            edge_label: torch.Tensor of shape (num_edge, Irrep, Irrep), for complete graph, num_edge = natm * (natm - 1)
        """
        num_nodes = len(data.at_no)
        folded_X = self.twobodypad(at_no, feat_matrice)
        
        diagnol_mask = torch.eye(num_nodes, dtype=torch.bool, device=feat_matrice.device)
        node_label = folded_X[diagnol_mask, :, :]
        if edge_index is None:
            # as a fully connected graph
            edge_mask = torch.logical_not(diagnol_mask)
            edge_index = torch.nonzero(edge_mask).T.long()
            data.fc_edge_index = edge_index
        edge_label = folded_X[edge_index[0], edge_index[1], ...]
        return node_label, edge_label


class BuildMatPerMole:
    r"""
    module to convert model output to QC matrice for each molecule.
    """
    def __init__(
        self, 
        irreps_out: Union[str, Irreps, Iterable], 
        possible_elements: List[str],
        basisname: str = "def2-svp",
        map_type: str = "pyscf",
    ):
        super().__init__() 
        self.irreps_out = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out) 
        self.num_channels_1d = self.irreps_out.num_irreps
        out_repid_map = torch.zeros(self.irreps_out.dim, dtype=torch.long)
        self.elem_num_basis = {}
        # out_repid_map 
        m_idx_map = resolve_m_idx_type(map_type)
        src_id, offset_per_l = 0, [0]
        for mul, ir in self.irreps_out:
            cur_num_reps = mul * ir.dim
            offset_per_l.append(cur_num_reps + offset_per_l[-1])
            offset = 0
            for dst_n_current in range(mul):
                for m_qc in range(2*ir.l + 1):
                    m_out = m_idx_map[ir.l][m_qc] 
                    out_repid_map[src_id] = offset_per_l[ir.l] + offset + m_out 
                    src_id += 1 
                offset += 2 * ir.l + 1 
        # num_basis for each element 
        for ele in possible_elements:
            if isinstance(ele, str):
                ele = ELEMENTS_DICT[ele]
            self.elem_num_basis[ele] = 0
            l_list = get_l_from_basis(basisname, ele)
            for l in l_list:
                self.elem_num_basis[ele] += 2 * l + 1
        # register buffer
        self.num_reps_1d = self.irreps_out.dim 
        self.out_repid_map = out_repid_map

    def __call__(
        self, 
        res_node: torch.Tensor,
        res_edge: torch.Tensor,
        node_mask: torch.BoolTensor,
        edge_mask: torch.BoolTensor,
        at_no: torch.LongTensor, 
        edge_index: torch.LongTensor, 
    ):
        out_repid_map = self.out_repid_map.to(edge_index.device)
        # transform e3nn's rep layout to qc interleaved layout along the two dims where the matrice is stored.
        # notice: no need to swap mask tensor, as ordering only differs in m index. 
        # dimension -1
        node_ten_tmp = torch.index_select(res_node, dim=2, index=out_repid_map)
        edge_ten_tmp = torch.index_select(res_edge, dim=2, index=out_repid_map)
        # dimension -2 
        node_ten = torch.index_select(node_ten_tmp, dim=1, index=out_repid_map)
        edge_ten = torch.index_select(edge_ten_tmp, dim=1, index=out_repid_map)
        # create a sratch tensor to hold output, shape(natm, natm, num_reps_1d, num_reps_1d)
        natms = at_no.shape[0]
        res = torch.zeros(natms, natms, self.num_reps_1d, self.num_reps_1d, device=res_node.device, dtype=res_node.dtype)
        res_mask = torch.zeros(natms, natms, self.num_reps_1d, self.num_reps_1d, device=res_node.device).bool()
        # get the huge mask 
        diagnol_mask = torch.eye(natms, device=res.device).bool()
        # diagonal res 
        res[diagnol_mask, :, :] = node_ten 
        res_mask[diagnol_mask, :, :] = node_mask
        # scatter edge sub-blocks onto off-diagnol part of the res matrix 
        res[edge_index[0], edge_index[1], :, : ] = edge_ten
        res_mask[edge_index[0], edge_index[1], :, : ] = edge_mask
        # transpose to (natm, num_reps_1d, natm, num_reps_1d)
        res = res.transpose(1, 2)
        res_mask = res_mask.transpose(1, 2)
        # mask and reshape
        tot_num_basis = sum(self.elem_num_basis[ele.item()] for ele in at_no)
        return res[res_mask].reshape(tot_num_basis, tot_num_basis)


class QCMatriceBuilder(torch.nn.Module):
    """
    Build block diagnol QC Matrice in a batch.
    """
    def __init__(
        self, 
        irreps_out: Union[str, Irreps, Iterable], 
        possible_elements: List[str],
        basisname: str = "def2svp",
        map_type: str = "pyscf",
    ):
        super().__init__() 
        self.irreps_out = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out) 
        self.num_channels_1d = self.irreps_out.num_irreps
        # metadata
        m_idx_map = resolve_m_idx_type(map_type)
        num_reps_1d = 0
        num_channels = torch.LongTensor([0 for _ in range(self.irreps_out.lmax+1)]) 
        dim_per_l = torch.LongTensor([2*l + 1 for l in range(self.irreps_out.lmax+1)])
        for mul, ir in self.irreps_out:
            cur_num_reps = mul * ir.dim
            num_reps_1d += cur_num_reps
            num_channels[ir.l] = mul 
        self.num_reps_1d = num_reps_1d
        offset_per_l = torch.cumsum(num_channels * dim_per_l, dim=0)
        offset_per_l = torch.concat((torch.LongTensor([0]), offset_per_l))
        # out_repid_map 
        out_repid_map = torch.zeros(self.irreps_out.dim, dtype=torch.long)
        src_id = 0
        for mul, ir in self.irreps_out:
            offset = 0
            for dst_n_current in range(mul):
                for m_qc in range(2*ir.l + 1):
                    m_out = m_idx_map[ir.l][m_qc] 
                    out_repid_map[src_id] = offset_per_l[ir.l] + offset + m_out 
                    src_id += 1 
                offset += 2 * ir.l + 1 
        # out_repid_mask & num_basis 
        out_repid_mask = {}
        elem_num_basis = {}
        for ele in possible_elements:
            if isinstance(ele, str):
                ele = ELEMENTS_DICT[ele]
            l_list = get_l_from_basis(basisname, ele)
            ele_repid_mask = torch.zeros(self.num_reps_1d, dtype=torch.int32)
            elem_num_basis[ele] = 0
            offset, cur_l = 0, l_list[0]
            for l in l_list:
                elem_num_basis[ele] += 2 * l + 1
                if l > cur_l:   # new shell 
                    offset = 0 
                    cur_l = l 
                pos_idx = torch.arange(2*l+1) + offset + offset_per_l[l]
                ele_repid_mask[pos_idx] = 1
                offset += 2 * l + 1
            out_repid_mask[ele] = ele_repid_mask
        # register buffer
        self.register_buffer("out_repid_map", out_repid_map)
        # self.out_repid_map = out_repid_map
        self.out_repid_mask = out_repid_mask 
        self.elem_num_basis = elem_num_basis

    def forward(
        self, 
        res_node: torch.Tensor,
        res_edge: torch.Tensor,
        at_no: torch.LongTensor, 
        edge_index: torch.LongTensor, 
    ):
        # generate huge fully connected edge_index 
        batch_num_nodes = res_node.shape[0]
        batch_diagnol_mask = torch.eye(batch_num_nodes, dtype=torch.bool, device=res_node.device)    
        batch_offdiag_mask = torch.logical_not(batch_diagnol_mask)
        batch_edge_index = torch.nonzero(batch_offdiag_mask).T.long()
        # huge mask
        dst_rep_1d_mask = torch.stack([self.out_repid_mask[ele.item()] for ele in at_no]).to(res_node.device)
        ## Diagnol mask for node 
        dst_rep_node_mask = dst_rep_1d_mask.unsqueeze(1) * dst_rep_1d_mask.unsqueeze(2) 
        ## Off-diagnol mask for edge 
        dst_rep_ket_mask_1d = torch.index_select(dst_rep_1d_mask, dim=0, index=batch_edge_index[1]).view(batch_edge_index.shape[1], 1, self.num_reps_1d)
        dst_rep_bra_mask_1d = torch.index_select(dst_rep_1d_mask, dim=0, index=batch_edge_index[0]).view(batch_edge_index.shape[1], self.num_reps_1d, 1) 
        dst_rep_edge_mask = (dst_rep_bra_mask_1d * dst_rep_ket_mask_1d)
        batch_node_mask = dst_rep_node_mask.bool()
        batch_edge_mask = dst_rep_edge_mask.bool()
        # transform e3nn's rep layout to qc interleaved layout along the two dims where the matrice is stored.
        # dimension -1
        node_ten_tmp = torch.index_select(res_node, dim=2, index=self.out_repid_map)
        edge_ten_tmp = torch.index_select(res_edge, dim=2, index=self.out_repid_map)
        # dimension -2 
        node_ten = torch.index_select(node_ten_tmp, dim=1, index=self.out_repid_map)
        edge_ten = torch.index_select(edge_ten_tmp, dim=1, index=self.out_repid_map)
        # create a scratch tensor to hold output, shape(natm, natm, num_reps_1d, num_reps_1d)
        res = torch.zeros(batch_num_nodes, batch_num_nodes, self.num_reps_1d, self.num_reps_1d, device=res_node.device, dtype=res_node.dtype)
        res_mask = torch.zeros(batch_num_nodes, batch_num_nodes, self.num_reps_1d, self.num_reps_1d, device=res_node.device).bool() 
        # diagonal res 
        res[batch_diagnol_mask, :, :] = node_ten 
        res_mask[batch_diagnol_mask, :, :] = batch_node_mask
        # scatter edge sub-blocks onto off-diagnol part of the res matrix 
        res[edge_index[0], edge_index[1], :, : ] = edge_ten
        res_mask[batch_edge_index[0], batch_edge_index[1], :, : ] = batch_edge_mask
        # transpose to (natm, num_reps_1d, natm, num_reps_1d)
        res = res.transpose(1, 2)
        res_mask = res_mask.transpose(1, 2)
        # mask and reshape
        tot_num_basis = sum(self.elem_num_basis[ele.item()] for ele in at_no)
        return res[res_mask].reshape(tot_num_basis, tot_num_basis)


class OrbitalCalculator:
    def __init__(self, basisname, lenth_unit, ortho_transformed=False):
        self.basisname = basisname 
        self.lenth_unit = lenth_unit
        self.ortho_trans = ortho_transformed
    
    def __call__(
        self, 
        fock:np.ndarray, 
        at_no:np.ndarray, 
        coords:np.ndarray, 
        charge=0
    ):
        """
        Calculate the orbitals of a molecule.
        """
        mol = gto.M(
            atom=[(str(ele), tuple(coords[i])) for i, ele in enumerate(at_no)], 
            charge=charge,
            basis=self.basisname,
            unit=self.lenth_unit
        )
        mol.build()
        # diagonalize the fock matrix to get orbital energies and coefficients
        if self.ortho_trans:
            orb_energies, orb_coeffs = eigh(fock)
        else:
            ovlp:np.ndarray = mol.intor("int1e_ovlp")
            orb_energies, orb_coeffs = eigh(fock, ovlp)
        idx = np.argmax(abs(orb_coeffs.real), axis=0)
        orb_coeffs[:,orb_coeffs[idx,np.arange(len(orb_energies))].real<0] *= -1
        # occupied number of orbitals
        nocc = mol.nelectron // 2
        nvirt = mol.nao - nocc
    
        return orb_coeffs, nocc, nvirt



"""
The following is implemented for inference purpose only for large and extended systems in the batch-loop manner. 
The input is a bipartite graph, i.e., the two-body feature tensor is not symmetric along the two dimensions, centering atom and its neighbors.
"""

class TwoBodyBlockPadAsym:
    r"""
    Class for converting 2-body feature from QC calculation to e3nn format 2d layout.
    """
    def __init__(
        self,
        irreps_out: Union[str, Irreps, Iterable],
        rep_dims: Tuple[int, int],
        possible_elements: list,
        basisname: str = "def2svp",
        m_idx_map: Dict[int, List[int]] = resolve_m_idx_type("pyscf")
    ):
        """
        Args:
            `irreps_out`: e3nn.o3.Irreps, layout of the output padded 2d tensor.
            `rep_dims`: original two dimensions of the two-body feature tensor.
            `possible_elements`: list of exisisting elements in the dataset.
            `basisname`: name of the target basis set.
            `m_idx_map`: mapping from output m index from the QC program to torch_gauge's m index. 
        """
        self.irreps = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out)
        self.rep_dims = rep_dims
        self._generate_buffers(possible_elements, basisname, m_idx_map) 

    def _generate_buffers(self, possible_elements, basisname, m_idx_map):
        self.out_repid_map = {}
        self.num_channels_1d = self.irreps.num_irreps 
        num_reps_1d = 0
        num_channels = torch.LongTensor([0 for _ in range(self.irreps.lmax+1)]) 
        dim_per_l = torch.LongTensor([2*l + 1 for l in range(self.irreps.lmax+1)])
        for mul, ir in self.irreps:
            cur_num_reps = mul * ir.dim 
            num_reps_1d += cur_num_reps
            num_channels[ir.l] = mul 
        self.num_reps_1d = num_reps_1d
        offset_per_l = torch.cumsum(num_channels * dim_per_l, dim=0)
        offset_per_l = torch.concat((torch.LongTensor([0]), offset_per_l))
        for ele in possible_elements:
            if isinstance(ele, str):
                ele = ELEMENTS_DICT[ele]
            repid_map = []
            l_list = get_l_from_basis(basisname, ele)
            offset, cur_l = 0, l_list[0]
            for l in l_list:
                if l > cur_l:  # new shell 
                    offset = 0
                    cur_l = l 
                repid_map.append(torch.LongTensor(m_idx_map[l]) + offset + offset_per_l[l])
                offset += 2 * l + 1
            self.out_repid_map[ele] = torch.cat(repid_map)
    
    def __call__(self, at_no_i: torch.LongTensor, at_no_j: torch.LongTensor, feat_ten: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `at_no_i`: source atoms.
            `at_no_j`: destination atoms.
            `feat_ten`: input 2-body feature tensor. 
        Return:
            torch.Tensor: padded tensor of shape (natm_i, natm_j, Irrep, Irrep), natms_i refers to number of source atoms in the system,
                natms_j refers to number of destination atoms in the system, Irrep is the output irreps dimension.
        """
        natms_i = len(at_no_i)
        natms_j = len(at_no_j)
        # generate dst idx for padding
        dst_rep_ids_1d_j = torch.cat([self.out_repid_map[ele.item()] for ele in at_no_j]) 
        dst_offsets_1d_j = torch.arange(
            natms_j, dtype=torch.long, device=feat_ten.device
        ).repeat_interleave(
            torch.tensor([len(self.out_repid_map[ele.item()]) for ele in at_no_j], dtype=torch.long, device=feat_ten.device) 
        )
        dst_rep_ids_1d_i = torch.cat([self.out_repid_map[ele.item()] for ele in at_no_i])
        dst_offsets_1d_i = torch.arange(
            natms_i, dtype=torch.long, device=feat_ten.device
        ).repeat_interleave(
            torch.tensor([len(self.out_repid_map[ele.item()]) for ele in at_no_i], dtype=torch.long, device=feat_ten.device) 
        )
        # print(dst_offsets_1d)
        dst_flat_ids_1d_j = dst_offsets_1d_j * self.num_reps_1d + dst_rep_ids_1d_j
        dst_flat_ids_1d_i = dst_offsets_1d_i * self.num_reps_1d + dst_rep_ids_1d_i
        # print(dst_flat_ids_1d)
        # prepare for output 
        sp_2body_feat_ten_flat = torch.zeros(
            *feat_ten.shape[: self.rep_dims[0]], (self.num_reps_1d * natms_i) * (self.num_reps_1d * natms_j) , *feat_ten.shape[self.rep_dims[1] + 1 :],
            dtype = feat_ten.dtype,
            device = feat_ten.device
        )
        # scatter interleaved reps to the padded 2d layout
        dst_flat_ids_2d = (
            dst_flat_ids_1d_i.unsqueeze(1) * (self.num_reps_1d * natms_j) + dst_flat_ids_1d_j.unsqueeze(0)
        ).view(-1) 
        # print(dst_flat_ids_2d, dst_flat_ids_2d.shape)
        sp_2body_feat_ten_flat.index_add_(self.rep_dims[0], dst_flat_ids_2d, feat_ten.flatten(*self.rep_dims)) 
        sp_2body_feat_ten = (
            sp_2body_feat_ten_flat.view(
                *feat_ten.shape[:self.rep_dims[0]], 
                natms_i, 
                self.num_reps_1d, 
                natms_j, 
                self.num_reps_1d,
                *feat_ten.shape[self.rep_dims[1]+1 :] 
            ).transpose(self.rep_dims[1], self.rep_dims[1] + 1)
            .contiguous()
        )
        return sp_2body_feat_ten


class Mat2GraphLabelAsym:
    r"""
    module to convert QC matrice to PyG data label for each atom (node) and 
    atom pair (edge of fully connected graph).
    """
    def __init__(
        self, 
        target_irreps: Union[str, Irreps, Iterable], 
        possible_elements: list, 
        basisname: str = "def2-svp",
        map_type: str = "pyscf",
    ):
        """
        Args:
            `target_irreps`: e3nn layout of the 2d tensor being masked.
            `possible_elements`: list of existing elements in the dataset.
            `basisname`: target basis set. 
        """
        target_irreps = target_irreps if isinstance(target_irreps, Irreps) else Irreps(target_irreps)
        self.twobodypad = TwoBodyBlockPadAsym(
            target_irreps,
            rep_dims=(0, 1),
            possible_elements=possible_elements,
            basisname=basisname,
            m_idx_map=resolve_m_idx_type(map_type),
        )

    def __call__(self, feat_matrice: torch.Tensor, at_no_i: torch.LongTensor = None, at_no_j: torch.LongTensor = None, edge_index: torch.LongTensor = None) -> torch.Tensor:
        
        folded_X = self.twobodypad(at_no_i, at_no_j, feat_matrice)
        if edge_index is not None:
            edge_label = folded_X[edge_index[0], edge_index[1], ...]
        else:
            edge_label = folded_X.flatten(0, 1)
        return edge_label