import torch
import torch.nn as nn
from torch_geometric.data import Data

from .xpainn import (
    XEmbedding, XPainnMessage, XPainnUpdate
)
from .painn import (
    Embedding, PainnMessage, PainnUpdate,
)
from .output import resolve_output
from .xe3net import (
    GraphConv, XE3embedding
)
from .xpainnorb import (
    XMatTrans, ZMatTrans, FullEdgeKernel, FullEdgeSPH, BlockNorm2d
)
from .xqhnet import (
    XMatEmbedding, NodewiseInteraction, MatTrans
)
from .matlayer import (
    MatriceOut, ZMatriceOut
)
from ..utils import NetConfig


class XPaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
        self.embed = XEmbedding(
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
        self.out = resolve_output(config)
    
    def forward(self, data: Data):
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        # embed input
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        # initialize vector with zeros
        x_spherical = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        # message passing and node update
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_spherical = msg(x_scalar, x_spherical, rbf, fcut, rsh, edge_index)
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
        # output
        result = self.out(data, x_scalar, x_spherical)
        return result


class PaiNN(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.embed = Embedding(
            node_dim=config.node_dim,
            num_basis=config.num_basis,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.message = nn.ModuleList([
            PainnMessage(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                num_basis=config.num_basis,
                actfn=config.activation,
            )
            for _ in range(config.action_blocks)
        ])
        self.update = nn.ModuleList([
            PainnUpdate(
                node_dim=config.node_dim,
                edge_dim=config.edge_dim,
                actfn=config.activation,
            )
            for _ in range(config.action_blocks)
        ])
        self.out = resolve_output(config)

    def forward(self, data: Data):
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index; batch_idx=data.batch
        # embed input
        x_scalar, rbf, envelop, rsh = self.embed(at_no, pos, edge_index)
        # initialize vector with zeros
        x_vector = torch.zeros((x_scalar.shape[0], 3, 128), device=x_scalar.device)
        # message passing and node update
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, envelop, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        # output
        result = self.out(data, x_scalar, x_vector)
        return result



# for test 
class XE3Net(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
        self.embed = XE3embedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        irreps_in = f"{config.node_dim}x0e"
        self.message = nn.ModuleList([
            GraphConv(
                irreps_node_in=irreps_in if idx == 0 else config.edge_irreps,
                irreps_node_out=config.edge_irreps,
                num_basis=config.num_basis,
                actfn=config.activation,
                norm_type="nonorm" if idx == 0 else config.norm_type,
                use_gate_activation=False if idx == 0 else True,
            )
            for idx in range(config.action_blocks)
        ])
        self.out = resolve_output(config)
    
    def forward(self, data: Data):
        """
        Args:
            `data`: Input data.
        Returns:
            `result`: Output.
        """
        # get required input from data
        at_no = data.at_no; pos=data.pos; edge_index=data.edge_index; batch=data.batch
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        # embed input
        x_embed, rbf, fcut, rsh = self.embed(at_no, pos, edge_index, shifts)
        edge_attr = rbf * fcut 
        node_feat = x_embed
        # message passing and node update
        for conv_layer in self.message:
            node_feat = conv_layer(node_feat, edge_attr, rsh, edge_index)
        # output
        result = self.out(data, x_embed, node_feat)
        return result


class XQHNet(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        assert config.num_mat_conv <= config.action_blocks
        self.begin_read_idx = config.action_blocks - config.num_mat_conv
        self.pbc = config.pbc
        self.embed = XMatEmbedding(
            node_dim=config.node_dim,
            edge_irreps=config.edge_irreps,
            embed_basis=config.embed_basis,
            aux_basis=config.aux_basis,
            num_basis=config.num_basis,
            rbf_kernel=config.rbf_kernel,
            cutoff=config.cutoff,
            pair_cutoff=config.pair_cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.mat_conv = nn.ModuleList() 
        for idx in range(config.action_blocks):
            irreps_in = f"{config.node_dim}x0e" if idx == 0 else config.edge_irreps 
            self.mat_conv.append(
                NodewiseInteraction(
                    irreps_node_in=irreps_in,
                    irreps_node_out=config.edge_irreps,
                    node_dim=config.node_dim,
                    edge_attr_dim=config.num_basis,
                    actfn=config.activation,
                    use_normgate=False if idx == 0 else True,
                )
            )
        self.mat_trans = nn.ModuleList([
            MatTrans(
                node_dim=config.node_dim,
                hidden_dim=config.mat_hidden_dim,
                max_l=config.max_l,
                edge_dim=config.num_basis,
                actfn=config.activation,
            )
            for _ in range(config.num_mat_conv)
        ])
        self.output = MatriceOut(
            config.irreps_out,
            node_dim=config.node_dim,
            hidden_dim=config.mat_hidden_dim,
            block_dim=config.mat_block_dim,
            max_l=config.max_l,
            actfn=config.activation,
            symmetrize=config.symmetrize,
            pbc=self.pbc, 
        )
    
    def forward(self, data: Data):
        """
        Args:
            `data`: Input Data.
        Returns:
            `result`: Output.
        """
        at_no = data.at_no; pos=data.pos
        edge_index=data.edge_index; edge_index_full=data.fc_edge_index
        node_feat, rbfs, rshs, full_rbfs = self.embed(at_no, pos, edge_index, edge_index_full)
        node_0, node_sph_ten, edge_sph_ten = node_feat, None, None 
        for idx, matconv in enumerate(self.mat_conv):
            node_feat = matconv(node_feat, rbfs, rshs, edge_index)
            if idx >= self.begin_read_idx:
                node_sph_ten, edge_sph_ten = self.mat_trans[idx - self.begin_read_idx](node_feat, full_rbfs, edge_index_full, node_sph_ten, edge_sph_ten)
        if self.pbc and hasattr(data, "cell_edge_index"):
            edge_index_cell = data.cell_edge_index
        else:
            edge_index_cell = edge_index_full 
        result = self.output(node_sph_ten, edge_sph_ten, node_0, edge_index_full, edge_index_cell)
        
        return result


class XPaiNNOrb(nn.Module):
    def __init__(self, config: NetConfig):
        super().__init__()
        assert config.num_mat_conv <= config.action_blocks
        self.begin_read_idx = config.action_blocks - config.num_mat_conv
        # xpainn block
        self.config = config
        self.cutoff = config.cutoff
        self.pbc = config.pbc
        self.embed = XEmbedding(
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
        # XPaiNNOrb block
        self.edge_full_embed = FullEdgeKernel(
            rbf_kernel=config.pair_rbf_kernel,
            num_basis=config.pair_num_basis,
            cutoff=config.pair_cutoff,
            cutoff_fn=config.cutoff_fn,
        )
        self.mat_transform = nn.ModuleList([
            XMatTrans(
                irreps_node=config.edge_irreps,
                hidden_dim=config.mat_hidden_dim,
                max_l=config.max_l,
                edge_dim=config.pair_num_basis,
                actfn=config.activation,
                rbf_bias=True,
            )
            for _ in range(config.num_mat_conv)
        ])
        self.output = MatriceOut(
            irreps_out=config.irreps_out,
            node_dim=config.node_dim,
            hidden_dim=config.mat_hidden_dim,
            block_dim=config.mat_block_dim,
            max_l=config.max_l,
            actfn=config.activation, 
            symmetrize=config.symmetrize,
            pbc=self.pbc,
        )

    def forward(self, data:Data):
        """
        Args:
            `data`: Input Data.
        Returns:
            `result`: Output.
        """
        # raw data 
        at_no=data.at_no; pos=data.pos
        edge_index=data.edge_index; edge_index_full=data.fc_edge_index
        # embedding
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        x_scalar, rbf, fcut, rshs = self.embed(at_no, pos, edge_index, shifts)
        x_spherical = torch.zeros((x_scalar.shape[0], rshs.shape[1]), device=x_scalar.device)
        node_sph_ten, edge_sph_ten = None, None
        if hasattr(data, "fc_shifts"):
            full_shifts = data.fc_shifts
        else:
            full_shifts = torch.zeros((edge_index_full.shape[1], 3), device=pos.device) 
        full_rbfs = self.edge_full_embed(pos, edge_index_full, full_shifts)
        # message convolution & representation generation 
        for idx, (msg, upd) in enumerate(zip(self.message, self.update)):
            x_scalar, x_spherical = msg(x_scalar, x_spherical, rbf, fcut, rshs, edge_index)
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
            if idx >= self.begin_read_idx:
                node_sph_ten, edge_sph_ten = self.mat_transform[idx - self.begin_read_idx](x_spherical, full_rbfs, edge_index_full, node_sph_ten, edge_sph_ten)
        # output
        if self.pbc and hasattr(data, "cell_edge_index"):
            edge_index_cell = data.cell_edge_index
        else:
            edge_index_cell = edge_index_full 
        result = self.output(node_sph_ten, edge_sph_ten, x_scalar, edge_index_full, edge_index_cell)

        return result


class XPaiNN3Orb(nn.Module):
    """
    XPaiNN-Orb/X2 model 
    """
    def __init__(self, config: NetConfig):
        super().__init__()
        assert config.num_mat_conv <= config.action_blocks
        self.begin_read_idx = config.action_blocks - config.num_mat_conv
        self.pbc = config.pbc 
        # xpainn block
        self.config = config
        self.embed = XEmbedding(
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
        # XPaiNNOrb block
        self.edge_full_embed = BlockNorm2d(irreps_out=config.irreps_out)
        self.edge_full_rshs = FullEdgeSPH(irreps_node=config.edge_irreps)
        self.mat_transform = nn.ModuleList([
            ZMatTrans(
                irreps_node=config.edge_irreps,
                hidden_dim=config.mat_hidden_dim,
                max_l=config.max_l,
                edge_dim=config.pair_num_basis,
                actfn=config.activation,
                rbf_bias=False,
            )
            for _ in range(config.num_mat_conv)
        ])
        self.output = ZMatriceOut(
            irreps_out=config.irreps_out,
            node_dim=config.node_dim,
            edge_dim=config.pair_num_basis,
            hidden_dim=config.mat_hidden_dim,
            block_dim=config.mat_block_dim,
            max_l=config.max_l,
            actfn=config.activation,
            symmetrize=config.symmetrize, 
            pbc=self.pbc,
        )

    def forward(self, data:Data):
        """
        Args:
            `data`: Input Data.
        Returns:
            `result`: Output.
        """
        # raw data 
        at_no=data.at_no; pos=data.pos
        edge_index=data.edge_index; edge_index_full=data.fc_edge_index
        assert hasattr(data, "fc_edge_attr")
        # embedding
        if hasattr(data, "shifts"):
            shifts = data.shifts
        else:
            shifts = torch.zeros((edge_index.shape[1], 3), device=pos.device)
        x_scalar, rbfs, fcut, rshs = self.embed(at_no, pos, edge_index, shifts)
        x_spherical = torch.zeros((x_scalar.shape[0], rshs.shape[1]), device=x_scalar.device)
        node_sph_ten, edge_sph_ten = None, None 
        if hasattr(data, "fc_shifts"):
            full_shifts = data.fc_shifts
        else:
            full_shifts = torch.zeros((edge_index_full.shape[1], 3), device=pos.device) 
        full_edge_attr_ten = data.fc_edge_attr
        full_rbfs = self.edge_full_embed(full_edge_attr_ten)
        full_rshs = self.edge_full_rshs(pos, edge_index_full, full_shifts)
        # message convolution & representation generation 
        for idx, (msg, upd) in enumerate(zip(self.message, self.update)):
            x_scalar, x_spherical = msg(x_scalar, x_spherical, rbfs, fcut, rshs, edge_index)
            x_scalar, x_spherical = upd(x_scalar, x_spherical)
            if idx >= self.begin_read_idx:
                node_sph_ten, edge_sph_ten = self.mat_transform[idx - self.begin_read_idx](x_spherical, full_rbfs, full_rshs, edge_index_full, node_sph_ten, edge_sph_ten)
        # output
        if self.pbc and hasattr(data, "cell_edge_index"):
            edge_index_cell = data.cell_edge_index
        else:
            edge_index_cell = edge_index_full 
        result = self.output(node_sph_ten, edge_sph_ten, x_scalar, full_rbfs, edge_index_full, edge_index_cell)

        return result



def resolve_model(config: NetConfig) -> nn.Module:
    version = config.version.lower()
    if version in ["xpainn", "xpainn-pbc"]:
        return XPaiNN(config)
    elif version in ["test", "test-pbc"]:
        return XE3Net(config)
    elif version == "painn":
        return PaiNN(config)
    elif version == "xqhnet-mat":
        return XQHNet(config)
    elif version == "xpainn-mat":
        return XPaiNNOrb(config)
    elif version == "xpainn3-mat":
        return XPaiNN3Orb(config)
    else:
        raise NotImplementedError(f"Unsupported model {config.version}")
