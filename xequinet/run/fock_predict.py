from typing import Tuple 
import os 

import numpy as np
import torch 
from torch_cluster import radius_graph
from scipy.linalg import inv, sqrtm, eigh 

from xequinet.nn import resolve_model 
from xequinet.data import TextDataset 
from xequinet.utils import set_default_unit, unit_conversion, NetConfig
from xequinet.utils import BuildMatPerMole, TwoBodyBlockMask, Mat2GraphLabel

from pyscf import gto, scf, dft, lib

def cal_orbital_and_energies(ovlp: np.ndarray, fock: np.ndarray, ortho_transform:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    if not ortho_transform:
        orb_energies, c = eigh(fock, ovlp)
    else:
        orb_energies, c_prime = eigh(fock)
        ovlp_inv_sqrt = sqrtm(inv(ovlp))
        c = ovlp_inv_sqrt @ c_prime 
    idx = np.argmax(abs(c.real), axis=0)
    c[:, c[idx, np.arange(len(orb_energies))].real < 0] *= -1
    return orb_energies, c 


def save_chk(args):
    # set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.model_validate(ckpt["config"])
    # determine some metadata 
    if config.version == "xpainn3-mat":
        has_edge_attr = True 
        edge_attr_generator = Mat2GraphLabel(config.irreps_out, config.possible_elements, config.target_basisname, "pyscf")
    else:
        has_edge_attr = False
    ortho_transformed:bool = args.transform

    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)
    # set default dtype
    if config.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif config.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)

    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load input data
    dataset = TextDataset(args.input)

    matrix_builder = BuildMatPerMole(config.irreps_out, config.possible_elements, config.target_basisname, "pyscf")
    mask_generator = TwoBodyBlockMask(config.irreps_out, config.possible_elements, config.target_basisname)
    
    verbose = args.verbose 
    out_prefix = f"{args.output}_predict" if args.output is not None else f"{config.run_name}_predict"

    for imol, data in enumerate(dataset, start=1):
        at_no = data.at_no.cpu().numpy()
        coord = data.pos.cpu().numpy().astype(np.float64)
        charge = data.charge.to(torch.long).item()
        mol = gto.Mole()
        t = [(a, c) for a, c in zip(at_no, coord)]
        mol.build(
            atom=t,
            charge=charge,
            basis=config.target_basisname,
            unit=config.default_length_unit,
            verbose=verbose,
        )

        ovlp = mol.intor("int1e_ovlp")

        with torch.no_grad():
            data.batch = torch.zeros_like(data.at_no, dtype=torch.long)
            data.edge_index = radius_graph(data.pos, r=config.cutoff, max_num_neighbors=config.max_edges)
            if not config.full_edge_index:
                num_nodes = len(data.at_no)
                diagnol_mask = torch.eye(num_nodes, dtype=torch.bool)
                off_diagnol_mask = torch.logical_not(diagnol_mask)
                mat_edge_index = torch.nonzero(off_diagnol_mask).T.long()
                data.fc_edge_index = mat_edge_index 
            else:
                mat_edge_index = data.edge_index
                data.fc_edge_index = mat_edge_index 
            data.node_mask, data.edge_mask = mask_generator(data.at_no, mat_edge_index)
            if has_edge_attr:
                ovlp_ten = torch.from_numpy(ovlp.copy()).to(torch.float64)
                _, full_edge_attr = edge_attr_generator(data, ovlp_ten, data.at_no, mat_edge_index)
                data.fc_edge_attr = full_edge_attr
            data = data.to(device)
            node_padded, edge_padded = model(data)
            pred_fock = matrix_builder(node_padded, edge_padded, data.node_mask, data.edge_mask, data.at_no, data.fc_edge_index)
            pred_fock = pred_fock.cpu().numpy() * unit_conversion(config.default_property_unit, "Hartree")
        
        xc_method = args.xc 
        fake_method = dft.RKS(mol, xc=xc_method)
        if args.delta:
            dm0 = fake_method.get_init_guess(mol, key=args.method)
            init_fock = fake_method.get_fock(dm=dm0)
            if ortho_transformed:
                s_half_inv = sqrtm(inv(ovlp))
                init_fock = s_half_inv @ init_fock @ s_half_inv
            fock = init_fock + pred_fock 
        else:
            fock = pred_fock 
        orb_energies, orb_coeff = cal_orbital_and_energies(ovlp, fock, ortho_transformed)

        occ = mol.nelectron // 2
        nao = mol.nao  # total numbers of orbitals
        mo_occ = [2] * occ + [0] * (nao - occ) 
        mo_occ = np.array(mo_occ) 
               
        fake_method.mo_occ = mo_occ
        fake_method.mo_energy = orb_energies
        fake_method.mo_coeff = orb_coeff
        fake_method.verbose = verbose

        chk_file_name = f"{out_prefix}_{imol:04d}.chk"
        scf.chkfile.dump_scf(fake_method.mol, chk_file_name, 0.0, fake_method.mo_energy, fake_method.mo_coeff, fake_method.mo_occ)
        lib.chkfile.save(chk_file_name, "scf/fock", pred_fock)


import argparse 
def main():
    parser = argparse.ArgumentParser(description="XequiNet test script")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--xc", type=str, default="b3lyp",
    )
    parser.add_argument(
        "input",  type=str,
        help="Input xyz file or PySCF checkpoint file. (*.xyz or *.chk)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file name."
    )
    parser.add_argument(
        "--transform", action="store_true", default=False,
        help="Whether the Fock matrix is orthogonalized by S^-1/2. Default is False.",
    )
    parser.add_argument(
        "--delta", action="store_true", default=False,
        help="Whether to use delta model. Default is False.",
    )
    parser.add_argument(
        "--method", "-m", type=str, choices=["minao", "vsap"], default="minao",
        help="Method to generate initial guess for delta model. Default is 'minao'."
    )
    parser.add_argument(
        "--memory", "-M", type=int, default=4000,
        help="Memory size for SCF calculation in MB. Default is 4000.",
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
    )
    parser.add_argument(
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0,
        help="Verbose level for PySCF. Default is None.",
    )
    args = parser.parse_args()

    save_chk(args)


if __name__ == "__main__":
    main()

