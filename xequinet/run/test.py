import argparse
import math

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from xequinet.data import create_dataset
from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig,
    unit_conversion, set_default_unit, get_default_unit,
    gen_3Dinfo_str,
)
from xequinet.utils.qc import ELEMENTS_DICT


@torch.no_grad()
def test_scalar(model, test_loader, device, outfile, output_dim=1, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = torch.zeros(output_dim, device=device)
    num_mol = 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        if hasattr(data, "base_y"):
            pred += data.base_y
        real = data.y
        error = real - pred
        sum_loss += error.abs().sum(dim=0)
        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                wf.write(f"Real:")
                wf.write("".join([f"{r.item():15.9f} " for r in real[imol]]))
                wf.write(f"    Predict:")
                wf.write("".join([f"{p.item():15.9f}" for p in pred[imol]]))
                wf.write(f"    Error:")
                wf.write("".join([f"{l.item():15.9f}" for l in error[imol]]))
                wf.write(f"    ({p_unit})\n\n")
                wf.flush()
        num_mol += len(data.y)
    avg_loss = sum_loss / num_mol
    wf.write(f"Test MAE:")
    wf.write("".join([f"{l:15.9f}" for l in avg_loss]))
    wf.write(f"  {p_unit}\n")
    wf.close()


def test_grad(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_lossE, sum_lossF, num_mol, num_atom = 0.0, 0.0, 0, 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        data.pos.requires_grad = True
        predE, predF = model(data)
        with torch.no_grad():
            if hasattr(data, "base_y"):
                predE += data.base_y
            if hasattr(data, "base_force"):
                predF += data.base_force
            realE, realF = data.y, data.force
            errorE = realE - predE
            errorF = realF - predF
            sum_lossE += errorE.abs().sum()
            sum_lossF += errorF.abs().sum()
        if verbose >= 1:
            for imol in range(len(data.y)):
                idx = (data.batch == imol)
                at_no = data.at_no[idx]
                coord = data.pos[idx] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    info_3ds = [coord, predF[idx], realF[idx], errorF[idx]]
                    titles = [
                        "Coordinates (Angstrom)",
                        f"Predicted Forces ({p_unit}/{l_unit})",
                        f"Real Forces ({p_unit}/{l_unit})",
                        f"Error Forces ({p_unit}/{l_unit})"
                    ]
                    precisions = [6, 9, 9, 9]
                    wf.write(gen_3Dinfo_str(at_no, info_3ds, titles, precisions))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                wf.write(f"Energy | Real: {realE[imol].item():15.9f}    ")
                wf.write(f"Predict: {predE[imol].item():15.9f}    ")
                wf.write(f"Error: {errorE[imol].item():15.9f}    {p_unit}\n")
                wf.write(f"Force  | MAE : {errorF[idx].abs().mean():15.9f}   {p_unit}/{l_unit}\n\n")
                wf.flush()
        num_mol += data.y.numel()
        num_atom += data.at_no.numel()
    wf.write(f"Energy MAE : {sum_lossE / num_mol:15.9f}    {p_unit}\n")
    wf.write(f"Force  MAE : {sum_lossF / (3*num_atom):15.9f}    {p_unit}/{l_unit}\n")
    wf.close()


@torch.no_grad()
def test_vector(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = 0.0
    num_mol = 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        real = data.y
        error = real - pred
        sum_loss += error.abs().sum().item()
        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                values = [
                    f"X{vec[imol][0].item():12.6f}  Y{vec[imol][1].item():12.6f}  Z{vec[imol][2].item():12.6f}"
                    for vec in [real, pred, error]
                ]
                titles = [f"Real ({p_unit})", f"Predict ({p_unit})", f"Error ({p_unit})"]
                filled_t = [f"{t: <{len(v)}}" for t, v in zip(titles, values)]
                wf.write("    ".join(filled_t) + "\n")
                wf.write("    ".join(values) + "\n\n")
                wf.flush()
        num_mol += len(data.y)
    wf.write(f"Test MAE: {sum_loss / num_mol / 3 :12.6f} {p_unit}\n")
    wf.close()


@torch.no_grad()
def test_polar(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = 0.0
    num_mol = 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        real = data.y
        error = real - pred
        sum_loss += error.abs().sum().item()
        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                tri_values = []
                for i, D in enumerate(['X', 'Y', 'Z']):
                    tri_values.append([
                        f"{D}X{pol[imol][i,0].item():12.6f}  {D}Y{pol[imol][i,1].item():12.6f}  {D}Z{pol[imol][i,2].item():12.6f}"
                        for pol in [real, pred, error]
                    ])
                titles = [f"Real ({p_unit})", f"Predict ({p_unit})", f"Error ({p_unit})"]
                filled_t = [f"{t: <{len(v)}}" for t, v in zip(titles, tri_values[0])]
                wf.write("    ".join(filled_t) + "\n")
                for values in tri_values:
                    wf.write("    ".join(values) + "\n")
                wf.write("\n")
                wf.flush()
        num_mol += len(data.y)
    wf.write(f"Test MAE: {(sum_loss / num_mol / 9) :12.6f} {p_unit}\n")
    wf.close()


@torch.no_grad()
def test_cart_tensor(model, test_loader, device, outfile):
    p_unit, _ = get_default_unit()
    sum_mae, sum_mse = 0.0, 0.0
    sum_l2_norm_dist = 0.0
    mean, m2 = 0.0, 0.0
    count = 0
    num_samples = 0 
    wf = open(outfile, 'a')
    for data in test_loader:  
        data = data.to(device)
        pred = model(data)
        real = data.y
        error = real - pred
        sum_mae += error.abs().sum().item()
        sum_mse += error.pow(2).sum().item()
        # l2 metric 
        cur_num_sample = real.shape[0]
        batch_l2_norm_dist = torch.sqrt(error.pow(2).view(cur_num_sample, -1).sum(-1))
        sum_l2_norm_dist += batch_l2_norm_dist.sum().item()
        # 
        batch_size = real.numel() 
        new_count = count + batch_size 
        batch_mean = torch.mean(real.view(-1)) 
        batch_m2 = torch.sum((real.view(-1) - batch_mean) ** 2)
        delta = batch_mean - mean 
        mean += delta * batch_size / new_count 
        corr = batch_size * count / new_count 
        m2 += batch_m2 + delta ** 2 * corr 
        count = new_count  
        num_samples += cur_num_sample
    var = m2 / count
    mae = sum_mae / count 
    mse = sum_mse / count
    # calculate Frobenius norm
    l2_norm_dist = sum_l2_norm_dist / num_samples
    rmse = math.sqrt(mse)
    r2 = 1 - mse / var
    wf.write(f"Test MAE: {mae:12.7f} {p_unit}, RMSE {rmse:12.7f} {p_unit}, L2 distance {l2_norm_dist:12.7f} {p_unit}, R2  {r2:12.7f} \n")
    wf.close()


@torch.no_grad()
def test_cart_tensor_fnorm(model, order, test_loader, device, outfile):
    p_unit, _ = get_default_unit()
    sum_fnorm = 0.0
    wt_count = [0, 0, 0]
    num_samples = 0 
    symmetry_index = [0, 1, 2, 4, 5, 8]
    wf = open(outfile, 'a')
    for data in test_loader:  
        data = data.to(device)
        pred = model(data)
        real = data.y
        cur_num_sample = real.shape[0]
        if order == 3:
            pred = pred.view(cur_num_sample, 3, 9)[:, :, symmetry_index] 
            real = real.view(cur_num_sample, 3, 9)[:, :, symmetry_index] 
        elif order == 4:
            pred = pred.view(cur_num_sample, 9, 9)[:, symmetry_index, symmetry_index]
            real = real.view(cur_num_sample, 9, 9)[:, symmetry_index, symmetry_index] 
        error = real - pred
        # fnorm metric 
        batch_fnorm_err = torch.sqrt(error.pow(2).view(cur_num_sample, -1).sum(-1))
        batch_fnorm_real = torch.sqrt(real.pow(2).view(cur_num_sample, -1).sum(-1))
        # fetch zero 
        batch_eq_index = torch.where(batch_fnorm_real.abs() <= 1e-4)
        batch_err_index = torch.where(batch_fnorm_real.abs() > 1e-4)
        cur_eq = torch.count_nonzero(torch.where((batch_fnorm_err[batch_eq_index]).abs() <= 1e-4, 1, 0)).item()      
        cur_wt_count_25 = torch.count_nonzero(torch.where(batch_fnorm_err[batch_err_index] / batch_fnorm_real[batch_err_index] < 0.25, 1, 0)).item()
        cur_wt_count_10 = torch.count_nonzero(torch.where(batch_fnorm_err[batch_err_index] / batch_fnorm_real[batch_err_index] < 0.10, 1, 0)).item() 
        cur_wt_count_5 = torch.count_nonzero(torch.where(batch_fnorm_err[batch_err_index] / batch_fnorm_real[batch_err_index] < 0.05, 1, 0)).item()
        wt_count[0] += cur_wt_count_25
        wt_count[1] += cur_wt_count_10
        wt_count[2] += cur_wt_count_5
        wt_count[0] += cur_eq 
        wt_count[1] += cur_eq 
        wt_count[2] += cur_eq 
        sum_fnorm += torch.sqrt(error.pow(2).view(cur_num_sample, -1).sum(-1)).sum().item()
        num_samples += cur_num_sample
    # calculate Frobenius norm
    fnorm = sum_fnorm / num_samples
    EwT25, EwT10, EwT5 = 100 * wt_count[0] / num_samples, 100 * wt_count[1] / num_samples, 100 * wt_count[2] / num_samples
    wf.write(f"Fnorm {fnorm:12.6f} {p_unit} \n")
    wf.write(f"EwT 25% {EwT25:6.2f},\t 10% {EwT10:6.2f},\t 5% {EwT5:6.2f} \n")
    wf.close()


@torch.no_grad()
def test_csc(model, test_loader, device, outfile, required_elements=None):
    p_unit, _ = get_default_unit() 
    error_dict = {"total":{"sum_mae": 0.0, "sum_mse": 0.0, "count": 0, "mean": 0.0, "m2": 0.0}} 
    for ele in required_elements:
        error_dict[ele] = {"sum_mae": 0.0, "sum_mse": 0.0, "count": 0, "mean": 0.0, "m2": 0.0} 
    wf = open(outfile, 'a')
    for data in test_loader:  
        data = data.to(device)
        res = model(data) 
        label = data.y 
        # pred = model(data)
        # real = data.y
        if hasattr(data, "label_mask"):
            batch_label_mask = data.label_mask 
        else:
            batch_label_mask = torch.ones_like(data.at_no).bool().to(device)
        for key_name in error_dict.keys():
            if key_name == "total":
                node_mask = batch_label_mask 
            else:
                cur_at_no = ELEMENTS_DICT[key_name]
                node_mask = torch.logical_and(batch_label_mask, data.at_no == cur_at_no)
            real, pred = label[node_mask], res[node_mask]
            error = real - pred
            error_dict[key_name]["sum_mae"] += error.abs().sum().item()
            error_dict[key_name]["sum_mse"] += error.pow(2).sum().item()
            batch_size = real.numel()
            if batch_size == 0:
                continue
            new_count = error_dict[key_name]["count"] + batch_size 
            batch_mean = torch.mean(real.view(-1)) 
            batch_m2 = torch.sum((real.view(-1) - batch_mean) ** 2)
            delta = batch_mean - error_dict[key_name]["mean"] 
            error_dict[key_name]["mean"] += delta * batch_size / new_count 
            corr = batch_size * error_dict[key_name]["count"] / new_count 
            error_dict[key_name]["m2"] += batch_m2 + delta ** 2 * corr 
            error_dict[key_name]["count"] = new_count  
    for key_name in error_dict.keys():
        m2 = error_dict[key_name]["m2"] 
        count = error_dict[key_name]["count"] 
        sum_mae = error_dict[key_name]["sum_mae"] 
        sum_mse = error_dict[key_name]["sum_mse"] 
        if count == 0:
            wf.write(f"Current Nuclei {key_name} is not contained in the test set. \n")
            continue
        var = m2 / count
        mae = sum_mae / count 
        mse = sum_mse / count
        rmse = math.sqrt(mse)
        r2 = 1 - mse / var
        wf.write(f"{key_name:<{8}} Test MAE: {mae :12.7f} {p_unit}, RMSE {rmse:12.7f} {p_unit}, R2  {r2:12.7f} \n")
    wf.close()


@torch.no_grad()
def test_matrix(model, test_loader, device, output_file):
    model.eval()
    sum_loss_node, sum_loss_edge, sum_loss_total = 0.0, 0.0, 0.0
    num_node, num_edge, num_total = 0, 0, 0
    for data in test_loader:
        data = data.to(device)
        res_pad_node, res_pad_edge = model(data)
        pred_pad_node = res_pad_node + data.node_base if hasattr(data, 'node_base') else res_pad_node
        pred_pad_edge = res_pad_edge + data.edge_base if hasattr(data, 'edge_base') else res_pad_edge
        batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
        pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
        real_node, real_edge = data.node_label[batch_mask_node], data.edge_label[batch_mask_edge] 
        batch_pred = torch.cat([pred_node, pred_edge], dim=0)
        batch_real = torch.cat([real_node, real_edge], dim=0)
        node_l1loss = F.l1_loss(pred_node, real_node, reduce=False)
        edge_l1loss = F.l1_loss(pred_edge, real_edge, reduce=False)
        total_l1loss = F.l1_loss(batch_pred, batch_real, reduce=False)
        # loss accumulation 
        sum_loss_node += node_l1loss.sum().item()
        sum_loss_edge += edge_l1loss.sum().item()
        sum_loss_total += total_l1loss.sum().item()
        # count numel 
        num_node += real_node.numel()
        num_edge += real_edge.numel()
        num_total += batch_real.numel()
    
    with open(output_file, 'a') as wf:
        wf.write(f"Test MAE: node {sum_loss_node / num_node:10.8f}, edge {sum_loss_edge / num_edge:10.8f}, total {sum_loss_total / num_total:10.8f}.\n")


@torch.no_grad()
def test_matrix_wavefunc(model, test_loader, device, build_matrix, basis, output_file, transform_type):
    from xequinet.utils import cal_orbital_and_energies
    from pyscf import gto
    model.eval()
    p_unit, l_unit = get_default_unit()
    sum_mae_e, sum_mae_e_occ, sum_cosine_similarity = 0.0, 0.0, 0.0
    sum_mae_homo, sum_mae_lumo, sum_mae_gap, sum_mse_gap = 0.0, 0.0, 0.0, 0.0
    sum_fnorm_density, sum_fnorm_orbgrad, sum_wa_loss = 0.0, 0.0, 0.0
    num_moles = 0 
    for test_batch_idx, data in enumerate(test_loader, start=1):
        data = data.to(device) 
        res_node, res_edge = model(data) 
        pred_node = res_node + data.node_base if hasattr(data, 'node_base') else res_node
        pred_edge = res_edge + data.edge_base if hasattr(data, 'edge_base') else res_edge
        node_mask, edge_mask = data.onsite_mask, data.offsite_mask
        mol = gto.Mole()
        t = [
            [data.at_no[atom_idx].cpu().item(), data.pos[atom_idx].cpu().numpy()]
            for atom_idx in range(data.num_nodes)
        ]
        mol.build(verbose=0, atom=t, basis=basis, unit=l_unit)
        overlap = torch.from_numpy(mol.intor("int1e_ovlp"))
        overlap = overlap.to(torch.get_default_dtype()).to(device)
        pred_fock = build_matrix(pred_node, pred_edge, node_mask, edge_mask, data.at_no, data.fc_edge_index)
        real_fock = build_matrix(data.node_label, data.edge_label, node_mask, edge_mask, data.at_no, data.fc_edge_index)
        pred_e, pred_coeffs, _ = cal_orbital_and_energies(overlap, pred_fock, transform_type)
        real_e, real_coeffs, s_half = cal_orbital_and_energies(overlap, real_fock, transform_type) 
        # orbital_coeffs are of shape (nbasis, norb)
        nelectron = torch.sum(data.at_no).item()
        norb_occ = nelectron // 2 
        # orbital energy related errors
        mae_e = F.l1_loss(pred_e, real_e)
        mae_e_occ = F.l1_loss(pred_e[:norb_occ], real_e[:norb_occ])
        pred_homo, pred_lumo = pred_e[norb_occ-1], pred_e[norb_occ]
        real_homo, real_lumo = real_e[norb_occ-1], real_e[norb_occ] 
        mae_homo = F.l1_loss(pred_homo, real_homo) 
        mae_lumo = F.l1_loss(pred_lumo, real_lumo) 
        pred_gap = pred_lumo - pred_homo 
        real_gap = real_lumo - real_homo 
        mae_gap = F.l1_loss(pred_gap, real_gap)
        # wavefunction related errors 
        ## occupied orbitals 
        pred_occ_orbs = pred_coeffs[:, :norb_occ]
        real_occ_orbs = real_coeffs[:, :norb_occ]
        ## density 
        pred_density = pred_occ_orbs @ pred_occ_orbs.T
        real_density = real_occ_orbs @ real_occ_orbs.T
        if transform_type == 0:
            real_coeffs_prime = s_half @ real_coeffs
            cur_wa_loss = real_coeffs_prime.T @ (pred_fock - real_fock) @ real_coeffs_prime
            cur_orb_grad = real_coeffs_prime[:, norb_occ:].T @ pred_fock @ real_coeffs_prime[:, :norb_occ]
        elif transform_type == 1:
            cur_wa_loss = real_coeffs.T @ (pred_fock - real_fock) @ real_coeffs
            cur_orb_grad = real_coeffs[:, norb_occ:].T @ pred_fock @ real_coeffs[:, :norb_occ]
        fnorm_density = torch.norm(pred_density - real_density, p="fro") 
        fnorm_orb_grad = torch.norm(cur_orb_grad, p="fro")
        fnorm_wa_loss = torch.norm(cur_wa_loss, p="fro")
        cosine_similarity = torch.cosine_similarity(pred_occ_orbs, real_occ_orbs, dim=0).abs().mean()
        # accumulation
        sum_mae_e += mae_e.item()
        sum_mae_e_occ += mae_e_occ.item()
        sum_mae_homo += mae_homo.item() 
        sum_mae_lumo += mae_lumo.item()
        sum_mae_gap += mae_gap.item()
        sum_cosine_similarity += cosine_similarity.item()
        sum_fnorm_density += fnorm_density.item()
        sum_fnorm_orbgrad += fnorm_orb_grad.item()
        sum_wa_loss += fnorm_wa_loss.item()
        num_moles += 1 
        # write to log file 
        with open(output_file, 'a') as wf:
            wf.write(f"Test error for mole {test_batch_idx:06d}: \n")
            wf.write(f"mae_e: {mae_e.item():6.4f} {p_unit}, cosine similarity: {cosine_similarity.item():6.4f}\n")
            wf.write(f"mae_homo: {mae_homo.item():6.4f}, mae_lumo: {mae_lumo.item():6.4f}\n")
            wf.write(f"mae_gap: {mae_gap.item():6.4f}\n")
            wf.write(f"Density error:  {fnorm_density.item():10.4f} e \n")
            wf.write(f"Wavefunction alignment: {fnorm_wa_loss.item():10.4f} {p_unit}\n")
            wf.write(f"Orbital Gradient:       {fnorm_orb_grad.item():10.4f} {p_unit}\n")
        
    with open(output_file, 'a') as wf:
        wf.write(f"\nAverage Prediction Error: \n")
        wf.write(f"Orbital Energy MAE: {sum_mae_e/num_moles:12.8f} {p_unit}.\n")
        wf.write(f"Occupied Orbital Energy MAE: {sum_mae_e_occ/num_moles:12.8f} {p_unit}.\n")
        wf.write(f"Wavefunction Cosine Similarity: {100 * sum_cosine_similarity/num_moles:6.2f} %.\n")
        wf.write(f"HOMO MAE: {sum_mae_homo / num_moles:10.6f} {p_unit}.\n")
        wf.write(f"LUMO MAE: {sum_mae_lumo / num_moles:10.6f} {p_unit}.\n")
        wf.write(f"Gap  MAE: {sum_mae_gap  / num_moles:10.6f} {p_unit}.\n")
        wf.write(f"\nAverage Frobenius Norm Distance: \n")
        wf.write(f"Density Norm Distance:   {sum_fnorm_density / num_moles:12.8f} e.\n")
        wf.write(f"Wavefunction Alignment:  {sum_wa_loss / num_moles:12.8f} {p_unit}.\n")
        wf.write(f"Orbital Gradient:        {sum_fnorm_orbgrad / num_moles:12.8f} {p_unit}.\n")


def main():
    # parse config
    parser = argparse.ArgumentParser(description="XequiNet test script")
    parser.add_argument(
        "--config", "-C", type=str, default="config.json",
        help="Configuration file (default: config.json).",
    )
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Whether testing force additionally when the output mode is 'scalar'",
    )
    parser.add_argument(
        "--no-force", "-nf", action="store_true",
        help="Whether not testing force when the output mode is 'grad'",
    )
    parser.add_argument(
        "--atomic", "-A", default=False, action="store_true",
        help="Whether to save atomic info.",
    )
    parser.add_argument(
        "--diag", default=False, action="store_true",
        help="Whether to diagonalize the Fock matrix for orbital energy and wavefunction.",
    )
    parser.add_argument(
        "--transform", type=int, default=1, choices=[0, 1],
        help="Transform type to apply in wavefunction diagonalization. (default: 1)",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, choices=[0, 1, 2],
        help="Verbose level. (default: 0)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size. (default: 32)",
    )
    parser.add_argument(
        "--mode", "-m" , type=str, default="test",
        help="Mode. (default: test)",
    )
    parser.add_argument(
        "--info", type=str, default="info",
        help="Info. (default: info)",
    )
    parser.add_argument(
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    args = parser.parse_args()
    
    # open warning or not
    if not args.warning:
        import warnings
        warnings.filterwarnings("ignore")
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    with open(args.config, 'r') as json_file:
        config = NetConfig.model_validate_json(json_file.read())
    ckpt = torch.load(args.ckpt, map_location=device)
    config.model_validate(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    test_dataset = create_dataset(config, args.mode)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    
    # adjust some configurations
    if args.force == True and config.output_mode == "scalar":
        config.output_mode = "grad"
    if args.no_force == True and config.output_mode == "grad":
        config.output_mode = "scalar"
    
    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # test
    subset = f"{args.mode}"
    if args.info != "info":
        comments = args.info
        output_file = f"{config.run_name}_{comments}_{subset}.log"
    else: 
        output_file = f"{config.run_name}_{subset}.log"
        
    with open(output_file, 'w') as wf:
        wf.write("XequiNet Testing\n")
        wf.write(f"Unit: {config.default_property_unit} {config.default_length_unit}\n")

    if config.output_mode == "grad":
        test_grad(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode == "vector" and config.output_dim == 3:
        test_vector(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode == "polar" and config.output_dim == 9:
        test_polar(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode in ["chemical_shielding", "chemical_shifts", "atomic_shielding"]:
        if config.output_dim == 1:
            required_elements = config.target_elem if config.target_elem is not None else ["H", "C"]
            test_csc(model, test_loader, device, output_file, required_elements)
        else:
            test_cart_tensor(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode in ["cart_tensor", "cart_tensor_tp", "cart_tensor_gate", "cart_tensor_lin", "cart_tensor_mix"]:
        if config.output_dim == 1:
            if config.target_elem is not None:
                required_elements = config.target_elem
                test_csc(model, test_loader, device, output_file, required_elements)
            else:
                test_scalar(model, test_loader, device, output_file, config.output_dim, args.verbose)
        elif args.verbose == 1:
            test_cart_tensor_fnorm(model, config.order, test_loader, device, output_file)
        elif args.atomic:
            required_elements = config.target_elem if config.target_elem is not None else ["O"]
            test_csc(model, test_loader, device, output_file, required_elements)
        else:
            test_cart_tensor(model, test_loader, device, output_file)
    elif "mat" in config.version:
        if args.diag:
            from xequinet.utils import BuildMatPerMole
            matrix_builder = BuildMatPerMole(config.irreps_out, config.possible_elements, config.target_basisname, "pyscf")
            test_matrix_wavefunc(model, test_loader, device, matrix_builder, config.target_basisname, output_file, args.transform)
        else:
            test_matrix(model, test_loader, device, output_file)
    else:
        test_scalar(model, test_loader, device, output_file, config.output_dim, args.verbose)


if __name__ == "__main__":
    main()