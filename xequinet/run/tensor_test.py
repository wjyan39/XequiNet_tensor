import argparse
import math  

import torch 
import numpy as np 
import torch.nn.functional as F 
from torch_geometric.loader import DataLoader 

from xequinet.data import create_dataset 
from xequinet.nn import resolve_model 
from xequinet.utils import (
    NetConfig, 
    unit_conversion,
    set_default_unit,
    get_default_unit,
    gen_3Dinfo_str,
)

# test for n-order tensor output 
@torch.no_grad() 
def test_1_tensor(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit() 
    sum_mae, sum_rmse = 0.0, 0.0
    num_mole = 0

    wf = open(outfile, "a") 
    for data in test_loader:
        data = data.to(device) 
        pred = model(data)
        real = data.y 
        deviation = pred - real
        
        pred_norm = torch.linalg.vector_norm(pred, dim=-1, keepdim=True)
        real_norm = torch.linalg.vector_norm(real, dim=-1, keepdim=True)
        norm_error = pred_norm - real_norm
        sum_mae += norm_error.abs().sum().item() 
        sum_rmse += norm_error.pow(2).sum().item()

        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mole + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")

                titles = [f"Real    ({p_unit}):", f"Predict ({p_unit}):", f"Error   ({p_unit}):"]
                values = [
                    f"X{vec[imol][0].item():12.6f}  Y{vec[imol][1].item():12.6f}  Z{vec[imol][2].item():12.6f}  Norm{norm[imol][1].item():12.6f}"
                    for (vec, norm) in [(real, real_norm), (pred, pred_norm), (deviation, norm_error)]
                ]
                for t, v in zip(titles, values):
                    wf.write("  ".join(t, v) + "\n") 
                wf.write("\n")
                wf.flush() 
        num_mole += len(data.y) 
    mae = sum_mae / num_mole
    rmse = math.sqrt(sum_rmse / num_mole)
    wf.write(f"Test Norm Error: MAE {mae:12.7f} {p_unit}, RMSE {rmse:12.7f} {p_unit} \n") 
    wf.close() 


@torch.no_grad()
def test_2_tensor(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit() 
    # sum_mae, sum_rmse = 0.0, 0.0
    sum_iso_abs_e, sum_ani_abs_e = 0.0, 0.0
    sum_iso_sqr_e, sum_ani_sqr_e = 0.0, 0.0
    num_mole = 0
    wf = open(outfile, "w") 
    for data in test_loader:
        data = data.to(device)
        pred = model(data) 
        real = data.y
        deviation = pred - real
        # sum_mae += deviation.abs().sum().item()
        # sum_rmse += deviation.pow(2).sum().item() 
        # diagnol term 
        pred_diag = torch.diagonal(pred, dim1=1, dim2=2)
        real_diag = torch.diagonal(real, dim1=1, dim2=2)
        ## trace 
        pred_iso = pred_diag.sum(-1, keepdim=True) / 3 
        real_iso = real_diag.sum(-1, keepdim=True) / 3
        error_iso = pred_iso - real_iso
        sum_iso_abs_e += error_iso.abs().sum().item()
        sum_iso_sqr_e += error_iso.pow(2).sum().item()
        ## anistropy 
        pred_ani = torch.pow(pred_diag - pred_diag[:, [1, 2, 0]], 2).sum(-1, keepdim=True) / 2
        pred_ani = torch.sqrt(pred_ani) 
        real_ani = torch.pow(real_diag - real_diag[:, [1, 2, 0]], 2).sum(-1, keepdim=True) / 2
        real_ani = torch.sqrt(real_ani) 
        error_ani = pred_ani - real_ani
        sum_ani_abs_e += error_ani.abs().sum().item()
        sum_ani_sqr_e += error_ani.pow(2).sum().item()

        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mole + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")

                titles = [f"Real    ({p_unit}):\n", f"Predict ({p_unit}):\n", f"Error   ({p_unit}):\n"]
                values = [
                    f"  XX{vec[imol][0,0].item():12.6f}  YY{vec[imol][1,1].item():12.6f}  ZZ{vec[imol][2,2].item():12.6f} \n\
                        XY{vec[imol][0,1].item():12.6f}  XZ{vec[imol][0,2].item():12.6f}  YZ{vec[imol][1,2].item():12.6f} \n\
                        Iso{iso[imol][1].item():12.6f}     Aniso{ani[imol][1].item():12.6f}" \
                    for (vec, iso, ani) in [(real, real_iso, real_ani), (pred, pred_iso, pred_ani), (deviation, error_iso, error_ani)]
                ]
                for t, v in zip(titles, values):
                    wf.write("  ".join(t, v) + "\n") 
                wf.write("\n")
                wf.flush() 
        num_mole += len(data.y) 
    iso_mae = sum_iso_abs_e / num_mole 
    iso_rmse = math.sqrt(sum_iso_sqr_e / num_mole)
    ani_mae = sum_ani_abs_e / num_mole
    ani_rmse = math.sqrt(sum_ani_sqr_e / num_mole)
    wf.write(f"Test Error  Unit:{p_unit} \n") 
    wf.write(f"Isotropic  MAE {iso_mae:12.7f}, RMSE {iso_rmse:12.7f} {p_unit}. \n")
    wf.write(f"Anisotropy MAE {ani_mae:12.7f}, RMSE {ani_rmse:12.7f} {p_unit}. \n")
    wf.close() 


@torch.no_grad()
def test_3_tensor(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit() 
    # sum_mae, sum_rmse = 0.0, 0.0
    sum_beta_error, sum_beta_pz_error, sum_beta_vz_error = 0.0, 0.0, 0.0
    sum_beta_var, sum_beta_pz_var, sum_beta_vz_var = 0.0, 0.0, 0.0
    num_mole = 0
    wf = open(outfile, "w") 
    for data in test_loader:
        data = data.to(device)
        pred = model(data) 
        real = data.y
        deviation = pred - real
        # sum_mae += deviation.abs().sum().item()
        # sum_rmse += deviation.pow(2).sum().item() 
        # beta scalars 
        pred_beta = torch.zeros(*real.shape[:2], device=device, dtype=real.dtype) 
        real_beta = torch.zeros(*real.shape[:2], device=device, dtype=real.dtype)
        for i in range(3):
            for j in range(3):
                pred_beta[:, i] += pred[:, i, j, j] + pred[:, j, i, j] + pred[:, j, j, i] 
                real_beta[:, i] += real[:, i, j, j] + real[:, j, i, j] + real[:, j, j, i]
        pred_beta = pred_beta / 3.0 
        real_beta = real_beta / 3.0
        pred_beta_norm = torch.norm(pred_beta, dim=1)
        real_beta_norm = torch.norm(real_beta, dim=1) 
        error_beta_norm = pred_beta_norm - real_beta_norm 
        sum_beta_error += error_beta_norm.abs().sum().item()
        sum_beta_var += error_beta_norm.pow(2).sum().item()
        pred_beta_pz = 0.6 * pred_beta[:, 2]
        real_beta_pz = 0.6 * real_beta[:, 2] 
        error_beta_pz = pred_beta_pz - real_beta_pz
        sum_beta_pz_error += error_beta_pz.abs().sum().item() 
        sum_beta_pz_var += error_beta_pz.pow(2).sum().item()
        pred_beta_vz = torch.zeros_like(pred_beta_pz).to(device) 
        real_beta_vz = torch.zeros_like(real_beta_pz).to(device) 
        for j in range(3):
            pred_beta_vz += 0.4*pred[:, 2,j,j] - 0.6*pred[:, j,2,j] + 0.4*pred[:, j,j,2] 
            real_beta_vz += 0.4*real[:, 2,j,j] - 0.6*real[:, j,2,j] + 0.4*real[:, j,j,2] 
        error_beta_vz = pred_beta_vz - real_beta_vz 
        sum_beta_vz_error += error_beta_vz.abs().sum().item() 
        sum_beta_vz_var += error_beta_vz.pow(2).sum().item()
        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mole + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")

                titles = [f"Real    ({p_unit}):\n", f"Predict ({p_unit}):\n", f"Error   ({p_unit}):\n"]

                values = [
                    f"  XXX{vec[imol][0,0,0].item():12.6f}  XXY{vec[imol][0,0,1].item():12.6f}  XXZ{vec[imol][0,0,2].item():12.6f} \n\
                        XYY{vec[imol][0,1,1].item():12.6f}  YYY{vec[imol][1,1,1].item():12.6f}  YYZ{vec[imol][1,1,2].item():12.6f} \n\
                        XZZ{vec[imol][0,1,1].item():12.6f}  YZZ{vec[imol][1,2,2].item():12.6f}  ZZZ{vec[imol][2,2,2].item():12.6f} \n\
                        XYZ{vec[imol][0,1,2].item():12.6f}  \n \
                        Beta Magnitude{beta[imol][1].item():12.6f}   Beta ||(z){pz[imol][1].item():12.6f}   Beta _|_(z){vz[imol][1].item():12.6f}" \
                    for (vec, beta, pz, vz) in \
                        [(real, real_beta, real_beta_pz, real_beta_vz), (pred, pred_beta, pred_beta_pz, pred_beta_vz), (deviation, error_beta_norm, error_beta_pz, error_beta_vz)]
                ]
                for t, v in zip(titles, values):
                    wf.write("  ".join(t, v) + "\n") 
                    wf.write("\n")
                    wf.flush() 
        num_mole += len(data.y) 
    mae_beta = sum_beta_error / num_mole
    rmse_beta = math.sqrt(sum_beta_var / num_mole)
    mae_pz = sum_beta_pz_error / num_mole
    rmse_pz = math.sqrt(sum_beta_pz_var / num_mole)
    mae_vz = sum_beta_vz_error / num_mole
    rmse_vz = math.sqrt(sum_beta_vz_var / num_mole)
    wf.write(f"Test Error  Unit:{p_unit} \n") 
    wf.write(f"Beta magnitude: MAE {mae_beta:12.7f}, RMSE {rmse_beta:12.7f}. \n") 
    wf.write(f"Beta ||(z):     MAE {mae_pz:12.7f}, RMSE {rmse_pz:12.7f}. \n")
    wf.write(f"Beta _|_(z):    MAE {mae_vz:12.7f}, RMSE {rmse_vz:12.7f}. \n")
    wf.close() 


@torch.no_grad()
def test_4_tensor(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit() 
    sum_v_error, sum_K_error, sum_G_error, sum_E_error = 0.0, 0.0, 0.0, 0.0
    sum_v_var, sum_K_var, sum_G_var, sum_E_var = 0.0, 0.0, 0.0, 0.0 
    num_mole = 0
    wf = open(outfile, "w") 
    symm_index = torch.LongTensor([0, 4, 8, 1, 5, 2]).to(device)
    zero_padding = np.zeros((6, 6), dtype=np.float32)
    thresh = 1e-4

    for data in test_loader:
        data = data.to(device)
        pred = model(data) 
        pred = pred.view(-1, 9, 9)
        pred_tmp = torch.index_select(pred, dim=2, index=symm_index)
        # 6x6 tensor 
        pred_C = torch.index_select(pred_tmp, dim=1, index=symm_index).cpu().numpy()
        if verbose == 1:
            pred_C = np.where(np.abs(pred_C) < thresh, zero_padding, pred_C)
        pred_s = np.linalg.inv(pred_C)
        # metadata 
        pred_C_0 = pred_C[:, 0, 0] + pred_C[:, 1, 1] + pred_C[:, 2, 2]
        pred_C_1 = pred_C[:, 0, 1] + pred_C[:, 1, 2] + pred_C[:, 2, 0] 
        pred_C_2 = pred_C[:, 3, 3] + pred_C[:, 4, 4] + pred_C[:, 5, 5]
        pred_s_0 = pred_s[:, 0, 0] + pred_s[:, 1, 1] + pred_s[:, 2, 2]
        pred_s_1 = pred_s[:, 0, 1] + pred_s[:, 1, 2] + pred_s[:, 2, 0]
        pred_s_2 = pred_s[:, 3, 3] + pred_s[:, 4, 4] + pred_s[:, 5, 5]
        # pred 
        pred_K_V = (pred_C_0 + 2 * pred_C_1) / 9.0
        pred_G_V = (pred_C_0 -  pred_C_1 + 3 * pred_C_2) / 15.0
        pred_K_R = 1.0 /  (pred_s_0 + 2 * pred_s_1)
        pred_G_R = 15.0 / (4 * pred_s_0 - 4 * pred_s_1 + 3 * pred_s_2)
        pred_K_VRH = (pred_K_V + pred_K_R) / 2.0 
        pred_G_VRH = (pred_G_V + pred_G_R) / 2.0
        pred_v = (3*pred_K_VRH - 2*pred_G_VRH) / (6*pred_K_VRH + 2*pred_G_VRH)
        pred_E = 9*pred_K_VRH*pred_G_VRH / (3*pred_K_VRH + pred_G_VRH)
        # real 
        real = data.y.view(-1, 9, 9)
        real_tmp = torch.index_select(real, dim=2, index=symm_index)
        real_C = torch.index_select(real_tmp, dim=1, index=symm_index).cpu().numpy() 
        real_s = np.linalg.inv(real_C)
        # metadata
        real_C_0 = real_C[:, 0, 0] + real_C[:, 1, 1] + real_C[:, 2, 2]
        real_C_1 = real_C[:, 0, 1] + real_C[:, 1, 2] + real_C[:, 2, 0]
        real_C_2 = real_C[:, 3, 3] + real_C[:, 4, 4] + real_C[:, 5, 5]
        real_s_0 = real_s[:, 0, 0] + real_s[:, 1, 1] + real_s[:, 2, 2]
        real_s_1 = real_s[:, 0, 1] + real_s[:, 1, 2] + real_s[:, 2, 0]
        real_s_2 = real_s[:, 3, 3] + real_s[:, 4, 4] + real_s[:, 5, 5]
        # real 
        real_K_V = (real_C_0 + 2 * real_C_1) / 9.0
        real_G_V = (real_C_0 -  real_C_1 + 3 * real_C_2) / 15.0
        real_K_R = 1.0 /  (real_s_0 + 2 * real_s_1)
        real_G_R = 15.0 / (4 * real_s_0 - 4 * real_s_1 + 3 * real_s_2)
        real_K_VRH = (real_K_V + real_K_R) / 2.0
        real_G_VRH = (real_G_V + real_G_R) / 2.0
        real_v = (3*real_K_VRH - 2*real_G_VRH) / (6*real_K_VRH + 2*real_G_VRH)
        real_E = 9*real_K_VRH*real_G_VRH / (3*real_K_VRH + real_G_VRH)

        if verbose >= 2:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mole + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")

                titles = [f"Real Elastic Tensor ({p_unit}):\n", f"Predict Elastic Tensor ({p_unit}):\n"]

                values = [
                    f"  {vec[imol][0, 0]:12.4f} {vec[imol][0, 1]:12.4f} {vec[imol][0, 2]:12.4f} {vec[imol][0, 3]:12.4f}, {vec[imol][0, 4]:12.4f} {vec[imol][0, 5]:12.4f} \n\
                        {vec[imol][1, 0]:12.4f} {vec[imol][1, 1]:12.4f} {vec[imol][1, 2]:12.4f} {vec[imol][1, 3]:12.4f}, {vec[imol][1, 4]:12.4f} {vec[imol][1, 5]:12.4f} \n\
                        {vec[imol][2, 0]:12.4f} {vec[imol][2, 1]:12.4f} {vec[imol][2, 2]:12.4f} {vec[imol][2, 3]:12.4f}, {vec[imol][2, 4]:12.4f} {vec[imol][2, 5]:12.4f} \n\
                        {vec[imol][3, 0]:12.4f} {vec[imol][3, 1]:12.4f} {vec[imol][3, 2]:12.4f} {vec[imol][3, 3]:12.4f}, {vec[imol][3, 4]:12.4f} {vec[imol][3, 5]:12.4f} \n\
                        {vec[imol][4, 0]:12.4f} {vec[imol][4, 1]:12.4f} {vec[imol][4, 2]:12.4f} {vec[imol][4, 3]:12.4f}, {vec[imol][4, 4]:12.4f} {vec[imol][4, 5]:12.4f} \n\
                        {vec[imol][5, 0]:12.4f} {vec[imol][5, 1]:12.4f} {vec[imol][5, 2]:12.4f} {vec[imol][5, 3]:12.4f}, {vec[imol][5, 4]:12.4f} {vec[imol][5, 5]:12.4f} \n\n\
                        Bulk modulus{K[imol].item():9.3f}   Shear modulus{G[imol].item():9.3f}   Poisson ratio{v[imol].item():9.3f}  Young's modulus{E[imol].item():9.3f} \n\n" \
                    for (vec, K, G, v, E) in \
                        [(real_C, real_K_VRH, real_G_VRH, real_v, real_E), (pred_C, pred_K_VRH, pred_G_VRH, pred_v, pred_E)]
                ]
                for t, v in zip(titles, values):
                    wf.write("  ".join(t, v) + "\n") 
                    wf.write("\n")
                    wf.flush() 
        num_mole += len(data.y) 

        error_K_VRH = torch.from_numpy(pred_K_VRH - real_K_VRH)
        error_G_VRH = torch.from_numpy(pred_G_VRH - real_G_VRH)
        error_v = torch.from_numpy(pred_v - real_v) 
        error_E = torch.from_numpy(pred_E - real_E)

        sum_K_error += error_K_VRH.abs().sum().item() 
        sum_G_error += error_G_VRH.abs().sum().item()
        sum_v_error += error_v.abs().sum().item()
        sum_E_error += error_E.abs().sum().item()
        
        sum_K_var += error_K_VRH.pow(2).sum().item() 
        sum_G_var += error_G_VRH.pow(2).sum().item()
        sum_v_var += error_v.pow(2).sum().item()
        sum_E_var += error_E.pow(2).sum().item()

    mae_K = sum_K_error / num_mole
    mae_G = sum_G_error / num_mole
    mae_v = sum_v_error / num_mole
    mae_E = sum_E_error / num_mole
    rmse_v = math.sqrt(sum_v_var / num_mole)
    rmse_K = math.sqrt(sum_K_var / num_mole)
    rmse_G = math.sqrt(sum_G_var / num_mole)
    rmse_E = math.sqrt(sum_E_var / num_mole)
    wf.write(f"Test Error  Unit:{p_unit} \n") 
    wf.write(f"Bulk Modulus: MAE {mae_K:10.4f}, RMSE {rmse_K:10.4f}. \n")
    wf.write(f"Shear Modulus: MAE {mae_G:10.4f}, RMSE {rmse_G:10.4f}. \n")
    wf.write(f"Poisson Ratio: MAE {mae_v:10.4f}, RMSE {rmse_v:10.4f}. \n") 
    wf.write(f"Young's Modulus: MAE {mae_E:10.4f}, RMSE {rmse_E:10.4f}. \n")


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
        "--verbose", "-v", type=int, default=0, choices=[0, 1, 2],
        help="Verbose level. (default: 0)",
    )
    parser.add_argument(
        "--batch-size", "-bz", type=int, default=32,
        help="Batch size. (default: 32)",
    )
    args = parser.parse_args()
    

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    config = NetConfig.parse_file(args.config)
    ckpt = torch.load(args.ckpt, map_location=device)
    config.parse_obj(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    test_dataset = create_dataset(config, "test")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    
    # adjust some configurations
    if args.force == True and config.output_mode == "scalar":
        config.output_mode = "grad"
    
    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # test
    output_file = f"{config.run_name}_analysis.log"
        
    with open(output_file, 'w') as wf:
        wf.write("XequiNet testing\n")
        wf.write(f"Unit: {config.default_property_unit} {config.default_length_unit}\n")

    if config.order == 1:
        test_1_tensor(model, test_loader, device, output_file, args.verbose)
    elif config.order == 2:
        test_2_tensor(model, test_loader, device, output_file, args.verbose)
    elif config.order == 3:
        test_3_tensor(model, test_loader, device, output_file, args.verbose)
    elif config.order == 4:
        test_4_tensor(model, test_loader, device, output_file, args.verbose)
    else:
        raise NotImplementedError(f"Unsupported cartesian tensor Error Analysis for required order {config.order}.")


if __name__ == "__main__":
    main()
       