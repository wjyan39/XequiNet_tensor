import argparse

import numpy as np 
import torch
from torch_geometric.loader import DataLoader

from xequinet.data import create_dataset
from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig, unit_conversion, set_default_unit, get_default_unit,
)

from torch_scatter import scatter 

@torch.no_grad()
def predict_csc(model, test_loader, device):
    at_no = []
    res_all = []
    for data in test_loader:  
        data = data.to(device)
        pred = model(data)
        res_batch = pred.cpu().numpy() 
        at_no_batch = data.at_no.cpu().numpy()
        res_all.append(res_batch) 
        at_no.append(at_no_batch)
    
    res_all = np.concatenate(res_all, axis=0) 
    at_no_all = np.concatenate(at_no, axis=0)
    res = {}
    res['at_no'] = at_no_all
    res['atomic_res'] = res_all 
    np.savez("nmr_tsne.npz", **res)


@torch.no_grad()
def predict_shift(model, test_loader, target_elem, device, save_path:str="predict.pt"):
    at_no = []
    pred = []
    real = []
    for data in test_loader:
        data = data.to(device) 
        pred_batch = model(data)
        real_batch = data.y
        at_no_batch = data.at_no
        if hasattr(data, "label_mask"):
            label_mask = data.label_mask 
        else:
            label_mask = torch.ones_like(real_batch, device=device).bool()
        batch_elem_mask = torch.zeros(real_batch.shape[0], dtype=torch.bool, device=real_batch.device)
        for ele in target_elem:
            batch_elem_mask = torch.logical_or(batch_elem_mask, data.at_no == ele)
        # ele_mask = data.at_no == target_elem 
        batch_mask = torch.logical_and(batch_elem_mask, label_mask)
        pred_batch = pred_batch[batch_mask]
        real_batch = real_batch[batch_mask]
        at_no_batch = at_no_batch[batch_mask]
        pred.append(pred_batch.cpu())
        real.append(real_batch.cpu())
        at_no.append(at_no_batch.cpu())
    pred_all = torch.cat(pred, dim=0)
    real_all = torch.cat(real, dim=0)
    at_no_all = torch.cat(at_no, dim=0)
    save_items = {"x": pred_all, "y": real_all, "at_no": at_no_all}
    torch.save(save_items, save_path)


@torch.no_grad()
def predict_ten(model, test_loader, device, save_path:str="predict.pt", atomic_info:bool=False):
    res_all = []
    gt_all = []
    if atomic_info:
        at_no_all = []
    for data in test_loader:
        data = data.to(device)
        pred = model(data) 
        res_batch = pred.cpu() 
        gt_batch = data.y.cpu() 
        res_all.append(res_batch)
        gt_all.append(gt_batch) 
        if atomic_info:
            at_no_batch = data.at_no.cpu()
            at_no_all.append(at_no_batch)
    res_all = torch.cat(res_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0) 
    save_items = {"x": res_all, "y": gt_all}
    if atomic_info:
        at_no_all = torch.cat(at_no_all, dim=-1)
        save_items["at_no"] = at_no_all
    torch.save(save_items, save_path)


@torch.no_grad()
def predict_elastic(model, test_loader, device, save_path:str="predict.pt"):
    pred_data_list = []
    thresh = 1e-4
    symm_index = torch.LongTensor([0, 4, 8, 1, 5, 2]).to(device)
    for data in test_loader:
        data = data.to(device) 
        pred = model(data).view(-1, 9, 9)
        tmp = torch.index_select(pred, dim=2, index=symm_index)
        pred = torch.index_select(tmp, dim=1, index=symm_index)
        zero_padding = torch.zeros_like(pred, device=device)     
        res = torch.where(torch.abs(pred) > thresh, pred, zero_padding).cpu()
        pred_data_list.append(res) 
    pred_data = torch.cat(pred_data_list, dim=0)
    save_ten = {"elastic_matrix": pred_data}
    torch.save(save_ten, save_path)
            

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
        "--mode", "-m", type=str, default="test",
        help="Mode. (default: test)",
    )
    parser.add_argument(
        "--batch-size", "-bz", type=int, default=32,
        help="Batch size. (default: 32)",
    )
    parser.add_argument(
        "--elastic", "-E", default=False, action="store_true",
        help="Option to save elastic matrix.",
    )
    parser.add_argument(
        "--atomic", "-A", default=False, action="store_true",
        help="Whether to save atomic info.",
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

    test_dataset = create_dataset(config, args.mode)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    
    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    outfile = f"{config.run_name}_predict.pt"
    atomic_info = args.atomic or "atomic" in config.output_mode
    
    if config.output_mode in ["chemical_shielding", "chemical_shifts"]:
        target_elem = config.target_elem[0]
        assert type(target_elem) == int, "Put atomic number directly in target_elem."
        predict_shift(model, test_loader, target_elem, device, outfile)
    elif args.elastic:
        predict_elastic(model, test_loader, device, outfile)
    else:
        predict_ten(model, test_loader, device, outfile, atomic_info) 


if __name__ == "__main__":
    main()