"""
Run a JIT-scripted XequiNet Orb model and save the predicted Fock matrix(es)
to NPZ file(s).

The input can be any file readable by ASE (``.xyz``, ``.pdb``, ...) or a PySCF
checkpoint, and is handled by ``xequinet.data.TextDataset``. The overlap matrix
required by the XPaiNN-Orb/X2 model is computed on the fly with PySCF in
``config.target_basisname``.

The saved NPZ contains:
    fock     (nbasis, nbasis)  predicted Fock matrix, in Hartree
    ovlp     (nbasis, nbasis)  AO overlap matrix used as input
    at_no    (natm,)           atomic numbers
    coord    (natm, 3)         coordinates in config.default_length_unit
    charge   ()                total charge

Usage:
    python -m xequinet.run.jit_fock_predict -c model.pt -j model.jit input.xyz -o pred
"""

import argparse
import os

import numpy as np
import torch
from pyscf import gto

from xequinet.data import TextDataset
from xequinet.utils import NetConfig, set_default_unit


def output_path(prefix: str, imol: int, total: int) -> str:
    if prefix.endswith(".npz"):
        if total == 1:
            return prefix
        return f"{prefix[:-4]}_{imol:04d}.npz"
    return f"{prefix}_{imol:04d}.npz"


def main():
    parser = argparse.ArgumentParser(description="Run JIT XequiNet Orb inference and save Fock to NPZ")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'config')",
    )
    parser.add_argument(
        "--jit", "-j", type=str, default=None,
        help="JIT model file. (default: <ckpt stem>.jit)",
    )
    parser.add_argument(
        "input", type=str,
        help="Input file readable by ASE (*.xyz, *.pdb, ...) or a PySCF checkpoint.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output prefix or *.npz path. (default: <jit stem>_predict)",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on. (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0,
        help="Verbose level for PySCF. (default: 0)",
    )
    args = parser.parse_args()

    jit_path = args.jit if args.jit is not None else os.path.splitext(args.ckpt)[0] + ".jit"
    if not os.path.exists(jit_path):
        raise FileNotFoundError(f"JIT model not found: {jit_path}")
    device = torch.device(args.device)

    # load config from checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = NetConfig.model_validate(ckpt["config"])
    if config.version not in ("xpainn-mat", "xpainn3-mat"):
        raise NotImplementedError(f"Unsupported model version for JIT Fock prediction: {config.version}")

    # set default unit / dtype (must precede TextDataset, which reads the default unit)
    set_default_unit(config.default_property_unit, config.default_length_unit)
    if config.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif config.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    has_ovlp = config.version == "xpainn3-mat"

    # load JIT model
    model = torch.jit.load(jit_path, map_location=device)
    model.eval()

    dataset = TextDataset(args.input)
    total = len(dataset)
    out_prefix = args.output if args.output is not None else \
        os.path.splitext(os.path.basename(jit_path))[0] + "_predict"

    for imol, data in enumerate(dataset, start=1):
        at_no = data.at_no.to(device).to(torch.long)
        coord = data.pos.to(device).to(torch.get_default_dtype())
        charge = int(data.charge.item())

        # AO overlap in the target basis
        mol = gto.Mole()
        mol.build(
            atom=[(int(a), tuple(c)) for a, c in zip(at_no.cpu().numpy(), coord.cpu().numpy())],
            charge=charge,
            basis=config.target_basisname,
            unit=config.default_length_unit,
            verbose=args.verbose,
        )
        ovlp = torch.tensor(mol.intor("int1e_ovlp"), dtype=torch.float64, device=device)

        with torch.no_grad():
            if has_ovlp:
                result = model(at_no, coord, ovlp)
            else:
                result = model(at_no, coord)
        fock = result["Fock"].detach().cpu().numpy()

        path = output_path(out_prefix, imol, total)
        np.savez(
            path,
            fock=fock,
            ovlp=ovlp.detach().cpu().numpy(),
            at_no=at_no.detach().cpu().numpy(),
            coord=coord.detach().cpu().numpy(),
            charge=np.asarray(charge),
        )
        print(f"[{imol}/{total}] {config.version}: Fock {fock.shape} -> {path}")


if __name__ == "__main__":
    main()
