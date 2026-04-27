#!/usr/bin/env python3
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
import argparse
from pathlib import Path
import tqdm


def xx_ham(L, J=1.0, boundary_coupling=1.0):
    A = np.zeros((L, L), dtype=complex)
    for i in range(L - 1):
        A[i, i + 1] = J
        A[i + 1, i] = J
    mid = L // 2 - 1
    A[mid, mid + 1] *= boundary_coupling
    A[mid + 1, mid] *= boundary_coupling
    return A


def ground_state(A, Nf):
    eps, V = eigh(A)
    idx = np.argsort(eps)
    V = V[:, idx]
    Phi = V[:, :Nf]
    return Phi, eps[idx]


def slater_fidelity(Phi1, Phi2):
    overlap = np.linalg.det(Phi1.conj().T @ Phi2)
    return np.abs(overlap) ** 2


def adiabatic_fusion(L, Nf, dt, T):
    steps = int(T / dt)
    A0 = xx_ham(L, boundary_coupling=0.0)
    Phi, _ = ground_state(A0, Nf)
    for s in range(steps):
        lam = (s + 1) / steps
        A = xx_ham(L, boundary_coupling=lam)
        Phi = expm_multiply(-1j * A * dt, Phi)
        U, _, Vh = np.linalg.svd(Phi, full_matrices=False)
        Phi = U @ Vh
    return Phi


def run_sim(L, dt, filling, ramp_times, output_file):
    Nf = int(L * filling)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        data = dict(np.load(output_file, allow_pickle=True))
        existing_times = list(data["ramp_times"])
        existing_fidelities = list(data["fidelities"])
        print(f"Loaded {len(existing_times)} existing ramp times.")
    else:
        existing_times = []
        existing_fidelities = []

    A_full = xx_ham(L, boundary_coupling=1.0)
    Phi_target, _ = ground_state(A_full, Nf)

    for T in tqdm.tqdm(ramp_times):
        T_rounded = round(T, 8)  
        existing_times_rounded = [round(t, 8) for t in existing_times]
        if T_rounded in existing_times_rounded:
            continue

        Phi = adiabatic_fusion(L, Nf, dt, T)
        fid = slater_fidelity(Phi, Phi_target)

        existing_times.append(T)
        existing_fidelities.append(fid)

        sort_idx = np.argsort(existing_times)
        existing_times = [existing_times[i] for i in sort_idx]
        existing_fidelities = [existing_fidelities[i] for i in sort_idx]

        np.savez(
            output_file,
            L=L,
            dt=dt,
            filling=filling,
            ramp_times=np.array(existing_times),
            fidelities=np.array(existing_fidelities)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--filling", type=float, required=True)

    parser.add_argument(
        "--ramp_times",
        type=float,
        nargs="+",
        default=None,
        help="Explicit ramp times."
    )
    parser.add_argument("--ramp_min", type=float, default=None, help="Minimum ramp time")
    parser.add_argument("--ramp_max", type=float, default=None, help="Maximum ramp time")
    parser.add_argument("--ramp_steps", type=int, default=None, help="Number of ramp times between min and max")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.ramp_times is not None:
        ramp_times = np.array(args.ramp_times)
    elif None not in (args.ramp_min, args.ramp_max, args.ramp_steps):
        ramp_times = np.logspace(
            np.log10(args.ramp_min),
            np.log10(args.ramp_max),
            args.ramp_steps
        )
    else:
        raise ValueError(
            "Specify either --ramp_times or (--ramp_min, --ramp_max, --ramp_steps)"
        )

    dt_str = str(args.dt).replace('.', 'p')
    fill_str = str(args.filling).replace('.', 'p')

    output_file = Path("data/adiabatic") / f"adiabatic_L{args.L}_dt{dt_str}_filling{fill_str}.npz"

    run_sim(args.L, args.dt, args.filling, ramp_times, output_file)