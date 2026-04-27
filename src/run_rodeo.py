#!/usr/bin/env python3
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
import argparse
from pathlib import Path
import tqdm


# -------------------------------------------------------
# Hamiltonian / state helpers  (same as run_adiabatic.py)
# -------------------------------------------------------

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


def compute_gap(L, Nf):
    """Single-particle gap above the Fermi level for the full (fused) chain."""
    A_full = xx_ham(L, boundary_coupling=1.0)
    _, eps_full = ground_state(A_full, Nf)
    _, eps_up = ground_state(A_full, Nf + 1)
    gap = eps_up[-1] - eps_full[-2]
    return gap


# -------------------------------------------------------
# Rodeo helpers
# -------------------------------------------------------

def superiteration(T_total, alpha, dt=1e-6):
    """Build geometric sequence of rodeo times summing to T_total."""
    first_term = T_total * (1 - 1 / alpha)
    times = []
    while first_term > dt:
        times.append(first_term)
        first_term /= alpha
    times.append(first_term)
    times_rounded = [(t + 1e-7) // dt * dt for t in times]
    return times_rounded


def rodeo_free_fermion(initial_orbitals, times, h_matrix, e_target, target_orbitals):
    """
    Free-fermion Rodeo algorithm (Eq. 1):
        |psi> = 1/2^N * Prod_k (I + exp[i E_t t_k] exp[-i H t_k]) |psi_i>

    Returns
    -------
    fidelity : float
    psi_norm_sq : float  (success probability)
    """
    num_times = len(times)
    # List of (complex_coefficient, orbital_matrix)
    final_states = [(1.0 / (2 ** num_times), initial_orbitals.copy())]

    for t in times:
        mb_phase = np.exp(1j * e_target * t)
        new_states = []
        for coeff, orbitals in final_states:
            # Branch 1: identity
            new_states.append((coeff, orbitals))
            # Branch 2: exp[i(E_t - H) t]
            evolved = expm_multiply(-1j * h_matrix * t, orbitals)
            new_states.append((coeff * mb_phase, evolved))
        final_states = new_states

    # Overlap <target|psi>
    total_overlap = sum(
        coeff * np.linalg.det(target_orbitals.conj().T @ orbs)
        for coeff, orbs in final_states
    )

    # Norm <psi|psi>
    psi_norm_sq = sum(
        np.conj(c1) * c2 * np.linalg.det(orb1.conj().T @ orb2)
        for c1, orb1 in final_states
        for c2, orb2 in final_states
    ).real

    fidelity = np.abs(total_overlap) ** 2 / psi_norm_sq
    return float(fidelity), float(psi_norm_sq)


# -------------------------------------------------------
# Preconditioned-state cache
# -------------------------------------------------------

def precond_cache_path(cache_dir, L, filling, adiabatic_dt, adiabatic_time):
    dt_str = str(adiabatic_dt).replace(".", "p")
    fill_str = str(filling).replace(".", "p")
    T_str = str(adiabatic_time).replace(".", "p")
    fname = f"precond_L{L}_fill{fill_str}_adt{dt_str}_T{T_str}.npz"
    return Path(cache_dir) / fname


def load_or_compute_precond(L, Nf, filling, adiabatic_dt, adiabatic_time,
                             cache_dir="data/precond"):
    """
    Return the preconditioned orbital matrix Phi and the single-particle gap.
    If adiabatic_time == 0 the initial state is the disconnected-chain ground
    state (no time evolution needed).  The result is cached on disk so that
    multiple rodeo runs sharing the same parameters never re-run the ramp.
    """
    path = precond_cache_path(cache_dir, L, filling, adiabatic_dt, adiabatic_time)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"[cache] Loading preconditioned state from {path}")
        data = np.load(path)
        Phi = data["Phi"]
        gap = float(data["gap"])
        return Phi, gap

    print(f"[cache] Computing preconditioned state (adiabatic_time={adiabatic_time}) ...")
    A_full = xx_ham(L, boundary_coupling=1.0)
    Phi_target, eps = ground_state(A_full, Nf)
    E0 = np.sum(eps[:Nf])

    gap = compute_gap(L, Nf)
    print(f"  gap = {gap:.6f}")

    if adiabatic_time == 0.0:
        # No preconditioning: start from disconnected ground state
        A0 = xx_ham(L, boundary_coupling=0.0)
        Phi, _ = ground_state(A0, Nf)
    else:
        Phi = adiabatic_fusion(L, Nf, dt=adiabatic_dt, T=adiabatic_time)

    np.savez(path, Phi=Phi, gap=gap, E0=E0,
             L=L, Nf=Nf, filling=filling,
             adiabatic_dt=adiabatic_dt, adiabatic_time=adiabatic_time)
    print(f"[cache] Saved to {path}")
    return Phi, gap


# -------------------------------------------------------
# Main simulation
# -------------------------------------------------------

def run_sim(L, filling, adiabatic_dt, adiabatic_time,
            rodeo_dt, alpha_coeff,
            rodeo_total_times, time_cuts,
            output_dir, cache_dir,
            times_in_T0=False):
    """
    For each (rodeo_total_time, time_cut) pair compute the rodeo fidelity
    starting from the (cached) preconditioned state.

    Results are saved per time_cut in separate .npz files so that cuts can
    be run independently / in parallel.
    """
    Nf = int(L * filling)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- preconditioned state (cached) ----------
    Phi_init, gap = load_or_compute_precond(
        L, Nf, filling, adiabatic_dt, adiabatic_time, cache_dir=cache_dir
    )

    # ---------- optionally convert T0-multiples to absolute times ----------
    if times_in_T0 == 'true':
        T0 = np.pi / gap
        print(f"[T0 scaling] gap = {gap:.6f}, T0 = pi/gap = {T0:.6f}")
        print(f"[T0 scaling] time range in T0: [{rodeo_total_times[0]:.4f}, {rodeo_total_times[-1]:.4f}]")
        rodeo_total_times = rodeo_total_times * T0
        print(f"[T0 scaling] absolute time range: [{rodeo_total_times[0]:.4f}, {rodeo_total_times[-1]:.4f}]")

    # ---------- target state ----------
    A_full = xx_ham(L, boundary_coupling=1.0)
    Phi_target, eps = ground_state(A_full, Nf)
    E0 = float(np.sum(eps[:Nf]))

    # ---------- per-cut output files ----------
    dt_str = str(adiabatic_dt).replace(".", "p")
    fill_str = str(filling).replace(".", "p")
    T_str = str(adiabatic_time).replace(".", "p")
    rdt_str = str(rodeo_dt).replace(".", "p")

    def output_path(cut):
        return output_dir / (
            f"rodeo_L{L}_fill{fill_str}_adt{dt_str}_aT{T_str}"
            f"_rdt{rdt_str}_cut{cut}.npz"
        )

    # Load existing results for each cut
    existing = {}
    for cut in time_cuts:
        p = output_path(cut)
        if p.exists():
            d = dict(np.load(p, allow_pickle=True))
            existing[cut] = {
                "times": list(d["rodeo_total_times"]),
                "fidelities": list(d["fidelities"]),
                "success_probs": list(d["success_probs"]),
            }
            print(f"[cut={cut}] Loaded {len(existing[cut]['times'])} existing points.")
        else:
            existing[cut] = {"times": [], "fidelities": [], "success_probs": []}

    # ---------- main loop ----------
    for T in tqdm.tqdm(rodeo_total_times, desc="rodeo_total_times"):
        T_rounded = round(float(T), 8)

        # alpha from optimized formula (scales as 1/T, coefficient set by user)
        alpha = 1.0 + alpha_coeff * np.pi / (gap * T) * 1/L
        alpha = max(alpha, 1.0 + 1e-6)  # guard against alpha <= 1

        # build time sequence once (shared across cuts; each cut just truncates)
        rodeo_times_full = superiteration(T, alpha, dt=rodeo_dt)
        # sort descending (largest times first) as in notebook
        rodeo_times_sorted = sorted(rodeo_times_full, reverse=True)

        for cut in time_cuts:
            ex = existing[cut]
            existing_rounded = [round(t, 8) for t in ex["times"]]
            if T_rounded in existing_rounded:
                continue

            rodeo_times = rodeo_times_sorted[:cut]
            if len(rodeo_times) == 0:
                continue

            fid, prob = rodeo_free_fermion(
                Phi_init, rodeo_times, A_full, E0, Phi_target
            )

            ex["times"].append(float(T))
            ex["fidelities"].append(fid)
            ex["success_probs"].append(prob)

            # keep sorted by rodeo_total_time
            sort_idx = np.argsort(ex["times"])
            ex["times"] = [ex["times"][i] for i in sort_idx]
            ex["fidelities"] = [ex["fidelities"][i] for i in sort_idx]
            ex["success_probs"] = [ex["success_probs"][i] for i in sort_idx]

            np.savez(
                output_path(cut),
                L=L, filling=filling,
                adiabatic_dt=adiabatic_dt, adiabatic_time=adiabatic_time,
                rodeo_dt=rodeo_dt, alpha_coeff=alpha_coeff,
                time_cut=cut, gap=gap, E0=E0,
                rodeo_total_times=np.array(ex["times"]),
                fidelities=np.array(ex["fidelities"]),
                success_probs=np.array(ex["success_probs"]),
            )


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rodeo algorithm for free fermions with optional adiabatic preconditioning."
    )
    # System
    parser.add_argument("--L", type=int, required=True, help="System size")
    parser.add_argument("--filling", type=float, default=0.5, help="Filling fraction")

    # Adiabatic preconditioning
    parser.add_argument("--adiabatic_time", type=float, default=0.0,
                        help="Adiabatic ramp time. Set to 0 for no preconditioning. "
                             "Ignored if --adiabatic_time_auto is set.")
    parser.add_argument("--adiabatic_time_auto", action="store_true", default=False,
                        help="Compute adiabatic ramp time automatically as "
                             "TAE = tae_slope * L + tae_intercept. "
                             "Overrides --adiabatic_time.")
    parser.add_argument("--tae_slope", type=float, default=0.84,
                        help="Slope in TAE = slope * L + intercept (default: 0.84).")
    parser.add_argument("--tae_intercept", type=float, default=1.19,
                        help="Intercept in TAE = slope * L + intercept (default: 1.19).")
    parser.add_argument("--adiabatic_dt", type=float, default=0.5,
                        help="Time step for adiabatic ramp.")

    # Rodeo parameters
    parser.add_argument("--rodeo_dt", type=float, default=0.01,
                        help="Minimum time resolution in superiteration.")
    parser.add_argument("--alpha_coeff", type=float, default=0.35 / 8,
                        help="Coefficient c in alpha = 1 + c*pi/(gap*T). "
                             "Default 0.35/8 matches notebook cell [86].")

    # Rodeo total times
    parser.add_argument("--rodeo_times", type=float, nargs="+", default=None,
                        help="Explicit list of total rodeo times T_R.")
    parser.add_argument("--rodeo_min", type=float, default=None)
    parser.add_argument("--rodeo_max", type=float, default=None)
    parser.add_argument("--rodeo_steps", type=int, default=None)
    parser.add_argument("--rodeo_spacing", choices=["linear", "log"], default="linear",
                        help="Spacing for rodeo_min/max/steps grid.")
    parser.add_argument("--rodeo_times_in_T0", choices=['true', 'false'], default='false',
                        help="Interpret rodeo time arguments as multiples of T0 = pi/gap. "
                             "E.g. --rodeo_min 1 --rodeo_max 100 --rodeo_steps 50 "
                             "gives 50 times linearly spaced between 1*T0 and 100*T0. "
                             "The gap is computed from the system before scaling. "
                             "Saved output always stores absolute times.")

    # Time cuts
    parser.add_argument("--time_cuts", type=int, nargs="+", required=True,
                        help="List of cut values (max number of rodeo steps to use).")

    # I/O
    parser.add_argument("--output_dir", type=str, default="data/rodeo")
    parser.add_argument("--cache_dir", type=str, default="data/precond",
                        help="Directory for cached preconditioned states.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build rodeo_total_times
    if args.rodeo_times is not None:
        rodeo_total_times = np.array(args.rodeo_times)
    elif None not in (args.rodeo_min, args.rodeo_max, args.rodeo_steps):
        if args.rodeo_spacing == "log":
            rodeo_total_times = np.logspace(
                np.log10(args.rodeo_min), np.log10(args.rodeo_max), args.rodeo_steps
            )
        else:
            rodeo_total_times = np.linspace(args.rodeo_min, args.rodeo_max, args.rodeo_steps)
    else:
        raise ValueError(
            "Specify either --rodeo_times or (--rodeo_min, --rodeo_max, --rodeo_steps)."
        )

    # Resolve adiabatic time
    if args.adiabatic_time_auto:
        adiabatic_time = args.tae_slope * args.L + args.tae_intercept
        print(f"[TAE] auto: {args.tae_slope} * {args.L} + {args.tae_intercept} = {adiabatic_time:.4f}")
    else:
        adiabatic_time = args.adiabatic_time

    run_sim(
        L=args.L,
        filling=args.filling,
        adiabatic_dt=args.adiabatic_dt,
        adiabatic_time=adiabatic_time,
        rodeo_dt=args.rodeo_dt,
        alpha_coeff=args.alpha_coeff,
        rodeo_total_times=rodeo_total_times,
        time_cuts=args.time_cuts,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        times_in_T0=args.rodeo_times_in_T0,
    )