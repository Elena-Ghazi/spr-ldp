"""
Same experiment as reproduce_kl_vs_eps_k6_with_spr.py but at k = 100,
where the Kairouz LP has 2^100 variables and is infeasible to solve.

We compare three mechanisms:
    - SPR dynamic program (this paper, O(L k^2))
    - binary mechanism (closed form)
    - randomized response (closed form)

The SPR DP uses a vectorized inner loop. The script produces the k=100
panels of Figures 1 and 3, with pointwise bootstrap confidence intervals.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 12,
})


# -------------------------- shared helpers --------------------------

def kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


# -------------------------- closed-form mechanisms --------------------------

def binary_kl(P0, P1, eps):
    e = np.exp(eps)
    hi = e / (1.0 + e)
    lo = 1.0 / (1.0 + e)
    favor0 = P0 >= P1
    Q = np.zeros((len(P0), 2))
    Q[:, 0] = np.where(favor0, hi, lo)
    Q[:, 1] = 1.0 - Q[:, 0]
    return kl(P0 @ Q, P1 @ Q)


def randomized_response_kl(P0, P1, eps):
    k = len(P0)
    e = np.exp(eps)
    diag = e / (k - 1 + e)
    off = 1.0 / (k - 1 + e)
    Q = np.full((k, k), off)
    np.fill_diagonal(Q, diag)
    return kl(P0 @ Q, P1 @ Q)


# -------------------------- SPR dynamic program (vectorized) --------------------------

def spr_dp_kl(P0, P1, eps, L=None):
    """O(L k^2) DP. Vectorized over the inner ell-argmax via cummax.

    For KL the column score on interval [a:b] (1-indexed) is
        mu(a,b) = (1 + (alpha-1) p_ab) * log( (1 + (alpha-1) p_ab)
                                              / (1 + (alpha-1) q_ab) ).
    The recurrence is
        F[s, i] = max_{s-1 <= ell < i} F[s-1, ell] + mu(ell+1, i),
    and the SPR optimum is V* = max_s F[s, k] / (alpha + s - 1).
    """
    k = len(P0)
    if L is None:
        L = k
    if eps <= 0:
        return 0.0

    # Sort by likelihood ratio P0/P1 ascending.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(P1 > 0, P0 / np.maximum(P1, 1e-300), np.inf)
    sigma = np.argsort(ratios, kind="stable")
    p = P0[sigma]
    q = P1[sigma]

    alpha = np.exp(eps)
    a1 = alpha - 1.0

    # Prefix sums of the sorted distributions.
    P_pref = np.concatenate([[0.0], np.cumsum(p)])  # length k+1
    Q_pref = np.concatenate([[0.0], np.cumsum(q)])

    # Build M[i, j] = mu(i+1, j) for 0 <= i < j <= k:
    #     interval [i+1 : j], so masses = P_pref[j] - P_pref[i].
    P_diff = P_pref[None, :] - P_pref[:, None]   # (k+1, k+1), >= 0 for j>=i
    Q_diff = Q_pref[None, :] - Q_pref[:, None]
    num = 1.0 + a1 * P_diff
    den = 1.0 + a1 * Q_diff
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_ratio = np.where((num > 0) & (den > 0),
                             np.log(num) - np.log(den),
                             0.0)
        M = num * log_ratio  # (k+1, k+1); only entries with i<j are meaningful

    # DP table: F[s, i] = max raw utility partitioning [1:i] into s blocks.
    F = np.full((L + 1, k + 1), -np.inf)
    # Base case s = 1: F[1, i] = mu(1, i) = M[0, i].
    F[1, 1:] = M[0, 1:]

    # Recurrence: F[s, i] = max_{ell in [s-1, i-1]} F[s-1, ell] + M[ell, i].
    # For each fixed s, build A[r, i] = F[s-1, r] + M[r, i], then take
    # cummax over rows starting at row s-1, and read off i->i-s.
    for s in range(2, L + 1):
        A = F[s - 1, :, None] + M                  # (k+1, k+1)
        A_sub = A[s - 1:, :]                       # rows s-1, ..., k
        cmax = np.maximum.accumulate(A_sub, axis=0)
        i_vals = np.arange(s, k + 1)
        r_vals = i_vals - s
        F[s, s:k + 1] = cmax[r_vals, i_vals]

    s_arr = np.arange(1, L + 1)
    vals = F[1:L + 1, k] / (alpha + s_arr - 1.0)
    return float(np.max(vals))


# -------------------------- experiment --------------------------

def run(num_trials=100, k=100, eps_grid=None, seed=0):
    """Returns per-trial normalized utilities of shape (T, E) for each mechanism.

    Trials with D_kl(P_0 || P_1) <= 0 are dropped (probability zero under
    Dirichlet draws, but guarded against). All mechanisms see the same
    surviving trial set, so the comparison stays paired.
    """
    if eps_grid is None:
        eps_grid = np.linspace(0.0, 10.0, 101)
    rng = np.random.default_rng(seed)
    alpha = np.ones(k)
    P0_list = rng.dirichlet(alpha, size=num_trials)
    P1_list = rng.dirichlet(alpha, size=num_trials)
    DPP = np.array([kl(P0_list[t], P1_list[t]) for t in range(num_trials)])
    keep = np.where(DPP > 0)[0]
    if len(keep) < num_trials:
        print(f"  dropped {num_trials - len(keep)} trials with D_kl(P0||P1)<=0")
    P0_list, P1_list, DPP = P0_list[keep], P1_list[keep], DPP[keep]
    T = len(keep)
    E = len(eps_grid)

    per_trial = {name: np.zeros((T, E))
                 for name in ["spr_dp", "binary", "rr"]}

    for i, eps in enumerate(eps_grid):
        for t in range(T):
            P0, P1 = P0_list[t], P1_list[t]
            d_pp = DPP[t]
            v_dp  = spr_dp_kl(P0, P1, eps, L=k)
            v_bin = binary_kl(P0, P1, eps)
            v_rr  = randomized_response_kl(P0, P1, eps)
            per_trial["spr_dp"][t, i] = v_dp / d_pp
            per_trial["binary"][t, i] = v_bin / d_pp
            per_trial["rr"][t, i]     = v_rr / d_pp
        if (i % 10) == 0:
            print(f"  eps = {eps:5.2f}  done")

    return eps_grid, per_trial


# -------------------------- bootstrap CI --------------------------

def bootstrap_mean_ci(values, confidence=0.95, n_boot=5000, seed=0,
                      batch_size=500):
    """Pointwise percentile-bootstrap CIs; values has shape (T, E)."""
    values = np.asarray(values, dtype=float)
    T, E = values.shape
    rng = np.random.default_rng(seed)
    mean = values.mean(axis=0)
    boot = np.empty((n_boot, E))
    for start in range(0, n_boot, batch_size):
        end = min(start + batch_size, n_boot)
        idx = rng.integers(0, T, size=(end - start, T))
        boot[start:end] = values[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return mean, lo, hi


STYLES_K100 = [
    ("spr_dp", dict(color="orange", linestyle="--", linewidth=1.8,
                    label="SPR mechanism (DP, ours)")),
    ("binary", dict(color="blue",   linestyle=":",  linewidth=1.8,
                    label="binary mechanism")),
    ("rr",     dict(color="red",    linestyle="-.", linewidth=1.8,
                    label=r"$k$-ary randomized response")),
]


def _decorate(ax):
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$D_{\mathrm{kl}}(M_0\|M_1)\,/\,D_{\mathrm{kl}}(P_0\|P_1)$")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25)


def plot(eps_grid, per_trial, out_pdf):
    """Plot mean curves only (no CI bands)."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for name, style in STYLES_K100:
        ax.plot(eps_grid, per_trial[name].mean(axis=0), **style)
    _decorate(ax)
    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"wrote {out_pdf}")


def plot_with_ci(eps_grid, per_trial, out_pdf, n_boot=5000, ci_seed=0):
    """Plot mean curves with shaded pointwise 95% bootstrap CI bands.

    Same bootstrap seed for all curves: trials are paired (same Dirichlet
    pair across mechanisms), so identical bootstrap indices preserve the
    paired structure at the band level.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for name, style in STYLES_K100:
        mean, lo, hi = bootstrap_mean_ci(per_trial[name],
                                         confidence=0.95,
                                         n_boot=n_boot, seed=ci_seed)
        ax.fill_between(eps_grid, lo, hi, color=style["color"],
                        alpha=0.18, linewidth=0)
        ax.plot(eps_grid, mean, **style)
    _decorate(ax)
    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    eps_grid, per_trial = run(num_trials=100, k=100,
                              eps_grid=np.linspace(0.0, 10.0, 101),
                              seed=0)
    plot(eps_grid, per_trial,
         os.path.join(out_dir, "kl_vs_eps_k100_panel.pdf"))
    plot_with_ci(eps_grid, per_trial,
                 os.path.join(out_dir, "kl_vs_eps_k100_panel_with_ci.pdf"))
