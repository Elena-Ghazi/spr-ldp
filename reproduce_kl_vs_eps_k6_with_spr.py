"""
Reproduce the k=6 panels of Figures 1 and 3 in
"Sort, Partition, Randomize: Optimal Binary Hypothesis Testing under
Local Differential Privacy" (NeurIPS 2026).

Compare SPR, the Kairouz--Oh--Viswanath staircase LP, the binary
mechanism, and k-ary randomized response, with pointwise bootstrap CIs.
"""
import os
import numpy as np
from scipy.optimize import linprog
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


# -------------------------- mechanism builders --------------------------

def staircase_pattern_matrix(k, eps):
    e = np.exp(eps)
    cols = []
    for j in range(2 ** k):
        bits = np.array([(j >> i) & 1 for i in range(k)], dtype=float)
        cols.append(np.where(bits == 1, e, 1.0))
    return np.stack(cols, axis=1)


def kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def optimal_staircase_kl(P0, P1, eps):
    k = len(P0)
    S = staircase_pattern_matrix(k, eps)
    a = P0 @ S
    b = P1 @ S
    mu = a * (np.log(a) - np.log(b))
    res = linprog(c=-mu, A_eq=S, b_eq=np.ones(k),
                  bounds=[(0, None)] * (2 ** k), method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    theta = res.x
    Q = S * theta[np.newaxis, :]
    M0 = P0 @ Q
    M1 = P1 @ Q
    keep = (M0 + M1) > 1e-15
    return kl(M0[keep], M1[keep])


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


# -------------------------- SPR dynamic program --------------------------

def spr_dp_kl(P0, P1, eps, L=None):
    """O(L k^2) DP from Algorithm 1 of the paper; L is the output budget.

    For KL the column score on interval [a:b] (1-indexed) is
        mu[a:b] = (1 + (alpha-1) p_ab) * log( (1 + (alpha-1) p_ab)
                                              / (1 + (alpha-1) q_ab) ),
    where alpha = e^eps and p_ab, q_ab are interval masses on the
    likelihood-ratio-sorted alphabet, with f(t) = t log t.

    F[s, i] = max raw utility over partitions of [1:i] into exactly s
    contiguous blocks.  Optimal value V* = max_s F[s, k] / (alpha + s - 1).
    """
    k = len(P0)
    if L is None:
        L = k
    if eps <= 0:
        return 0.0

    # Sort by likelihood ratio p_i / q_i ascending.  P1 > 0 a.s. under
    # Dirichlet, but guard with the paper's convention P0/0 = inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(P1 > 0, P0 / np.maximum(P1, 1e-300), np.inf)
    sigma = np.argsort(ratios, kind="stable")
    p = P0[sigma]
    q = P1[sigma]

    alpha = np.exp(eps)
    a1 = alpha - 1.0

    # Prefix sums; P_pref[i] = sum_{j<=i} p_j, with P_pref[0] = 0.
    P_pref = np.concatenate([[0.0], np.cumsum(p)])
    Q_pref = np.concatenate([[0.0], np.cumsum(q)])

    def mu(a, b):
        # 1-indexed interval [a:b]; uses P_pref[b] - P_pref[a-1].
        p_ab = P_pref[b] - P_pref[a - 1]
        q_ab = Q_pref[b] - Q_pref[a - 1]
        num = 1.0 + a1 * p_ab
        den = 1.0 + a1 * q_ab
        # f(t) = t log t, so mu = den * (num/den) * log(num/den)
        #                        = num * log(num/den).
        if num <= 0.0 or den <= 0.0:
            return 0.0
        return num * (np.log(num) - np.log(den))

    NEG_INF = -np.inf
    F = np.full((L + 1, k + 1), NEG_INF)
    for i in range(1, k + 1):
        F[1, i] = mu(1, i)
    for s in range(2, L + 1):
        for i in range(s, k + 1):
            best = NEG_INF
            for ell in range(s - 1, i):  # s-1 <= ell < i
                val = F[s - 1, ell] + mu(ell + 1, i)
                if val > best:
                    best = val
            F[s, i] = best

    best_val = -np.inf
    for s in range(1, L + 1):
        v = F[s, k] / (alpha + s - 1.0)
        if v > best_val:
            best_val = v
    return float(best_val)


# -------------------------- experiment --------------------------

def run(num_trials=100, k=6, eps_grid=None, seed=0):
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
                 for name in ["optimal", "spr_dp", "binary", "rr"]}
    max_lp_vs_dp_gap = 0.0

    for i, eps in enumerate(eps_grid):
        for t in range(T):
            P0, P1 = P0_list[t], P1_list[t]
            d_pp = DPP[t]
            v_lp  = optimal_staircase_kl(P0, P1, eps)
            v_dp  = spr_dp_kl(P0, P1, eps, L=k)
            v_bin = binary_kl(P0, P1, eps)
            v_rr  = randomized_response_kl(P0, P1, eps)
            max_lp_vs_dp_gap = max(max_lp_vs_dp_gap, abs(v_lp - v_dp))
            per_trial["optimal"][t, i] = v_lp / d_pp
            per_trial["spr_dp"][t, i]  = v_dp / d_pp
            per_trial["binary"][t, i]  = v_bin / d_pp
            per_trial["rr"][t, i]      = v_rr / d_pp
        if (i % 10) == 0:
            print(f"  eps = {eps:5.2f}  done"
                  f"  (max |LP - SPR-DP| so far: {max_lp_vs_dp_gap:.3e})")

    print(f"Final max |LP - SPR-DP| over all (trial, eps): {max_lp_vs_dp_gap:.3e}")
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


STYLES_K6 = [
    # Draw KOV first; SPR is dashed and lays on top so both are visible
    # despite agreeing to ~1e-13.
    ("optimal", dict(color="black",  linestyle="-",  linewidth=1.2,
                     label="optimal (KOV LP)")),
    ("spr_dp",  dict(color="orange", linestyle="--", linewidth=1.8,
                     label="SPR mechanism (DP, ours)")),
    ("binary",  dict(color="blue",   linestyle=":",  linewidth=1.8,
                     label="binary mechanism")),
    ("rr",      dict(color="red",    linestyle="-.", linewidth=1.8,
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
    for name, style in STYLES_K6:
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
    for name, style in STYLES_K6:
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
    eps_grid, per_trial = run(num_trials=100, k=6,
                              eps_grid=np.linspace(0.0, 10.0, 101),
                              seed=0)
    spr_vs_kov = float(np.max(np.abs(per_trial["spr_dp"] - per_trial["optimal"])))
    spr_vs_kov_mean = float(np.max(np.abs(per_trial["spr_dp"].mean(axis=0)
                                          - per_trial["optimal"].mean(axis=0))))
    print(f"max |SPR - KOV| over (trials, eps): {spr_vs_kov:.3e}")
    print(f"max |mean SPR - mean KOV| over eps: {spr_vs_kov_mean:.3e}")

    plot(eps_grid, per_trial,
         os.path.join(out_dir, "kl_vs_eps_k6_panel.pdf"))
    plot_with_ci(eps_grid, per_trial,
                 os.path.join(out_dir, "kl_vs_eps_k6_panel_with_ci.pdf"))
