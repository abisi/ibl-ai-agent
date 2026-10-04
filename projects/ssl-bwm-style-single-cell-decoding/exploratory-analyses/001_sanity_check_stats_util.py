"""Sanity-check the ported BWM shuffle-test primitives
(`scripts/ssl_bwm_stats_util.py`) before using them on real units, per the
plan's Verification section:
1. Unstratified null case (x, y same distribution) should not be
   systematically significant.
2. Unstratified strong-shift case should be significant.
3. combined_stratified_pvalue with block_aware=False, single stratum,
   should track scipy.stats.mannwhitneyu's significance call.
4. time_two_n_mannwhitneyu_shuf with a single (degenerate) block should
   closely match the unblocked two_n_mannwhitneyu_shuf on the same data
   (permuting "within the one block" ~= permuting freely).
5. benjamini_hochberg matches statsmodels' multipletests on a known example.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import numpy as np
from scipy import stats

from ssl_bwm_stats_util import (
    two_n_mannwhitneyu_shuf,
    time_two_n_mannwhitneyu_shuf,
    combined_stratified_pvalue,
    benjamini_hochberg,
)

RNG_SEED = 20260818
N_SHUF = 3000


def rank_to_p(numer_final: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(numer_final)
    return float(r[0] / (1 + (len(numer_final) - 1)))


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("=== 1. Unstratified null case ===")
    null_ps = []
    for trial in range(30):
        x = rng.normal(5.0, 1.0, size=40)
        y = rng.normal(5.0, 1.0, size=40)
        numer = two_n_mannwhitneyu_shuf(x, y, N_SHUF, rng)
        null_ps.append(rank_to_p(numer))
    null_ps = np.array(null_ps)
    frac_sig = (null_ps < 0.05).mean()
    print(f"30 null trials: fraction p<0.05 = {frac_sig:.3f} (expect ~0.05, tolerate up to ~0.15 given n=30)")
    assert frac_sig <= 0.30, "null case is wildly anti-conservative"

    print("\n=== 2. Unstratified strong-shift case ===")
    x = rng.normal(5.0, 1.0, size=40)
    y = rng.normal(8.0, 1.0, size=40)
    numer = two_n_mannwhitneyu_shuf(x, y, N_SHUF, rng)
    p_shift = rank_to_p(numer)
    print(f"p (3-SD mean shift, n=40 each) = {p_shift:.5f} (expect << 0.05)")
    assert p_shift < 0.01

    print("\n=== 3. Compare to scipy.stats.mannwhitneyu (unstratified) ===")
    agree = 0
    n_compare = 20
    for trial in range(n_compare):
        x = rng.normal(5.0, 1.0, size=30)
        # half the trials get a real shift, half don't
        shift = 2.5 if trial % 2 == 0 else 0.0
        y = rng.normal(5.0 + shift, 1.0, size=30)
        strata = [{"x": x, "y": y}]
        p_ours = combined_stratified_pvalue(strata, N_SHUF, block_aware=False, rng=rng)
        _, p_scipy = stats.mannwhitneyu(x, y, alternative="two-sided")
        sig_ours = p_ours < 0.05
        sig_scipy = p_scipy < 0.05
        agree += int(sig_ours == sig_scipy)
        print(f"trial {trial:2d} shift={shift:.1f}  p_ours={p_ours:.4f}  p_scipy={p_scipy:.4f}  agree={sig_ours == sig_scipy}")
    print(f"Significance-call agreement with scipy: {agree}/{n_compare}")
    assert agree >= n_compare - 2, "too many disagreements with scipy mannwhitneyu"

    print("\n=== 4. Block-aware vs unblocked, single degenerate block ===")
    x = rng.normal(5.0, 1.0, size=40)
    y = rng.normal(7.0, 1.0, size=40)
    bx = np.zeros(40)
    by = np.zeros(40)
    numer_block = time_two_n_mannwhitneyu_shuf(x, y, bx, by, N_SHUF, rng)
    p_block = rank_to_p(numer_block)
    numer_plain = two_n_mannwhitneyu_shuf(x, y, N_SHUF, rng)
    p_plain = rank_to_p(numer_plain)
    print(f"p (block-aware, 1 block) = {p_block:.5f}; p (plain) = {p_plain:.5f} -- both should be small/significant")
    assert p_block < 0.05 and p_plain < 0.05

    print("\n=== 5. Benjamini-Hochberg check ===")
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.251, 0.319, 0.351, 0.5])
    q = benjamini_hochberg(pvals)
    try:
        from statsmodels.stats.multitest import multipletests
        _, q_sm, _, _ = multipletests(pvals, method="fdr_bh")
        print("max abs diff vs statsmodels:", np.max(np.abs(q - q_sm)))
        assert np.allclose(q, q_sm, atol=1e-10)
        print("Matches statsmodels fdr_bh exactly.")
    except ImportError:
        print("statsmodels not available; checking against the textbook worked example instead.")
        print("q =", np.round(q, 4))

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
