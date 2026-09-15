# Resolving Axel Bisi's external repo/analysis-share paths across machines

## The rule

Never hardcode `M:\analysis\Axel_Bisi\...` (or any other one-machine spelling of this path) in a script. Resolve it via `scripts/axel_bisi_paths.py`'s `axel_bisi_root()` / `axel_bisi_path(*parts)` instead.

## Why it matters

Axel Bisi's external analysis share — his `Github` checkouts (`ephys_utilities`, `allen_utils`, etc.), raw NWB trees, the mouse-reference xlsx, and anything else under `.../analysis/Axel_Bisi/...` — is the same NAS share, but it mounts at a **different path per machine**:

- This local Windows machine: `M:\analysis\Axel_Bisi\...`
- `haas056.rcp.epfl.ch` (and presumably any other lab machine): `/mnt/lsens-analysis/Axel_Bisi/...`

A script with `M:\...` hardcoded works on this one Windows machine and silently fails (or falls back to a degraded default, e.g. grey plot colors instead of Allen-atlas colors) everywhere else — including on `haas`, now that it's a compute target for this repo's decoding pipelines (see `ssl-analyze/references` for that project's remote-compute setup). This surfaced concretely in `032_plot_decode_results.py`/`034_area_window_quant_grid.py`: their `allen_utils` import worked locally but silently degraded to grey area coloring, both because `M:` wasn't mounted locally at the time *and* because it would never have worked on `haas` at all with a hardcoded `M:\` path.

## Where implemented

`scripts/axel_bisi_paths.py`:

```python
from axel_bisi_paths import axel_bisi_root, axel_bisi_path

ALLEN_UTILS_PATH = axel_bisi_path("Github", "ephys_utilities", "ephys_utilities", "allen_utils")
```

- `axel_bisi_root()` checks, in order: the `SSL_AXEL_BISI_ROOT` env var override, then `M:\analysis\Axel_Bisi`, then `/mnt/lsens-analysis/Axel_Bisi` — returns the first that exists, raises `FileNotFoundError` if none do.
- `axel_bisi_path(*parts)` is the same, joined with `parts`, returning `None` instead of raising when the share isn't mounted at all — for callers that already have a defined fallback (e.g. grey plot colors) and just want "give me the path if it's there, else None."

Reference usage: `032_plot_decode_results.py` and `034_area_window_quant_grid.py`'s `ALLEN_UTILS_PATH` definitions.

## Known scripts not yet migrated

These still hardcode `M:\...` directly (found during a 2026-09-14 SSH/remote-compute setup investigation) — migrate them the same way before running any of them on a non-this-machine host:

- `scripts/build_ssl_area_labels.py`
- `scripts/compute_ssl_quality_label.py`
- `scripts/extract_ssl_mouse_reference.py`
- `scripts/extract_ssl_target_region.py`
- `scripts/ssl_ephys_utilities_qc_report.py`

## Quality gate

- Reject any new or edited script that hardcodes `M:\...` (or `/mnt/lsens-analysis\...`) for a path under Axel Bisi's share — it must resolve via `axel_bisi_paths.py` instead, so it works unmodified on both this machine and `haas`.
