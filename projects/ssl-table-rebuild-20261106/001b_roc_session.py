"""001b -- Main ROC (roc_utils_new.compute_unit_roc, rates) for one session: python 001b_roc_session.py <nwb> <results_path>.
A script file, not stdin: roc_utils_new starts worker processes, which must re-import __main__."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
from roc_analysis import roc_utils_new as ru  # noqa: E402

if __name__ == "__main__":
    assert ru.ROC_USES_RATES
    ru.compute_unit_roc(sys.argv[1], sys.argv[2], make_psth_plots=False, n_workers=20)
    print(f"{pathlib.Path(sys.argv[1]).stem} ROC done", flush=True)
