"""Shared paths, session sample and module import for the ssl-dlc-utils-validation scripts."""
import pathlib
import socket
import sys

if 'haas' in socket.gethostname():
    sys.path.insert(0, '/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities')
else:
    sys.path.insert(0, r'M:\analysis\Axel_Bisi\Github\ephys_utilities')
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402,F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'scripts'))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

NWB_DIR = axel_bisi_path('NWB_ks4')
OUT = axel_bisi_path('combined_results_ks4') / 'ssl-dlc-utils-validation'
(OUT / 'figures').mkdir(parents=True, exist_ok=True)
(OUT / 'tables').mkdir(parents=True, exist_ok=True)

# AB and MH mice, both rigs, 2023-2026; in register (ssl-dlc-alignment-debug) unless noted
SESSIONS = [
    'AB127_20240823_094915',  # top end-aligned (clause)
    'AB151_20250130_131124',
    'AB153_20250209_153912',  # end-aligned (clause)
    'AB095_20231212_141728',
    'MH006_20250123_131233',
    'MH009_20250206_104829',
    'MH013_20250218_110530',  # end-aligned (clause)
    'MH015_20250222_194156',
    'MH028_20250502_123728',
    'MH038_20250524_105329',  # camera timestamps from the log; jaw reference at origin
    'MH020_20250304_161732',  # top DLC incomplete; jaw + nose reference at origin
    'MH064_20260115_104148',  # last camera pulse run (clause)
]


def nwb(session_id):
    return str(NWB_DIR / f'{session_id}.nwb')
