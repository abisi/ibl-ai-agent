"""019 -- Final NWB of the joined AB155 day-0 session, written to NWB_ks4 with NWB_converter (session exception
utils/joined_sessions.py, tested in 018), then compared with the validated test NWB (010 + 012) as in 018.
The previous NWB_ks4/AB155_20250208_175103.nwb (part 1 only) was moved to SessionToConcat/..._joined/nwb_old (2026-11-06).
Run (local, conda env nwb): python 019_final_conversion.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t018', os.path.join(HERE, '018_converter_clause_test.py'))
t018 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t018)

NWB_KS4 = r'M:\analysis\Axel_Bisi\NWB_ks4'


def main():
    out = os.path.join(NWB_KS4, f'{t018.S}.nwb')
    assert not os.path.exists(out), f'{out} exists: move it away first'
    t018.ncp.convert_data_to_nwb(config_file=t018.CONFIG, output_folder=NWB_KS4, with_time_string=False,
                                 experimenter='Axel_Bisi')
    t018.compare(out, t018.REF)


if __name__ == '__main__':
    sys.exit(main())
