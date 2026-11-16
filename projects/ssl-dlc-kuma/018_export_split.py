"""Export the shared train/test split of the merged DLC model (training_plan.md section 1) with the ground-truth labels,
indexed like DLC's evaluation predictions (loader.df.index: labeled-data / folder / image), for the analysis on haas.

Output (Kuma): /work/lsens/bisi/dlc_merged/analysis/split_ground_truth.h5 (key 'df': ground truth, column-level
('split', '', '') = train / test); copied to DLCProjects/merged_training_2026-11-09/ with rclone.
Run on Kuma (login node, no GPU needed):
  source /work/lsens/bisi/dlc_project/venv_dlc/bin/activate; python 018_export_split.py
"""

from pathlib import Path

from deeplabcut.pose_estimation_pytorch.data import DLCLoader

P = Path("/work/lsens/bisi/dlc_merged/Tracking2D-Axel-2026-11-04")
OUT = Path("/work/lsens/bisi/dlc_merged/analysis")


def main():
    OUT.mkdir(exist_ok=True)
    loader = DLCLoader(str(P / "config.yaml"), shuffle=1, trainset_index=0)
    df = loader.df.copy()
    test = set(map(int, loader.split["test"]))
    df[("split", "", "")] = ["test" if i in test else "train" for i in range(len(df))]
    for s in range(2, 7):  # the split is identical for shuffles 1-6 (check_split.py); verified again here
        other = DLCLoader(str(P / "config.yaml"), shuffle=s, trainset_index=0)
        assert set(map(int, other.split["test"])) == test, f"shuffle {s} has another split"
    df.to_hdf(OUT / "split_ground_truth.h5", key="df")
    print(df[("split", "", "")].value_counts().to_dict(), df.shape)


if __name__ == "__main__":
    main()
