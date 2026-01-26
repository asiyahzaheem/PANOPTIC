from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import datetime
import time


def try_read_nifti(path: Path) -> tuple[bool, str]:
    """
    Drive-safe check:
    - load header
    - avoid random voxel access (Drive streaming can false-fail)
    - do a small, sequential read of a single slice via dataobj
    """
    try:
        img = nib.load(str(path))
        _ = img.shape
        d = img.dataobj
        # read a small slice (sequential-ish) rather than one voxel
        _ = d[..., 0]
        return True, ""
    except Exception as e:
        return False, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="artifacts/labels.csv")
    ap.add_argument("--log", default="artifacts/data_qc_log.csv")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    labels_path = Path(args.labels).resolve()
    log_path = Path(args.log).resolve()
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_path)
    ts = datetime.datetime.now().isoformat()

    bad = []
    print(f"[QC] Checking {len(df)} scans (Drive-safe, no deletions)...")

    for fp in tqdm(df["filepath"].tolist(), desc="qc", leave=False):
        p = Path(fp)
        if not p.exists():
            bad.append({"filepath": fp, "error": "FileNotFound", "timestamp": ts})
            continue

        ok, err = (False, "")
        for _ in range(args.retries + 1):
            ok, err = try_read_nifti(p)
            if ok:
                break
            time.sleep(args.sleep)

        if not ok:
            bad.append({"filepath": fp, "error": err, "timestamp": ts})

    bad_df = pd.DataFrame(bad)

    if len(bad_df) == 0:
        print("[QC] No problematic scans. Not writing qc log.")
        return


    # Append log (never delete / never modify labels)
    if log_path.exists():
        old = pd.read_csv(log_path)
        out = pd.concat([old, bad_df], ignore_index=True) if len(bad_df) else old
    else:
        out = bad_df

    out.to_csv(log_path, index=False)

    print(f"[QC] Found {len(bad_df)} problematic scans")
    print(f"[QC] Logged to {log_path}")
    print("[QC] labels.csv NOT changed (log-and-skip policy)")


if __name__ == "__main__":
    main()
