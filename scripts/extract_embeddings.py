from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config, ensure_dir
from src.data.dataset import ScanKSliceDataset
from src.models.cnn_backbone import ResNet18Embedder


def repeat_to_3ch(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(3, 1, 1)

def make_transform(resize: int, crop: int):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        repeat_to_3ch,
    ])

@torch.no_grad()
def main():
    cfg = load_config("configs/config.yaml")
    artifacts = ensure_dir(Path(cfg["data"]["artifacts_dir"]))

    index_csv = artifacts / cfg["data"]["imaging_index_csv"]
    #qc_log = artifacts / cfg["data"]["imaging_qc_log_csv"]
    out_csv = artifacts / cfg["data"]["imaging_embeddings_csv"]
    out_pt = artifacts / cfg["data"]["imaging_embeddings_pt"]

    df = pd.read_csv(index_csv)

    qc_log = artifacts / cfg["data"]["imaging_qc_log_csv"]
    bad_set = set()
    if qc_log.exists() and qc_log.stat().st_size > 0:
        try:
            bad_df = pd.read_csv(qc_log)
            if "filepath" in bad_df.columns:
                bad_set = set(bad_df["filepath"].astype(str).tolist())
        except EmptyDataError:
            bad_set = set()
    else:
            print("[EMB] No QC log found/usable; using all scans")

    if bad_set:
        before = len(df)
        df = df[~df["filepath"].astype(str).isin(bad_set)].reset_index(drop=True)
        print(f"[EMB] Skipping QC-logged scans: {before} -> {len(df)}")

    tfm = make_transform(cfg["preprocess"]["patch_resize"], cfg["preprocess"]["patch_crop"])
    ds = ScanKSliceDataset(df, cfg, transform=tfm)

    loader = DataLoader(
        ds,
        batch_size=int(cfg["embeddings"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["embeddings"]["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = ResNet18Embedder().to(device).eval()

    # store in dict for easy GNN usage later
    emb_dict = {}

    rows = []
    for xk, pid, source in tqdm(loader, desc="Embeddings"):
        # xk: [B,K,3,224,224]
        B, K, C, H, W = xk.shape
        xk = xk.to(device)

        xflat = xk.view(B * K, C, H, W)
        zflat = embedder(xflat)                 # [B*K,512]
        zscan = zflat.view(B, K, -1).mean(1)    # [B,512]
        zscan_cpu = zscan.cpu()

        for i in range(B):
            p = pid[i]
            s = source[i]
            vec = zscan_cpu[i]
            emb_dict[p] = vec

            row = {"patient_id": p, "source": s}
            row.update({f"z{j}": float(vec[j].item()) for j in range(vec.shape[0])})
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    torch.save(emb_dict, out_pt)

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_pt}  (dict patient_id -> tensor[512])")

if __name__ == "__main__":
    main()
