from pathlib import Path
from src.utils.io import load_config, resolve_path
from src.data.dicom_convert import convert_tcia_dicom_to_nifti
from src.data.index_build import build_labels_csv

def main():
    cfg = load_config("configs/config.yaml")
    base = Path(cfg["data"]["base_dir"])

    tcia_raw   = resolve_path(base, cfg["data"]["tcia_raw_dir"])
    tcia_nifti = resolve_path(base, cfg["data"]["tcia_nifti_dir"])
    nih_data   = resolve_path(base, cfg["data"]["nih_data_dir"])
    labels_csv = resolve_path(base, cfg["data"]["labels_csv"])
    print("[prepare_data] labels_csv ->", labels_csv)


    convert_tcia_dicom_to_nifti(tcia_raw, tcia_nifti)
    build_labels_csv(tcia_nifti, nih_data, labels_csv)

if __name__ == "__main__":
    main()
