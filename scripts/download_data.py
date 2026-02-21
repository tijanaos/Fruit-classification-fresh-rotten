# scripts/download_data.py
from pathlib import Path
import subprocess
import shutil

from fruitcls.config import Config


def main():
    cfg = Config()

    kaggle_exe = shutil.which("kaggle")
    if not kaggle_exe:
        raise RuntimeError("Kaggle CLI nije pronađen. Probaj: python -m pip install kaggle")

    out_dir = cfg.project_root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ako su klase već tu, ne skidamo opet
    expected_root = cfg.data_raw_dir 
    expected = [expected_root / c for c in cfg.class_names]
    if all(p.exists() for p in expected):
        print("✅ Dataset već postoji:", expected_root)
        return

    cmd = [
        kaggle_exe,
        "datasets",
        "download",
        "-d",
        "swoyam2609/fresh-and-stale-classification",
        "-p",
        str(out_dir),
        "--unzip",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    missing = [c for c in cfg.class_names if not (expected_root / c).exists()]
    if missing:
        print("WARNING: Missing class folders:", missing)
        print("Check dataset structure in:", expected_root)
        print("Tip: uradi 'dir /b /ad data\\raw' i vidi kako se raspakovalo.")
    else:
        print("✅ Dataset looks OK in:", expected_root)


if __name__ == "__main__":
    main()