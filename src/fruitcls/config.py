from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_raw_dir: Path = project_root / "data" / "raw"
    reports_dir: Path = project_root / "reports"
    reports_metrics_dir: Path = reports_dir / "metrics"
    reports_figures_dir: Path = reports_dir / "figures"
    models_dir: Path = project_root / "models"

    # dataset
    class_names = [
        "freshapples",
        "freshbananas",
        "freshoranges",
        "rottenapples",
        "rottenbananas",
        "rottenoranges",
    ]

    # training defaults
    seed: int = 42
    img_size: int = 224
    batch_size: int = 32