from pathlib import Path
import pytest
from fruitcls.config import Config
from fruitcls.data.loader import build_train_val_loaders

def test_data_loader_builds():
    cfg = Config()
    if not Path(cfg.data_raw_dir).exists():
        pytest.skip("Dataset not found; skipping.")
    loaders = build_train_val_loaders(cfg.data_raw_dir, cfg.img_size, 4, cfg.seed, val_split=0.2)
    assert len(loaders.class_names) == 6