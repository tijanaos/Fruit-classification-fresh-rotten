from pathlib import Path
import pytest
import tensorflow as tf
from fruitcls.config import Config
from fruitcls.data.loader import build_train_val_loaders
from fruitcls.models.custom_cnn import build_custom_cnn

def test_one_train_step():
    cfg = Config()
    if not Path(cfg.data_raw_dir).exists():
        pytest.skip("Dataset not found; skipping.")

    loaders = build_train_val_loaders(cfg.data_raw_dir, cfg.img_size, 4, cfg.seed, val_split=0.2)
    model = build_custom_cnn((cfg.img_size, cfg.img_size, 3), 6)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # samo 1 batch da proverimo da nista ne puca
    model.fit(loaders.train.take(1), epochs=1, verbose=0)
    assert True