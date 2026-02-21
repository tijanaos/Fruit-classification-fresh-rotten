import argparse
import tensorflow as tf

from fruitcls.config import Config
from fruitcls.utils.seed import set_seed
from fruitcls.data.loader import build_train_val_loaders
from fruitcls.models.custom_cnn import build_custom_cnn

def main():
    cfg = Config()
    set_seed(cfg.seed)

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--img-size", type=int, default=cfg.img_size)
    p.add_argument("--batch", type=int, default=cfg.batch_size)
    args = p.parse_args()

    loaders = build_train_val_loaders(cfg.data_raw_dir, args.img_size, args.batch, cfg.seed)
    num_classes = len(loaders.class_names)

    model = build_custom_cnn((args.img_size, args.img_size, 3), num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    out_dir = cfg.models_dir / "custom_cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    model.fit(loaders.train, validation_data=loaders.val, epochs=args.epochs, callbacks=callbacks)
    print("✅ Saved best model to:", ckpt_path)

if __name__ == "__main__":
    main()