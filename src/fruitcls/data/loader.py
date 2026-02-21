from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class DataLoaders:
    train: tf.data.Dataset
    val: tf.data.Dataset
    class_names: List[str]


def _augment() -> tf.keras.Model:
    # augment samo za train
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomBrightness(0.15),
        ],
        name="augment",
    )


def build_train_val_loaders(
    train_dir: Path,
    img_size: int,
    batch_size: int,
    seed: int,
    val_split: float = 0.2,
) -> DataLoaders:
    train_dir = Path(train_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)

    norm = tf.keras.layers.Rescaling(1.0 / 255)
    aug = _augment()

    train_ds = train_ds.map(
        lambda x, y: (aug(norm(x), training=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (norm(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return DataLoaders(train=train_ds, val=val_ds, class_names=class_names)


def build_test_loader(
    test_dir: Path,
    img_size: int,
    batch_size: int,
) -> Tuple[tf.data.Dataset, List[str]]:
    test_dir = Path(test_dir)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(test_ds.class_names)

    norm = tf.keras.layers.Rescaling(1.0 / 255)
    test_ds = test_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return test_ds, class_names