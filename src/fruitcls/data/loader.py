from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf

@dataclass
class DataLoaders:
    train: tf.data.Dataset
    val: tf.data.Dataset
    class_names: list

def _augment():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomBrightness(0.15),
    ], name="augment")

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

    norm = tf.keras.layers.Rescaling(1.0 / 255)
    aug = _augment()

    train_ds = train_ds.map(lambda x, y: (aug(norm(x), training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return DataLoaders(train=train_ds, val=val_ds, class_names=train_ds.class_names)

def build_test_loader(
    test_dir: Path,
    img_size: int,
    batch_size: int,
):
    test_dir = Path(test_dir)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    test_ds = test_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return test_ds.prefetch(tf.data.AUTOTUNE), test_ds.class_names