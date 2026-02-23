import tensorflow as tf

def build_mobilenet_transfer(input_shape=(224, 224, 3), num_classes=6, trainable_base=False) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=input_shape)

    x = (inputs * 2.0) - 1.0   

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")