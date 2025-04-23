import tensorflow as tf


def build_encoder(inp):
    '''
    inp must be a keras tensor 
    '''
    e1 = tf.keras.layers.Conv2D(32, 3, padding="same", name="Encoder-Start",
                                activation="relu")(inp)
    e2 = tf.keras.layers.MaxPooling2D()(e1)
    e3 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(e2)
    e4 = tf.keras.layers.MaxPooling2D()(e3)
    e5 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(e4)
    e6 = tf.keras.layers.MaxPooling2D()(e5)
    e7 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(e6)
    e8 = tf.keras.layers.MaxPooling2D()(e7)
    e9 = tf.keras.layers.Conv2D(2, 3, padding="same", activation="relu")(e8)
    e10 = tf.keras.layers.Flatten()(e9)
    return tf.keras.layers.Dense(80, activation="relu", name="Encoder-End")(e10)


def build_decoder(inp):
    '''
    inp must be a keras tensor 
    '''
    bd1 = tf.keras.layers.Dense(
        220, activation="relu", name="Decoder-Start")(inp)
    rebuilt = tf.keras.layers.Reshape((10, 11, 2))(bd1)
    d1 = tf.keras.layers.Conv2D(
        256, 3, activation='relu', padding='same')(rebuilt)
    d2 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2))(d1)
    d3 = tf.keras.layers.Conv2D(
        128, 3, activation='relu', padding='same')(d2)
    d4 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2))(d3)
    d5 = tf.keras.layers.Conv2D(
        64, 3, activation='relu', padding='same')(d4)
    d6 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2))(d5)
    d7 = tf.keras.layers.Conv2D(
        32, 3, activation='relu', padding='same')(d6)
    d8 = tf.keras.layers.Conv2DTranspose(32, 2, strides=(2, 2))(d7)
    return tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same",
                                  name="Decoder-End")(d8)


def build(inp_shape):
    '''
    Builds Auto-Encoder

    Parameters:
    - inp_shape (tuple): (height, width, channels)

    Returns:
    - Keras model
    '''
    inp = tf.keras.Input(shape=inp_shape)
    b = build_encoder(inp)

    outp = build_decoder(b)

    return tf.keras.models.Model(inp, outp, name="Reconstruct")
