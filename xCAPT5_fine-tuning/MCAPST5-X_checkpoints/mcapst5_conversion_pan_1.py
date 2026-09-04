"""
To be run with TensorFlow 2.13.
"""

import tensorflow as tf

old_model_path = "mcapst5_pan_epoch_20.hdf5"

old_model = tf.keras.models.load_model(old_model_path, compile=False)
# Remove the last layer
old_model = tf.keras.Model(
    inputs=old_model.inputs, outputs=old_model.layers[-2].output
)
old_model.save("mcapst5_pan_epoch_20_converted")