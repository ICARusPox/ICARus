"""
To be run with TensorFlow 2.16.
"""

import tensorflow as tf

intermediate_model_path = "mcapst5_sled_epoch_20_converted"

# Load SavedModel
loaded = tf.saved_model.load(intermediate_model_path)
infer = loaded.signatures["serving_default"]

# Custom wrapper layer
@tf.keras.utils.register_keras_serializable()
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self._loaded = None
        self._infer = None

    def build(self, input_shape):
        # Load SavedModel lazily (important for serialization)
        self._loaded = tf.saved_model.load(self.model_path)
        self._infer = self._loaded.signatures["serving_default"]

    def call(self, inputs):
        seq1, seq2 = inputs
        outputs = self._infer(seq1=seq1, seq2=seq2)
        return list(outputs.values())[0]  # extract tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_path": self.model_path
        })
        return config

# Define inputs
input_1 = tf.keras.Input(shape=(1200, 1024), name="seq1")
input_2 = tf.keras.Input(shape=(1200, 1024), name="seq2")

# Use custom layer
outputs = MyLayer(intermediate_model_path)([input_1, input_2])

model = tf.keras.Model(inputs=[input_1, input_2], outputs=outputs)

model.save("mcapst5_sled_epoch_20.keras")