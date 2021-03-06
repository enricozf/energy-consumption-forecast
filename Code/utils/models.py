import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, InputLayer, Dense, LSTM, GRU, Conv1D, Dropout, Flatten, 
    Bidirectional, TimeDistributed, GlobalAveragePooling1D, Add
)
from tensorflow.keras import activations
from tensorflow.keras.models import Model

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size=4, num_patches=6, *args, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)

    def call(self, sequences):
        patches = tf.image.extract_patches(
            images=tf.expand_dims(sequences, axis=-1),
            sizes=[1,self.patch_size, self.patch_size,1],
            strides=[1,self.patch_size,self.patch_size,1],
            rates=[1,1,1,1],
            padding='VALID'
        )
        # patches = tf.squeeze(patches, axis=2)
        patches = tf.reshape(patches, [-1,self.num_patches,self.patch_size**2])
        return patches
    
    def get_config(self):
        config = super(Patches, self).get_config()
        config.update(
            {
                "patch_size" : self.patch_size,
                "num_patches" : self.num_patches
            }
        )
        return config

class MLPMixerLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_patches=6, hidden_units=16, dropout_rate=0, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate

    # def build(self, ):
        self.mlp1 = tf.keras.Sequential(
            [
            InputLayer(input_shape=(hidden_units, num_patches)),
            TimeDistributed(Dense(units=num_patches, activation='gelu')),
            TimeDistributed(Dense(units=num_patches)),
            Dropout(rate=self.dropout_rate)
            ]
        )

        self.mlp2 = tf.keras.Sequential(
            [
            InputLayer(input_shape=(num_patches, hidden_units)),
            TimeDistributed(Dense(units=num_patches, activation='gelu')),
            TimeDistributed(Dense(units=hidden_units)),
            Dropout(rate=self.dropout_rate)
            ]
        )
        self.normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x

    def get_config(self):
        config = super(MLPMixerLayer, self).get_config()
        config.update(
            {
                "mlp1" : self.mlp1,
                "mlp2" : self.mlp2,
                "normalize" : self.normalize
            }
        )
        return config

class GatedActivationUnit(tf.keras.layers.Layer):
    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)

    def call(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "activation" : self.activation
        })

        return config

def wavenet_residual_block(inputs, n_filters, dilation_rate):
    z = Conv1D(2 * n_filters, kernel_size=2, padding="causal",
                dilation_rate=dilation_rate)(inputs)
    z = GatedActivationUnit()(z)
    z = Conv1D(n_filters, kernel_size=1)(z)
    return Add()([z, inputs]), z