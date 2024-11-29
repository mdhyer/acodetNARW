"""
Working to implement Shiu pipeline correctly
"""
import tensorflow as tf


class LinearSpecTFLayer(tf.keras.layers.Layer):
    """
    Austin's tensorflow spectrogram layer. Shiu paper uses 128 ms Hann window with 50 ms advance

    When using 1 kHz sampling rate audio, 128 ms = 128 frames, 50 ms = 50 frames
    """

    def __init__(self, frame_step=42,  # 50,  # 280,  # frame_step=374,
                 frame_length=256,  # 128,  # 512,
                 kernel=None, data_format='channels_last', dtype=tf.float64, **kwargs):
        super(LinearSpecTFLayer, self).__init__(**kwargs)
        self.frame_step = frame_step
        self.frame_length = frame_length
        # self.kernel = tf.constant(kernel, dtype=tf.float32)  # tf.convert_to_tensor or variable, too
        self.data_format = data_format

    def call(self, inputs, training=None):
        # Perform STFT
        spec = tf.signal.stft(inputs,
                              self.frame_length,
                              self.frame_step,
                              fft_length=self.frame_length,
                              window_fn=tf.signal.hann_window,
                              pad_end=False)  # Results in correct length

        # Crop to 64x64 and prepare spectrogram
        spec = spec[:64, 13:77]  # REAL VERSION (frame_length=256, frame_step=42)

        spec = tf.square(spec)

        spec = tf.abs(spec)

        # Pseudo-db scale
        spec = tf.math.log(spec + 1e-6)

        spec = tf.expand_dims(spec, -1)

        return spec

class BatchedLinearSpecTFLayer(tf.keras.layers.Layer):
    """
    Austin's tensorflow spectrogram layer. Shiu paper uses 128 ms Hann window with 50 ms advance

    When using 1 kHz sampling rate audio, 128 ms = 128 frames, 50 ms = 50 frames
    """

    def __init__(self, frame_step=42,  # 50,  # 280,  # frame_step=374,
                 frame_length=256,  # 128,  # 512,
                 kernel=None, data_format='channels_last', **kwargs):
        super(BatchedLinearSpecTFLayer, self).__init__(**kwargs)
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.data_format = data_format

    def call(self, inputs, training=None):
        # Perform STFT
        spec = tf.signal.stft(tf.squeeze(inputs,axis=-1),  # inputs,
                              self.frame_length,
                              self.frame_step,
                              fft_length=self.frame_length,
                              window_fn=tf.signal.hann_window,
                              pad_end=False)  # Results in correct length

        # Crop and prepare spectrogram
        spec = spec[:, :64, 13:77]

        spec = tf.square(spec)

        spec = tf.abs(spec)

        # Pseudo-db scale
        spec = tf.math.log(spec + 1e-6)

        spec = tf.expand_dims(spec, -1)

        return spec

class SpecVariable(tf.keras.layers.Layer):
    """
    Austin's tensorflow spectrogram layer. Shiu paper uses 128 ms Hann window with 50 ms advance

    When using 1 kHz sampling rate audio, 128 ms = 128 frames, 50 ms = 50 frames
    """

    def __init__(self, frame_step=42,
                 frame_length=256,
                 kernel=None, data_format='channels_last', dtype=tf.float64, **kwargs):
        super(SpecVariable, self).__init__(**kwargs)
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.data_format = data_format

    def call(self, inputs, training=None):
        # Perform STFT
        spec = tf.signal.stft(inputs,
                              self.frame_length,
                              self.frame_step,
                              fft_length=self.frame_length,
                              window_fn=tf.signal.hann_window,
                              pad_end=False)  # Results in correct length

        spec = spec[:, 13:77]

        # Skipping magnitude to power conversion because rescaling with batchnorm
        spec = tf.square(spec)

        spec = tf.abs(spec)

        # Pseudo-db scale
        spec = tf.math.log(spec + 1e-6)

        spec = tf.expand_dims(spec, -1)

        return spec
