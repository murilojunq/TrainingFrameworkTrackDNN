import tensorflow as tf

class OneHotLayer(tf.keras.layers.Layer):
    '''
    A custom layer class that takes one input and one-hot encodes it into a vector of length n_categories
    '''

    def __init__(self, **kwargs):
        super(OneHotLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OneHotLayer, self).build(input_shape)

    def call(self, input):
        input = tf.cast(input, dtype=tf.int32)
        hot_encoded = tf.one_hot(input, 25)
        hot_encoded = tf.reshape(hot_encoded, ([-1, 25]))
        hot_encoded = tf.cast(hot_encoded, dtype=tf.float32)
        return hot_encoded

    # def get_config(self):
    #     return {'n_categories': self._n_categories}