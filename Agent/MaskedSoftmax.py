import tensorflow as tf

class MaskedSoftmax(tf.keras.layers.Layer):
  def call(self, inputLayer, mask):
    mask = tf.where(tf.equal(mask, 1))
    return tf.sparse.to_dense(
      tf.sparse.softmax(
        tf.sparse.SparseTensor(
          indices=mask,
          values=tf.gather_nd(inputLayer, mask),
          dense_shape=tf.shape(inputLayer, out_type=tf.int64)
        )
      )
    )
