import tensorflow as tf
import math

class Bilinear(tf.keras.layers.Layer):
  """A Keras Layer makes bilinear interaction of two vectors.
  This Keras Layer implements the bilinear interaction of two vectors of
  embedding dimensions. The bilinear, linear and scalar parameters of the
  interaction are trainable.
  The bilinear interaction are used in the work "Revisiting two tower models
  for unbiased learning to rank" by Yan et al, see
  https://research.google/pubs/pub51296/.
  In this work, the bilinear interaction appears to be helpful in model the
  complex interaction between position and relevance in unbiased LTR.
  """

  def __init__(self, embedding_dim: int, output_dim: int, **kwargs):
    """Initializer.
    Args:
      embedding_dim: An integer that indicates the embedding dimension of the
        interacting vectors.
      output_dim: An integer that indicates the output dimension of the layer.
      **kwargs: A dict of keyword arguments for the tf.keras.layers.Layer.
    """
    super().__init__(**kwargs)
    self._embedding_dim = embedding_dim
    self._output_dim = output_dim

  def build(self, input_shape: tf.TensorShape):
    """See tf.keras.layers.Layer."""
    # Create a trainable weight variable for this layer.
    self._bilinear_weight = self.add_weight(
        name='bilinear_term',
        shape=(self._embedding_dim, self._embedding_dim, self._output_dim),
        initializer=tf.keras.initializers.RandomNormal(stddev=1. /
                                                       self._embedding_dim),
        trainable=True)
    self._linear_weight_1 = self.add_weight(
        name='linear_term_1',
        shape=(self._embedding_dim, self._output_dim),
        initializer=tf.keras.initializers.RandomNormal(
            stddev=1. / math.sqrt(self._embedding_dim)),
        trainable=True)
    self._linear_weight_2 = self.add_weight(
        name='linear_term_2',
        shape=(self._embedding_dim, self._output_dim),
        initializer=tf.keras.initializers.RandomNormal(
            stddev=1. / math.sqrt(self._embedding_dim)),
        trainable=True)
    self._bias = self.add_weight(
        name='const_term',
        shape=(self._output_dim),
        initializer=tf.keras.initializers.Zeros(),
        trainable=True)
    super().build(input_shape)

  def call(self, inputs: tuple[tf.Tensor]) -> tf.Tensor:
    """Computes bilinear interaction between two vector tensors.
    Args:
      inputs: A pair of tensors of the same shape [batch_size, embedding_dim].
    Returns:
      A tensor, of shape [batch_size, output_dim], computed by the bilinear
      interaction.
    """
    # Input of the function must be a list of two tensors.
    vec_1, vec_2 = inputs
    return tf.einsum(
        'bi,ijk,bj->bk', vec_1, self._bilinear_weight, vec_2) + tf.einsum(
            'bi,ik->bk', vec_1, self._linear_weight_1) + tf.einsum(
                'bi,ik->bk', vec_2, self._linear_weight_2) + self._bias

  def compute_output_shape(self, input_shape: tf.TensorShape):
    """See tf.keras.layers.Layer."""
    return (input_shape[0], self._output_dim)

  def get_config(self):
    """See tf.keras.layers.Layer."""
    config = super().get_config()
    config.update({
        'embedding_dim': self._embedding_dim,
        'output_dim': self._output_dim
    })
    return config