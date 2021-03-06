"""Define the self-attention encoder."""

import tensorflow as tf

from opennmt.layers import transformer

from opennmt.encoders.encoder import Encoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers import common


class SelfAttentionEncoder(Encoder):
  """Encoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               maximum_relative_position=None,
               **kwargs):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      ffn_dropout: The probability to drop units from the activation output in
        the feed forward layer.
      ffn_activation: The activation function to apply between the two linear
        transformations of the feed forward layer.
      position_encoder_class: The :class:`opennmt.layers.PositionEncoder`
        class to use for position encoding (or a callable that returns an
        instance).
      maximum_relative_position: Maximum relative position representation
        (from https://arxiv.org/abs/1803.02155).
      **kwargs: Additional layer arguments.
    """
    super(SelfAttentionEncoder, self).__init__(**kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            maximum_relative_position=maximum_relative_position)
        for i in range(num_layers)]
    self.num_heads= num_heads

  def call(self, inputs, sequence_length=None, training=None,  return_attn=False, inject=None):
    inputs *= self.num_units**0.5
    attn_list = []
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i,layer in enumerate(self.layers):
      inject_i = (inject[0][i,:,:,:], inject[1][i,:,:,:]) if inject is not None else None
      inputs, attn = layer(inputs, mask=mask, training=training,
                           return_attn=return_attn, inject=inject_i)
      if return_attn:
        attn_list.append(attn)
    attention = tf.concat(attn_list, axis=0, name="Attention") if return_attn else None
    outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length, attention

  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
