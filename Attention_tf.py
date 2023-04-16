import tensorflow as tf
from tensorflow.keras import layers


class Attention(layers.Layer):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = layers.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = layers.Dense(self.embed_dim)
        self.att2 = layers.Dense(self.embed_dim)
        self.att3 = layers.Dense(1)
        self.softmax = layers.Softmax(axis=0)

    def call(self, node1, u_rep, num_neighs, training=None):
        uv_reps = tf.repeat(u_rep, num_neighs, axis=0)
        x = tf.concat([node1, uv_reps], axis=1)
        x = tf.nn.relu(self.att1(x))
        x = layers.Dropout(0.5)(x, training=training)
        x = tf.nn.relu(self.att2(x))
        x = layers.Dropout(0.5)(x, training=training)
        x = self.att3(x)
        att = tf.nn.softmax(x, axis=0)
        return att
