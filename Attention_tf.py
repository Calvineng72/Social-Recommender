import tensorflow as tf
from Bilinear_tf import Bilinear


class Attention(tf.keras.layers.Layer):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = Bilinear(self.embed_dim, 1)
        self.att1 = tf.keras.layers.Dense(self.embed_dim, name="Att_D1")
        self.att2 = tf.keras.layers.Dense(self.embed_dim, name="Att_D2")
        self.att3 = tf.keras.layers.Dense(1, name="Att_D3")
        self.softmax = tf.keras.layers.Softmax(axis=0)

    def call(self, node1, u_rep, num_neighs, att_training):
        with tf.GradientTape() as tape:
            uv_reps = tf.repeat(u_rep, num_neighs)
            x = tf.concat([node1, tf.reshape(uv_reps, tf.shape(node1))], axis=1)
            x = self.bilinear((node1, tf.reshape(uv_reps, tf.shape(node1))))
            x = tf.nn.relu(self.att1(x))
            x = tf.keras.layers.Dropout(0.5)(x, training=att_training)
            x = tf.nn.relu(self.att2(x))
            x = tf.keras.layers.Dropout(0.5)(x, training=att_training)
            x = self.att3(x)
            tape.watch(x)
            att = tf.nn.softmax(x, axis=0)
            return att
