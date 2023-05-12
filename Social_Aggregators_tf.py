import numpy as np
import random
from Attention_tf import Attention
import tensorflow as tf

class Social_Aggregator(tf.keras.layers.Layer): 
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="/device:CPU:0"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def call(self, nodes, to_neighs, training):
        embed_matrix = np.zeros((len(nodes), self.embed_dim))

        listTensors = []
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            with tf.GradientTape() as tape:
                e_u = self.u2e(np.array(list(tmp_adj)).astype('float32')) # fast: user embedding 
                u_rep = self.u2e(np.array(nodes[i]))

                att_w = self.att.call(e_u, u_rep, num_neighs, training)
                att_history = tf.transpose(tf.matmul(tf.transpose(e_u), att_w))
                embed_matrix[i] = att_history.numpy()
            listTensors.append(att_history)
        to_feats = embed_matrix
        to_feats = tf.stack(listTensors)

        return tf.squeeze(to_feats)
