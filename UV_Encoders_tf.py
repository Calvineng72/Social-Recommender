# import torch
# import torch.nn as nn
# from torch.nn import init
# import torch.nn.functional as F
import tensorflow as tf


class UV_Encoder(tf.keras.layers.Layer):

    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists, aggregator, cuda="/device:CPU:0", uv=True):
        super(UV_Encoder, self).__init__()

        self.features = features
        self.uv = uv
        self.history_uv_lists = history_uv_lists
        self.history_r_lists = history_r_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = tf.keras.layers.Dense(self.embed_dim, name="UVEnc_D1")

    def call(self, nodes, training):
        tmp_history_uv = []
        tmp_history_r = []
        with tf.GradientTape() as tape:
            for node in nodes:
                tmp_history_uv.append(self.history_uv_lists[int(node)])
                tmp_history_r.append(self.history_r_lists[int(node)])

            neigh_feats = self.aggregator.call(nodes, tmp_history_uv, tmp_history_r, training)  # user-item network

            self_feats = self.features(nodes)
            combined = tf.concat([self_feats, neigh_feats], axis=1)
            combined = tf.nn.relu(self.linear1(combined))
            tape.watch(combined)

            return combined
