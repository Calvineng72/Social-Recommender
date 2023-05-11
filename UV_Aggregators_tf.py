# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
import numpy as np
import random
from Attention_tf import Attention
import tensorflow as tf


class UV_Aggregator(tf.keras.layers.Layer):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="/device:CPU:0", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = tf.keras.layers.Dense(self.embed_dim, name="UVAgg_D1")
        self.w_r2 = tf.keras.layers.Dense(self.embed_dim, name="UVAgg_D2")
        self.att = Attention(self.embed_dim)

    def call(self, nodes, history_uv, history_r, training):
        
        embed_matrix = tf.Variable(tf.zeros((len(history_uv), self.embed_dim)))
        with tf.GradientTape() as tape:
            for i in range(len(history_uv)):
                history = history_uv[i]
                num_history_item = len(history)
                tmp_label = history_r[i]
                
                if self.uv == True:
                    # user component
                    e_uv = self.v2e(tf.convert_to_tensor(history))
                    uv_rep = self.u2e(nodes[i])
                else:
                    # item component
                    e_uv = self.u2e(tf.convert_to_tensor(history))
                    uv_rep = self.v2e(nodes[i])

                e_r = self.r2e(tf.convert_to_tensor(tmp_label))
                x = tf.concat((e_uv, e_r), 1)
                x = tf.nn.relu(self.w_r1(x))
                tape.watch(x)
                o_history = tf.nn.relu(self.w_r2(x))

                att_w = self.att.call(o_history, uv_rep, num_history_item, training)
                att_history = tf.matmul(tf.transpose(o_history), att_w)
                att_history = tf.transpose(att_history)
                tape.watch(att_history)

                embed_matrix = embed_matrix[i].assign(tf.squeeze(att_history))
                tape.watch(embed_matrix)

        to_feats = embed_matrix
        return to_feats
