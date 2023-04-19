# import torch
# import torch.nn as nn
# from torch.nn import init
# import torch.nn.functional as F

import tensorflow as tf


class Social_Encoder(tf.keras.layers.Layer): #layers.Layer?

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        # self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #
        # NOTE: torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.linear1 = tf.Dense(self.embed_dim, input_shape=2 * self.embed_dim)

    def call(self, nodes):

        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])
        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network

        # self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = tf.Variable(nodes, dtype=tf.int16)  #TODO: unsure about this
        self_feats = self_feats.T #transpose
        
        # self-connection could be considered.
        combined = tf.concat([self_feats, neigh_feats], dim=1)
        combined = tf.nn.relu(self.linear1(combined))

        return combined
