# import torch
# import torch.nn as nn
# from torch.nn import init
# import torch.nn.functional as F

import tensorflow as tf


class Social_Encoder(tf.keras.layers.Layer): #layers.Layer?

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="/device:CPU:0"):
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
        self.linear1 = tf.keras.layers.Dense(self.embed_dim)

    def call(self, nodes, training):

        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])
        neigh_feats = self.aggregator.call(nodes, to_neighs, training)  # user-user network

        # self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self.features(tf.convert_to_tensor(nodes))  #TODO: unsure about this
        self_feats = tf.transpose(self_feats) #transpose
        
        # self-connection could be considered.
        combined = tf.concat([self_feats, neigh_feats], axis=1)
        combined = tf.nn.relu(self.linear1(combined))

        return combined
