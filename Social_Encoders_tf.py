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
        self.linear1 = tf.keras.layers.Dense(self.embed_dim, name="SocEnc_D1")

    def call(self, nodes, training):

        to_neighs = []
        with tf.GradientTape() as tape:
            for node in nodes:
                to_neighs.append(self.social_adj_lists[int(node)])
            neigh_feats = self.aggregator.call(nodes, to_neighs, training) 

            self_feats = self.features(tf.convert_to_tensor(nodes))  
            self_feats = tf.transpose(self_feats) 
            
            combined = tf.concat([self_feats, neigh_feats], axis=1)
            combined = tf.nn.relu(self.linear1(combined))
            tape.watch(combined)

        return combined
