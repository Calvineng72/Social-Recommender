import numpy as np
import random
from Attention import Attention
import tensorflow as tf

class Social_Aggregator(tf.keras.layers.Layer): #TODO: or is it layers.Layer
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def call(self, nodes, to_neighs):
        # embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        # arguments to torch.empty: 
        # size (int...) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
        # out (Tensor, optional) – the output tensor.
        # dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_tensor_type()).

        #alternatively, use with tf.device("/gpu=0"): above this line to set the device
        embed_matrix = tf.Tensor(tf.zeros(len(nodes), self.embed_dim, dtype=tf.float32), device="cuda")
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            # 
            e_u = self.u2e.weight[list(tmp_adj)] # fast: user embedding 
            #slow: item-space user latent factor (item aggregation)
            #feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            #e_u = torch.t(feature_neigbhors)

            u_rep = self.u2e.weight[nodes[i]]

            att_w = self.att(e_u, u_rep, num_neighs)
            att_history = tf.matmul(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats
