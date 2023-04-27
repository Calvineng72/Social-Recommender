import numpy as np
import random
from Attention_tf import Attention
import tensorflow as tf

class Social_Aggregator(tf.keras.layers.Layer): #TODO: or is it layers.Layer
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
        # embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        # arguments to torch.empty: 
        # size (int...) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
        # out (Tensor, optional) – the output tensor.
        # dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_tensor_type()).

        #alternatively, use with tf.device("/gpu=0"): above this line to set the device
        # embed_matrix = tf.Tensor(tf.zeros(len(nodes), self.embed_dim, dtype=tf.float32), device="/device:CPU:0")
        embed_matrix = np.zeros((len(nodes), self.embed_dim))
        for i in range(len(nodes)):
            # print(nodes[i])
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            # 
            # print(type(nodes[i]))
            # print(tmp_adj)
            # print(len(tmp_adj))
            e_u = self.u2e(np.array(list(tmp_adj)).astype('float32')) # fast: user embedding 
            #slow: item-space user latent factor (item aggregation)
            #feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
            #e_u = torch.t(feature_neigbhors)
            u_rep = self.u2e(np.array(nodes[i]))

            att_w = self.att.call(e_u, u_rep, num_neighs, training)
            att_history = tf.transpose(tf.matmul(tf.transpose(e_u), att_w))
            embed_matrix[i] = att_history
        to_feats = embed_matrix
        to_feats = tf.Variable(tf.convert_to_tensor(embed_matrix, dtype=tf.float32), dtype=tf.float32)

        return to_feats
