# import torch
# import torch.nn as nn
# from torch.nn import init
# from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders_tf import UV_Encoder
from UV_Aggregators_tf import UV_Aggregator
from Social_Encoders_tf import Social_Encoder
from Social_Aggregators_tf import Social_Aggregator
# import torch.nn.functional as F
# import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os

import tensorflow as tf

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(tf.keras.Model):

    def __init__(self, num_users, embed_dim, num_items, num_ratings, device, 
                 history_u_lists, history_ur_lists, history_v_lists, 
                 history_vr_lists, social_adj_lists):
        super(GraphRec, self).__init__()
        # self.num_users = num_users
        # self.embed_dim 


        # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
        # u2e = nn.Embedding(num_users, embed_dim).to(device)
        self.u2e = tf.keras.layers.Embedding(num_users, embed_dim)
        # v2e = nn.Embedding(num_items, embed_dim).to(device)
        self.v2e = tf.keras.layers.Embedding(num_items, embed_dim)
        # r2e = nn.Embedding(num_ratings, embed_dim).to(device)
        self.r2e = tf.keras.layers.Embedding(num_ratings, embed_dim)

        # user feature
        # features: item * rating
        self.agg_u_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, embed_dim, cuda=device, uv=True)
        self.enc_u_history = UV_Encoder(self.u2e, embed_dim, history_u_lists, history_ur_lists, self.agg_u_history, cuda=device, uv=True)
        # neighobrs
        self.agg_u_social = Social_Aggregator(lambda nodes: tf.transpose(self.enc_u_history.call(nodes)), self.u2e, embed_dim, cuda=device)
        self.enc_u = Social_Encoder(lambda nodes: tf.transpose(self.enc_u_history.call(nodes)), embed_dim, social_adj_lists, self.agg_u_social,
                            base_model=self.enc_u_history, cuda=device)

        # item feature: user * rating
        self.agg_v_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, embed_dim, cuda=device, uv=False)
        self.enc_v_history = UV_Encoder(self.v2e, embed_dim, history_v_lists, history_vr_lists, self.agg_v_history, cuda=device, uv=False)

        # self.enc_u = enc_u
        # self.enc_v_history = enc_v_history
        self.embed_dim = self.enc_u.embed_dim

        # self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur1 = tf.keras.layers.Dense(self.embed_dim)
        # self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = tf.keras.layers.Dense(self.embed_dim)
        # self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = tf.keras.layers.Dense(self.embed_dim)
        # self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = tf.keras.layers.Dense(self.embed_dim)
        # self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv1 = tf.keras.layers.Dense(self.embed_dim)
        # self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv2 = tf.keras.layers.Dense(16)
        # self.w_uv3 = nn.Linear(16, 1)
        self.w_uv3 = tf.keras.layers.Dense(1)
        # self.r2e = r2e
        # torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        # self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.5)
        # self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.5)
        # self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.5)
        # self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.5)
        # self.criterion = nn.MSELoss()
        self.criterion = tf.keras.losses.MeanSquaredError()

    def call(self, nodes_u, nodes_v, call_training):
        embeds_u = self.enc_u.call(nodes_u, call_training)
        embeds_v = self.enc_v_history.call(nodes_v, call_training)

        # x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = tf.nn.relu(self.bn1(self.w_ur1(embeds_u)))
        # x_u = F.dropout(x_u, training=self.training)
        x_u = tf.keras.layers.Dropout(rate=0.5)(x_u, training=call_training)
        x_u = self.w_ur2(x_u)


        
        # x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = tf.nn.relu(self.bn2(self.w_vr1(embeds_v)))
        # x_v = F.dropout(x_v, training=self.training)
        x_v = tf.keras.layers.Dropout(rate=0.5)(x_v, training=call_training)
        x_v = self.w_vr2(x_v)

        # x_uv = torch.cat((x_u, x_v), 1)
        x_uv = tf.concat((x_u, x_v), 1)
        # x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = tf.nn.relu(self.bn3(self.w_uv1(x_uv)))
        # x = F.dropout(x, training=self.training)
        x = tf.keras.layers.Dropout(rate=0.5)(x, training=call_training)
        # x = F.relu(self.bn4(self.w_uv2(x)))
        x = tf.nn.relu(self.bn4(self.w_uv2(x)))
        # x = F.dropout(x, training=self.training)
        x = tf.keras.layers.Dropout(rate=0.5)(x, training=call_training)
        scores = self.w_uv3(x)
        return tf.squeeze(scores)

    # def loss(self, nodes_u, nodes_v, labels_list, training):
    #     scores = self.call(nodes_u, nodes_v, training)
    #     return self.criterion(scores, labels_list)
    def loss(self, scores, labels_list):
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, training=True):
    # model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        with tf.GradientTape() as tape:
            # deleted the to device stufff
            scores = model.call(batch_nodes_u, batch_nodes_v, training)
            loss = model.loss(scores, labels_list)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        running_loss += loss
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0

        # optimizer.zero_grad()
        # # loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        # loss.backward(retain_graph=True)
        # optimizer.step()
        # running_loss += loss.item()
        # if i % 100 == 0:
        #     print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
        #         epoch, i, running_loss / 100, best_rmse, best_mae))
        #     running_loss = 0.0
    return 0


def test(model, device, test_loader, training=False):
    # model.eval()
    tmp_pred = []
    target = []
    # with torch.no_grad():
    #     for test_u, test_v, tmp_target in test_loader:
    #         test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
    #         val_output = model.forward(test_u, test_v)
    #         tmp_pred.append(list(val_output.data.cpu().numpy()))
    #         target.append(list(tmp_target.data.cpu().numpy()))
    # tmp_pred = np.array(sum(tmp_pred, []))
    # target = np.array(sum(target, []))
    # expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    # mae = mean_absolute_error(tmp_pred, target)
    # return expected_rmse, mae




    #TODO: maybe pass a boolean is_training to call
    for test_u, test_v, tmp_target in test_loader:
        val_output = model.call(test_u, test_v, training)
        tmp_pred.append(list(val_output))
        target.append(list(tmp_target))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


#TODO: changed default epochs from 100

def main():
    # Training settings #TODO og batch size was 128, making it 32
    # embed dim was 64 making it 32
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=32, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    #TODO: set device with TF
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # use_cuda = False
    # if tf.test.is_gpu_available():
    #     use_cuda = True
    use_cuda = True
    device = tf.device("/GPU:0" if use_cuda else "/device:CPU:0")

    embed_dim = args.embed_dim
    dir_data = './data/toy_dataset'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    
    # please add the validation set
    
    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    # trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
    #                                           torch.FloatTensor(train_r))
    # print(tf.convert_to_tensor(train_r).dtype)
    trainset = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(train_u, dtype=tf.float32), tf.convert_to_tensor(train_v, dtype=tf.float32), tf.convert_to_tensor(train_r, dtype=tf.float32)])
    # testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
    #                                          torch.FloatTensor(test_r))
    testset = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(test_u, dtype=tf.float32), tf.convert_to_tensor(test_v, dtype=tf.float32), tf.convert_to_tensor(test_r, dtype=tf.float32)])

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    train_loader = trainset.batch(args.batch_size)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = testset.batch(args.test_batch_size)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    # print(num_users)

#TODO: 
    # # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
    # # u2e = nn.Embedding(num_users, embed_dim).to(device)
    # u2e = tf.keras.layers.Embedding(num_users, embed_dim)
    # # v2e = nn.Embedding(num_items, embed_dim).to(device)
    # v2e = tf.keras.layers.Embedding(num_items, embed_dim)
    # # r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    # r2e = tf.keras.layers.Embedding(num_ratings, embed_dim)

    # # user feature
    # # features: item * rating
    # agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    # enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # # neighobrs
    # agg_u_social = Social_Aggregator(lambda nodes: tf.transpose(enc_u_history(nodes)), u2e, embed_dim, cuda=device)
    # enc_u = Social_Encoder(lambda nodes: tf.transpose(enc_u_history(nodes)), embed_dim, social_adj_lists, agg_u_social,
    #                        base_model=enc_u_history, cuda=device)

    # # item feature: user * rating
    # agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    # enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    # graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    graphrec = GraphRec(num_users, embed_dim, num_items, num_ratings, device, history_u_lists, 
                        history_ur_lists, history_v_lists, history_vr_lists, 
                        social_adj_lists)
    # optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr, epsilon=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    # graphrec.build(tf.shape(train_loader))
    # print(graphrec.summary())

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae, training=True)

        #TODO: move testing outside???
        expected_rmse, mae = test(graphrec, device, test_loader, training=False)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break


if __name__ == "__main__":
    main()
