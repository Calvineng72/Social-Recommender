import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders_tf import UV_Encoder
from UV_Aggregators_tf import UV_Aggregator
from Social_Encoders_tf import Social_Encoder
from Social_Aggregators_tf import Social_Aggregator
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

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = tf.keras.layers.Dense(self.embed_dim)
        self.w_ur2 = tf.keras.layers.Dense(self.embed_dim)
        self.w_vr1 = tf.keras.layers.Dense(self.embed_dim)
        self.w_vr2 = tf.keras.layers.Dense(self.embed_dim)
        self.w_uv1 = tf.keras.layers.Dense(self.embed_dim)
        self.w_uv2 = tf.keras.layers.Dense(16)
        self.w_uv3 = tf.keras.layers.Dense(1)
        self.r2e = r2e
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.criterion = tf.keras.losses.MeanSquaredError()

    # def compile(self, optimizer, loss, metrics):
    #     self.optimizer = optimizer
    #     # self.compiled_loss = loss 
        # self.accuracy_function = metrics[0]

    def call(self, nodes_u, nodes_v, call_training):
        embeds_u = self.enc_u.call(nodes_u, call_training)
        embeds_v = self.enc_v_history.call(nodes_v, call_training)

        x_u = tf.nn.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = tf.keras.layers.Dropout(rate=0.5)(x_u, training=call_training)
        x_u = self.w_ur2(x_u)
        x_v = tf.nn.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = tf.keras.layers.Dropout(rate=0.5)(x_v, training=call_training)
        x_v = self.w_vr2(x_v)

        x_uv = tf.concat((x_u, x_v), 1)
        x = tf.nn.relu(self.bn3(self.w_uv1(x_uv)))
        x = tf.keras.layers.Dropout(rate=0.5)(x, training=call_training)
        x = tf.nn.relu(self.bn4(self.w_uv2(x)))
        x = tf.keras.layers.Dropout(rate=0.5)(x, training=call_training)
        scores = self.w_uv3(x)
        return tf.squeeze(scores)



    def train_step(self, device, train_loader, optimizer, epoch, best_rmse, best_mae, training=True):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            batch_nodes_u, batch_nodes_v, labels_list = data
            with tf.GradientTape() as tape:
                scores = self.call(batch_nodes_u, batch_nodes_v, training)
                # loss = self.loss_func(scores, labels_list)
                loss = self.compiled_loss(scores, labels_list)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            running_loss += loss
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                    epoch, i, running_loss / 100, best_rmse, best_mae))
                running_loss = 0.0
        # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(labels_list, scores)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}


    def test_step(self, device, test_loader, training=False):
        tmp_pred = []
        target = []
        for test_u, test_v, tmp_target in test_loader:
            val_output = self.call(test_u, test_v, training)
            tmp_pred.append(list(val_output))
            target.append(list(tmp_target))
        tmp_pred = np.array(sum(tmp_pred, []))
        target = np.array(sum(target, []))
        expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
        mae = mean_absolute_error(tmp_pred, target)
        return expected_rmse, mae
    
def loss_func(self, nodes_u, nodes_v, labels_list, training):
    scores = self.call(nodes_u, nodes_v, training)
    return self.criterion(scores, labels_list)

def my_loss_func(scores, labels_list):
    # scores = self.call(nodes_u, nodes_v, training)
    return tf.keras.losses.MeanSquaredError(scores, labels_list)

#TODO: changed default epochs from 100

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    #TODO: set device with TF
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # use_cuda = False
    # if tf.test.is_gpu_available():
    #     use_cuda = True
    use_cuda = False
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

    trainset = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(train_u, dtype=tf.float32), tf.convert_to_tensor(train_v, dtype=tf.float32), tf.convert_to_tensor(train_r, dtype=tf.float32)])
    testset = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(test_u, dtype=tf.float32), tf.convert_to_tensor(test_v, dtype=tf.float32), tf.convert_to_tensor(test_r, dtype=tf.float32)])

    train_loader = trainset.batch(args.batch_size, deterministic=False)
    test_loader = testset.batch(args.test_batch_size, deterministic=False)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = tf.keras.layers.Embedding(num_users, embed_dim)
    v2e = tf.keras.layers.Embedding(num_items, embed_dim)
    r2e = tf.keras.layers.Embedding(num_ratings, embed_dim)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: tf.transpose(enc_u_history(nodes)), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: tf.transpose(enc_u_history(nodes)), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e)
    optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=args.lr, epsilon=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    graphrec.compile(
        optimizer=optimizer, 
        loss=my_loss_func,
        metrics=tf.keras.metrics.MeanSquaredError
    )

    graphrec.fit(
        train_loader,
        epochs = args.epochs,
        batch_size = args.batch_size,
        validation_data={test_loader}
    )

    # for epoch in range(1, args.epochs + 1):

    #     train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae, training=True)
    #     expected_rmse, mae = test(graphrec, device, test_loader, training=False)
    #     # please add the validation set to tune the hyper-parameters based on your datasets.

    #     # early stopping (no validation set in toy dataset)
    #     if best_rmse > expected_rmse:
    #         best_rmse = expected_rmse
    #         best_mae = mae
    #         endure_count = 0
    #     else:
    #         endure_count += 1
    #     print("TESTING SET!!!!")
    #     print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

    #     if endure_count > 5:
    #         break


if __name__ == "__main__":
    main()
