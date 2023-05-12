import pickle
import numpy as np
import time
from collections import defaultdict
from UV_Encoders_tf import UV_Encoder
from UV_Aggregators_tf import UV_Aggregator
from Social_Encoders_tf import Social_Encoder
from Social_Aggregators_tf import Social_Aggregator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf


class GraphRec(tf.keras.Model):

    def __init__(self, num_users, embed_dim, num_items, num_ratings, device, 
                 history_u_lists, history_ur_lists, history_v_lists, 
                 history_vr_lists, social_adj_lists):
        super(GraphRec, self).__init__()


        self.u2e = tf.keras.layers.Embedding(num_users, embed_dim, name="u2e")
        self.v2e = tf.keras.layers.Embedding(num_items, embed_dim, name="v2e")
        self.r2e = tf.keras.layers.Embedding(num_ratings, embed_dim, name="r2e")

        # user feature
        # features: item * rating
        self.agg_u_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, embed_dim, cuda=device, uv=True)
        self.enc_u_history = UV_Encoder(self.u2e, embed_dim, history_u_lists, history_ur_lists, self.agg_u_history, cuda=device, uv=True)
        self.agg_u_social = Social_Aggregator(lambda nodes: tf.transpose(self.enc_u_history(nodes, True)), self.u2e, embed_dim, cuda=device)
        self.enc_u = Social_Encoder(lambda nodes: tf.transpose(self.enc_u_history(nodes, True)), embed_dim, social_adj_lists, self.agg_u_social,
                            base_model=self.enc_u_history, cuda=device)

        # item feature: user * rating
        self.agg_v_history = UV_Aggregator(self.v2e, self.r2e, self.u2e, embed_dim, cuda=device, uv=False)
        self.enc_v_history = UV_Encoder(self.v2e, embed_dim, history_v_lists, history_vr_lists, self.agg_v_history, cuda=device, uv=False)

        self.embed_dim = self.enc_u.embed_dim

        self.w_ur1 = tf.keras.layers.Dense(self.embed_dim)
        self.w_ur2 = tf.keras.layers.Dense(self.embed_dim)
        self.w_vr1 = tf.keras.layers.Dense(self.embed_dim)
        self.w_vr2 = tf.keras.layers.Dense(self.embed_dim)
        self.w_uv1 = tf.keras.layers.Dense(self.embed_dim)
        self.w_uv2 = tf.keras.layers.Dense(16)
        self.w_uv3 = tf.keras.layers.Dense(1)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.criterion = tf.keras.losses.MeanSquaredError()

    def call(self, nodes_u, nodes_v, call_training):
        embeds_u = self.enc_u(nodes_u, call_training)
        embeds_v = self.enc_v_history(nodes_v, call_training)

        x_u = tf.nn.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = tf.keras.layers.Dropout(rate=0.5)(x_u, training=call_training)
        x_u = self.w_ur2(x_u)


        x_v = tf.nn.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = tf.keras.layers.Dropout(rate=0.5)(x_v, training=call_training)
        x_v = self.w_vr2(x_v)

        with tf.GradientTape() as tape:
            x_uv = tf.concat((x_u, x_v), 1)
            x = tf.nn.relu(self.bn3(self.w_uv1(x_uv)))
            x = tf.keras.layers.Dropout(rate=0.5)(x, training=call_training)
            x = tf.nn.relu(self.bn4(self.w_uv2(x)))
            x = tf.keras.layers.Dropout(rate=0.5)(x, training=call_training)
            scores = self.w_uv3(x)
            return tf.squeeze(scores)
        
    def loss(self, scores, labels_list):
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, training=True):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        with tf.GradientTape() as tape:
            scores = model(batch_nodes_u, batch_nodes_v, training)
            loss = model.loss(scores, labels_list)
            print("scores + labels list")
            print(scores)
            print(labels_list)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        running_loss += loss
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0

    return 0


def test(model, device, test_loader, training=False):
    tmp_pred = []
    target = []

    for test_u, test_v, tmp_target in test_loader:
        val_output = model(test_u, test_v, training)
        tmp_pred.append(list(val_output))
        target.append(list(tmp_target))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def main():
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.015, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    use_cuda = True
    device = tf.device("/GPU:0" if use_cuda else "/device:CPU:0")

    embed_dim = args.embed_dim
    dir_data = './data/douban_final'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list, val_u, val_v, val_r = pickle.load(
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
    valset = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(val_u, dtype=tf.float32), tf.convert_to_tensor(val_v, dtype=tf.float32), tf.convert_to_tensor(val_r, dtype=tf.float32)])
    train_loader = trainset.batch(args.batch_size)
    test_loader = testset.batch(args.test_batch_size)
    val_loader = valset.batch(args.test_batch_size)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    graphrec = GraphRec(num_users, embed_dim, num_items, num_ratings, device, history_u_lists, 
                        history_ur_lists, history_v_lists, history_vr_lists, 
                        social_adj_lists)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr, epsilon=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    epochs = []
    rmses = []
    maes = []
    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae, training=True)

        expected_rmse, mae = test(graphrec, device, val_loader, training=False)

        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("Val: rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
        epochs.append(epoch)
        rmses.append(expected_rmse)
        maes.append(mae)

        if endure_count > 5:
            break
    
    x = epochs
    y = rmses
    plt.plot(x, y)
    plt.title("RMSE per Epoch on Douban Dataset")
    atime = str(time.time())
    plt.xlabel("Epoch")
    plt.ylabel("Root-Mean-Square Error")
    plt.savefig("/home/ewang96/DLFinal/finplots/RMSEs_douban" + atime + ".jpg")
    plt.clf()
    ymaes = maes
    plt.plot(x, ymaes)
    plt.title("MAE per Epoch on Douban Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.savefig("/home/ewang96/DLFinal/finplots/MAEs_douban" + atime + ".jpg")
    expected_rmse, mae = test(graphrec, device, test_loader, training=False)
    print("Final testing: rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
    print("ID: " + atime)

if __name__ == "__main__":
    main()
