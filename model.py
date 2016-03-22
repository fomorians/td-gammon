from __future__ import division

import os
import time
import random
import numpy as np
import tensorflow as tf

from functools import partial, reduce

from game import Game
from player import Player
from player_strategy import PlayerStrategy
from strategy import td_gammon_strategy, random_strategy

model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/{0}'.format(int(time.time())))

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

def weight_bias(input_size, output_size):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    return W, b

def dense_layer(x, input_size, output_size, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(input_size, output_size)
        return activation(tf.matmul(x, W) + b, name='activation')

class Model(object):
    def __init__(self, restore=False):
        # setup our session and graph
        self.sess = tf.Session()
        self.graph = tf.Graph()

        # setup some constants
        alpha = 0.1 # learning rate
        lm = 0.7 # lambda
        gamma = 1.0 # discount
        reward = 0.0

        # describe network size
        input_layer_size = 478
        hidden_layer_size = 60
        output_layer_size = 1

        # placeholders for input and target output
        self.x = tf.placeholder("float", [1, input_layer_size], name="x")
        self.V_next = tf.placeholder("float", [1, output_layer_size], name="V_next")

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, input_layer_size, hidden_layer_size, tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid, name='layer2')

        tf.scalar_summary(self.V_next.name + '/sum', tf.reduce_sum(self.V_next))
        tf.scalar_summary(self.V.name + '/sum', tf.reduce_sum(self.V))

        tf.histogram_summary(self.V_next.name, self.V_next)
        tf.histogram_summary(self.V.name, self.V)

        # TODO: take the difference of vector containing win scenarios (incl. gammons)
        self.sigma = tf.reduce_sum(reward + (gamma * self.V_next) - self.V, name='sigma')
        tf.scalar_summary('sigma', self.sigma)

        self.loss = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')
        tf.scalar_summary(self.loss.name, self.loss)

        self.accuracy = tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float', name='accuracy')
        accuracy_ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        accuracy_avg = accuracy_ema.apply([self.accuracy])

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars) # ys wrt x in xs

        for grad, var in zip(grads, tvars):
            tf.histogram_summary(var.op.name, var)
            tf.histogram_summary(var.op.name + '/gradients', grad)

        with tf.variable_scope('grad_updates'):
            grad_updates = []
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = gamma * lm * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign(gamma * lm * trace + grad)
                    tf.histogram_summary(var.op.name + '/traces', trace)

                assign_op = var.assign_add(alpha * self.sigma * trace_op)
                grad_updates.append(assign_op)

        # applies gradient updates
        self.train_op = tf.group(*grad_updates, name="train")

        self.summaries = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=1)

        self.sess.run(tf.initialize_all_variables())

        if restore:
            latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint_path:
                self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, game):
        return self.sess.run(self.V, feed_dict={ self.x: game.to_array() })

    def play(self):
        strategy = partial(td_gammon_strategy, self)

        white = PlayerStrategy(Player.WHITE, strategy)
        black = PlayerHuman()

        game = Game(white, black)
        game.play()

    def test(self, episodes=100):
        wins_td = 0 # TD-gammon
        wins_rand = 0 # random

        player_td = PlayerStrategy(Player.WHITE, partial(td_gammon_strategy, self))
        player_gammon = PlayerStrategy(Player.BLACK, random_strategy)

        for episode in range(episodes):
            white, black = random.sample([player_td, player_gammon], 2)
            game = Game(white, black)

            while not game.board.finished():
                game.next(draw_board=False)

            if (game.winner == Player.WHITE and game.white == player_td) \
            or (game.winner == Player.BLACK and game.black == player_td):
                wins_td += 1
            else:
                wins_rand += 1

            win_ratio = wins_td / wins_rand if wins_rand > 0 else wins_td
            print('TEST GAME [{0}] => Ratio: {1}, TD-Gammon: {2}, Random: {3}'.format(episode, win_ratio, wins_td, wins_rand))

    def train(self):
        self.sess.run(tf.initialize_all_variables())

        tf.train.write_graph(self.sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter(summary_path, self.sess.graph_def)

        model_strategy = partial(td_gammon_strategy, self)
        white = PlayerStrategy(Player.WHITE, model_strategy)
        black = PlayerStrategy(Player.BLACK, model_strategy)

        global_step = 0
        test_interval = 100
        episodes = 2000

        for episode in range(episodes):
            if episode != 0 and episode % test_interval == 0:
                self.test()

            game = Game(white, black)
            episode_step = 0

            while not game.board.finished():
                x = game.to_array()

                game.next(draw_board=False)
                x_next = game.to_array()
                V_next = self.sess.run(self.V, feed_dict={ self.x: x_next })

                v, sigma, loss, _, summaries = self.sess.run([
                    self.V,
                    self.sigma,
                    self.loss,
                    self.train_op,
                    self.summaries
                ], feed_dict={
                    self.x: x,
                    self.V_next: V_next
                })
                summary_writer.add_summary(summaries, global_step)

                global_step += 1
                episode_step += 1

            x = game.to_array()
            z = game.to_win_array()

            v, accuracy, sigma, loss, _, summaries = self.sess.run([
                self.V,
                self.accuracy,
                self.sigma,
                self.loss,
                self.train_op,
                self.summaries
            ], feed_dict={
                self.x: x,
                self.V_next: z
            })
            summary_writer.add_summary(summaries, global_step)
            print('TRAIN GAME [{0}] => {1} {2} (accuracy: {3}, sigma: {4}, loss: {5})'.format(episode, np.around(v), z, accuracy, sigma, loss))
            self.saver.save(self.sess, checkpoint_path + 'checkpoint', global_step=global_step)

        summary_writer.close()
        self.test()
