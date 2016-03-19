from __future__ import division

import os
import time
import random
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
        return activation(tf.matmul(x, W) + b)

class Model(object):
    def __init__(self, restore=False):
        self.sess = tf.Session()
        self.graph = tf.Graph()

        input_layer_size = 476
        hidden_layer_size = 120
        output_layer_size = 4

        self.x = tf.placeholder("float", [1, input_layer_size])
        self.Y_next = tf.placeholder("float", [1, output_layer_size])

        prev_y = dense_layer(self.x, input_layer_size, hidden_layer_size, tf.sigmoid, 'layer1')
        self.Y = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid, 'layer2')

        # sigma = r + gamma * V(s') - V(s)
        self.sigma_op = self.Y_next - self.Y
        self.loss_op = self.get_loss_op(self.sigma_op)
        self.train_op = self.get_train_op(self.loss_op)

        self.summaries = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=1)

        self.sess.run(tf.initialize_all_variables())

        if restore:
            latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint_path:
                self.saver.restore(self.sess, latest_checkpoint_path)

    def get_loss_op(self, sigma_op):
        loss_op = tf.reduce_mean(tf.square(sigma_op), name='loss')
        loss_summary = tf.scalar_summary('loss', loss_op)
        return loss_op

    def get_train_op(self, loss_op):
        # TODO: this is actually wrong but I _want_ it to work because its easier ;)
        lr = 1e-1
        optimizer = tf.train.GradientDescentOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(loss_op)

        # from the computed gradients we setup eligibility traces
        traces_and_vars = []
        for grad, var in grads_and_vars:
            trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')

            # e-> = lambda * e-> + <grad of output w.r.t weights>
            with tf.variable_scope('update_trace'):
                lm = 0.7
                update_trace_op = trace.assign(lm * trace + grad)
                traces_and_vars.append((update_trace_op, var))

            tf.histogram_summary(var.op.name, var)
            tf.histogram_summary(var.op.name + '/eligibility_traces', trace)
            tf.histogram_summary(var.op.name + '/gradients', grad)

        # apply gradients
        return optimizer.apply_gradients(traces_and_vars)

    def get_output(self, board):
        return self.sess.run(self.Y, feed_dict={
            self.x: board.to_array()
        })

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

            if wins_rand > 0:
                print('TEST GAME => [{0}] Wins: {1}, TD-Gammon: {2}, Random: {3}'.format(episode, wins_td / wins_rand, wins_td, wins_rand))
            else:
                print('TEST GAME => [{0}] Wins: {1}, TD-Gammon: {2}, Random: {3}'.format(episode, wins_td, wins_td, wins_rand))

        if wins_rand > 0:
            print('TEST => [{0}] Wins: {1}, TD-Gammon: {2}, Random: {3}'.format(episode, wins_td / wins_rand, wins_td, wins_rand))
        else:
            print('TEST => [{0}] Wins: {1}, TD-Gammon: {2}, Random: {3}'.format(episode, wins_td, wins_td, wins_rand))

    def train(self):
        self.sess.run(tf.initialize_all_variables())

        tf.train.write_graph(self.sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter(summary_path, self.sess.graph_def)

        model_strategy = partial(td_gammon_strategy, self)
        white = PlayerStrategy(Player.WHITE, model_strategy)
        black = PlayerStrategy(Player.BLACK, model_strategy)

        global_step = 0
        episodes = 1000

        for episode in range(episodes):
            game = Game(white, black)
            step = 0

            if episode % 100 == 0:
                self.test(episodes=10)

            while not game.board.finished():
                x = game.board.to_array()

                game.next(draw_board=False)

                x_next = game.board.to_array()
                Y_next = self.sess.run(self.Y, feed_dict={ self.x: x_next })

                _, summaries = self.sess.run([self.train_op, self.summaries], feed_dict={
                    self.x: x,
                    self.Y_next: Y_next
                })
                summary_writer.add_summary(summaries, global_step)

                global_step += 1
                step += 1

            x = game.board.to_array()
            loss, _, summaries = self.sess.run([self.loss_op, self.train_op, self.summaries], feed_dict={
                self.x: x,
                self.Y_next: game.to_outcome_array()
            })
            summary_writer.add_summary(summaries, global_step)

            print('GAME LOSS => [{0}] Loss: {1}'.format(episode, loss))

            self.saver.save(self.sess, checkpoint_path + 'checkpoint', global_step=global_step)

        summary_writer.close()

        self.test(episodes=100)
