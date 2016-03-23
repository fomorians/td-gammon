from __future__ import division

import os
import time
import random
import numpy as np
import tensorflow as tf

from functools import partial, reduce

from backgammon.game import Game
from backgammon.player import Player
from backgammon.player_strategy import PlayerStrategy
from backgammon.strategy import random_strategy

from strategy import td_gammon_strategy

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
    def __init__(self, sess, restore=False):
        # setup our session
        self.sess = sess

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # learning rate and lambda decay
        self.alpha = tf.train.exponential_decay(0.1, self.global_step, \
            15000, 0.95, staircase=True, name='alpha') # learning rate
        self.lm = tf.train.exponential_decay(0.9, self.global_step, \
            15000, 0.95, staircase=True, name='lambda') #lambda

        tf.scalar_summary(self.alpha.name, self.alpha)
        tf.scalar_summary(self.lm.name, self.lm)

        # setup some constants
        gamma = 1.0 # discount
        reward = 0.0

        # describe network size
        input_layer_size = 478
        hidden_layer_size = 120
        output_layer_size = 1

        # placeholders for input and target output
        self.x = tf.placeholder("float", [1, input_layer_size], name="x")
        self.V_next = tf.placeholder("float", [1, output_layer_size], name="V_next")

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, input_layer_size, hidden_layer_size, tf.nn.relu, name='layer1')
        self.V = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.scalar_summary(self.V_next.name + '/sum', tf.reduce_sum(self.V_next))
        tf.scalar_summary(self.V.name + '/sum', tf.reduce_sum(self.V))

        # tf.histogram_summary(self.V_next.name, self.V_next)
        # tf.histogram_summary(self.V.name, self.V)

        # TODO: take the difference of vector containing win scenarios (incl. gammons)
        # sigma = r + gamma * V_next - V
        self.sigma_op = tf.reduce_sum(reward + (gamma * self.V_next) - self.V, name='sigma')
        tf.scalar_summary(self.sigma_op.name, self.sigma_op)
        sigma_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        sigma_ema_op = sigma_ema.apply([self.sigma_op])
        tf.scalar_summary('sigma_avg', sigma_ema.average(self.sigma_op))

        # mean squared error of the difference between the next state and the current state
        self.loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')
        tf.scalar_summary(self.loss_op.name, self.loss_op)
        loss_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        loss_ema_op = loss_ema.apply([self.loss_op])
        tf.scalar_summary('loss_avg', loss_ema.average(self.loss_op))

        # check if the model predicts the correct winner
        self.accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'), name='accuracy')
        accuracy_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        accuracy_ema_op = accuracy_ema.apply([self.accuracy_op])
        tf.scalar_summary('accuracy_avg', accuracy_ema.average(self.accuracy_op))

        # perform gradient updates using TD-lambda and eligibility traces

        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars) # ys wrt x in xs

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            tf.histogram_summary(var.op.name, var)
            tf.histogram_summary(var.op.name + '/gradients', grad)

        # for each variable, define operations to update the var with sigma,
        # taking into account the gradient as part of the eligibility trace
        with tf.variable_scope('grad_updates'):
            grad_updates = []
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = gamma * lm * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign(gamma * self.lm * trace + grad)
                    tf.histogram_summary(var.op.name + '/traces', trace)

                assign_op = var.assign_add(self.alpha * self.sigma_op * trace_op)
                grad_updates.append(assign_op)

        # define single operation to apply all gradient updates
        with tf.control_dependencies([global_step_op, sigma_ema_op, loss_ema_op, accuracy_ema_op]):
            self.train_op = tf.group(*grad_updates, name="train")

        # merge summaries for TensorBoard
        self.summary_op = tf.merge_all_summaries()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.initialize_all_variables())

        # after training a model, we can restore checkpoints here
        if restore:
            latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint_path:
                print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
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
        tf.train.write_graph(self.sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter(summary_path, self.sess.graph_def)

        model_strategy = partial(td_gammon_strategy, self)
        white = PlayerStrategy(Player.WHITE, model_strategy)
        black = PlayerStrategy(Player.BLACK, model_strategy)

        test_interval = 1000
        episodes = 10000

        for episode in range(episodes):
            if episode != 0 and episode % test_interval == 0:
                self.test(episodes=100)

            game = Game(white, black)
            episode_step = 0

            while not game.board.finished():
                x = game.to_array()

                game.next(draw_board=False)
                x_next = game.to_array()
                V_next = self.sess.run(self.V, feed_dict={ self.x: x_next })

                global_step, v, sigma, loss, _, summaries = self.sess.run([
                    self.global_step,
                    self.V,
                    self.sigma_op,
                    self.loss_op,
                    self.train_op,
                    self.summary_op
                ], feed_dict={
                    self.x: x,
                    self.V_next: V_next
                })
                summary_writer.add_summary(summaries, global_step=global_step)

                episode_step += 1

            x = game.to_array()
            z = game.to_win_array()

            global_step, v, accuracy, sigma, loss, _, summaries = self.sess.run([
                self.global_step,
                self.V,
                self.accuracy_op,
                self.sigma_op,
                self.loss_op,
                self.train_op,
                self.summary_op
            ], feed_dict={
                self.x: x,
                self.V_next: z
            })
            summary_writer.add_summary(summaries, global_step=global_step)
            print('TRAIN GAME [{0}] => {1} {2} (accuracy: {3}, sigma: {4}, loss: {5})'.format(episode, np.around(v), z, accuracy, sigma, loss))
            self.saver.save(self.sess, checkpoint_path + 'checkpoint', global_step=global_step)

        summary_writer.close()
        self.test(episodes=1000)
