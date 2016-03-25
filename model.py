from __future__ import print_function
from __future__ import division

import os
import time
import random
import numpy as np
import tensorflow as tf

from backgammon.game import Game
from backgammon.agents.human_agent import HumanAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.agents.td_gammon_agent import TDAgent

model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

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
        self.alpha = tf.maximum(0.02, tf.train.exponential_decay(0.1, self.global_step, \
            10000, 0.96, staircase=True), name='alpha') # learning rate
        self.lm = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step, \
            10000, 0.96, staircase=True, name='lambda')) # lambda

        alpha_summary = tf.scalar_summary('alpha', self.alpha)
        lm_summary = tf.scalar_summary('lambda', self.lm)

        # describe network size
        input_layer_size = 294
        hidden_layer_size = 50
        output_layer_size = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, input_layer_size], name='x')
        self.V_next = tf.placeholder('float', [1, output_layer_size], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, input_layer_size, hidden_layer_size, tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.scalar_summary('V_next', tf.reduce_sum(self.V_next))
        tf.scalar_summary('V', tf.reduce_sum(self.V))

        # sigma = V_next - V
        sigma_op = tf.reduce_sum(self.V_next - self.V, name='sigma')
        sigma_summary = tf.scalar_summary('sigma', sigma_op)

        sigma_ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        sigma_ema_op = sigma_ema.apply([sigma_op])
        sigma_ema_summary = tf.scalar_summary('sigma_ema', sigma_ema.average(sigma_op))

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')
        loss_summary = tf.scalar_summary('loss', loss_op)

        loss_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        loss_ema_op = loss_ema.apply([loss_op])
        loss_ema_summary = tf.scalar_summary('loss_ema', loss_ema.average(loss_op))

        # check if the model predicts the correct winner
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'), name='accuracy')
        accuracy_summary = tf.scalar_summary('accuracy', accuracy_op)

        accuracy_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        accuracy_ema_op = accuracy_ema.apply([accuracy_op])
        accuracy_ema_summary = tf.scalar_summary('accuracy_ema', accuracy_ema.average(accuracy_op))

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            loss_sum_op = loss_sum.assign_add(loss_op)
            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            loss_avg_summary = tf.scalar_summary('game/loss_avg', loss_avg_op)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            loss_avg_ema_summary = tf.scalar_summary('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))

            # reset per-game tracking variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # perform gradient updates using TD-lambda and eligibility traces

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars) # ys wrt x in xs

        # watch the weight and gradient distributions
        for grad, tvar in zip(grads, tvars):
            tf.histogram_summary(tvar.name, tvar)
            tf.histogram_summary(tvar.name + '/gradients/original', grad)

        # for each variable, define operations to update the tvar with sigma,
        # taking into account the gradient as part of the eligibility trace
        grad_updates = []
        with tf.variable_scope('grad_updates'):
            for grad, tvar in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((self.lm * trace) + grad)
                    tf.histogram_summary(tvar.name + '/traces', trace)

                # final grad = alpha * sigma * e
                final_grad = self.alpha * sigma_op * trace_op
                tf.histogram_summary(tvar.name + '/gradients/final', final_grad)

                assign_op = tvar.assign_add(final_grad)
                grad_updates.append(assign_op)

        # as part of training we want to update our step and other tracking variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            sigma_ema_op,
            loss_ema_op,
            accuracy_ema_op,
            loss_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*grad_updates, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.merge_all_summaries()

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

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={ self.x: x })

    def play(self):
        players = [TDAgent(Game.TOKENS[0], self), HumanAgent(Game.TOKENS[1])]
        game = Game()
        game.reset()

        player_num = random.randint(0, 1)
        while not game.is_over():
            game.next_step(players[player_num], player_num, draw=True)
            player_num = (player_num + 1) % 2

    def test(self, episodes=100):
        players = [TDAgent(Game.TOKENS[0], self), RandomAgent(Game.TOKENS[1])]
        winners = [0, 0]
        for _ in range(episodes):
            game = Game()
            game.reset()

            player_num = random.randint(0, 1)
            while not game.is_over():
                game.next_step(players[player_num], player_num, draw=False)
                player_num = (player_num + 1) % 2

            winner = game.winner()
            winners[not winner] += 1

            os.system('clear')
            print("Player %s (TD-Gammon): %d/%d" % (players[0].player, winners[0], sum(winners)))
            print("Player %s (Random): %d/%d" % (players[1].player, winners[1], sum(winners)))

    def train(self):
        tf.train.write_graph(self.sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter('{0}{1}'.format(summary_path, int(time.time()), self.sess.graph_def))

        players = [TDAgent(Game.TOKENS[0], self), TDAgent(Game.TOKENS[1], self)]

        validation_interval = 1000
        episodes = 20000

        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                self.test(episodes=100)

            game = Game()
            game.reset()

            player_num = random.randint(0, 1)
            player = players[player_num]

            x = game.extract_features(player.player)

            game_step = 0
            while not game.is_over():
                game.next_step(player, player_num)

                x_next = game.extract_features(player.player)
                V_next = self.get_output(x_next)
                _, global_step, summaries = self.sess.run([
                    self.train_op,
                    self.global_step,
                    self.summaries_op
                ], feed_dict={ self.x: x, self.V_next: V_next })

                summary_writer.add_summary(summaries, global_step=global_step)

                player_num = (player_num + 1) % 2
                player = players[player_num]

                x = x_next
                game_step += 1

            winner = game.winner()

            _, global_step, summaries, _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.summaries_op,
                self.reset_op
            ], feed_dict={ self.x: x, self.V_next: np.array([[winner]]) })
            summary_writer.add_summary(summaries, global_step=episode)

            print("Game: %d/%d in %d turns" % (episode, episodes, game_step))
            self.saver.save(self.sess, checkpoint_path + 'checkpoint', global_step=global_step)

        summary_writer.close()
        self.test(episodes=1000)
