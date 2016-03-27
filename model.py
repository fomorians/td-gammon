import time
import random
import numpy as np
import tensorflow as tf

from backgammon.game import Game
from backgammon.agents.human_agent import HumanAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.agents.td_gammon_agent import TDAgent

# helper to initialize a weight and bias variable
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
    return W, b

# helper to create a dense, fully-connected layer
def dense_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(shape)
        return activation(tf.matmul(x, W) + b, name='activation')

class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay
        lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step, \
            10000, 0.9, staircase=True), name='lambda')

        # learning rate decay
        alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step, \
            10000, 0.9, staircase=True), name='alpha')

        tf.scalar_summary('lambda', lamda)
        tf.scalar_summary('alpha', alpha)

        # describe network size
        input_layer_size = 294
        hidden_layer_size = 50
        output_layer_size = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, input_layer_size], name='x')
        self.V_next = tf.placeholder('float', [1, output_layer_size], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, [input_layer_size, hidden_layer_size], tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, [hidden_layer_size, output_layer_size], tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.scalar_summary('V_next', tf.reduce_sum(self.V_next))
        tf.scalar_summary('V', tf.reduce_sum(self.V))

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'), name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)

            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)
            accuracy_avg_op = accuracy_sum / tf.maximum(game_step, 1.0)

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])

            tf.scalar_summary('game/loss_avg', loss_avg_op)
            tf.scalar_summary('game/delta_avg', delta_avg_op)
            tf.scalar_summary('game/accuracy_avg', accuracy_avg_op)
            tf.scalar_summary('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))
            tf.scalar_summary('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))
            tf.scalar_summary('game/accuracy_avg_ema', accuracy_avg_ema.average(accuracy_avg_op))

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            tf.histogram_summary(var.name, var)
            tf.histogram_summary(var.name + '/gradients/grad', grad)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)
                    tf.histogram_summary(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                grad_trace = alpha * delta_op * trace_op
                tf.histogram_summary(var.name + '/gradients/trace', grad_trace)

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            accuracy_sum_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
            accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.merge_all_summaries()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.initialize_all_variables())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={ self.x: x })

    def play(players):
        game = Game.new()
        game.play([TDAgent(Game.TOKENS[0], self), HumanAgent(Game.TOKENS[1])], draw=True)

    def test(episodes=100, draw=False):
        players = [TDAgent(Game.TOKENS[0], self), RandomAgent(Game.TOKENS[1])]
        winners = [0, 0]
        for episode in range(episodes):
            game = Game.new()

            winner = game.play(players, draw=draw)
            winners[winner] += 1

            winners_total = sum(winners)
            print("[Episode %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode, \
                players[0].name, players[0].player, \
                players[1].name, players[1].player, \
                winners[0], winners[1], winners_total, \
                (winners[0] / winners_total) * 100.0))

    def train(self):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter('{0}{1}'.format(self.summary_path, int(time.time()), self.sess.graph_def))

        # the agent plays against itself, making the best move for each player
        players = [TDAgent(Game.TOKENS[0], self), TDAgent(Game.TOKENS[1], self)]
        players_test = [TDAgent(Game.TOKENS[0], self), RandomAgent(Game.TOKENS[1])]

        validation_interval = 100
        episodes = 200

        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                self.test(players_test, episodes=100)

            game = Game.new()
            player_num = random.randint(0, 1)

            x = game.extract_features(players[player_num].player)

            game_step = 0
            while not game.is_over():
                game.next_step(players[player_num], player_num)
                player_num = (player_num + 1) % 2

                x_next = game.extract_features(players[player_num].player)
                V_next = self.get_output(x_next)
                self.sess.run(self.train_op, feed_dict={ self.x: x, self.V_next: V_next })

                x = x_next
                game_step += 1

            winner = game.winner()

            _, global_step, summaries, _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.summaries_op,
                self.reset_op
            ], feed_dict={ self.x: x, self.V_next: np.array([[winner]], dtype='float') })
            summary_writer.add_summary(summaries, global_step=global_step)

            print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, players[winner].player, game_step))
            self.saver.save(self.sess, self.checkpoint_path + 'checkpoint', global_step=global_step)

        summary_writer.close()

        test(players_test, episodes=1000)
