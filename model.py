import os
import time
import tensorflow as tf

from functools import partial, reduce

from game import Game
from player import Player
from player_strategy import PlayerStrategy
from strategy import td_gammon_strategy, random_strategy

model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/{0}'.format(int(time.time())))

def weight_bias(input_size, output_size):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    return W, b

def dense_layer(x, input_size, output_size, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(input_size, output_size)
        return activation(tf.matmul(x, W) + b)

class Model(object):
    def __init__(self):
        self.sess = tf.Session()
        self.graph = tf.Graph()

        input_layer_size = 476
        hidden_layer_size = 40 # use ~71 for fully-connected (plain) layers, 50 for highway layers
        output_layer_size = 4

        self.x = tf.placeholder("float", [1, input_layer_size])
        self.Y_next = tf.placeholder("float", [1, output_layer_size])

        prev_y = dense_layer(self.x, input_layer_size, hidden_layer_size, tf.nn.relu, 'layer1')
        self.Y = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid, 'layer2')

        # sigma = r + gamma * V(s') - V(s)
        loss_op = self.get_loss_op(self.Y_next - self.Y)
        self.train_op = self.get_train_op(loss_op)

        self.summaries = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=1)

    def get_loss_op(self, sigma_op):
        loss_op = tf.reduce_sum(tf.square(sigma_op), name='loss')
        loss_summary = tf.scalar_summary('loss', loss_op)
        return loss_op

    def get_train_op(self, loss_op):
        lr = 1e-2
        optimizer = tf.train.GradientDescentOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(loss_op)

        traces_and_vars = []
        for grad, var in grads_and_vars:
            trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False)

            # e-> = lambda * e-> + <grad w.r.t output>
            lm = 0.9
            trace_op = trace.assign(lm * trace + grad)
            traces_and_vars.append((trace_op, var))

            tf.histogram_summary(var.op.name, var)
            tf.histogram_summary(var.op.name + '/eligibility_traces', trace)
            tf.histogram_summary(var.op.name + '/gradients', grad)

        # apply gradients
        return optimizer.apply_gradients(traces_and_vars)

    def get_output(self, board):
        return self.sess.run(self.Y, feed_dict={
            self.x: board.to_array()
        })

    def restore(self):
        self.sess.run(tf.initialize_all_variables())

        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(self.sess, latest_checkpoint_path)

    def test(self):
        self.sess.run(tf.initialize_all_variables())

        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(self.sess, latest_checkpoint_path)

        white = PlayerStrategy(Player.WHITE, partial(td_gammon_strategy, self))
        black = PlayerStrategy(Player.BLACK, random_strategy)

        global_step = 0
        episodes = 100

        wins_white = 0 # TD-gammon
        wins_black = 0 # random

        for episode in range(episodes):
            game = Game(white, black)
            step = 0

            while not game.board.finished():
                game.next(draw_board=False)
                global_step += 1
                step += 1

            if game.winner == Player.WHITE:
                wins_white += 1
            else:
                wins_black += 1

            print('[{0}] Wins TD: {1}, Wins Random: {2}'.format(episode, wins_white, wins_black))

    def train(self):
        self.sess.run(tf.initialize_all_variables())

        tf.train.write_graph(self.sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter(summary_path, self.sess.graph_def)

        model_strategy = partial(td_gammon_strategy, self)
        white = PlayerStrategy(Player.WHITE, model_strategy)
        black = PlayerStrategy(Player.BLACK, model_strategy)

        global_step = 0
        episodes = 100

        for episode in range(episodes):
            game = Game(white, black)
            step = 0

            while not game.board.finished():
                print('Episode: {0}, Step: {1}, Global Step: {2}'.format(episode, step, global_step))

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

            print('END => Episode: {0}, Step: {1}, Global Step: {2}'.format(episode, step, global_step))
            x = game.board.to_array()
            _, summaries = self.sess.run([self.train_op, self.summaries], feed_dict={
                self.x: x,
                self.Y_next: game.to_outcome_array()
            })
            summary_writer.add_summary(summaries, global_step)

            self.saver.save(self.sess, 'td_gammon', global_step=global_step)

        summary_writer.close()
