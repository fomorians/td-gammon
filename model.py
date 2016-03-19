import os
import tensorflow as tf

from functools import partial, reduce

from game import Game
from player import Player
from player_strategy import PlayerStrategy
from strategy import td_gammon_strategy, random_strategy

model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

def weight_bias(input_size, output_size):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    return W, b

def dense_layer(x, input_size, output_size, activation):
    W, b = weight_bias(input_size, output_size)
    return activation(tf.matmul(x, W) + b)

class Model(object):
    def __init__(self, restore=False):
        self.sess = tf.Session()
        self.graph = tf.Graph()

        input_layer_size = 476
        hidden_layer_size = 40 # use ~71 for fully-connected (plain) layers, 50 for highway layers
        output_layer_size = 4

        self.x = tf.placeholder("float", [1, input_layer_size])
        self.Y_next = tf.placeholder("float", [1, output_layer_size])

        prev_y = dense_layer(self.x, input_layer_size, hidden_layer_size, tf.nn.sigmoid)
        self.Y = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid)

        # column vector of eligibility traces, one for each component of theta
        self.e1 = tf.Variable(tf.zeros([hidden_layer_size]))
        self.e2 = tf.Variable(tf.zeros([output_layer_size]))

        # sigma = r + gamma * V(s') - V(s)
        loss_op = self.get_loss_op(self.Y_next - self.Y)
        self.train_op = self.get_train_op(loss_op)

        if restore:
            self.sess.run(tf.initialize_all_variables())

            latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint_path:
                saver.restore(self.sess, latest_checkpoint_path)

        self.summaries = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=1)

    def get_loss_op(self, sigma_op):
        loss_op = -tf.reduce_sum(sigma_op)
        loss_summary = tf.scalar_summary("loss", loss_op)
        return loss_op

    def get_train_op(self, loss_op):
        lr = 1e-2
        optimizer = tf.train.GradientDescentOptimizer(lr)
        gradients = optimizer.compute_gradients(loss_op)

        for grad, var in gradients:
            tf.histogram_summary(var.op.name, var)
            if grad:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # TODO: update eligibility traces
        # e-> = lambda * e-> + <grad w.r.t output>

        # lm = 0.99
        # self.e1.assign(lm * self.e1 + gradients)
        # self.e2.assign(lm * self.e2 + gradients)

        # apply gradients
        return optimizer.apply_gradients(gradients)

    def get_output(self, board):
        return self.sess.run(self.Y, feed_dict={
            self.x: board.to_array()
        })

    def train(self):
        self.sess.run(tf.initialize_all_variables())

        tf.train.write_graph(self.sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter(summary_path, self.sess.graph_def)

        # strategy = partial(td_gammon, self)
        strategy = random_strategy
        white = PlayerStrategy(Player.WHITE, strategy)
        black = PlayerStrategy(Player.BLACK, strategy)

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

                step += 1
                global_step += 1

            print('END => Episode: {0}, Step: {1}, Global Step: {2}'.format(episode, step, global_step))
            x = game.board.to_array()
            _, summaries = self.sess.run([self.train_op, self.summaries], feed_dict={
                self.x: x,
                self.Y_next: game.to_outcome_array()
            })
            summary_writer.add_summary(summaries, global_step)

        summary_writer.close()
