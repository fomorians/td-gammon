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

        input_layer_size = 442
        hidden_layer_size = 40 # use ~71 for fully-connected (plain) layers, 50 for highway layers
        output_layer_size = 4

        self.board = tf.placeholder("float", [1, input_layer_size])

        prev_y = dense_layer(self.board, input_layer_size, hidden_layer_size, tf.nn.relu)
        self.output = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid)

        # TODO: use ET: https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node87.html#fig:GDTDl
        loss = -tf.reduce_sum(1.0 * tf.log(self.output))
        loss_summary = tf.scalar_summary("loss", loss)

        self.train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

        if restore:
            latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint_path:
                saver.restore(self.sess, latest_checkpoint_path)

        self.summaries = tf.merge_all_summaries()
        self.saver = tf.train.Saver(max_to_keep=1)

    def get_output(self, board):
        return self.sess.run(self.output, feed_dict={
            self.board: board.to_array()
        })

    def train(self):
        self.sess.run(tf.initialize_all_variables())

        tf.train.write_graph(self.sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.train.SummaryWriter(summary_path, self.sess.graph_def)

        # strategy = partial(td_gammon, self)
        strategy = random_strategy
        white = PlayerStrategy(Player.WHITE, strategy)
        black = PlayerStrategy(Player.BLACK, strategy)

        episodes = 100
        for episode in range(episodes):
            game = Game(white, black)

            while not game.board.finished():
                print('Episode: {0}'.format(episode))
                game.next()

                _, summaries = self.sess.run([self.train_step, self.summaries], feed_dict={
                    self.board: game.board.to_array()
                })
                summary_writer.add_summary(summaries, episode)

        summary_writer.close()
