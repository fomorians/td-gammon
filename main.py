import tensorflow as tf

from functools import partial, reduce

from backgammon.game import Game
from backgammon.player import Player
from backgammon.player_human import PlayerHuman
from backgammon.player_strategy import PlayerStrategy
from backgammon.strategy import safe_strategy

from model import Model
from strategy import td_gammon_strategy

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('play', False, 'If true, play against the trained model.')
flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint when training.')

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        if FLAGS.play:
            model = Model(sess, restore=True)
            model.play()
        elif FLAGS.test:
            model = Model(sess, restore=True)
            model.test()
        else:
            model = Model(sess, restore=FLAGS.restore)
            model.train()
