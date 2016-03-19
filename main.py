import tensorflow as tf

from functools import partial, reduce

from model import Model
from game import Game
from player import Player
from player_human import PlayerHuman
from player_strategy import PlayerStrategy
from strategy import td_gammon_strategy, safe_strategy

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('play', False, 'If true, play against the trained model.')
flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')

if __name__ == '__main__':
    model = Model()

    if FLAGS.play:
        model.play()
    elif FLAGS.test:
        model.test()
    else:
        model.train()
