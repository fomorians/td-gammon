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

def play():
    model = Model()
    model.restore()

    strategy = partial(td_gammon_strategy, model)

    white = PlayerStrategy(Player.WHITE, strategy)
    black = PlayerHuman()

    game = Game(white, black)
    game.play()

def train():
    model = Model()
    model.train()

if __name__ == '__main__':
    if FLAGS.play:
        play()
    else:
        train()
