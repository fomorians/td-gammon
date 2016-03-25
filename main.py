import evaluation
import tensorflow as tf

from model import Model

from backgammon.game import Game
from backgammon.agents.td_gammon_agent import TDAgent

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        if FLAGS.test:
            from backgammon.agents.random_agent import RandomAgent
            model = Model(sess, restore=True)
            players = [TDAgent(Game.TOKENS[0], model), RandomAgent(Game.TOKENS[1])]
            evaluation.test(players, episodes=1000)
        elif FLAGS.play:
            from backgammon.agents.human_agent import HumanAgent
            model = Model(sess, restore=True)
            players = [TDAgent(Game.TOKENS[0], model), HumanAgent(Game.TOKENS[1])]
            evaluation.play(players)
        else:
            model = Model(sess, restore=FLAGS.restore)
            model.train()
