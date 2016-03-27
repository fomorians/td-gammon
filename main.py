import os
import tensorflow as tf

from model import Model
from evaluation import play, test, train

from backgammon.game import Game
from backgammon.agents.human_agent import HumanAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.agents.td_gammon_agent import TDAgent

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        if FLAGS.test:
            model = Model(sess, restore=True)
            players = [TDAgent(Game.TOKENS[0], model), RandomAgent(Game.TOKENS[1])]
            test(players, episodes=1000)
        elif FLAGS.play:
            model = Model(sess, restore=True)
            players = [TDAgent(Game.TOKENS[0], model), HumanAgent(Game.TOKENS[1])]
            play(players)
        else:
            model = Model(sess, restore=FLAGS.restore)
            train(model, model_path, summary_path, checkpoint_path)
