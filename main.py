import tensorflow as tf

from model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint when training.')

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        if FLAGS.test:
            model = Model(sess, restore=True)
            model.test()
        else:
            model = Model(sess, restore=FLAGS.restore)
            model.train()
