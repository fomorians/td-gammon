import numpy as np
import tensorflow as tf

def weight_bias(input_size, output_size):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    return W, b

def dense_layer(x, input_size, output_size, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(input_size, output_size)
        return activation(tf.matmul(x, W) + b, name='activation')

class Model(object):
    # TODO: remove sess
    def __init__(self, sess, checkpoint_path, restore=False):
        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # learning rate and lambda decay
        self.alpha = tf.maximum(0.02, tf.train.exponential_decay(0.1, self.global_step, \
            40000, 0.96), name='alpha') # learning rate
        self.lm = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step, \
            30000, 0.96), name='lambda') # lambda

        alpha_summary = tf.scalar_summary('alpha', self.alpha)
        lm_summary = tf.scalar_summary('lambda', self.lm)

        # describe network size
        input_layer_size = 294
        hidden_layer_size = 50
        output_layer_size = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, input_layer_size], name='x')
        self.V_next = tf.placeholder('float', [1, output_layer_size], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, input_layer_size, hidden_layer_size, tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.scalar_summary('V_next', tf.reduce_sum(self.V_next))
        tf.scalar_summary('V', tf.reduce_sum(self.V))

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')
        tf.scalar_summary('delta', delta_op)

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')
        tf.scalar_summary('loss', loss_op)

        # check if the model predicts the correct winner
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'), name='accuracy')
        accuracy_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        accuracy_ema_op = accuracy_ema.apply([accuracy_op])
        accuracy_ema_summary = tf.scalar_summary('accuracy_ema', accuracy_ema.average(accuracy_op))

        delta_ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        delta_ema_op = delta_ema.apply([delta_op])
        delta_ema_summary = tf.scalar_summary('delta_ema', delta_ema.average(delta_op))

        loss_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        loss_ema_op = loss_ema.apply([loss_op])
        loss_ema_summary = tf.scalar_summary('loss_ema', loss_ema.average(loss_op))

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            loss_sum_op = loss_sum.assign_add(loss_op)
            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            loss_avg_summary = tf.scalar_summary('game/loss_avg', loss_avg_op)

            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            delta_sum_op = delta_sum.assign_add(delta_op)
            delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)
            delta_avg_summary = tf.scalar_summary('game/delta_avg', delta_avg_op)

            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)
            accuracy_avg_op = accuracy_sum / tf.maximum(game_step, 1.0)
            accuracy_avg_summary = tf.scalar_summary('game/accuracy_avg', accuracy_avg_op)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            loss_avg_ema_summary = tf.scalar_summary('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))

            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            delta_avg_ema_summary = tf.scalar_summary('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))

            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])
            accuracy_avg_ema_summary = tf.scalar_summary('game/accuracy_avg_ema', accuracy_avg_ema.average(accuracy_avg_op))

            # reset per-game tracking variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # perform gradient updates using TD-lambda and eligibility traces

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars) # ys wrt x in xs

        # watch the weight and gradient distributions
        for grad, tvar in zip(grads, tvars):
            tf.histogram_summary(tvar.name, tvar)
            tf.histogram_summary(tvar.name + '/gradients/original', grad)

        # for each variable, define operations to update the tvar with delta,
        # taking into account the gradient as part of the eligibility trace
        grad_updates = []
        with tf.variable_scope('grad_updates'):
            for grad, tvar in zip(grads, tvars):
                # final grad = alpha * sigma * e
                final_grad = self.alpha * sigma_op * grad
                tf.histogram_summary(tvar.name + '/gradients/final', final_grad)

                assign_op = tvar.assign_add(final_grad)
                grad_updates.append(assign_op)

        # as part of training we want to update our step and other tracking variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            accuracy_sum_op,
            loss_ema_op,
            delta_ema_op,
            accuracy_ema_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
            accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*grad_updates, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.merge_all_summaries()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.initialize_all_variables())

        # after training a model, we can restore checkpoints here
        if restore:
            latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint_path:
                print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
                self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        # TODO: use tf.get_default_session()
        return self.sess.run(self.V, feed_dict={ self.x: x })
