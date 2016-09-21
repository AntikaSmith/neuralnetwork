from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import time

from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import numpy as np
import model
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.2, 'Initial learning rate.')
flags.DEFINE_float("dropout_rate", 0.5, "output layer dropout rate")
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
DOC_FEATURE_SIZE = model.DOC_FEATURE_SIZE


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    docs_placeholder: docs placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    docs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, DOC_FEATURE_SIZE))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1))#todo: float or int? that is a question
    keep_prob_placeholder = tf.placeholder(tf.float32)
    return docs_placeholder, labels_placeholder, keep_prob_placeholder


def fill_feed_dict(data_set, docs_pl, labels_pl, keep_prob_pl, keep_prob):
    docs_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        docs_pl: docs_feed,
        labels_pl: labels_feed,
        keep_prob_pl: keep_prob
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            docs_placeholder,
            labels_placeholder,
            keep_prob_placeholder,
            data_set,
            keep_prob=1):
    """Runs one evaluation against the full epoch of data.
    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    docs_placeholder: The docs placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of docs and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, docs_placeholder, labels_placeholder, keep_prob_placeholder, 1)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
    """train model for a number of steps"""
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "  start reading data")
    data_sets = input_data.read_data("invited_info_trainoutput.txt")
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "  end reading data")
    with tf.Graph().as_default():
        docs_placeholder, labels_placeholder, keep_prob_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = model.inference(docs_placeholder, FLAGS.hidden1, FLAGS.hidden2, keep_prob_placeholder)
        loss = model.loss(logits, labels_placeholder)
        train_op = model.training(loss, FLAGS.learning_rate)
        eval_correct = model.evaluation(logits, labels_placeholder)
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        sess.run(init)
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train, docs_placeholder, labels_placeholder, keep_prob_placeholder, 0.5)
            _, loss_value = sess.run([train_op, loss], feed_dict)
            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        docs_placeholder,
                        labels_placeholder,
                        keep_prob_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        docs_placeholder,
                        labels_placeholder,
                        keep_prob_placeholder,
                        data_sets.validation)
            # # Evaluate against the test set.
            # print('Test Data Eval:')
            # do_eval(sess,
            #         eval_correct,
            #         docs_placeholder,
            #         labels_placeholder,
            #         data_sets.test)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()