import tensorflow as tf

import math

DOC_FEATURE_SIZE = 233
NUM_CLASS = 2
def inference(docs, hidden1_units, hidden2_units):
    """build the classification model, it contains one tanh hidden layer.
    one RELU hidden layer and linear sigmoid output layer
    Args:
        input: the input vector/matrix of the question - user pair,
            [20-d onehot question tag, 100-d word vec question description,
            3-d normalized question agrees, answer numbers, boutique answer numbers,
            10-d word vec user tags, 100-d word vec user description]total 233
        hidden1_units: size of first hidden layer
        hidden2_units: size of second hidden layer
    :returns
        output the compute probablistic
    """
    with tf.name_scope("hidden1"):
        weights = tf.Variable(
            tf.truncated_normal([DOC_FEATURE_SIZE, hidden1_units],
                                stddev=1.0/math.sqrt(float(DOC_FEATURE_SIZE))),
            name="weights")
        biases = tf.Variable(tf.zeros([hidden1_units]), name="biases")
        hidden1 = tf.nn.relu(tf.matmul(docs, weights) + biases)
    #hidden layer 2
    with tf.name_scope("hidden2"):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0/math.sqrt(float(hidden1_units)),
                                name="weights")
        )
        biases = tf.Variable(tf.zeros([hidden2_units], name="biases"))
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    #hidden layer 3
    with tf.name_scope("logistic_layer"):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden2_units, NUM_CLASS],stddev=1.0/math.sqrt(float(hidden2_units)),
                name="weights"
            )
        )
        biases = tf.Variable(tf.zeros([NUM_CLASS]), name="biases")
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
      Returns:
        loss: Loss tensor of type float.
      """
    labels = tf.to_int64(labels)
    print(labels.get_shape())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def training(loss, learningrate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, lables):
    correct = tf.nn.in_top_k(logits, lables, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


