import tensorflow as tf
from basic_config import Config


def _get_tf_rand_uniform(shape):
    """
    TensorFlow Random Uniform Variable Generator

    Initialize a random set of weights uniformly at random
    between -1 and 1.

    :param shape: Shape of the random object to generate
    :type shape: tf.shape

    :return: Uniform random TensorFlow variable of the specified shape
    :rtype: tf.Variable
    """
    obj = tf.random_uniform(shape, minval=-1., maxval=1.)
    return tf.Variable(obj)


def _get_tf_rand_normal(shape):
    """
    Random Normal Variable Generator

    Generates a random object of the specified shape that is
    using the normal distribution with mean 0 and standard deviation 1.

    :param shape: Shape of the random object to generate
    :type shape: tf.shape

    :return: Normal random TensorFlow variable of the specified shape
    :rtype: tf.Variable
    """
    obj = tf.random_normal(shape, mean=0., stddev=1.)
    return tf.Variable(obj)


def _build_feed_forward(ff_input, rand_func):
  """
  Feed-Forward Network Builder

  Constructs and initializes the feed-forward network.

  :param ff_input: Input to the feed-forward network.
  :type ff_input: tf.Tensor

  :param rand_func: Function to generate the
  :type rand_func: Callable

  :return: Output of the feed-forward network.
  :rtype: tf.Tensor
  """

  # Separate from the for loop in case no hidden layers
  input_width = int(ff_input.shape[1])
  ff_in = ff_input
  for i in range(0, Config.FF.depth):
    if i > 0:
      input_width = Config.FF.hidden_width
      # noinspection PyUnboundLocalVariable
      ff_in = hidden_out

    bias_input = rand_func([Config.FF.hidden_width])
    hidden_layer = rand_func([input_width, Config.FF.hidden_width])
    a_hidden = tf.add(tf.matmul(ff_in, hidden_layer), bias_input)
    hidden_out = tf.nn.relu(a_hidden)

  # Construct the output layer
  bias_input = rand_func([Config.vocab_size()])
  out_layer = rand_func([input_width, Config.vocab_size()])
  # noinspection PyUnboundLocalVariable
  a_out = tf.nn.relu(tf.add(tf.matmul(ff_in, out_layer), bias_input))
  return a_out


def setup_feed_forward(ff_input):
  """
  Feed-Forward Builder

  Builds the feed-forward network.

  :param ff_input: Item to be fed into the feed-forward network
  :type ff_input: tf.Tensor

  :return: Output from the softmax
  :rtype: tf.Tensor
  """
  ff_output = _build_feed_forward(ff_input, _get_tf_rand_normal)

  return ff_output
