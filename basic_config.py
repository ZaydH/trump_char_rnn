import logging
import argparse
import __main__
import os
import pickle
from enum import Enum
import copy
import tensorflow as tf
import math
import sys


class DecisionFunction(Enum):
  ArgMax = 1
  WeightRand = 10
  WeightRandAfterSpace = 11
  WeightRandTopK = 20
  WeightRandTopKAfterSpace = 21


class Config(object):
  """
  Master configuration class containing all settings related to
  the network, training, etc.
  """

  """
  Directory to export the trained model.
  """
  model_dir = '.' + os.sep + 'model' + os.sep
  """
  Name assigned to the TensorFlow model
  """
  model_name = "trump"
  """
  Character to integer look up.
  """
  char2int = None
  """
  Character dictionary exported to a pickle file so that it
  does not need to be reconstructed during text generation.
  """
  char2int_pk_file = "char2int.pk"

  sequence_length = 50
  """
  Stores whether training is being executed.
  """
  _is_train = False
  """
  Name of the main file.
  """
  _main = ""
  """
  Split between training and verification
  sets.
  """
  # training_split_ratio = 0.8
  training_split_ratio = 1

  @staticmethod
  def perform_validation():
    """
    Returns whether validation should be performed.

    :return: True validation is to be performed.
    """
    return Config.training_split_ratio != 1

  batch_size = 50

  word_count = -1
  """
  Number of sequences in the training and verification sets.
  Enter "-1" to import all possible samples.
  """
  dataset_size = -1

  class Validation(object):
    x = None
    t = None
    """
    For each training object, it is the number of vectors before
    the output is expected
    """
    depth = None
    """
    Pickle file to store the verify_x and verify_t objects.
    """
    pk_file = "verify.pk"

    _num_batch = -1

    @staticmethod
    def size():
      """
      Number of elements in the verification set

      :return: Size of the verification set
      :rtype: int
      """
      if Config.Validation.t is None:
        return 0

      return len(Config.Validation.t)

    @staticmethod
    def num_batch():
      if Config.Validation._num_batch <= 0:
        Config.Validation._num_batch = int(math.ceil(1. * Config.Validation.size() /
                                                     Config.batch_size))
      return Config.Validation._num_batch

    dataset_size = None

  class Train(object):
    """
    Stores all configuration settings and objects related to the training of
    the neural network.
    """

    """
    File containing the text training set.
    """
    training_file = "." + os.sep + "trump_speeches.txt"
    """
    Input training data
    """
    x = None
    """
    Training Labels
    """
    t = None
    """
    For each training object, it is the number of vectors before
    the output is expected
    """
    depth = None
    """
    Pickle file to export the input training set
    """
    pk_file = "train.pk"

    num_epochs = 100
    """
    If true, restore the previous settings
    """
    restore = True
    """
    Number of epochs between model checkpoint.
    """
    checkpoint_frequency = 2
    learning_rate = 0.0005
    _num_batch = -1

    @staticmethod
    def size():
      """
      Number of elements in the training set

      :return: Size of the training set
      :rtype: int
      """
      return len(Config.Train.t)

    @staticmethod
    def num_batch():
      """
      Number of batches to test.

      :return: Number of batches
      :rtype: int
      """
      if Config.Train._num_batch <= 0:
        Config.Train._num_batch = int(math.ceil(1. * Config.Train.size() /
                                                Config.batch_size))
      return Config.Train._num_batch

  class Generate(object):
    """
    Reverse operation of the char2int.  This maps the output
    integer back to a character.
    """
    _int2char = []
    """
    Text used to seed the text generator.
    """
    seed_text = ""
    """
    Minimum length for the seed text.  That ensures the learner
    has some valid text to learn.
    """
    min_seed_len = 10

    seed_x = []
    """
    Length of the text to generate
    """
    output_len = 250
    """
    Last selected character by the learner.
    """
    prev_char = ""

    enable_dropout = False

    loop = False

    @staticmethod
    def build_seed_x():
      """
      Converts the seed text to a list of integers for use to seed
      the text generator.
      """
      if Config.Generate.seed_x:
        return
      assert len(Config.char2int) > 0
      Config.Generate.seed_x = []
      for char in Config.Generate.seed_text:
        Config.Generate.seed_x.append(Config.char2int[char])

    @staticmethod
    def build_initial_x():
      Config.Generate.build_seed_x()

      extended_seed_x = copy.copy(Config.Generate.seed_x)
      while len(extended_seed_x) < Config.sequence_length:
        extended_seed_x.append(0)

      batch_x = []
      while len(batch_x) < Config.batch_size:
        batch_x.append(copy.copy(extended_seed_x))

      return batch_x

    @staticmethod
    def int2char():
      if not Config.Generate._int2char:
        return Config.Generate.build_int2char()
      return Config.Generate._int2char

    @staticmethod
    def build_int2char():
      """
      Maps integers to a character.

      :return: Mapping from integer to a character
      :rtype: List[str]
      """
      if Config.Generate._int2char:
        return Config.Generate._int2char
      assert len(Config.char2int) > 0

      Config.Generate._int2char = ["a"] * len(Config.char2int)
      for key in Config.char2int.keys():
        Config.Generate._int2char[Config.char2int[key]] = key
      return Config.Generate._int2char

  class FF(object):
    """
    Configuration settings for the feed-forward network.
    """
    depth = 1
    hidden_width = 256

  class DecisionEngine(object):
    """
    Configuration settings for the decision engine.
    """
    function = DecisionFunction.ArgMax

  class RNN(object):
    num_layers = 2
    hidden_size = 128

  @staticmethod
  def main():
    """
    Main Python file

    :return: Name of the main file.
    :rtype: str
    """
    if not Config._main:
      Config._main = os.path.basename(__main__.__file__)
    return Config._main

  @staticmethod
  def vocab_size():
    """
    Vocabulary size accessor.

    :return: Number of characters in the input and output vocabulary.
    """
    return len(Config.char2int)

  @staticmethod
  def parse_args():
    """
    Input Argument Parser

    Parses the command line input arguments.
    """
    # Select the arguments based on what program is running
    if Config.main() == "train.py":
      Config._is_train = True
      Config._train_args()
    elif Config.main() == "trump.py":
      Config._is_train = False
      Config._trump_args()
    else:
      raise ValueError("Unknown main file.")
    Config.setup_logger()

  @staticmethod
  def _train_args():
    """
    Training Command Line Argument Parser

    Parsers the command line arguments when performing training.
    """
    parser = argparse.ArgumentParser("Character-Level RNN Trainer")
    parser.add_argument("--train", type=str, required=False,
                        default=Config.Train.training_file,
                        help="Path to the training set file.")
    parser.add_argument("--restore", action="store_true",
                        help="Continue training the existing model")
    parser.add_argument("--model", type=str, required=False,
                        default=Config.model_dir,
                        help="Directory to which to export the trained network")
    parser.add_argument("--rnn_layers", type=int, required=False,
                        default=Config.RNN.num_layers,
                        help="Number of RNN layers")
    parser.add_argument("--hidden_size", type=int, required=False,
                        default=Config.RNN.hidden_size,
                        help="Number of neurons in the RNN hidden layer")
    parser.add_argument("--seqlen", type=int, required=False,
                        default=Config.sequence_length,
                        help="RNN sequence length")
    parser.add_argument("--epochs", type=int, required=False,
                        default=Config.Train.num_epochs,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, required=False,
                        default=Config.batch_size,
                        help="Batch size")
    args = parser.parse_args()

    Config.sequence_length = args.seqlen

    Config.model_dir = args.model
    if Config.model_dir[-1] != os.sep:
      Config.model_dir += os.sep

    Config.RNN.hidden_size = args.hidden_size
    Config.RNN.num_layers = args.rnn_layers

    Config.Train.training_file = args.train
    Config.Train.restore = args.restore
    Config.Train.num_epochs = args.epochs

    Config.batch_size = args.batch

  @staticmethod
  def _trump_args():
    parser = argparse.ArgumentParser("Character-Level Trump Text Generator")
    parser.add_argument("--seed", type=str, required=True,
                        help="Text with which to seed the generator")
    parser.add_argument("--len", type=int, required=False,
                        default=Config.Generate.output_len,
                        help="Length of the string to generate")

    parser.add_argument("--model", type=str, required=False, default=Config.model_dir,
                        help="Directory containing the trained model")

    parser.add_argument("--loop", action="store_true",
                        help="Loop the text generator to allow multiple text seeds.")

    parser.add_argument("--dropout", action="store_true",
                        help="Enable dropout during speech generation")

    parser.add_argument("--rnn_layers", type=int, required=False,
                        default=Config.RNN.num_layers,
                        help="Number of RNN layers")
    parser.add_argument("--hidden_size", type=int, required=False,
                        default=Config.RNN.hidden_size,
                        help="Number of neurons in the RNN hidden layer")
    parser.add_argument("--seqlen", type=int, required=False,
                        default=Config.sequence_length,
                        help="RNN sequence length")

    help_msg = """
               Function of the decision engine.  Set to \"%d\" to always greedily select
               the character with maximum probability. Set to \"%d\" to always select
               a character based off a weight random value of all characters. 
               Set to \"%d\" to make a weighted random selection ONLY for the first
               character after a space and perform greedy sampling otherwise.  
               Set to \"%d\" to take a weight random selection amongst only the  
               top 5 characters.  Set to \"%d\" to perform top-K selection after only
               a space and use greedy sampling otherwise.
               """
    help_msg = help_msg.replace("  ", " ")
    text_params = (DecisionFunction.ArgMax.value,
                   DecisionFunction.WeightRand.value,
                   DecisionFunction.WeightRandAfterSpace.value,
                   DecisionFunction.WeightRandTopK.value,
                   DecisionFunction.WeightRandTopKAfterSpace.value)
    parser.add_argument("--decision", type=int, required=False,
                        default=DecisionFunction.ArgMax.value,
                        help=help_msg % text_params)
    args = parser.parse_args()

    Config.model_dir = args.model

    import decision_engine  # Prevent circular dependencies
    if args.decision == DecisionFunction.ArgMax.value:
      dec_func = decision_engine.select_max_probability
    elif args.decision == DecisionFunction.WeightRand.value:
      dec_func = decision_engine.select_weighted_random_probability
    elif args.decision == DecisionFunction.WeightRandAfterSpace.value:
      dec_func = decision_engine.select_weighted_random_after_space
    elif args.decision == DecisionFunction.WeightRandTopK.value:
      dec_func = decision_engine.select_random_from_top_k
    elif args.decision == DecisionFunction.WeightRandTopKAfterSpace.value:
      dec_func = decision_engine.select_top_k_after_space
    else:
      raise ValueError("Unknown decision function selected.")
    Config.DecisionEngine.function = dec_func

    Config.Generate.output_len = args.len
    Config.Generate.seed_text = args.seed
    Config.Generate.enable_dropout = args.dropout
    Config.Generate.loop = args.loop

    if len(Config.Generate.seed_text) < Config.Generate.min_seed_len:
      raise ValueError("Seed text must be at least %d characters long"
                       % Config.Generate.min_seed_len)

    Config.RNN.hidden_size = args.hidden_size
    Config.RNN.num_layers = args.rnn_layers
    Config.sequence_length = args.seqlen

    # Always a batch size of one during generation.
    Config.batch_size = 1

  @staticmethod
  def parse_seed_text():
    assert len(Config.char2int) > 0

  @staticmethod
  def import_train_and_verification_data():
    logging.info("Importing the training and verification datasets.")
    Config.Train.x, Config.Train.t, Config.Train.depth \
        = _pickle_import(Config.model_dir + Config.Train.pk_file)
    Config.Validation.x, Config.Validation.t, Config.Validation.depth \
        = _pickle_import(Config.model_dir + Config.Validation.pk_file)
    logging.info("COMPLETED: Importing the training and verification datasets.")

  @staticmethod
  def export_train_and_verification_data():
    logging.info("Importing the training dataset and the character to integer map.")
    _pickle_export([Config.Train.x, Config.Train.t, Config.Train.depth],
                   Config.model_dir + Config.Train.pk_file)
    _pickle_export([Config.Validation.x, Config.Validation.t, Config.Validation.depth],
                   Config.model_dir + Config.Validation.pk_file)
    logging.info("COMPLETED: Importing the training dataset.")

  @staticmethod
  def export_character_to_integer_map():
    logging.info("Exporting the character to integer map...")
    _pickle_export(Config.char2int, Config.model_dir + Config.char2int_pk_file)
    logging.info("COMPLETED: Exporting the character to integer map")

  @staticmethod
  def import_character_to_integer_map():
    logging.info("Importing the character to integer map...")
    Config.char2int = _pickle_import(Config.model_dir + Config.char2int_pk_file)
    logging.info("COMPLETED: Importing the character to integer map")

  @staticmethod
  def import_model(sess):
    """
    Imports the weights of the training network.  This can be used
    to continue training or when generating text.

    :param sess: TensorFlow session to which to restore
    :type sess: tf.Session
    """
    logging.info("Importing the trained model...")
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(Config.model_dir))
    # model_file = (Config.model_dir + Config.model_name
    #               + "-" + str(Config.Train.checkpoint_frequency) + ".meta")
    # new_saver = tf.train.import_meta_graph(model_file)
    # new_saver.restore(sess, tf.train.latest_checkpoint(Config.model_dir))

    logging.info("COMPLETED: Importing the trained model")

  @staticmethod
  def export_model(sess, epoch):
    """
    Exports the network weights.
    """
    logging.info("Checkpoint: Exporting the trained model...")
    saver = tf.train.Saver(max_to_keep=20)
    # Only write the meta for the first checkpoint
    write_meta = (not Config.Train.restore) and (epoch == Config.Train.checkpoint_frequency)
    saver.save(sess, Config.model_dir + Config.model_name, global_step=epoch,
               write_meta_graph=write_meta)
    logging.info("COMPLETED Checkpoint: Exporting the trained model")

  @staticmethod
  def is_train():
    """
    Gets whether the current run is training.

    :return: true if training is being performed.
    :rtype: bool
    """
    if not Config._main:
      Config.main()
    return Config._is_train

  @staticmethod
  def setup_logger(log_level=logging.DEBUG):
    """
    Logger Configurator

    Configures the logger.

    :param log_level: Level to log
    :type log_level: int
    """
    data_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM

    period_loc = Config.main().rfind(".")
    filename = Config.main()[:period_loc] + ".log"
    logging.basicConfig(filename=filename, level=log_level,
                        format='%(asctime)s -- %(message)s', datefmt=data_format)

    # Also print to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.info("**********************  New Run Beginning  **********************")


def _pickle_export(obj, filename):
  """
  Pickle Exporter

  Pickles the specified object and writes it to the specified file name.

  :param obj: Object to be pickled.
  :type obj: Object

  :param filename: File to write the specified object to.
  :type filename: str
  """
  try:
    os.makedirs(os.path.dirname(filename))
  except FileExistsError:
    pass
  with open(filename, 'wb') as f:
    pickle.dump(obj, f)


def _pickle_import(filename):
  """
  Pickle Importer

  Helper function for importing pickled objects.

  :param filename: Name and path to the pickle file.
  :type filename: str

  :return: The pickled object
  :rtype: Object
  """
  with open(filename, 'rb') as f:
    obj = pickle.load(f)
  return obj
