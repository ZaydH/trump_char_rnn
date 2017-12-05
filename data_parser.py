"""
    data_parser.py

    Functions for text processing for trump speech generator.
"""
import logging
import random
import numpy as np
from basic_config import Config
import re


def read_input():
  """
  Reads the input file, get rid of newlines and empty lines, replace with space
  """
  with open(Config.Train.training_file, "r") as f:
    input_text = f.read()
  input_text = re.sub("SPEECH\s+\d+", "", input_text)  # Remove the speech headers
  input_text = re.sub("\n+", "\n", input_text)
  input_text = re.sub("\d+\\\d+\\\d+", "", input_text)  # Remove the dates
  input_text = re.sub(" +", " ", input_text)  # Remove double spaces.
  return ' '.join(input_text.splitlines())


def create_examples(input_string, ):
  """
  from the input, produce examples where the input is a sequence of integers
  representing a string of characters, and the target is the character immediately
  following the input sequence
  """

  sequences = []
  targets = []
  depths = []
  Config.char2int = {c: i for i, c in enumerate(sorted(set(input_string)))}

  # ToDo Discuss with Ben how we want to train on text shorter than the window size?

  # Get all examples
  if Config.dataset_size == -1:
    # iterate over the file window by window
    i = 0
    while i + Config.sequence_length + 1 < len(input_string):
      sequences += [[Config.char2int[c] for c in input_string[i: i + Config.sequence_length]]]
      depths.append(Config.sequence_length)
      targets += [Config.char2int[input_string[i + Config.sequence_length]]]
      i += 1

  else:
    # get size many examples
    for z in range(Config.dataset_size):
      # get a random starting point
      r = random.choice(range(len(input_string) - Config.sequence_length - 1))

      sequences.append([Config.char2int[c] for c in input_string[r: r + Config.sequence_length]])
      depths.append(Config.sequence_length)
      targets.append(Config.char2int[input_string[r + Config.sequence_length]])

  assert (len(sequences) == len(targets))
  # Define how to randomly split the input data into train and test
  shuffled_list = list(range(len(sequences)))
  random.shuffle(shuffled_list)
  # Determine whether to do a validation split
  if Config.perform_validation():
    split_point = int(Config.training_split_ratio * len(sequences))
  else:
    split_point = len(sequences)

  Config.Train.x = [sequences[idx] for idx in shuffled_list[:split_point]]
  Config.Train.depth = [depths[idx] for idx in shuffled_list[:split_point]]
  Config.Train.t = list(map(lambda idx: _build_target_vector(targets[idx]),
                            shuffled_list[:split_point]))

  if Config.perform_validation():
    Config.Validation.x = [sequences[idx] for idx in shuffled_list[split_point:]]
    Config.Validation.depth = [depths[idx] for idx in shuffled_list[split_point:]]
    Config.Validation.t = list(map(lambda idx: _build_target_vector(targets[idx]),
                                   shuffled_list[split_point:]))


def _build_input_sequence(int_sequence):
  """
  One-Hot Sequence Builder

  Converts a list of integers into a sequence of integers.

  :param int_sequence: List of the character indices
  :type int_sequence: List[int]

  :return: Input sequence converted into a matrix of one hot rows
  :rtype: np.ndarray
  """
  assert (0 < len(int_sequence) <= Config.sequence_length)
  one_hots = []
  while len(one_hots) < Config.sequence_length:
    idx = len(one_hots)
    char_id = 0  # This is used to pad the list as needed
    if idx < len(int_sequence):
      char_id = int_sequence[idx]
    vec = np.zeros([Config.vocab_size()])
    vec[char_id] = 1
    one_hots.append(vec)
  seq = np.vstack(one_hots)
  return seq


def _build_target_vector(idx):
  """
  Creates a one hot vector for the target with "1" in the correct character
  location and zero everywhere else.

  :param idx: Integer corresponding to the expected character
  :type idx: int

  :return: One hot vector for the target character
  :rtype: np.array
  """
  assert (0 <= idx < Config.vocab_size())
  one_hot = np.zeros([Config.vocab_size()])
  one_hot[idx] = 1
  return one_hot


def build_training_and_verification_sets():
  """
  Training and Verification Set Builder

  Builds the training and verification datasets.  Depending on the
  configuration, this may be from the source files or from pickled
  files.
  """
  if not Config.Train.restore:
    input_str = read_input()
    create_examples(input_str)
    # Character to integer map required during text generation
    Config.export_character_to_integer_map()
    # Export the training and verification data in case
    # the previous setup will be trained on aga
    Config.export_train_and_verification_data()

    Config.word_count = len(input_str.split(" "))
  else:
    Config.import_character_to_integer_map()
    Config.import_train_and_verification_data()

  Config.dataset_size = Config.Train.size() + Config.Validation.size()
  _print_basic_text_statistics()


def _print_basic_text_statistics():
  # Print basic statistics on the training set
  logging.info("Total Number of Characters: %d" % Config.dataset_size)
  if Config.word_count > 0:
    logging.info("Total Word Count: \t%d" % Config.word_count)
  logging.info("Vocabulary Size: \t%d" % Config.vocab_size())
  logging.info("Training Set Size: \t%d" % Config.Train.size())
  logging.info("Validation Set Size: \t%d" % Config.Validation.size())


# testing
if __name__ == '__main__':
  build_training_and_verification_sets()
