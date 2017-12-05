"""

"""
import network
import os
from basic_config import Config
import tensorflow as tf
import logging


def generate_text():
  """
  Generates the text that is hopefully Trumpian.
  """
  # Minimize TF warnings which are not helpful in generate mode.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tf.logging.set_verbosity(tf.logging.ERROR)

  net_features = network.construct()

  sess = tf.Session()
  Config.import_model(sess)
  # Enable the user to enter multiple text strings with a do-while loop
  while True:
    _generate_output(sess, net_features)

    if not Config.Generate.loop:
      break

    while True:
      print("")
      logging.info("Please supply a new seed text then press enter when complete: ")
      Config.Generate.seed_text = input("")
      if len(Config.Generate.seed_text) > Config.Generate.min_seed_len:
        print("You entered: \"" + Config.Generate.seed_text + "\"")
        print("")
        break
      logging.info("Invalid Seed Text.  Must be at least %d characters long"
                   % Config.Generate.min_seed_len)

  sess.close()

def _generate_output(sess, net_features):
  x = net_features["X"]
  get_softmax = tf.nn.softmax(net_features["output"])[0, :]
  seq_len = net_features["seq_len"]

  cur_seq_len = min(len(Config.Generate.seed_text), Config.sequence_length)
  input_x = Config.Generate.build_initial_x()
  generated_text = []
  # Generate the text character by character
  while len(generated_text) < Config.Generate.output_len:
    phrase_seq_len = [cur_seq_len] * Config.batch_size

    softmax_out = sess.run(get_softmax, feed_dict={x: input_x, seq_len: phrase_seq_len})

    pred_char_id = Config.DecisionEngine.function(sess, softmax_out)

    pred_char = Config.Generate.int2char()[pred_char_id]
    generated_text.append(pred_char)
    if cur_seq_len == Config.sequence_length:
      # Delete off the front of the list if it has reached the specified sequence length
      del input_x[0][0]
    else:
      # Shave last dummy element off since fixed batch size
      del input_x[0][Config.sequence_length - 1]
      cur_seq_len += 1
    input_x[0].insert(cur_seq_len - 1, pred_char_id)
    Config.Generate.prev_char = pred_char

  logging.info("Output Text: " + Config.Generate.seed_text + "".join(generated_text))


if __name__ == "__main__":
  Config.parse_args()
  Config.import_character_to_integer_map()
  Config.Generate.build_int2char()
  Config.Generate.build_seed_x()

  generate_text()
