import tensorflow as tf
import logging
import data_parser
import network
import random
from basic_config import Config


def run_training():
  net_features = network.construct()

  input_x = net_features["X"]
  target = net_features["target"]
  seq_len = net_features["seq_len"]

  # Setup the training procedure
  cross_h = tf.nn.softmax_cross_entropy_with_logits(logits=net_features["output"],
                                                    labels=target)
  
  loss_op = tf.reduce_sum(cross_h)
  optimizer = tf.train.AdamOptimizer(learning_rate=Config.Train.learning_rate)
  
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tvars), 5.)
  
  global_step = tf.get_variable('global_step', [],
                                initializer=tf.constant_initializer(0.0))
  
  train_op = optimizer.apply_gradients(zip(grads, tvars),
                                       global_step=global_step)
  # train_op = optimizer.minimize(loss_op)

  sess = tf.Session()
  if Config.Train.restore:
    Config.import_model(sess)
  else:
    sess.run(tf.global_variables_initializer())

  num_batches = 0

  for epoch in range(0, Config.Train.num_epochs):
    # Shuffle the batches for each epoch
    shuffled_list = list(range(Config.Train.size()))
    random.shuffle(shuffled_list)
    train_err = 0
    for batch in range(0, Config.Train.num_batch()):
      end_batch = min((batch + 1) * Config.batch_size, Config.Train.size())
      start_batch = max(0, end_batch - Config.batch_size)

      # Use the randomized batches
      train_x = list(map(lambda idx: Config.Train.x[idx], shuffled_list[start_batch:end_batch]))
      train_t = list(map(lambda idx: Config.Train.t[idx], shuffled_list[start_batch:end_batch]))
      seqlen = list(map(lambda idx: Config.Train.depth[idx], shuffled_list[start_batch:end_batch]))

      _, err = sess.run([train_op, loss_op], feed_dict={input_x: train_x, target: train_t,
                                                        seq_len: seqlen})
      train_err += err

      num_batches += 1
      BATCH_PRINT_FREQUENCY = 1000
      if num_batches % BATCH_PRINT_FREQUENCY == 0:
        print("Epoch %d: Total Batches %d: Last Batch Error: %0.3f" % (epoch, num_batches, err))

    # ToDo It would be nice to add perplexity here.
    logging.info("EPOCH #%05d COMPLETED" % epoch)
    train_err /= Config.Train.num_batch()
    logging.info("Epoch %05d: Average Batch Training Error: \t\t%0.3f" % (epoch, train_err))

    if Config.perform_validation():
      test_err = _calculate_validation_error(sess, loss_op, input_x, target, seq_len)
      logging.info("Epoch %05d: Average Batch Verification Error: \t%0.3f" % (epoch, test_err))

    if epoch % Config.Train.checkpoint_frequency == 0:
      Config.export_model(sess, epoch)

  sess.close()


def _calculate_validation_error(sess, loss_op, input_x, target, seq_len):
  """
  Determines the validation error
  """
  validation_err = 0
  for batch in range(0, Config.Validation.num_batch()):
    end_batch = min((batch + 1) * Config.batch_size, Config.Validation.size())
    start_batch = max(0, end_batch - Config.batch_size)

    # Use the randomized batches
    valid_x = Config.Validation.x[start_batch:end_batch]
    valid_t = Config.Validation.t[start_batch:end_batch]
    seqlen = Config.Validation.depth[start_batch:end_batch]
    err = sess.run(loss_op, feed_dict={input_x: valid_x, target: valid_t,
                                       seq_len: seqlen})
    validation_err += err
  validation_err /= Config.Validation.num_batch()
  return validation_err


if __name__ == "__main__":
  Config.parse_args()
  data_parser.build_training_and_verification_sets()

  run_training()
