"""
 network.py

 Construct RNN for character level text prediction
"""

import tensorflow as tf
import data_parser
from feed_forward import setup_feed_forward
from basic_config import Config


def construct():
    """
    Trump Neural Network Constructor

    Builds all layers of the neural network.
    """

    # create data input placeholder
    input_x = tf.placeholder(tf.int32, shape=[Config.batch_size, None])

    # create target input placeholder
    target = tf.placeholder(tf.float32, shape=[Config.batch_size, Config.vocab_size()])

    # Create the embedding matrix
    embed_matrix = tf.get_variable("word_embeddings",
                                   [Config.vocab_size(), Config.RNN.hidden_size])
    
    embedded = tf.nn.embedding_lookup(embed_matrix, input_x)

    # create RNN cell
    cells = []
    for _ in range(Config.RNN.num_layers):
        cells.append(tf.nn.rnn_cell.BasicLSTMCell(Config.RNN.hidden_size))
    if Config.is_train() or Config.Generate.enable_dropout:
        cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
                 for cell in cells]

    #     cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.8,
    #                                            state_keep_prob=1.0) for cell in cells]
    # else:
    #     cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.98, output_keep_prob=0.98,
    #                                            state_keep_prob=1.0) for cell in cells]
    # get rnn outputs
    seq_len = tf.placeholder(tf.int32, shape=[Config.batch_size])
    
    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    rnn_output, rnn_state = tf.nn.dynamic_rnn(multi_cell, embedded,
                                              sequence_length=seq_len,
                                              dtype=tf.float32)

    # transpose rnn_output into a time major form
    seq_end = tf.range(Config.batch_size) * tf.shape(rnn_output)[1] + (seq_len - 1)
    rnn_final_output = tf.gather(tf.reshape(rnn_output, [-1, Config.RNN.hidden_size]), seq_end)

    softmax_out = setup_feed_forward(rnn_final_output)

    final_output = softmax_out
    return {'X': input_x, 'target': target,
            'seq_len': seq_len, 'output': final_output}


def run():
    """
    run a tensor flow session and try feeding the network stuff.
    just for testing right now
    """
    # start the session
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)


# main
if __name__ == '__main__':

    data_parser.build_training_and_verification_sets()
    network_features = construct()

    #
    run()
