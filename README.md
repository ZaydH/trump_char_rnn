# trump_char_rnn
Character-Level RNN to Generate Speech in the Style of Donald Trump

## Authors
[Zayd Hammoudeh](https://users.soe.ucsc.edu/~zayd/) (zayd@ucsc.edu)  
[Ben Sherman](https://bcsherma.wordpress.com/) (bcsherma@ucsc.edu)

## Course
CMPS242 - Machine Learning  
University of California - Santa Cruz  
Prof. Manfred Warmuth  
Fall 2017

## Summary
Character-level recurrent neural network that generates text in the style of Donald Trump.


## Running the Program
1. Python Version: 3.5.\* or 3.6.\*
  * The code was specifically tested on 3.5.3 and 3.6.3 (Anaconda build).
2. Required Libraries
  * TensorFlow
  * Numpy
  * pickle
  * enum
  * logging
  * argparse
3. Training the Learner
  * Run the file `train.py`.  Default settings should be sufficient.
  * It will automatically checkpoint the model every two epochs.
  * We advise running at least 20 epochs (about 20 hours on our machine) to get an output of sufficient quality.
4. Generating the Text
  * Run the file trump.py`.  You must specify an input text seed and the character decision engine algorithm.
  * An example entry is:  
`python3 trump.py --seed "We will make America " --decision 11`


**Note**: For more details on the parameters for both `train.py` and `trump.py`, you can run the program with the flag `--help`.
