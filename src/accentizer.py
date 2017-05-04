from keras.models import load_model
import tensorflow as tf
import os.path
import numpy as np

import preprocessor.feedforward_preprocessor as feedforward_preprocessor
import preprocessor.lstm_baseline_preprocessor as lstm_baseline_preprocessor

# VOWELS = ['a', 'e', 'i', 'o', 'u']
VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
MODEL_PATH = '../models'

def accentize(text, network_type, model_id):
    if network_type == 'feedforward':
        accentize_with_feedforward(text, model_id)
    if network_type == 'lstm_baseline':
        return accentize_with_lstm_baseline(text, model_id)

def accentize_with_feedforward(text, model_id):
    windows = feedforward_preprocessor.FeedforwardPreprocessor.preprocess([text], 4)
    accents = {}

    for vowel in VOWEL_TABLE.keys():
        # output = tf.Variable(-1.0, validate_shape=False, name='output')
        
        # saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, 'feedforward', model_id, vowel + '.model.ckpt.meta'))

        # with tf.Session() as sess:
        #     # n_input = tf.placeholder(tf.float32, [None, 240])
        #     # saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, 'feedforward', model_id, vowel + '.model.meta'))

        #     # saver.restore(sess, os.path.join(MODEL_PATH, 'feedforward', model_id, vowel + '.model.ckpt'))
        #     # saver.restore(sess, tf.train.latest_checkpoint(os.path.join(MODEL_PATH, 'feedforward', model_id)))
        #     # output = tf.get_collection("output")[0]
        #     # graph = tf.get_default_graph()
        #     # graph.get

        #     with sess.graph.as_default():
        #         path = os.path.join(MODEL_PATH, 'feedforward', model_id, vowel + '.model')
        #         saver = tf.train.import_meta_graph(path + '.meta')
        #         saver.restore(sess, path)

        #     # saver.restore(sess, tf.train.latest_checkpoint('./'))
        #     # tmp = sess.run(tf.all_variables, feed_dict={n_input: windows[vowel]})
        #     tmp = sess.run(tf.global_variables_initializer())
        #     print(tmp)


        path = os.path.join(MODEL_PATH, 'feedforward', model_id, vowel + '.model')

        print(path)

        # if vowel in ['a', 'e', 'i']:
        #     continue

        # print('s;adflksdlkfsdaf sad       ' + vowel)

        tf.reset_default_graph()


            



        with tf.Session() as sess:

            # input_size = 240
            # output_size = 2

            # if vowel in 'ou':
            #     output_size = 4

            # n_input = tf.placeholder(tf.float32, [None, input_size])
            # n_output = tf.placeholder(tf.float32, [None, output_size])

            # hidden_neurons = 100
            # hidden_neurons2 = 10

            # b_hidden = tf.Variable(tf.random_normal([hidden_neurons]), name='b_hidden')
            # W_hidden = tf.Variable(tf.random_normal(
            #     [input_size, hidden_neurons]), name='W_hidden')
            # hidden = tf.sigmoid(
            #     tf.matmul(n_input, W_hidden) + b_hidden)

            # b_hidden2 = tf.Variable(tf.random_normal([hidden_neurons2]), name='b_hidden2')
            # W_hidden2 = tf.Variable(tf.random_normal(
            #     [hidden_neurons, hidden_neurons2]), name='W_hidden2')
            # hidden2 = tf.sigmoid(
            #     tf.matmul(hidden, W_hidden2) + b_hidden2)

            # W_output = tf.Variable(tf.random_normal(
            #     [hidden_neurons2, output_size]), name='W_output')
            # # output = tf.sigmoid(tf.matmul(hidden, W_output))
            # output = tf.nn.softmax(tf.matmul(hidden2, W_output))



            saver = tf.train.import_meta_graph(path + '.meta')
            # print('s;adflksdlkfsdaf sad       ' + vowel)
            saver.restore(sess, path)

            accents[vowel] = sess.run('output:0', feed_dict={'n_input:0': windows[vowel]})

    # for vowel in VOWEL_TABLE.keys():
    #     print(vowel)
    #     print(accents[vowel])

    vowel_counter = {}
    for vowel in VOWEL_TABLE.keys():
        vowel_counter[vowel] = 0

    accentized_text = ''
    for c in text.lower():
        if c in VOWEL_TABLE.keys():
            accentized_text += decode_accent(c, accents[c][vowel_counter[c]])
            vowel_counter[c] += 1
        else:
            accentized_text += c

    print('feedforward: ' + accentized_text)
    

def accentize_with_lstm_baseline(text, model_id):
    windows = lstm_baseline_preprocessor.LstmBaselinePreprocessor.preprocess([text], 1)
    accents = {}

    # for vowel in VOWEL_TABLE.keys():
    #     print(vowel)
    #     print(windows[vowel])

    for vowel in VOWEL_TABLE.keys():
        model = load_model(os.path.join(MODEL_PATH, 'lstm_baseline', model_id, vowel + '.model'))
        accents[vowel] = model.predict(np.array(windows[vowel]))

    # for vowel in VOWEL_TABLE.keys():
    #     print(vowel)
    #     print(accents[vowel])

    vowel_counter = {}
    for vowel in VOWEL_TABLE.keys():
        vowel_counter[vowel] = 0

    accentized_text = ''
    for c in text.lower():
        if c in VOWEL_TABLE.keys():
            accentized_text += decode_accent(c, accents[c][vowel_counter[c]])
            vowel_counter[c] += 1
        else:
            accentized_text += c

    print('lstm_baseline: ' + accentized_text)

        
def decode_accent(vowel, coded_accent):
    i = np.array(coded_accent).argmax(axis=0)
    return VOWEL_TABLE[vowel][i]

# accentize('Akar eleg ozek izu', 'feedforward', '2017-05-04-4_0')
# accentize('Akar eleg ozek izu', 'lstm_baseline', '2017-04-30_0')

accentize('arvizturo tukorfurogep', 'feedforward', '2017-05-04-4_0')
accentize('arvizturo tukorfurogep', 'lstm_baseline', '2017-04-30_0')