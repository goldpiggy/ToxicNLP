import datetime
import os
# import sys
# import codecs
# import csv
import argparse
import logging
import numpy as np
import pandas as pd
import re
import cPickle
import wordninja
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from string import punctuation

# from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Lambda, Dense, Input, LSTM, Embedding, Dropout, CuDNNLSTM, CuDNNGRU
from keras.regularizers import l1_l2, l1
from keras.layers import concatenate, GlobalMaxPooling1D, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Bidirectional

# from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

# from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from keras import optimizers
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
glove = cPickle.load(open("/home/rachel/.Experiment/Toxic/word_vector/glove.840B.300d.pkl"))


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


special_character_removal = re.compile(r'[^a-z\d\!\?\'\*\- ]', re.IGNORECASE)  # Regex to remove all Non-Alpha Numeric and space
replace_numbers = re.compile(r'\d+', re.IGNORECASE)    # regex to replace all numerics

'''
def text_to_wordlist(text, glove_vector, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # text = nltk.word_tokenize(unicode(text).encode('ascii', 'ignore'))

    # if remove_stopwords:
    #     stops = set(stopwords.words("english"))
    #     text = [w for w in text if not w in stops]
    #Remove Special Characters, numbers
    original_text = text
    try:
        text = special_character_removal.sub('', text)
        text = replace_numbers.sub('n', text)
    except Exception:
        # ipdb.set_trace()
        text = original_text
    # if str(text) != str(original_text):
    #     ipdb.set_trace()

    # if stem_words:
    #     text = text.split()
    #     stemmer = SnowballStemmer('english')
    #     stemmed_words = [stemmer.stem(word) for word in text]
    #     text = " ".join(stemmed_words)
    # print("Text: {}".format(text))
    words = []
    text_tokens = text.split(" ")
    #print(text_tokens)
    for token in text_tokens:
        if token in glove_vector:
        words += token
        else:
        words += wordninja.split(token.lower())
    text = " ".join(words)
    #print(text)
    return(text)
'''


def text_to_wordlist(text, remove_stopwords=False, Lemmatizer=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # text = nltk.word_tokenize(unicode(text).encode('ascii', 'ignore'))

    #Remove Special Characters, numbers
    original_text = text
    ## txt = "I will have the shittttttyyyyyy, bitch fu*k your! moTHer !!!!! do not donot who'll've."
    try:
        text = special_character_removal.sub('', text)
        # for word in text.split():
        #     if word.lower() in contractions:
        #         text = text.replace(word, contractions[word.lower()])
        # ipdb.set_trace()  # txt = 'I will have the shittttttyyyyyy bitch fu*k your! moTHer !!!!! do not donot who shall have who will have'
        text = text.replace('-', ' ')
        text = text.replace('!', ' !')
        text = text.replace('?', ' ?')
        text = text.replace("'", '')
        text = replace_numbers.sub('n', text)
    except Exception:
        # ipdb.set_trace()
        text = original_text
    if Lemmatizer:
        words = text.split()
        # words = [stemmer.stem(word) for word in words]
        words = [lem.lemmatize(word) for word in words]
        words = [lem.lemmatize(word, 'v') for word in words]
        clean_text = " ".join(words)
        # row.loc['lemmatuzed_text'] = text
    # if stem_words:
    #     text = text.split()
    #     stemmer = SnowballStemmer('english')
    #     stemmed_words = [stemmer.stem(word) for word in text]
    #     text = " ".join(stemmed_words)
    # print("Text: {}".format(text))
    words = []
    for token in text.split(" "):
        #if token.lower() in ["wtf"]:
        #    token = ["What", "the", "fuck"]
        #    words += token
        #elif glove.get(token) is None:  ####################################################### Things to do : add spell corrector
        #    words += wordninja.split(token.lower())
        #else :
        #    words.append(token)
        if glove.get(token) is not None:
            words.append(token)
        else:
            words += wordninja.split(token.lower())

    if remove_stopwords:
        words = [w for w in words if not w in stops]
    text = " ".join(words)
    # print(text)   # 'I will have the shit ttt tty yyyy y bitch fu k your ! mother ! ! ! ! ! do not donot who shall have who will have'
    return text

def save_training_prediction_corrected(x_train, model, train_df, list_classes, training_file_path, batch_size=1024, output_path="train_prediction.csv",):
    print "SAVE CODE CHANGE SUCCESSFULLY !!!"
    ## save train prediction to csv
    y_train_pred = model.predict([x_train], batch_size=batch_size, verbose=1)
    train_df_pred = train_df
    train_df_pred[list_classes] = y_train_pred
    train_df_pred.columns = ['id', 'comment_text'] + [c+'_pred' for c in list_classes]
    train_df = train_df.set_index(['id'])
    train_df_pred = train_df_pred.set_index(['id'])
    train_df_pred = train_df_pred[[c+'_pred' for c in list_classes]]

    train_df_ori = pd.read_csv(training_file_path,index_col='id')
    train_concat = train_df_ori.join(train_df_pred)
    train_concat.to_csv(output_path, index=True)
    print("Saving training set prediction to {}".format(output_path))


def save_training_prediction(x_train, model, train_df, list_classes, training_file_path, batch_size=1024, output_path="train_prediction.csv",):
    ## save train prediction to csv
    y_train_pred = model.predict([x_train], batch_size=batch_size, verbose=1)
    train_df_pred = train_df
    train_df_pred[list_classes] = y_train_pred
    train_df_pred.columns = ['id', 'comment_text'] + [c+'_pred' for c in list_classes]
    train_df = train_df.set_index(['id'])
    train_df_pred = train_df_pred.set_index(['id'])
    train_df_pred = train_df_pred[[c+'_pred' for c in list_classes]]
    train_df_ori = pd.read_csv(training_file_path)
    train_concat = pd.concat([train_df_ori, train_df_pred], axis=1)
    train_concat.to_csv(output_path, index=True)
    print("Saving training set prediction to {}".format(output_path))


def setup_logging(file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName("DEBUG"))
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.getLevelName("DEBUG"))
    file_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.getLevelName("DEBUG"))
    console_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
    logger.addHandler(console_handler)


def get_glove_vector(glove_vector_path):
    print('Indexing word vectors')
    embeddings_index = cPickle.load(open(glove_vector_path))
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index


def prepare_embedding_matrix(embeddings_index, word_index, embedding_dimension=300, max_nb_words=400000):
    #print('Indexing word vectors')
    #embeddings_index = cPickle.load(open(glove_vector_path))
    #print('Total %s word vectors.' % len(embeddings_index))
    # with open('glove.840B.300d.pkl', 'wb') as handle:
    #   cPickle.dump(embeddings_index, handle, protocol=cPickle.HIGHEST_PROTOCOL)
    # handle.close()

    print('Preparing embedding matrix')
    nb_words = min(max_nb_words, len(word_index)) + 1
    print("nb_words: {}".format(nb_words))
    embedding_matrix = np.zeros((nb_words, embedding_dimension))

    unkown_word = []
    for word, i in word_index.iteritems():
        if i >= max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            print word
            #split_words = wordninja.split(word)
                #split_words_check = [embeddings_index.get(word) is not None for word in split_words]
            #ipdb.set_trace()
            ## if split_words has over 50% can be found in glove, split the word
            unkown_word.append(word)

    print("Unknown words len {}, total unique words {}".format(len(unkown_word), len(word_index)))
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return nb_words, embedding_matrix


def tokenize_input_data(train_df, test_df, list_classes, comment_column="comment_text", max_nb_words=400000, max_sequence_length=150):
    # list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # y_train = train_df[list_classes].values
    X_train = [text_to_wordlist(txt) for txt in train_df[comment_column].fillna("NA").values]
    X_test = [text_to_wordlist(txt) for txt in test_df["comment_text"].fillna("NA").values]

    tokenizer = Tokenizer(num_words=max_nb_words, lower=False)
    tokenizer.fit_on_texts(X_train + X_test)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    word_index = tokenizer.word_index
    logging.info('Found %s unique tokens' % len(word_index))

    training_data = pad_sequences(train_sequences, padding="post", truncating="post", maxlen=max_sequence_length)
    logging.info('Shape of data tensor: {}'.format(training_data.shape))
    # logging.info('Shape of label tensor: {}'.format(y_train.shape))

    test_data = pad_sequences(test_sequences, padding="post", truncating="post", maxlen=max_sequence_length)
    logging.info('Shape of test_data tensor: {}'.format(test_data.shape))
    return training_data, test_data, word_index


def get_train_labels(train_df, list_classes):
    y_train = train_df[list_classes].values
    logging.info('Shape of label tensor: {}'.format(y_train.shape))
    return y_train


def parse_input_data(training_file_path, test_file_path):
    train_df = pd.read_csv(training_file_path)
    test_df = pd.read_csv(test_file_path)
    return train_df, test_df


def get_val_data(x_train, y_train, validation_split=0.1):
    perm = np.random.permutation(len(x_train))
    #idx_train = perm[:int(len(x_train)*(1-validation_split)) - 434]
    #idx_val = perm[int(len(x_train)*(1-validation_split)) - 434: -107]

    idx_train = perm[:int(len(x_train)*(1-validation_split))]
    idx_val = perm[int(len(x_train)*(1-validation_split)):]

    data_train = x_train[idx_train]
    labels_train = y_train[idx_train]
    logging.info("train_data shape: {}, train_label shape: {}".format(data_train.shape, labels_train.shape))

    data_val = x_train[idx_val]
    labels_val = y_train[idx_val]

    logging.info("val_data shape: {}, val_label shape: {}".format(data_val.shape, labels_val.shape))
    return data_train, labels_train, data_val, labels_val


def build_monitor(monitor='val_acc', patience=15):
    early_stopping = EarlyStopping(monitor=monitor, patience=patience)
    return early_stopping


def build_checkpoint(model_path='best_model.h5', save_best_only=True, save_weights_only=True):
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=save_best_only, save_weights_only=save_weights_only)
    return model_checkpoint


def auc_score(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)    
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def build_model_new(nb_words, embedding_matrix, embedding_dimension=300, max_sequence_length=150, num_lstm=300,
                rate_drop_lstm=0.25, recurrent_dropout=0.25, rate_drop_dense=0.25, num_dense=256,
                return_sequences=True, trainable=False, number_class=6):

    embedding_layer = Embedding(nb_words, embedding_dimension, weights=[embedding_matrix], input_length=max_sequence_length, trainable=trainable)
    lstm_layer_1 = LSTM(300, stateful=False, dropout=rate_drop_lstm, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences)
    # lstm_layer_1 = LSTM(num_lstm, stateful=False, dropout=rate_drop_lstm, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences)


    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    #x = Dense(128, activation='relu')(embedded_sequences)
    
    x = SpatialDropout1D(0.2)(embedded_sequences)
    x = lstm_layer_1(x)
    #merged = Attention(max_sequence_length)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    #merged = Dense(128, activation='relu')(conc)
    #merged = Dense(64, activation='relu')(merged)
    #merged = Dense(32, activation='relu')(merged)
    preds = Dense(6, activation='sigmoid')(conc)
    model = Model(inputs=[comment_input], outputs=preds)
    
    optimizer = optimizers.RMSprop(lr=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', auc_score])

    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', auc_score])
    model.summary()
    return model


def build_model(nb_words, embedding_matrix, embedding_dimension=300, max_sequence_length=150, num_lstm=300,
                rate_drop_lstm=0.25, recurrent_dropout=0.25, rate_drop_dense=0.25, num_dense=256,
                return_sequences=True, trainable=False, number_class=6):

    embedding_layer = Embedding(nb_words, embedding_dimension, weights=[embedding_matrix], input_length=max_sequence_length, trainable=trainable)
    lstm_layer_1 = Bidirectional(CuDNNLSTM(300, stateful=False, return_sequences=return_sequences), merge_mode='concat')
    #lstm_layer_1 = Bidirectional(LSTM(300, stateful=False, dropout=rate_drop_lstm, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences), merge_mode='concat')
    # lstm_layer_1 = LSTM(num_lstm, stateful=False, dropout=rate_drop_lstm, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences)


    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    #x = Dense(300, activation='relu')(embedded_sequences)
    x = SpatialDropout1D(0.5)(embedded_sequences)
    x = lstm_layer_1(x)
    x = Dropout(0.5)(x)
    merged = Attention(max_sequence_length)(x)
    merged = Dropout(0.5)(merged)
    #merged = Dense(128, activation='relu')(merged)
    #merged = Dropout(0.5)(merged)
    #merged = Dense(64, activation='relu')(merged)
    #merged = Dense(32, activation='relu')(merged)
    preds = Dense(6, activation='sigmoid')(merged)
    model = Model(inputs=[comment_input], outputs=preds)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', auc_score])
    model.summary()
    return model

def build_model_sigmoid_feb9(nb_words, embedding_matrix, embedding_dimension=300, max_sequence_length=128, num_lstm=128,
                rate_drop_lstm=0.25, recurrent_dropout=0.25, rate_drop_dense=0.25, num_dense=128, num_dense_2=64,
                return_sequences=True, trainable=False, number_class=6):

    embedding_layer = Embedding(nb_words, embedding_dimension, weights=[embedding_matrix], input_length=max_sequence_length, trainable=trainable)
    lstm_layer_1 = LSTM(num_lstm, recurrent_activation='tanh', stateful=False, dropout=rate_drop_lstm, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences)

    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    x = Dense(num_dense, activation='tanh')(embedded_sequences)
    #x = SpatialDropout1D(0.2)(x)
    x = lstm_layer_1(x)
    merged = Attention(max_sequence_length)(x)
    #merged = Dropout(0.25)(merged)
    merged = Dense(num_dense, activation='tanh')(merged)
    #merged = Dropout(0.5)(merged)
    merged = Dense(64, activation='tanh')(merged)
    #merged = Dense(32, activation='relu')(merged)
    preds = Dense(6, activation='sigmoid')(merged)
    model = Model(inputs=[comment_input], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', auc_score])
    model.summary()
    return model


def build_convd_model(nb_words, embedding_matrix, embedding_dimension=300, max_sequence_length=150, return_sequences=True, trainable=False, number_class=6):
    embedding_layer = Embedding(nb_words, embedding_dimension, weights=[embedding_matrix], input_length=max_sequence_length, trainable=trainable)
    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    x = Conv1D(512, 3, activation='relu')(embedded_sequences)
    x = Conv1D(512, 3, activation='relu')(x)
    x = Conv1D(512, 3, activation='relu')(x)
    x = Conv1D(512, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(512, 3, activation='relu')(x)
    x = Conv1D(512, 3, activation='relu')(x)
    x = Conv1D(512, 3, activation='relu')(x)
    x = Conv1D(512, 3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.25)(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[comment_input], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model


def model_train(model, data_train, labels_train, data_val, labels_val, epochs=25, batch_size=256, shuffle=True, callbacks=None):
    hist = model.fit(data_train, labels_train, validation_data=(data_val, labels_val),
                     epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                     callbacks=callbacks)

    return hist


def model_predict(model, model_path, test_data, batch_size=1024, verbose=1):
    model.load_weights(model_path)
    return model.predict([test_data], batch_size=batch_size, verbose=verbose)


def make_submission(y_test, list_classes, output_path="submission.csv", sample_path="./sample_submission.csv"):
    sample_submission = pd.read_csv(sample_path)
    sample_submission[list_classes] = y_test
    sample_submission.to_csv(output_path, index=False)
    logging.info("Saving test set prediction to {}".format(output_path))


def create_submission(sample_path="./sample_submission.csv"):
    sample_submission = pd.read_csv(sample_path)
    return sample_submission


def write_submission(df_submission, y_test, list_classes):
    df_submission[list_classes] = y_test
    return df_submission


def dump_submission(df_submission, output_path="submission.csv"):
    df_submission.to_csv(output_path, index=False)
    logging.info("Saving test set prediction to {}".format(output_path))


def create_working_dir(path):
    os.makedirs(path)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max-nb-words', dest="MAX_NB_WORDS", default=100000, type=int, help="Maximum number of words in Vocabulary")
    parser.add_argument('--num-lstm', dest="NUM_LSTM", default=128, type=int, help="Number of units for LSTM layer")
    parser.add_argument('--num-dense', dest="NUM_DENSE", default=128, type=int, help="Number of Dense Units")
    parser.add_argument('--max-sequence-length', dest="MAX_SEQUENCE_LENGTH", default=150, type=int, help="Length of input sequence length. i.e max number of words in a text")
    parser.add_argument('--rate-drop-lstm', dest="RATE_DROP_LSTM", default=0.5, type=float, help="Fraction of the units to drop for the linear transformation of the inputs in LSTM layer")
    parser.add_argument('--recurrent-dropout', dest="RECURRENT_DROPOUT", default=0.5, type=float, help="Fraction of the units to drop for the linear transformation of the recurrent state for LSTM layer")
    parser.add_argument('--rate-drop-dense', dest="RATE_DROP_DENSE", default=0.5, type=float, help="Dropout rate for the Dense layer")
    parser.add_argument('--embedding-dimension', dest="EMBEDDING_DIMENSION", default=300, type=int, help="Dimension of which word will be embedded. Should match the glove vector dimension")
    parser.add_argument('--epoch', dest="MAX_EPOCH", default=25, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch-size', dest="BATCH_SIZE", default=256, type=int, help="Number of samples per gradient update")
    parser.add_argument('--monitor', dest="MONITOR_TARGET", default="val_acc", type=str, help="Monitoring the val_acc/val_loss after each epoch, default val_acc")
    parser.add_argument('--patience', dest="MONITOR_PATIENCE", default=10, type=int, help="Stop the training process if val_acc/val_loss didn't improve after --patience epochs")
    parser.add_argument('--validation-split', dest="VALIDATION_SPLIT", default=0.1, type=float, help="Percentage of training set used for validation")
    parser.add_argument('--glove-vector-path', dest="GLOVE_PATH", default="/home/rachel/.Experiment/Toxic/word_vector/glove.840B.300d.txt", type=str, help="path to the glove vector")
    parser.add_argument('--training-data-path', dest="TRAIN_PATH", default="/home/rachel/.Experiment/Toxic/new_data/train.csv", type=str, help="path to the training data file")
    parser.add_argument('--test-data-path', dest="TEST_PATH", default="/home/rachel/.Experiment/Toxic/new_data/test.csv", type=str, help="path to the test data file")
    parser.add_argument('--sample-submission', dest="SAMPLE_PATH", default="/home/rachel/.Experiment/Toxic/new_data/sample_submission.csv", type=str, help="path to the sample submission file")
    parser.add_argument('--output-dir', dest="OUTPUT_DIR", default="./", type=str, help="Location where the output will be store")
    parser.add_argument('--reduce-learning', dest="REDUCE_LEARNING", action='store_true', help="Reduce learning-rate alpha if validation score hasn't improved during x patience")
    parser.add_argument('--reduce-learning-patience', dest="REDUCE_LEARNING_PATIENCE", type=int, default=5, help="Number of epoch to wait before reducing the learning rate")
    parser.add_argument('--reduce-learning-factor', dest="REDUCE_LEARNING_FACTOR", type=float, default=0.2, help="factor by which the learning rate will be reduced. new_lr = lr * factor")
    parser.add_argument('--debug', dest="DEBUG_ENABLE", action='store_true', help="Will activate debugger if breakpoints are set")
    parser.add_argument('--classes', dest="CLASSES", default=None, type=str, nargs='*', help="Will activate debugger if breakpoints are set")
    parser.add_argument('--comment', dest="COMMENT_COLUMN", default="comment_text", type=str, help="Header of comment text of the training csv")

    # TODO: Adding regularization parameters for Dense & Atthention layer. Or maybe it is not necessary

    args = parser.parse_args()

    # Create output directory
    working_dir = os.path.join(args.OUTPUT_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    create_working_dir(working_dir)

    log_path = os.path.join(working_dir, "training.log")
    setup_logging(log_path)

    logging.info("Training with the following parameters {}: ".format(args))
    logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!! Training Begin !!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Parse input data
    training_file_path = args.TRAIN_PATH
    test_file_path = args.TEST_PATH
    train_df, test_df = parse_input_data(training_file_path, test_file_path)

    # Tokenize the input data
    max_nb_words = args.MAX_NB_WORDS
    max_sequence_length = args.MAX_SEQUENCE_LENGTH
    x_train, test_data, word_index = tokenize_input_data(train_df, test_df, LIST_CLASSES, max_nb_words=max_nb_words, max_sequence_length=max_sequence_length)
    y_train = get_train_labels(train_df, LIST_CLASSES)

    # Construct Embedding Matrix using the glove vector
    glove_vector_path = args.GLOVE_PATH
    embedding_dimension = args.EMBEDDING_DIMENSION
    #glove_vector = get_glove_vector(glove_vector_path)
    nb_words, embedding_matrix = prepare_embedding_matrix(glove, word_index, embedding_dimension=embedding_dimension, max_nb_words=max_nb_words)

    # sample train/validation data
    validation_split = args.VALIDATION_SPLIT
    data_train, labels_train, data_val, labels_val = get_val_data(x_train, y_train, validation_split=validation_split)

    # create callbacks. One for early stoping, other for model saving
    monitor = args.MONITOR_TARGET
    patience = args.MONITOR_PATIENCE
    model_monitor = build_monitor(monitor=monitor, patience=patience)

    out_model_path = os.path.join(working_dir, "best_model.h5")
    model_checkpoint = build_checkpoint(model_path=out_model_path, save_best_only=True, save_weights_only=True)

    csv_file = os.path.join(working_dir, "metrics.log")
    csv_callback = CSVLogger(csv_file, separator=',', append=False)

    callbacks = [model_monitor, model_checkpoint, csv_callback]

    factor = args.REDUCE_LEARNING_FACTOR
    patience = args.REDUCE_LEARNING_PATIENCE
    if args.REDUCE_LEARNING:
        reduce_learning = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=0.001)
        callbacks.append(reduce_learning)

    # Build/Compile the Model
    num_lstm = args.NUM_LSTM
    rate_drop_lstm = args.RATE_DROP_LSTM
    recurrent_dropout = args.RECURRENT_DROPOUT
    rate_drop_dense = args.RATE_DROP_DENSE
    num_dense = args.NUM_DENSE
    model = build_model_new(nb_words, embedding_matrix, embedding_dimension=embedding_dimension, max_sequence_length=max_sequence_length, num_lstm=num_lstm,
                        rate_drop_lstm=rate_drop_lstm, recurrent_dropout=recurrent_dropout, rate_drop_dense=rate_drop_dense, num_dense=num_dense,
                        return_sequences=True, trainable=False)

    #model = build_model_sigmoid_feb9(nb_words, embedding_matrix, embedding_dimension=embedding_dimension, max_sequence_length=max_sequence_length, num_lstm=num_lstm,
    #                    rate_drop_lstm=rate_drop_lstm, recurrent_dropout=recurrent_dropout, rate_drop_dense=rate_drop_dense, num_dense=num_dense,
    #                    return_sequences=True, trainable=False)


    # Train the model
    epochs = args.MAX_EPOCH
    batch_size = args.BATCH_SIZE

    model.load_weights("/home/superuser/.Experiment/Toxic/output/0.9851-2018-02-26_23-15-49/best_model.h5")
    hist = model_train(model, data_train, labels_train, data_val, labels_val, epochs=epochs, batch_size=batch_size, callbacks=callbacks, shuffle=True)

    if monitor == "val_acc":
        bst_val_score = max(hist.history['val_acc'])
    else:
        bst_val_score = min(hist.history['val_loss'])

    logging.info("Best Validation score: {}".format(bst_val_score))

    # Test the Model
    STAMP = str('%.4f_' % (bst_val_score))
    out_test_predict = os.path.join(working_dir, "{}_pred_test.csv".format(STAMP))
    sample_path = args.SAMPLE_PATH
    y_test = model_predict(model, out_model_path, test_data, batch_size=1024, verbose=1)
    make_submission(y_test, LIST_CLASSES, output_path=out_test_predict, sample_path=sample_path)
    out_train_predict = os.path.join(working_dir, "{}_pred_train.csv".format(STAMP))
    try:
        save_training_prediction_corrected(x_train, model, train_df, LIST_CLASSES, training_file_path, batch_size=batch_size, output_path=out_train_predict)
    except:
        save_training_prediction(x_train, model, train_df, LIST_CLASSES, training_file_path, batch_size=batch_size, output_path=out_train_predict)

    if args.DEBUG_ENABLE:
        # pred outside dataset : insult data set
        # ipdb.set_trace()
        insulted_df = pd.read_csv("~/Experiment/Toxic/data/insult/train.csv")
        insulted_df['id'] = range(test_df_max_id + 1, test_df_max_id + 3947 + 1)
        insulted_df.columns = [u'Insult', u'Date', u'comment_text', u'id']
        # insulted_df_pred = pd.DataFrame(columns = ['comment_text']+LIST_CLASSES)
        out_insult_test_predict = os.path.join(working_dir, "{}_pred_insult_test.csv".format(STAMP))
        train_df = pd.read_csv(training_file_path)
        x_train_new, y_train_new, insult_test_data, word_index_new = tokenize_input_data(train_df, insulted_df, LIST_CLASSES, max_nb_words=max_nb_words, max_sequence_length=max_sequence_length)
        insult_y_test = model_predict(model, out_model_path, insult_test_data, batch_size=1024, verbose=1)
        insulted_df[LIST_CLASSES] = pd.DataFrame(insult_y_test)
        insulted_df.to_csv(out_insult_test_predict, index=True)
        logging.info("Saving testing set prediction to {}".format(out_insult_test_predict))

    logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!! DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    main()

