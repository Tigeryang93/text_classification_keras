import os
import sys
import numpy as np
from sklearn.metrics import make_scorer, f1_score, precision_score, accuracy_score, log_loss, classification_report, confusion_matrix
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing import sequence
from fasttext import FastText

sys.path.append('../')
import data_helpers

# 指定使用gpu显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    # add ngram into sequence
    new_sequences = []
    for input_list in sequences:
        new_list = list(input_list[:])
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


# ngram_range = 2 will add bi-grams features
ngram_range = 3
batch_size = 64
embedding_dim = 128
epochs = 100

print('Loading data...')
# get data
x_train, y_train, x_test, y_test, word2index = data_helpers.preprocess()
max_features = len(word2index)

print('get ngram feature...')
if ngram_range > 1:
    print(str(ngram_range)+'-gram')
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)

max_len = max(len(x) for x in x_train)
print(max_len)

print('Pad sequences...')
x_train = sequence.pad_sequences(x_train, maxlen=max_len, value=0)
x_test = sequence.pad_sequences(x_test, maxlen=max_len, value=0)

print('Build model...')
model = FastText(max_len, embedding_dim, batch_size=batch_size, class_num=2, max_features=max_features, epochs=epochs)

print('Train...')
model.fit(x_train, x_test, y_train, y_test)

print('Test...')
result = model.predict(x_test)
result = np.argmax(np.array(result), axis=1)
y_test = np.argmax(np.array(y_test), axis=1)

print('f1:', f1_score(y_test, result, average='macro'))
print('accuracy:', accuracy_score(y_test, result))
print('classification report:\n', classification_report(y_test, result))
print('confusion matrix:\n', confusion_matrix(y_test, result))
