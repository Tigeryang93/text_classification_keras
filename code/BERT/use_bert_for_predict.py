import re
import tensorflow as tf
import numpy as np
from sklearn.metrics import make_scorer, f1_score, precision_score, accuracy_score, log_loss, classification_report, confusion_matrix
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend.tensorflow_backend as KTF


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_bert_sent_embedding():
    # get data label
    positive_data_file = '../../data/rt-polaritydata/rt-polarity.pos'
    negative_data_file = '../../data/rt-polaritydata/rt-polarity.neg'
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    encoded_text = np.load('encoded_text.npy')

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = encoded_text[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_percentage = 0.1
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    return x_train, y_train, x_dev, y_dev


def bert_model(x_train, y_train, x_dev, y_dev):
    inputs = Input((x_train.shape[1],))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)

    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    print(model.summary())
    model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=64, epochs=100, callbacks=[early_stopping])

    print('Test...')
    result = model.predict(x_dev)
    result = np.argmax(np.array(result), axis=1)
    y_test = np.argmax(np.array(y_dev), axis=1)

    print('f1:', f1_score(y_test, result, average='macro'))
    print('accuracy:', accuracy_score(y_test, result))
    print('classification report:\n', classification_report(y_test, result))
    print('confusion matrix:\n', confusion_matrix(y_test, result))


x_train, y_train, x_dev, y_dev = get_bert_sent_embedding()

bert_model(x_train, y_train, x_dev, y_dev)




