import sys
import os
import numpy as np
from sklearn.metrics import make_scorer, f1_score, precision_score, accuracy_score, log_loss, classification_report, confusion_matrix
import tensorflow as tf
from keras.preprocessing import sequence
import keras.backend.tensorflow_backend as KTF
from textcnn import TextCNN
sys.path.append('../')
import data_helpers


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

batch_size = 64
embedding_dim = 128
epochs = 100

print('Loading data...')
# get data
x_train, y_train, x_test, y_test, word2index = data_helpers.preprocess()
max_features = len(word2index)

max_len = max(len(x) for x in x_train)
print(max_len)

print('Pad sequences...')
x_train = sequence.pad_sequences(x_train, maxlen=max_len, value=0)
x_test = sequence.pad_sequences(x_test, maxlen=max_len, value=0)

print('Build model...')
model = TextCNN(max_len, embedding_dim, batch_size=batch_size, class_num=2, max_features=max_features, epochs=epochs)

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

