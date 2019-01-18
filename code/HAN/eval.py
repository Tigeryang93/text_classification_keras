import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import make_scorer, f1_score, precision_score, accuracy_score, log_loss, classification_report, confusion_matrix
import keras.backend.tensorflow_backend as KTF
from han import create_han_model
from han import load_word2vec
from keras.callbacks import EarlyStopping
sys.path.append('../')
from data_helpers import preprocess_hierarchiacal_attention_network

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

print('Loading data...')
x_train, x_word_train, y_train, x_test, x_word_test, y_test, word2index, char2index = preprocess_hierarchiacal_attention_network()
word_num = len(word2index)
char_num = len(char2index)
print('word num', word_num)
print('char num', char_num)
print('train shape', x_train.shape)
print('x_word_train shape', x_word_train.shape)
sent_len = x_train.shape[1]
word_len = x_train.shape[2]
batch_size = 64
char_embedding_dim = 64
word_embedding_dim = 300
epochs = 100

print('Build model...')

word2vec_path = '../../../../Ek/GoogleNews-vectors-negative300.bin'
embedding_weights = load_word2vec(word2vec_path, word_num, word2index)

model = create_han_model(class_num=2, word_embedding_dim=char_embedding_dim, sent_embedding_dim=word_embedding_dim,
                         word_context_dim=word_len, sent_context_dim=sent_len, doc_len=sent_len, sent_len=word_len,
                         word_num=char_num, sent_num=word_num, embedding_weights=embedding_weights)

print('Compile model...')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('Train...')
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit([x_train, x_word_train], y_train, validation_data=([x_test, x_word_test], y_test), batch_size=64, epochs=100,
          callbacks=[early_stopping])

print('Test...')
result = model.predict([x_test, x_word_test])
result = np.argmax(np.array(result), axis=1)
y_test = np.argmax(np.array(y_test), axis=1)

print('f1:', f1_score(y_test, result, average='macro'))
print('accuracy:', accuracy_score(y_test, result))
print('classification report:\n', classification_report(y_test, result))
print('confusion matrix:\n', confusion_matrix(y_test, result))
