import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import make_scorer, f1_score, precision_score, accuracy_score, log_loss, classification_report, confusion_matrix
from keras.models import Model
from keras.layers import *
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping
from aiayn import load_word2vec
from aiayn import MultiHeadAttention

sys.path.append('..')
import data_helpers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

print('Loading data...')
x_train, y_train, x_test, y_test, word2index = data_helpers.preprocess()
word_num = len(word2index)
print('word num', word_num)
print('train shape', x_train.shape)
word_len = x_train.shape[1]
batch_size = 64
word_embedding_dim = 300
epochs = 100

print('Build model...')
word2vec_path = '../../../../Ek/GoogleNews-vectors-negative300.bin'
embedding_weights = load_word2vec(word2vec_path, word_num, word2index)

inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(word_num, word_embedding_dim, weights=[embedding_weights])(inputs)
# embeddings = PositionEmbedding()(embeddings)
o_seq = MultiHeadAttention(8, 32)([embeddings, embeddings, embeddings])
o_seq = concatenate([embeddings, o_seq], axis=-1)
o_seq = GlobalAveragePooling1D()(o_seq)
o_seq = Dropout(0.2)(o_seq)
outputs = Dense(2, activation='sigmoid')(o_seq)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
print(model.summary())
print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test),
          callbacks=[early_stopping])

print('Test...')
result = model.predict(x_test)
result = np.argmax(np.array(result), axis=1)
y_test = np.argmax(np.array(y_test), axis=1)

print('f1:', f1_score(y_test, result, average='macro'))
print('accuracy:', accuracy_score(y_test, result))
print('classification report:\n', classification_report(y_test, result))
print('confusion matrix:\n', confusion_matrix(y_test, result))
