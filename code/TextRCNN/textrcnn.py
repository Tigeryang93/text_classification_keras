from keras.models import *
from keras.layers import *
import keras.backend as K
from keras.callbacks import *


class TextRCNN(object):
    def __init__(self, word_num, embedding_dim=300, rnn_dim=32, cnn_dim=32, class_num=2, epochs=100,
                 batch_size=64, embedding_weight=None, ):
        self.word_num = word_num
        self.embedding_dim = embedding_dim
        self.embedding_weight = embedding_weight
        self.rnn_dim = rnn_dim
        self.cnn_dim = cnn_dim
        self.class_num = class_num
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x_train, left_cxt_train, right_cxt_train, x_test, left_ctx_test, right_ctx_test, y_train, y_test):
        doc = Input(shape=(None, ))
        left_context = Input(shape=(None, ))
        right_context = Input(shape=(None, ))

        embedding = Embedding(input_dim=self.word_num, output_dim=self.embedding_dim, weights=[self.embedding_weight],
                              trainable=True)
        doc_embedding = embedding(doc)
        left_cont_embedding = embedding(left_context)
        right_cont_embedding = embedding(right_context)

        forward = LSTM(self.rnn_dim, return_sequences=True)(left_cont_embedding)
        backward = LSTM(self.rnn_dim, return_sequences=True)(right_cont_embedding)
        backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)

        together = concatenate([forward, doc_embedding, backward], axis=2)

        semantic = Conv1D(self.cnn_dim, kernel_size=1, activation='tanh')(together)

        pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cnn_dim, ))(semantic)

        output = Dense(self.class_num, activation='softmax')(pool_rnn)

        self.model = Model(inputs=[doc, left_context, right_context], outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(self.model.summary())
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit([x_train, left_cxt_train, right_cxt_train], y_train,
                       validation_data=([x_test, left_ctx_test, right_ctx_test], y_test), batch_size=self.batch_size,
                       epochs=self.epochs, callbacks=[early_stopping])

        return self.model

    def predict(self, x, left_context, right_context):
        return self.model.predict([x, left_context, right_context])





