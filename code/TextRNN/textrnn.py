from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping


class TextRNN(object):
    """
    reference:
    """
    def __init__(self, maxlen, embedding_dim, batch_size=64, class_num=2, max_features=1000, epochs=1,
                 embedding_weights=None):
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.class_num = class_num
        self.max_features = max_features
        self.epochs = epochs
        self.embedding_weights = embedding_weights

    def fit(self, x_train, x_test, y_train, y_test):
        inputs = Input((self.maxlen,))
        embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dim, input_length=self.maxlen,
                              weights=[self.embedding_weights])(inputs)
        x = Bidirectional(LSTM(64))(embedding)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(self.class_num, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=output)

        self.model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        print(self.model.summary())
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=self.batch_size,
                       epochs=self.epochs, callbacks=[early_stopping])

        return self.model

    def predict(self, x):
        return self.model.predict(x)
