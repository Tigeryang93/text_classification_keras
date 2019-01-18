from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras.callbacks import EarlyStopping


class FastText(object):
    def __init__(self, maxlen, embedding_dim, batch_size=64, class_num=2, max_features=1000, epochs=1,
                 trainable=True, weights=None):
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.class_num = class_num
        self.trainable = trainable
        self.max_features = max_features
        self.epochs = epochs
        self.weights = weights

    def fit(self, x_train, x_test, y_train, y_test):
        # input->embedding->pooling->dense
        inputs = Input((self.maxlen,))
        embedding = Embedding(self.max_features, self.embedding_dim, input_length=self.maxlen, weights=self.weights,
                              trainable=self.trainable)(inputs)
        x = GlobalAveragePooling1D()(embedding)
        output = Dense(self.class_num, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=output)

        self.model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=self.batch_size,
                       epochs=self.epochs, callbacks=[early_stopping])

        return self.model

    def predict(self, x):
        return self.model.predict(x)






