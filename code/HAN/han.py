import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, GRU, Bidirectional, TimeDistributed, merge
from keras.engine.topology import Layer
from keras import backend as K
from gensim.models import KeyedVectors


class AttentionLayer(Layer):
    """
    attention mechanism
    reference:https://www.microsoft.com/developerblog/2018/03/06/sequence-intent-classification/
    """
    def __init__(self, regularizer=None, context_dim=100, **kwargs):
        self.regularizer = regularizer
        self.context_dim = context_dim
        self.supports_masking = True
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.context_dim), initializer='normal',
                                 trainable=True, regularizer=self.regularizer)
        self.b = self.add_weight(name='b', shape=(self.context_dim,), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u', shape=(self.context_dim,1), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))
        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        return None


def load_word2vec(word2vec_model, vocab_sz, word2index):
    print('Load Word2vec...')
    word2vec1 = KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
    embed_size = 300
    embedding_weight = np.zeros((vocab_sz, embed_size))
    for word, index in word2index.items():
        try:
            embedding_weight[index, :] = word2vec1[word]
        except KeyError:
            pass
    del word2vec1
    return embedding_weight


def create_han_model(class_num, word_embedding_dim, sent_embedding_dim, word_context_dim, sent_context_dim, doc_len,
                     sent_len, word_num, sent_num, embedding_weights):
    # hierarchical attention network
    # sent encoder
    sent_input = Input(shape=(sent_len, ))
    word_embedding = Embedding(word_num, word_embedding_dim, input_length=sent_len)(sent_input)
    sent_lstm = Bidirectional(GRU(32, return_sequences=True))(word_embedding)
    sent_att = AttentionLayer(context_dim=word_context_dim)(sent_lstm)
    sent_encoder = Model(sent_input, sent_att)

    # doc encoder with word2vec feature
    doc_input = Input(shape=(doc_len, sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)

    doc_word_input = Input(shape=(doc_len, ))
    sent_embedding = Embedding(sent_num, sent_embedding_dim, weights=[embedding_weights], input_length=doc_len)(doc_word_input)
    doc_encoder = merge([doc_encoder, sent_embedding], mode='concat')
    doc_lstm = Bidirectional(GRU(64, return_sequences=True))(doc_encoder)

    doc_att = AttentionLayer(context_dim=sent_context_dim)(doc_lstm)
    preds = Dense(class_num, activation='softmax')(doc_att)
    model = Model([doc_input, doc_word_input], preds)
    return model








