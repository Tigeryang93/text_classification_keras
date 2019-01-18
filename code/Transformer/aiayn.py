"""
Author:yanghu
Time:2019.1.2
User:All you need is attention
"""
import numpy as np
from keras.layers import Layer
from keras import backend as K
from gensim.models import KeyedVectors


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


class PositionEmbedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]

        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)

        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1)-1
        position_i = K.expand_dims(position_i, 2)

        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)

        if self.mode == 'sum':
            return position_ij+x
        elif self.mode == 'cancat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class MultiHeadAttention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', shape=(input_shape[0][-1], self.output_dim), initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', shape=(input_shape[1][-1], self.output_dim), initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', shape=(input_shape[2][-1], self.output_dim), initializer='glorot_uniform',
                                  trainable=True)
        super(MultiHeadAttention, self).build(input_shape)

    def mask(self, inputs, seq_len, mode='mul'):
        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x, **kwargs):
        q_seq, k_seq, v_seq, q_len, v_len = None, None, None, None, None

        if len(x) == 3:
            q_seq, k_seq, v_seq = x
            q_len, v_len = None, None
        elif len(x) == 5:
            q_seq, k_seq, v_seq, q_len, v_len = x

        q_seq = K.dot(q_seq, self.WQ)
        q_seq = K.reshape(q_seq, (-1, K.shape(q_seq)[1], self.nb_head, self.size_per_head))
        q_seq = K.permute_dimensions(q_seq, (0, 2, 1, 3))

        k_seq = K.dot(k_seq, self.WK)
        k_seq = K.reshape(k_seq, (-1, K.shape(k_seq)[1], self.nb_head, self.size_per_head))
        k_seq = K.permute_dimensions(k_seq, (0, 2, 1, 3))

        v_seq = K.dot(v_seq, self.WV)
        v_seq = K.reshape(v_seq, (-1, K.shape(v_seq)[1], self.nb_head, self.size_per_head))
        v_seq = K.permute_dimensions(v_seq, (0, 2, 1, 3))

        att = K.batch_dot(q_seq, k_seq, axes=[3, 3]) / self.size_per_head**0.5
        att = K.permute_dimensions(att, (0, 3, 2, 1))
        att = self.mask(att, v_len, 'add')
        att = K.permute_dimensions(att, (0, 3, 2, 1))
        att = K.softmax(att)

        o_seq = K.batch_dot(att, v_seq, axes=[3, 2])
        o_seq = K.permute_dimensions(o_seq, (0, 2, 1, 3))
        o_seq = K.reshape(o_seq, (-1, K.shape(o_seq)[1], self.output_dim))
        o_seq = self.mask(o_seq, q_len, 'mul')
        return o_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)