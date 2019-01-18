import numpy as np
import re
import os
import collections


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


def load_data_and_labels(positive_data_file, negative_data_file):
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def preprocess():
    print(os.getcwd())
    positive_data_file = '../../data/rt-polaritydata/rt-polarity.pos'
    negative_data_file = '../../data/rt-polaritydata/rt-polarity.neg'
    dev_sample_percentage = 0.1

    # load rt-polarity data
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file)

    # get word2index
    counter = collections.Counter()
    for text in x_text:
        text = text.split(' ')
        for word in text:
            counter[word] += 1
    word2index = collections.defaultdict(int)
    for wid, word in enumerate(counter.most_common()):
        word2index[word[0]] = wid+1
    word2index['PAD'] = 0

    # change word to index
    x_text_numeral = []
    for text in x_text:
        temp_text = []
        text = text.split(' ')
        for word in text:
            temp_text.append(word2index[word])
        x_text_numeral.append(temp_text)

    max_len = 0
    for i in range(len(x_text_numeral)):
        if len(x_text_numeral[i]) > max_len:
            max_len = len(x_text_numeral[i])

    for i in range(len(x_text_numeral)):
        if len(x_text_numeral[i]) < max_len:
            x_text_numeral[i] += (max_len - len(x_text_numeral[i])) * [word2index['PAD']]

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_text_numeral = np.array(x_text_numeral)
    x_shuffled = x_text_numeral[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x_text_numeral, y, x_shuffled, y_shuffled, x_text

    return x_train, y_train, x_dev, y_dev, word2index


def preprocess_hierarchiacal_attention_network():

    positive_data_file = '../../data/rt-polaritydata/rt-polarity.pos'
    negative_data_file = '../../data/rt-polaritydata/rt-polarity.neg'
    dev_sample_percentage = 0.1

    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    # Change data shape to (example_number, word_number, char_number)
    # Get word2index and char2index
    word_counter = collections.Counter()
    char_counter = collections.Counter()
    for text in x_text:
        text = text.split(' ')
        for word in text:
            word_counter[word] += 1
            word = list(word)
            for char in word:
                char_counter[char] += 1

    word2index = collections.defaultdict(int)
    for wid, word in enumerate(word_counter.most_common()):
        word2index[word[0]] = wid+1
    word2index['PAD'] = 0

    char2index = collections.defaultdict(int)
    for cid, char in enumerate(char_counter.most_common()):
        char2index[char[0]] = cid+1
    char2index['PAD'] = 0

    x_text_3d = []
    x_text_word = []
    for text in x_text:
        temp_text = []
        temp_text_word = []
        text = text.split(' ')
        for word in text:
            temp_text_word.append(word2index[word])
            temp_word = []
            word = list(word)
            for char in word:
                temp_word.append(char2index[char])
            temp_text.append(temp_word)
        x_text_word.append(temp_text_word)
        x_text_3d.append(temp_text)

    # Get sent_len and word_len
    sent_len = 0
    word_len = 0
    for sent in x_text_3d:
        if len(sent) > sent_len:
            sent_len = len(sent)
        for word in sent:
            if len(word) > word_len:
                word_len = len(word)

    print('sent len', sent_len)
    print('word len', word_len)

    for i in range(len(x_text_3d)):
        if len(x_text_3d[i]) < sent_len:
            x_text_3d[i][:] += (sent_len-len(x_text_3d[i]))*[word_len*[char2index['PAD']]]

        for j in range(len(x_text_3d[i])):
            if len(x_text_3d[i][j]) < word_len:
                x_text_3d[i][j][:] += (word_len-len(x_text_3d[i][j]))*[char2index['PAD']]

    for i in range(len(x_text_word)):
        if len(x_text_word[i]) < sent_len:
            x_text_word[i] += (sent_len-len(x_text_word[i]))*[word2index['PAD']]

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_text_3d = np.array(x_text_3d)
    x_text_word = np.array(x_text_word)
    x_shuffled = x_text_3d[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    x_word = x_text_word[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    x_word_train, x_word_dev = x_word[:dev_sample_index], x_word[dev_sample_index:]

    del x_text_3d, y, x_shuffled, y_shuffled, x_text_word, x_word

    return x_train, x_word_train, y_train, x_dev, x_word_dev, y_dev, word2index, char2index

