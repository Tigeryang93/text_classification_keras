"""
Using: 获得bert模型得到的数据word embedding和sent embedding
Author: yanghu
Time:2019/1/4
"""
import re
from bert_serving.client import BertClient


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


# pre deal data
positive_data_file = 'rt-polaritydata/rt-polarity.pos'
negative_data_file = 'rt-polaritydata/rt-polarity.neg'
positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
positive_examples = [s.strip() for s in positive_examples]
negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
negative_examples = [s.strip() for s in negative_examples]
x_text = positive_examples + negative_examples
x_text = [clean_str(sent) for sent in x_text]

bc = BertClient()
encoded_text_to_sent_embed = bc.encode(x_text)

filename = 'encoded_text.txt'
file = open(filename, 'w', encoding='utf-8')
for line in encoded_text_to_sent_embed:
    file.write(line+'\n')
file.close()
