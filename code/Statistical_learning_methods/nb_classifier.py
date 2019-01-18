import re
import numpy as np
import random
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import make_scorer, f1_score, precision_score, accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
from sklearn import metrics


def review_to_wordlist(review):
    review_text = re.sub("[^a-zA-Z]", " ", review)
    words = review_text.lower().split()
    return words


def load_data():
    positive_data_file = '../../data/rt-polaritydata/rt-polarity.pos'
    negative_data_file = '../../data/rt-polaritydata/rt-polarity.neg'
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    x_text = positive_examples + negative_examples

    positive_labels = [0 for _ in positive_examples]
    negative_labels = [1 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    train_data = []
    for i in range(len(x_text)):
        train_data.append(" ".join(review_to_wordlist(x_text[i])))

    randnum = random.randint(0, len(y))
    random.seed(randnum)
    random.shuffle(train_data)
    random.seed(randnum)
    random.shuffle(y)

    return train_data, y


print('data has loaded!')
train_data, y = load_data()

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
           ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
tfv.fit(train_data)
train_data = tfv.transform(train_data)

# split data into train and dev
dev_sample_percentage = 0.1
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = train_data[:dev_sample_index], train_data[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

model_nb = MultinomialNB()
model_nb.fit(x_train, y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

score = np.mean(cross_val_score(model_nb, x_train, y_train, cv=10, scoring='accuracy'))
print(score)
y_pred = model_nb.predict(x_dev)

result = y_pred
y_test = y_dev
print('f1:', f1_score(y_test, result, average='macro'))
print('accuracy:', accuracy_score(y_test, result))
print('classification report:\n', classification_report(y_test, result))
print('confusion matrix:\n', confusion_matrix(y_test, result))



