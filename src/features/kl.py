import nltk
from scipy.stats import entropy
import math


def get_freq_list(docs):
    back_fd = {}
    for doc in docs:
        a_doc = docs[doc]
        words = nltk.word_tokenize(' '.join(a_doc))
        for word in words:
            if word not in nltk.corpus.stopwords.words('english'):
                if word in back_fd:
                    back_fd[word] += 1
                else:
                    back_fd[word] = 1
    back_list = []
    vocab = sorted(list(back_fd.keys()), key=lambda x: back_fd[x], reverse=True)
    for word in vocab:
        back_list.append(back_fd[word])
    return back_list, vocab


def get_kl(sentence, back_list, vocab):
    sent_words = nltk.word_tokenize(sentence)
    fd = nltk.FreqDist(sent_words)
    sent_list = []
    for word in vocab:
        if word in fd:
            sent_list.append(fd[word] + 1)
        else:
            sent_list.append(1)
    ent = entropy(back_list, sent_list)
    return [ent, entropy(sent_list, back_list)]
