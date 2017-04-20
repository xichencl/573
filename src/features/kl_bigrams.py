import nltk
from scipy.stats import entropy
import math


def get_freq_list(docs):
    back_fd = {}
    for doc in docs.keys():
        a_doc = docs[doc]
        words = nltk.word_tokenize(' '.join(a_doc))
        stems = []
        stemmer = nltk.stem.PorterStemmer()
        for word in words:
            stems.append(stemmer.stem(word))
        bigrams = nltk.ngrams(stems, 2)
        for big in bigrams:
            if big[0] not in nltk.corpus.stopwords.words('english') or big[1] not in nltk.corpus.stopwords.words('english'):
                if big in back_fd.keys():
                    back_fd[big] += 1
                else:
                    back_fd[big] = 1
    back_list = []
    vocab = sorted(list(back_fd.keys()), key=lambda x: back_fd[x], reverse=True)
    for word in vocab:
        back_list.append(back_fd[word])
    return back_list, vocab


def get_kl(sentence, back_list, vocab):
    sent_words = nltk.word_tokenize(sentence)
    stems = []
    stemmer = nltk.stem.PorterStemmer()
    for word in sent_words:
        stems.append(stemmer.stem(word))
    bigrams = nltk.ngrams(sent_words, 2)
    fd = nltk.FreqDist(bigrams)
    sent_list = []
    for big in vocab:
        if big in fd.keys():
            sent_list.append(fd[big] + 1)
        else:
            sent_list.append(1)
    ent = entropy(back_list, sent_list)
    return [ent, entropy(sent_list, back_list)]
