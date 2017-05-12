import nltk
from scipy.stats import entropy
import math

def get_kl(sentence, query):
    vocab = list(set(sentence) or set(query))
    sent_words = sentence
    fd = nltk.FreqDist(sent_words)
    sent_list = []
    back_list = []
    for word in vocab:
        if word in fd:
            sent_list.append(fd[word] + 1)
        else:
            sent_list.append(1)
        if word in query:
            back_list.append(2)
        else:
            back_list.append(1)
    ent = entropy(back_list, sent_list)
    return [ent, entropy(sent_list, back_list)]
