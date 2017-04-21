import re
import nltk
import math


def get_rouge(sent, sum_bigs, sum_words):
    sent_words = nltk.word_tokenize(sent)
    rouge_1 = 0
    if len(set(sum_words) & set(sent_words)) > 0:
        rouge_1 = len(set(sum_words) & set(sent_words)) / float(len(set(sum_words) | set(sent_words)))
    sent_bigs = list(nltk.ngrams(sent_words, 2))
    rouge_2 = 0
    if len(set(sum_bigs) & set(sent_bigs)) > 0:
        rouge_2 = len(set(sum_bigs) & set(sent_bigs)) / float(len(set(sum_bigs) | set(sent_bigs)))
    return (rouge_2 + rouge_1) / 2
