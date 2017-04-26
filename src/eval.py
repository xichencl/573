import re
import nltk
import math


def get_rouge(sent, sum_bigs, sum_tri, sum_quad, sum_words):
    sent_words = nltk.word_tokenize(sent)
    sent_bigs = list(nltk.ngrams(sent_words, 2))
    sent_tri = list(nltk.ngrams(sent_words, 3))
    sent_quad = list(nltk.ngrams(sent_words, 4))

    rouge_1 = 0
    if len(set(sum_words) & set(sent_words)) > 0:
        rouge_1 = len(set(sum_words) & set(sent_words)) / float(len(set(sum_words) | set(sent_words)))
    rouge_2 = 0
    if len(set(sum_bigs) & set(sent_bigs)) > 0:
        rouge_2 = len(set(sum_bigs) & set(sent_bigs)) / float(len(set(sum_bigs) | set(sent_bigs)))
    rouge_3 = 0
    if len(set(sum_tri) & set(sent_tri)) > 0:
        rouge_3 = len(set(sum_tri) & set(sent_tri)) / float(len(set(sum_tri) | set(sent_tri)))
    rouge_4 = 0
    if len(set(sum_quad) & set(sent_quad)) > 0:
        rouge_4 = len(set(sum_quad) & set(sent_quad)) / float(len(set(sum_quad) | set(sent_quad)))
    return (rouge_4 + rouge_3 + rouge_2 + rouge_1) / 4
