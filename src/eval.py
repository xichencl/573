import re
import nltk
import math


def get_rouge(sent, sum_bigs, sum_tri, sum_quad, sum_words):
    sent_words = nltk.word_tokenize(sent)
    sent_words_fd = nltk.FreqDist(sent_words)
    sent_bigs = nltk.FreqDist(list(nltk.ngrams(sent_words, 2)))
    sent_tri = nltk.FreqDist(list(nltk.ngrams(sent_words, 3)))
    sent_quad = nltk.FreqDist(list(nltk.ngrams(sent_words, 4)))

    rouge_1 = 0
    for word in sent_words_fd:
        if word in sum_words:
            rouge_1 += sent_words_fd[word] + sum_words[word]
    rouge_1 /= sent_words_fd.N() + sum_words.N()

    rouge_2 = 0
    for word in sent_bigs:
        if word in sum_bigs:
            rouge_2 += sent_bigs[word] + sum_bigs[word]
    rouge_2 /= sent_bigs.N() + sum_bigs.N()

    rouge_3 = 0
    for word in sent_tri:
        if word in sum_tri:
            rouge_3 += sent_tri[word] + sum_tri[word]
    rouge_3 /= sent_tri.N() + sum_tri.N()

    rouge_4 = 0
    for word in sent_quad:
        if word in sum_quad:
            rouge_4 += sent_quad[word] + sum_quad[word]
    rouge_4 /= sent_quad.N() + sum_quad.N()
    return (rouge_4 + rouge_3 + rouge_2 + rouge_1) / 4
