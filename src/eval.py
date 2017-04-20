import re
import nltk
import math




def get_rouge2(sent, sum_bigs):

    sent_words = nltk.word_tokenize(sent)
    sent_bigs = list(nltk.ngrams(sent_words, 2))
    rouge_2 = 0
    if len(set(sum_bigs) & set(sent_bigs)) > 0:
        rouge_2 = len(set(sum_bigs) & set(sent_bigs)) / float(len(set(sum_bigs) | set(sent_bigs)))
    return rouge_2
