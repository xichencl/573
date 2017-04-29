import nltk
import math
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tagset = ['CC', 'DT', 'IN', 'JJ', 'NN', 'NNS', 'NNP', 'PRP', 'RB', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ']

# add linguistic information like POS and named entity probabilities
def add_feats(docs, sent, vec):
    words = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(words)
    num_cap = 0
    for word in words[1:]:
        if word.istitle():
            num_cap += 1
    nums = re.findall('[0-9]', sent)
    caps = re.findall('[A-Z]', sent)
    has_quote = bool('"' in sent)
    vec.append(int(has_quote))
    commas = re.findall(',', sent)
    vec.append(len(commas))
    vec.append(float(len(nums)) / len(sent))
    vec.append(float(len(caps)) / len(sent))
    vec.append(float(num_cap)/len(words))
    vec.append(num_cap)
    pos_counts = {}
    for pair in pos:
        if pair[1] in pos_counts:
            pos_counts[pair[1]] += 1
        else:
            pos_counts[pair[1]] = 1
    for tag in tagset:
        if tag in pos_counts:
            #vec.append(float(pos_counts[tag])/len(pos))
            vec.append(pos_counts[tag])
        else:
            #vec.append(0)
            vec.append(0)

    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sent)
    vec.append(ss['compound'])
    return vec