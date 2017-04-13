import nltk
import math

tagset = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

# add linguistic information like POS and named entity probabilities
def add_feats(docs, sent, vec):
    words = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(words)
    num_cap = 0
    for word in words[1:]:
        if word.istitle():
            num_cap += 1
    vec.append(float(num_cap)/len(words))
    vec.append(num_cap)
    pos_counts = {}
    for pair in pos:
        if pair[1] in pos_counts.keys():
            pos_counts[pair[1]] += 1
        else:
            pos_counts[pair[1]] = 1
    for tag in tagset:
        if tag in pos_counts.keys():
            vec.append(float(pos_counts[tag])/len(pos))
            vec.append(pos_counts[tag])
        else:
            vec.append(0)
            vec.append(0)
    return vec