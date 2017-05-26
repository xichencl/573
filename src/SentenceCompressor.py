import nltk
import re
import json

def compress(sent):
    months = '(January|February|March|April|May|June|July|August|September|October|November|December)'
    sent = re.sub(', age [0-9]+,?', '', sent)
    sent = re.sub('"', '', sent)
    sent = re.sub('[0-9]+-year-old', '', sent)
    sent = re.sub(', [0-9]+,', '', sent)
    sent = re.sub('(, )?said [a-zA-Z]+,', '', sent)
    sent = re.sub('(, )?[a-zA-Z]+ said,', '', sent)
    sent = re.sub('[iI]n [0-9]{4},?', '', sent)
    sent = re.sub('[iI]n ' + months, '', sent)
    sent = re.sub('[oO]n ' + months + ' [0-9]+[a-z]*,?', '', sent)
    sent = re.sub('[aA]t [0-9]{1,2}(:[0-9]{2})? (a\.m\.|p\.m\.)?,? ', ' ', sent)
    words = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(words)
    if pos[0][1] == 'CC' or pos[0][1] == 'IN':
        pos = pos[1:]
    new_sent = []
    '''for word in pos:
        if word[1] != 'RB' and word[1] != 'JJ':
            new_sent.append(word[0])'''
    proc_sent = sent
    if proc_sent[0].islower():
        proc_sent = proc_sent[:1].upper() + proc_sent[1:]
    proc_sent = re.sub(r' (\.|\?|!|,|\'|:)', r'\1', proc_sent)
    return proc_sent

docs = json.load(open('data/training.json', 'r'))
for key in docs:
    cluster = docs[key]
    for doc in cluster:
        a_doc = cluster[doc]
        for sent in a_doc:
            print(sent)
            print(compress(sent))
            print('\n')