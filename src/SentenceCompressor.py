import nltk
import re

def compress(sent):
    sent = re.sub(', age [0-9]+,', '', sent)
    words = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(words)
    if pos[0][1] == 'CC' or pos[0][1] == 'IN':
        pos = pos[1:]
    new_sent = []
    for word in pos:
        if word[1] != 'RB' and word[1] != 'JJ':
            new_sent.append(word[0])
    proc_sent = ' '.join(new_sent)
    return proc_sent

print(compress('the brown dog, age 12, runs quickly'))