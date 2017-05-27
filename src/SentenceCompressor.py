import nltk
import re

def compress(sent):
    return sent
    if len(sent.split()) > 1:
        months = '(January|February|March|April|May|June|July|August|September|October|November|December)'
        sent = re.sub(', age [0-9]+,?', '', sent)
        sent = re.sub('"', '', sent)
        sent = re.sub('[0-9]+-year-old', '', sent)
        sent = re.sub(', [0-9]+,', '', sent)
        sent = re.sub('(, )?said [a-zA-Z]+[,\.]', '', sent)
        sent = re.sub('(, )?[a-zA-Z]+ said[,\.]', '', sent)
        sent = re.sub('[iI]n [0-9]{4},?', '', sent)
        sent = re.sub('[iI]n ' + months, '', sent)
        sent = re.sub('[oO]n ' + months + ' [0-9]+[a-z]*,?', '', sent)
        sent = re.sub('[aA]t [0-9]{1,2}(:[0-9]{2})? (a\.m\.|p\.m\.)?,? ', ' ', sent)
        if len(sent):
            words = nltk.word_tokenize(sent)
            pos = nltk.pos_tag(words)
            if pos[0][1] == 'CC' or pos[0][1] == 'IN':
                pos = pos[1:]
            new_sent = []
            iterator = iter(range(len(pos) - 1))
            for i in iterator:
                if i < len(pos):
                    word = pos[i]
                    if word[1] == 'RB':
                        if i > len(pos) - 1:
                            new_sent.append(word[0])
                        elif pos[i + 1][1] == 'VB' or pos[i + 1][1] == 'VBZ' or pos[i + 1][1] == 'VBD':
                            new_sent.append(pos[i+1][0])
                            next(iterator, None)
                        else:
                            new_sent.append(word[0])
                    elif word[1] == 'JJ':
                        if i > len(pos) - 1:
                            new_sent.append(word[0])
                        elif pos[i+1][1] == 'NN' or pos[i+1][1] == 'NNS':
                            new_sent.append(pos[i+1][0])
                            next(iterator, None)
                        else:
                            new_sent.append(word[0])
                    else:
                        new_sent.append(word[0])
            proc_sent = ' '.join(new_sent)
            #proc_sent = sent
            if len(proc_sent):
                if proc_sent[0].islower():
                    proc_sent = proc_sent[:1].upper() + proc_sent[1:]
                proc_sent = re.sub(r' (\.|\?|!|,|\'|:)', r'\1', proc_sent)
                proc_sent = re.sub('\$ ', '\$', proc_sent)

                return proc_sent

