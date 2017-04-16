import nltk

def pre_proc(docs):
    sent2proc = {}
    sent2
    proc2sent = {}
    for event in docs.keys():
        an_event = docs[event]
        for document in an_event.keys():
            a_doc = an_event[document]
            for sentence in a_doc:
                proc = ''
                words = nltk.word_tokenize(sentence)
                pos = nltk.pos_tag(words)
