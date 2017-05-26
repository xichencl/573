import nltk

def compress(sent):
    pos = nltk.pos_tag(sent)

    return sent

print(compress(['this is a test']))