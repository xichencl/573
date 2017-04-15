


class ContentRealizer:
    def __init__(self):
        return

    #sort input sentences by their scores
    #sentences: list of sentences with each score
    #return: picked sentences
    def realize(self, sentences, length_limit):
        total_len = 0;
        results = list();
        for sent in sentences:
            if total_len < length_limit:
                text = sent[0];
                total_len += len(text.split());
                results.append((text,sent[1]));
        return results;
