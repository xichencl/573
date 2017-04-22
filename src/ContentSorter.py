class ContentSorter:
    def __init__(self):
        return

    #sort input sentences by their scores
    #sentences: list of sentences with each score
    #return: sorted sentences
    def sort(self, sentences):
        return sentences.sort(key=lambda x: x[1][0], reverse=True);