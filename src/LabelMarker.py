import numpy as np
from sklearn.semi_supervised import label_propagation

class LabelMarker:
    def __init__(self):
        return

    #LabelPropagation based learning
    #data: sentence vectors
    #labels: labels for each sentence, -1 for unclassified
    #return clustering labels for all sentences
    def label(self, data, labels):
        lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
        lp_model.fit(data, labels)
        return lp_model.transduction_