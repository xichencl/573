from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import numpy as np

forest = ExtraTreesRegressor(n_estimators=250,
                              random_state=0)

def get_feats(x, y):
    #x = np.array(x)
    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    labels = ['tf_idf_sum', 'tf_idf_avg', 'LLR_sum', 'LLR', 'sent_len', 'quote', 'commas', 'P(num)', 'P(cap)', 'P(cap_word)', '#cap_word',
              'CC', 'DT', 'IN', 'JJ', 'NN', 'NNS', 'NNP', 'PRP', 'RB', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ', 'sentiment', 'KL',
              'KL_rev', 'KL_big', 'KL_big_rev', 'early_first', 'avg_first', 'avg_pos', 'is_first', 'idx', 'lexrank', 'q_lexrank', 'q_kl', 'q_kl_rev']
    sorted_labels = []

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        sorted_labels.append(labels[indices[f]])
        print("%s\t%f" % (labels[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[1]), sorted_labels)
    plt.xlim([-1, x.shape[1]])
    #plt.show()