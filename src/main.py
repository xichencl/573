import json
import ContentSelector
import ContentRealizer
import sys
import re

gold = json.load(open(sys.argv[1], 'r'))
train_documents = json.load(open(sys.argv[2], 'r'))
test_documents = json.load(open(sys.argv[3], 'r'))
proc_test = json.load(open(sys.argv[4], 'r'))

selector = ContentSelector.ContentSelector()
selector.train(train_documents, gold)

realizer = ContentRealizer.ContentRealizer()

index = 0
for event in test_documents:
    index += 1
    an_event = test_documents[event]
    proc_event = proc_test[event]
    print("Testing " + str(index) + '/' + str(len(test_documents.keys())))
    results = selector.test(an_event, proc_event, event, 10)
    picked = realizer.realize(results, 100)
    summary = re.sub('\W+', ' ', ' '.join(picked))
    letter = event[-1]

    if 'Group' in event:
        out = open('/home2/mblac6/573/573/outputs/D2/' + event, 'w')
    else:
        out = open('/home2/mblac6/573/573/outputs/D2/' + event[:-1] + '-A.M.100.' + letter + '.A', 'w')
    out.write(summary)


