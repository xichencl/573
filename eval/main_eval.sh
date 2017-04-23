#!/usr/bin/env bash
python3 /home2/mblac6/573/573/src/main.py /home2/mblac6/573/573/src/data/new_human.json /home2/mblac6/573/573/src/data/new_training.json /home2/mblac6/573/573/src/data/devtest.json
./ROUGE-1.5.5.pl -e data -a -n 4 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d /home2/mblac6/573/573/eval/test.xml