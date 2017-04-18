#!/usr/bin/env bash
python3 /Users/mackie/PycharmProjects/573/src/main.py /Users/mackie/PycharmProjects/573/src/data/new_human.json /Users/mackie/PycharmProjects/573/src/data/new_training.json
./ROUGE-1.5.5.pl -e data -a -n 4 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d /Users/mackie/PycharmProjects/573/eval/new_conf.xml