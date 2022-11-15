#!/bin/bash

python3 ./src/create_mgc_src.py
python3 ./src/create_mgc_trg.py
python3 ./src/create_dat.py
mkdir -p conf
ls data/dat/src | head -45 | sed -e 's/\.dat//' > conf/train.list
ls data/dat/src | tail -5 | sed -e 's/\.dat//' > conf/eval.list
python3 ./src/train.py
