#!/bin/sh
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.50 -t data/small/en20.50) &> log.model1.50.txt
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.100 -t data/small/en20.100) &> log.model1.100.txt
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/large/train.clean.tok.true.es -t data/large/train.clean.tok.true.en) &> log.model1.full.txt
(time python featurized_em_wa.py -a LBFGS -m hmm -s data/small/es20.50 -t data/small/en20.50) &> log.hmm.50.txt
(time python featurized_em_wa.py -a LBFGS -m hmm -s data/small/es20.100 -t data/small/en20.100) &> log.hmm.100.txt
(time python featurized_em_wa.py -a LBFGS -m hmm -s data/large/train.clean.tok.true.es -t data/large/train.clean.tok.true.en) &> log.hmm.full.txt
