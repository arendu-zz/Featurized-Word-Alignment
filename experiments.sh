#!/bin/sh
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/es20.50 -t data/en20.50) &> log1.txt
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/es20.100 -t data/en20.100) &> log2.txt
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/es20 -t data/en20) &> log3.txt
(time python featurized_em_wa.py -a LBFGS -m hmm -s data/es20.50 -t data/en20.50) &> log4.txt
(time python featurized_em_wa.py -a LBFGS -m hmm -s data/es20.100 -t data/en20.100) &> log5.txt
(time python featurized_em_wa.py -a LBFGS -m hmm -s data/es20 -t data/en20) &> log6.txt
