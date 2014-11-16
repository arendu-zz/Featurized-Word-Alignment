#!/bin/sh
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.100 -t data/small/en20.100) &> log.lbfgs.model1.100.txt
(time python featurized_em_wa.py -a SGD -m model1 -s data/small/es20.100 -t data/small/en20.100) &> log.sgd.model1.100.txt

(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.200 -t data/small/en20.200) &> log.lbfgs.model1.200.txt
(time python featurized_em_wa.py -a SGD -m model1 -s data/small/es20.200 -t data/small/en20.200) &> log.sgd.model1.200.txt

(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.400 -t data/small/en20.400) &> log.lbfgs.model1.400.txt
(time python featurized_em_wa.py -a SGD -m model1 -s data/small/es20.400 -t data/small/en20.400) &> log.sgd.model1.400.txt
