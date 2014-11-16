#!/bin/sh
(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.100 -t data/small/en20.100) &> log.model1.full.txt
(time python featurized_em_wa_mp.py -a LBFGS -m model1 -s data/small/es20.100 -t data/small/en20.100) &> log.parallel.model1.full.txt

(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.200 -t data/small/en20.200) &> log.model1.full.txt
(time python featurized_em_wa_mp.py -a LBFGS -m model1 -s data/small/es20.200 -t data/small/en20.200) &> log.parallel.model1.full.txt

(time python featurized_em_wa.py -a LBFGS -m model1 -s data/small/es20.400 -t data/small/en20.400) &> log.model1.full.txt
(time python featurized_em_wa_mp.py -a LBFGS -m model1 -s data/small/es20.400 -t data/small/en20.400) &> log.parallel.model1.full.txt
