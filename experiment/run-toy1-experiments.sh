python initial_translation.py -t data/toy1.fr -s data/toy1.en -o initial.probs -m uniform
python model1.py -t data/toy1.fr -s data/toy1.en -i initial.probs -p model1.probs -a model1.align -as data/toy1.en -at data/toy1.fr
python ../featurized_em_wa.py -t data/toy1.fr -s data/toy1.en --iw initial.probs
