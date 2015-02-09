__author__ = 'arenduchintala'

from math import log
from optparse import OptionParser


if __name__ == "__main__":
    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/toy.fr")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/toy.en")
    opt.add_option("-o", dest="save_init_trans", default="init.trans")
    (options, _) = opt.parse_args()
    initial_translation = {}
    translations = {}
    save = options.save_init_trans
    corpus_source = open(options.source_corpus, 'r').readlines()
    corpus_target = open(options.target_corpus, 'r').readlines()

    for k, sentence_source in enumerate(corpus_source):
        sentence_target = corpus_target[k]
        tokens_source = sentence_source.split()
        tokens_source.insert(0, 'NULL')
        tokens_target = sentence_target.split()
        corpus_source[k] = tokens_source
        corpus_target[k] = tokens_target
        for e in tokens_source:
            n_e = initial_translation.get(e, set())
            n_e.update(tokens_target)
            initial_translation[e] = n_e

    for k, v in initial_translation.iteritems():
        for v_es in v:
            translations[v_es, k] = 1.0 / len(v)

    TYPE = "EMISSION"
    writer = open(save, 'w')
    for k in sorted(translations):
        v = translations[k]
        writer.write(TYPE + '\t' + str('\t'.join(k)) + '\t' + str(v) + '\n')
    writer.flush()
    writer.close()

    writer = open(save + ".log", 'w')
    for k in sorted(translations):
        v = log(translations[k])
        writer.write(TYPE + '\t' + str('\t'.join(k)) + '\t' + str(v) + '\n')
    writer.flush()
    writer.close()

