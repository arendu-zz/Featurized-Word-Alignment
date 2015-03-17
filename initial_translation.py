__author__ = 'arenduchintala'
from math import log
from optparse import OptionParser


if __name__ == "__main__":
    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/dev.es")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/dev.en")
    opt.add_option("-o", dest="save_init_trans", default="experiment/data/init.trans")
    (options, _) = opt.parse_args()
    print '#', options
    translations = {}
    save = options.save_init_trans
    corpus_source = open(options.source_corpus, 'r').readlines()
    corpus_target = open(options.target_corpus, 'r').readlines()
    target_sentences = []
    source_sentences = []
    for s, t in zip(corpus_source, corpus_target):
        s = ['NULL'] + s.strip().split()
        t = t.strip().split()
        source_sentences.append(s)
        target_sentences.append(t)

    from cyth.cyth_model1 import init_lex

    translations = init_lex(source_sentences, target_sentences)
    TYPE = "EMISSION"
    writer = open(save, 'w')
    for k in sorted(translations):
        val = translations[k]
        writer.write(TYPE + '\t' + str('\t'.join(k)) + '\t' + str(val) + '\n')
    writer.flush()
    writer.close()


