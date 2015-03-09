__author__ = 'arenduchintala'
from optparse import OptionParser

if __name__ == '__main__':
    opt = OptionParser()

    opt.add_option("-p", dest="probs", default="input.probs")
    opt.add_option("-o", dest="output_align", default="output.align")
    opt.add_option("-s", dest="source_corpus", default="data/toy1.en")
    opt.add_option("-t", dest="target_corpus", default="data/toy1.fr")
    (options, _) = opt.parse_args()

    translations = {}
    for l in open(options.probs, 'r').readlines():
        if l[0] == '#':
            pass
        else:
            (typ, tar, src, fp) = l.split()
            translations[tar, src] = float(fp)

    ali_out = options.output_align
    writer = open(ali_out, 'w')
    test_source = open(options.source_corpus, 'r').readlines()
    test_target = open(options.target_corpus, 'r').readlines()
    for dk in range(len(test_source)):
        source_tokens = test_source[dk].split()
        source_tokens.insert(0, 'NULL')
        target_tokens = test_target[dk].split()
        for i, token_target in enumerate(target_tokens):
            max_p = float('-inf')
            max_j = 0.0
            for j, token_source in enumerate(source_tokens):
                if translations[token_target, token_source] > max_p:
                    max_p = translations[token_target, token_source]
                    max_j = j
            if max_j > 0:
                writer.write(str(dk + 1) + ' ' + str(max_j) + ' ' + str(i + 1) + '\n')
    writer.flush()
    writer.close()

    writer = open(ali_out + '.token', 'w')
    test_source = open(options.source_corpus, 'r').readlines()
    test_target = open(options.target_corpus, 'r').readlines()
    for dk in range(len(test_source)):
        source_tokens = test_source[dk].split()
        source_tokens.insert(0, 'NULL')
        target_tokens = test_target[dk].split()
        for i, token_target in enumerate(target_tokens):
            max_p = float('-inf')
            max_j = 0.0
            for j, token_source in enumerate(source_tokens):
                if translations[token_target, token_source] > max_p:
                    max_p = translations[token_target, token_source]
                    max_j = j
            if max_j > 0:
                writer.write(str(dk + 1) + ' ' + str(source_tokens[max_j]) + ' ' + str(target_tokens[i]) + '\n')
    writer.flush()
    writer.close()