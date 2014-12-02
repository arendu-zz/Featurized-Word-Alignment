# -*- coding: latin1 -*-
__author__ = 'arenduchintala'
import numpy as np
from optparse import OptionParser


def edratio(a, b):
    ed = editdistance(a, b)
    edr = ed / float(max(len(a), len(b)))
    return 1.0 - edr


def editdistance(a, b):
    table = np.zeros((len(a) + 1, len(b) + 1))
    for i in range(len(a) + 1):
        table[i, 0] = i
    for j in range(len(b) + 1):
        table[0, j] = j
    # print 'start'
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i, j] = table[i - 1, j - 1]
            else:
                # print i, j
                diag = table[i - 1, j - 1] + 1
                # print 'diag', diag
                left = table[i - 1, j] + 1
                # print 'left', left
                top = table[i, j - 1] + 1
                # print 'top', top
                best = min(diag, top, left)
                # print 'best so far', best
                table[i, j] = best
                # print 'current cell', table[i, j]
    # print table
    return table[i, j]


if __name__ == "__main__":
    x = "de"  # "ALTRUISM"
    y = "onf"  # "PLASMA"
    ed = editdistance(x, y)
    # print 'final dist', ed, 1.0 / (ed + 1.0)
    edr = edratio(x, y)
    # print 'final edr', edr
    # eixit()
    opt = OptionParser()
    opt.add_option("-i", dest="basic_features", default="experiment/initial.trans")
    (options, _) = opt.parse_args()

    for l in open(options.basic_features, 'r').readlines():
        [t, fr, en, wt] = l.split()
        ed = str(round(1.0 / (editdistance(fr, en) + 1.0), 4))
        l2 = '\t'.join([t, fr, en, ed])
        print l2

