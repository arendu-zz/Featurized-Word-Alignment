__author__ = 'arenduchintala'
import sys
import pdb

if __name__ == '__main__':
    try:
        predicted = sys.argv[1]
        key_file = sys.argv[2]
    except IndexError, err:
        print "Usage: python eval.py [predicted alignment file] [key file]"
        exit()

    correct = 0
    prec_total_count = 0
    recall_total_count = 0
    predicted_lst = open(predicted, 'r').readlines()
    key_lst = open(key_file, 'r').readlines()
    for p, k in zip(predicted_lst, key_lst):
        p = set(p.split())
        k = set(k.split())
        correct += len(p.intersection(k))
        prec_total_count += len(p)
        recall_total_count += len(k)
    print "%10s  %10s  %10s  %10s   %10s" % (
        "Type", "Total", "Precision", "Recall", "F1-Score")
    print "==============================================================="
    prec = float(correct) / float(prec_total_count)
    recall = float(correct) / float(recall_total_count)
    fscore = (2 * prec * recall) / (prec + recall)
    print "%10s        %4d     %0.3f        %0.3f        %0.3f" % (
        "total", recall_total_count, prec, recall, fscore)




