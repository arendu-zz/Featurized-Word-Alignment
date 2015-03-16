__author__ = 'arenduchintala'

import sys

if __name__ == '__main__':
    try:
        key_file = sys.argv[1].strip()
    except IndexError, err:
        print "Usage: python convert_keys.py [name of key file]"
        exit()

    k = open(key_file, 'r').readlines()
    k_dict = {}
    for l in k:
        l_num, t1, t2 = l.split()
        l_num = int(l_num)
        t = (int(t1) - 1, int(t2) - 1)
        lst = k_dict.get(l_num, [])
        lst.append(t)
        k_dict[l_num] = lst

    for k in sorted(k_dict):
        lst = k_dict[k]
        lst = [str(i) + '-' + str(j) for i, j in sorted(lst)]
        print ' '.join(lst)

