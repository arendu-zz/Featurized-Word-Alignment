from optparse import OptionParser
import pdb

if __name__ == "__main__":
    """
    the spanish tags have come from http://stackoverflow.com/questions/27047450/meaning-of-stanford-spanish-pos-tagger-tags
    anacora tagset
    """
    opt = OptionParser()
    opt.add_option("-p", dest="tagged_file", default="experiment/data/train.en.pos")
    opt.add_option("-m", dest="map", default="experiment/data/en.map")
    opt.add_option("-l", dest="lang", default="en")
    (options, _) = opt.parse_args()
    map = {}
    for l in open(options.map, 'r').readlines():
        s, u = l.lower().split()
        if options.lang == "en":
            for sv in s.split('|'):
                map[sv.strip().lower()] = u.strip().lower()
        elif options.lang == "es":
            s = s.strip().lower()[:2]
            map[s.strip()] = u.strip().lower()

    if options.lang == "es":
        for l in open(options.tagged_file, 'r').readlines():
            t = [wt.lower().split("_")[1] for wt in l.lower().split()]
            # print ' '.join(t)
            ut = [i[:2] if i[:2] in map else i[:1] for i in t]
            #print ' '.join(ut)
            print ' '.join([map[i] for i in ut])

    else:
        for l in open(options.tagged_file, 'r').readlines():
            t = [wt.lower().split("_")[1] for wt in l.lower().split()]
            ut = [map[i] for i in t]
            print ' '.join(ut)