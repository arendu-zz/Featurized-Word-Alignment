__author__ = 'arenduchintala'
import const

global feature_values, dictionay_features
dictionay_features = {}
feature_values = {}


def load_feature_values(valpath=None):
    global feature_values
    try:
        feature_values = {}
        for l in open(valpath, 'r').readlines():
            [t, fr, en, val] = l.split()
            feature_values[t, fr, en] = float(val)
        print 'loaded feature values...'
    except BaseException:
        print 'binary feature values assumed...'


def load_dictionary_features(dict_features_path=None):
    global dictionay_features
    if dict_features_path is None:
        print 'no dictionary features...'
    else:
        df = open(dict_features_path, 'r').readlines()
        for line in df:
            line = line.strip()
            if line is not '':
                terms, v = line.strip().split('\t')
                t1, t2 = terms.split('|||')
                dictionay_features[t1, t2] = v
        print 'loaded ', len(dictionay_features), ' dictionary features...'
    return True


def get_wa_features_fired(type, decision, context):
    global feature_values, dictionay_features
    fired_features = []
    if type == const.E_TYPE:
        val = feature_values.get((const.E_TYPE, decision, context), 1.0)
        fired_features = [(val, (const.E_TYPE, decision, context))]

        if decision == context:
            fired_features += [(1.0, ("IS_SAME", decision, context))]

        if (decision, context) in dictionay_features:
            fired_features += [(1.0, ("IN_DICT", decision, context))]

        # if context == const.NULL:
        # fired_features += [(-1.0, ("IS_FROM_NULL", context))]

        # if const.has_pos:
        # decision_pos = decision.split("_")[1]
        # context_pos = context.split("_")[1]
        # if decision_pos == context_pos:
        # fired_features += [(1.0, ("IS_POS_SAME", decision_pos, context_pos))]

        """if decision[0].isupper() and context[0].isupper() and context != const.NULL:
            fired_features += [("IS_UPPER", decision, context)]"""
    elif type == const.T_TYPE:
        p = context
        if decision != const.NULL and p != const.NULL:
            jump = abs(decision - p)
        else:
            jump = const.NULL
        fired_features = [(1.0, (const.T_TYPE, jump))]

    return fired_features


