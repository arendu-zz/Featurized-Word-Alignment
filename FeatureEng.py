__author__ = 'arenduchintala'

import featurized_em as fe
import featurized_em_wa_mp as fe_w

global feature_values
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


def get_wa_features_fired(type, decision, context):
    global feature_values
    fired_features = []
    if type == fe_w.E_TYPE:
        val = feature_values.get((fe_w.E_TYPE, decision, context), 1.0)
        fired_features = [(val, (fe_w.E_TYPE, decision, context))]
        if context == fe_w.NULL:
            fired_features += [(-1.0, ("FROM_NULL",context))]
        #if decision == context:
        #    fired_features += [(1.0, ("IS_SAME", decision, context))]

        if fe_w.has_pos:
            decision_pos = decision.split("_")[1]
            context_pos = context.split("_")[1]
            if decision_pos == context_pos:
                fired_features += [(1.0, ("IS_POS_SAME", decision_pos, context_pos))]

        """if decision[0].isupper() and context[0].isupper() and context != fe_w.NULL:
            fired_features += [("IS_UPPER", decision, context)]"""
    elif type == fe_w.T_TYPE:
        p = context
        if decision != fe_w.NULL and p != fe_w.NULL:
            jump = abs(decision - p)
        else:
            jump = fe_w.NULL
        fired_features = [(1.0, (fe_w.T_TYPE, jump))]

    return fired_features


def get_pos_features_fired(type, decision, context):
    if decision is not fe.BOUNDARY_STATE and context is not fe.BOUNDARY_STATE:
        if type == fe.E_TYPE:
            # suffix feature, prefix feature
            fired_features = list(set([(fe.E_TYPE_SUF, str(decision[-3:]).lower(), context),
                                       (fe.E_TYPE_PRE, str(decision[:2]).lower(), context),
                                       (fe.E_TYPE_PRE, str(decision[:1]).lower(), context),
                                       ('IS_CAP', str(decision[0]).isupper(), context),
                                       ('IS_ALNUM', str(decision).isalnum(), context),
                                       ('IS_SAME', decision == context, context),
                                       (fe.E_TYPE, decision, context)]))
            # fired_features = [(fe.E_TYPE, decision, context)]
            return fired_features
        elif type == fe.T_TYPE:
            return [(fe.T_TYPE, decision, context)]
    else:
        return [(type, decision, context)]
