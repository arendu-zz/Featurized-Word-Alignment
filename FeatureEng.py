__author__ = 'arenduchintala'

import featurized_em as fe
import featurized_em_wa as fe_w


def get_wa_features_fired(type, decision, context):
    fired_features = []
    if type == fe_w.E_TYPE:
        fired_features = [(fe_w.E_TYPE, decision, context)]
        if decision == context:
            fired_features += [("IS_SAME", decision, context)]
    elif type == fe_w.T_TYPE:
        (p, L) = context
        if decision != fe_w.NULL and p != fe_w.NULL:
            jump = abs(decision - p)
        else:
            jump = fe_w.NULL
        fired_features = [(fe_w.T_TYPE, jump)]

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