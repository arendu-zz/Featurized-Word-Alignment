__author__ = 'arenduchintala'

import featurized_em as fe


def get_wa_features_fired(type, decision, context):
    # TODO: this has model 1 features only
    (d_pos, d_tok) = decision
    (c_pos, c_tok) = context
    fired_features = [("MODEL_1", d_tok, c_tok)]
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