__author__ = 'arenduchintala'

from optparse import OptionParser
import sys
from math import exp, log
from DifferentiableFunction import DifferentiableFunction
import numpy as np
import FeatureEng as FE
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import approx_fprime
import pdb
import utils
from pprint import pprint
from collections import defaultdict
import itertools

global BOUNDARY_STATE, END_STATE, SPLIT, E_TYPE, T_TYPE, possible_states, normalizing_decision_map
global cache_normalizing_decision, features_to_conditional_arcs, conditional_arcs_to_features
global trellis, max_jump_width
max_jump_width = 6  # creates a span of +/- span centered around current token
trellis = []
cache_normalizing_decision = {}
BOUNDARY_STATE = "###"
NULL = "NULL"
SPLIT = "###/###"
E_TYPE = "EMISSION"
E_TYPE_PRE = "PREFIX_FEATURE"
E_TYPE_SUF = "SUFFIX_FEATURE"
T_TYPE = "TRANSITION"
S_TYPE = "STATE"
ALL = "ALL_STATES"
fractional_counts = {}
conditional_arcs_to_features = {}
features_to_conditional_arcs = {}
feature_index = {}
conditional_arc_index = {}
possible_states = {}
possible_obs = {}
normalizing_decision_map = {}


def get_jump(aj, aj_1):
    if aj != NULL and aj_1 != 'NULL':
        jump = abs(aj_1 - aj)
        assert jump <= max_jump_width
    else:
        jump = NULL
    return jump


def populate_features():
    global trellis, feature_index
    for treli in trellis:
        for idx in treli:
            for t_idx, t_tok, s_idx, s_tok, L in treli[idx]:
                if idx > 0:
                    for prev_t_idx, prev_t_tok, prev_s_idx, prev_s_tok, L in treli[idx - 1]:
                        # an arc in an hmm can have the following information:
                        # prev_s_idx, prev_s_tok, s_idx, s_tok, t_idx, t_tok
                        prev_s = (prev_s_idx, prev_s_tok)
                        curr_s = (s_idx, s_tok)
                        curr_t = (t_idx, t_tok)
                        # arc = (prev_s, curr_s, curr_t)
                        """
                        transition features
                        """
                        jump = get_jump(s_idx, prev_s_idx)
                        ff_t = FE.get_wa_features_fired(type=T_TYPE, decision=jump, context=prev_s_idx)
                        transition_arc = (jump, prev_s_idx)
                        for f in ff_t:
                            feature_index[f] = feature_index.get(f, 0) + 1
                            ca2f = conditional_arcs_to_features.get(transition_arc, set([]))
                            ca2f.add(f)
                            conditional_arcs_to_features[transition_arc] = ca2f
                            f2ca = features_to_conditional_arcs.get(f, set([]))
                            f2ca.add(transition_arc)
                            features_to_conditional_arcs[f] = f2ca

                """
                emission features
                """
                emission_arc = (t_tok, s_tok)
                ff_e = FE.get_wa_features_fired(type=E_TYPE, decision=t_tok, context=s_tok)
                for f in ff_e:
                    feature_index[f] = feature_index.get(f, 0) + 1
                    ca2f = conditional_arcs_to_features.get(emission_arc, set([]))
                    ca2f.add(f)
                    conditional_arcs_to_features[emission_arc] = ca2f
                    f2ca = features_to_conditional_arcs.get(f, set([]))
                    f2ca.add(emission_arc)
                    features_to_conditional_arcs[f] = f2ca


def populate_normalizing_terms(target):
    global max_jump_width
    for treli in trellis:
        for idx in treli:
            tar = target[idx]
            for t_idx, t_tok, s_idx, s_tok, L in treli[idx]:
                s = (s_idx, s_tok)
                t = (t_idx, t_tok)
                ndm = normalizing_decision_map.get((E_TYPE, s_tok), set([]))
                ndm.add(t_tok)
                normalizing_decision_map[E_TYPE, s_tok] = ndm

                if s_idx == 0:
                    normalizing_decision_map[T_TYPE, s_idx] = set(range(max_jump_width) + [NULL])
                elif s_idx == L:
                    pass
                else:
                    normalizing_decision_map[T_TYPE, s_idx] = set(range(max_jump_width) + [NULL])


def get_transition_no_feature(jump):
    p = utils.normpdf(jump, 1.0, 3.0)  # TODO: what should the variance be?
    return log(p)


def get_decision_given_context(theta, type, decision, context):
    global normalizing_decision_map, cache_normalizing_decision
    fired_features = FE.get_wa_features_fired(type=type, context=context, decision=decision)
    theta_dot_features = sum([theta[f] for f in fired_features if f in theta])
    # TODO: weights theta are initialized to 0.0
    # TODO: this initialization should be done in a better way
    if (type, context) in cache_normalizing_decision:
        theta_dot_normalizing_features = cache_normalizing_decision[type, context]
    else:
        normalizing_decisions = normalizing_decision_map[type, context]
        theta_dot_normalizing_features = float('-inf')
        for d in normalizing_decisions:
            d_features = FE.get_wa_features_fired(type=type, context=context, decision=d)
            theta_dot_normalizing_features = utils.logadd(theta_dot_normalizing_features,
                                                          sum([theta[f] for f in d_features if f in theta]))

        cache_normalizing_decision[type, context] = theta_dot_normalizing_features
    log_prob = round(theta_dot_features - theta_dot_normalizing_features, 10)
    if log_prob > 0.0:
        print log_prob, type, decision, context
        raise Exception
    return log_prob


def get_possible_states(o):
    if o == BOUNDARY_STATE:
        return [BOUNDARY_STATE]
    # elif o in possible_states:
    # return list(possible_states[o])
    else:
        return list(possible_states[ALL] - set([BOUNDARY_STATE]))


def get_backwards(theta, obs, alpha_pi):
    n = len(obs) - 1  # index of last word
    COMPOSITE_BOUNDARY_STATE = obs[n][0]
    beta_pi = {(n, COMPOSITE_BOUNDARY_STATE): 0.0}
    S = alpha_pi[(n, COMPOSITE_BOUNDARY_STATE)]  # from line 13 in pseudo code
    for k in range(n, 0, -1):
        for v in obs[k]:
            tk, t_tok, aj, s_tok, L = v
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            pb = beta_pi[(k, v)]
            accumulate_fc(type=E_TYPE, alpha=alpha_pi[(k, v)], beta=beta_pi[k, v], S=S, d=t_tok, c=s_tok)
            for u in obs[k - 1]:
                tk_1, t_tok_1, aj_1, s_tok_1, L = u

                jump = get_jump(aj, aj_1)
                q = get_decision_given_context(theta, T_TYPE, decision=jump, context=aj_1)
                # q = log(1.0 / len(obs[k]))
                p = q + e
                beta_p = pb + p  # The beta includes the emission probability
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                # print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
                # alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
                accumulate_fc(type=T_TYPE, alpha=alpha_pi[k - 1, u], beta=beta_pi[k, v], q=q, e=e, d=jump, c=aj_1,
                              S=S)
    return S, beta_pi


def get_viterbi_and_forward(theta, obs):
    global max_jump_width
    COMPOSITE_BOUNDARY_STATE = obs[0][0]
    pi = {(0, COMPOSITE_BOUNDARY_STATE): 0.0}
    alpha_pi = {(0, COMPOSITE_BOUNDARY_STATE): 0.0}
    arg_pi = {(0, COMPOSITE_BOUNDARY_STATE): []}
    for k in range(1, len(obs)):  # the words are numbered from 1 to n, 0 is special start character
        for v in obs[k]:  # [1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            for u in obs[k - 1]:  # [1]:
                tk, t_tok, aj, s_tok, L = v
                tk_1, t_tok_1, aj_1, s_tok_1, L = u

                jump = get_jump(aj, aj_1)
                q = get_decision_given_context(theta, T_TYPE, decision=jump, context=aj_1)

                # q = log(1.0 / len(obs[k]))
                e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
                p = pi[(k - 1, u)] + q + e
                alpha_p = alpha_pi[(k - 1, u)] + q + e
                if len(arg_pi[(k - 1, u)]) == 0:
                    bt = [u]
                else:
                    bt = [arg_pi[(k - 1, u)], u]
                max_prob_to_bt[p] = bt
                sum_prob_to_bt.append(alpha_p)

            max_bt = max_prob_to_bt[max(max_prob_to_bt)]
            new_pi_key = (k, v)
            pi[new_pi_key] = max(max_prob_to_bt)
            # print 'mu   ', new_pi_key, '=', pi[new_pi_key], exp(pi[new_pi_key])
            alpha_pi[new_pi_key] = utils.logadd_of_list(sum_prob_to_bt)
            # print 'alpha', new_pi_key, '=', alpha_pi[new_pi_key], exp(alpha_pi[new_pi_key])
            arg_pi[new_pi_key] = max_bt

    max_bt = max_prob_to_bt[max(max_prob_to_bt)]
    max_p = max(max_prob_to_bt)
    max_bt = utils.flatten_backpointers(max_bt)
    return max_bt, max_p, alpha_pi


def reset_fractional_counts():
    global fractional_counts, cache_normalizing_decision
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}


def accumulate_fc(type, alpha, beta, d, S, c=None, k=None, q=None, e=None):
    global fractional_counts
    if type == T_TYPE:
        update = alpha + q + e + beta - S
        fractional_counts[T_TYPE, d, c] = utils.logadd(update, fractional_counts.get((T_TYPE, d, c,), float('-inf')))
    elif type == E_TYPE:
        update = alpha + beta - S  # the emission should be included in alpha
        fractional_counts[E_TYPE, d, c] = utils.logadd(update, fractional_counts.get((E_TYPE, d, c,), float('-inf')))
    elif type == S_TYPE:
        update = alpha + beta - S
        old = fractional_counts.get((S_TYPE, k, d), float('-inf'))
        fractional_counts[S_TYPE, k, d] = utils.logadd(update, old)
        old = fractional_counts.get((S_TYPE, None, d), float('-inf'))
        fractional_counts[S_TYPE, None, d] = utils.logadd(update, old)
    else:
        raise "Wrong type"


def write_alignments(theta, save_align):
    global trellis
    write_align = open(save_align, 'w')
    for idx, obs in enumerate(trellis[:]):
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, obs)
        w = ' '.join([str(tar_i - 1) + '-' + str(src_i - 1) for src_i, src, tar_i, tar, L in max_bt if
                      (tar_i != NULL and tar_i > 0 and src_i > 0)])
        write_align.write(w + '\n')
    write_align.flush()
    write_align.close()


def get_likelihood(theta):
    # theta = dict((k, theta_list[v]) for k, v in feature_index.items())
    global trellis
    reset_fractional_counts()
    data_likelihood = 0.0
    for idx, obs in enumerate(trellis[:]):
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, obs)
        S, beta_pi = get_backwards(theta, obs, alpha_pi)
        data_likelihood += S

    reg = sum([t ** 2 for k, t in theta.items()])
    print 'reg log likelihood:', data_likelihood - (0.5 * reg)  # , ' '.join(obs)
    return data_likelihood - (0.5 * reg)


def get_gradient(theta):
    fractional_count_grad = {}
    for event in fractional_counts:
        (type, d, c) = event
        # print event, fractional_counts[event]
        if type == E_TYPE:
            Adc = exp(fractional_counts.get(event, float('-inf')))
            a_dc = exp(get_decision_given_context(theta, type=E_TYPE, decision=d, context=c))
            fractional_count_grad[type, d, c] = Adc * (1 - a_dc)
        elif type == T_TYPE:
            Adc = exp(fractional_counts.get(event, float('-inf')))
            dd = get_decision_given_context(theta, type=T_TYPE, decision=d, context=c)
            a_dc = exp(dd)
            fractional_count_grad[type, d, c] = Adc * (1 - a_dc)

    grad = {}
    for fcg_event in fractional_count_grad:
        (type, d, c) = fcg_event
        if type == E_TYPE:
            for f in conditional_arcs_to_features[d, c]:
                grad[f] = fractional_count_grad[fcg_event] + grad.get(f, 0.0)
        elif type == T_TYPE:
            for f in conditional_arcs_to_features[d, c]:
                grad[f] = fractional_count_grad[fcg_event] + grad.get(f, 0.0)

    for t in theta:
        if t not in grad:
            grad[t] = 0.0 - (0.5 * theta[t])
        else:
            grad[t] -= (0.5 * theta[t])
    print 'max gradient      :', max(grad.values())
    return grad


def populate_trellis(source_corpus, target_corpus):
    global trellis, max_jump_width
    for s_sent, t_sent in zip(source_corpus, target_corpus):
        t_sent.insert(0, BOUNDARY_STATE)
        s_sent.insert(0, BOUNDARY_STATE)
        t_sent.append(BOUNDARY_STATE)
        s_sent.append(BOUNDARY_STATE)
        current_trellis = {}
        for t_idx, t_tok in enumerate(t_sent):
            if t_tok == BOUNDARY_STATE:
                current_trellis[t_idx] = [(t_idx, BOUNDARY_STATE, t_idx, BOUNDARY_STATE, len(s_sent))]
            else:
                start = t_idx - (max_jump_width / 2) if t_idx - (max_jump_width / 2) >= 0 else 0
                end = t_idx + (max_jump_width / 2)
                assert end - start <= max_jump_width
                current_trellis[t_idx] = [(t_idx, t_tok, s_idx + start, s_tok, len(s_sent)) for s_idx, s_tok in
                                          enumerate(s_sent[start:end]) if s_tok != BOUNDARY_STATE] + [
                                             (t_idx, t_tok, NULL, NULL, len(s_sent))]
        trellis.append(current_trellis)


if __name__ == "__main__":
    trellis = []
    possible_states = defaultdict(set)
    possible_obs = defaultdict(set)
    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="data/small/en20.50")
    opt.add_option("-s", dest="source_corpus", default="data/small/es20.50")
    opt.add_option("-o", dest="save", default="theta.out")
    opt.add_option("-a", dest="alignments", default="alignments.out")
    (options, _) = opt.parse_args()
    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines() if s.strip() != '']
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines() if s.strip() != '']
    populate_trellis(source, target)
    populate_features()
    populate_normalizing_terms(target)
    init_theta = dict((k, 1.0) for k in feature_index)

    # init_theta = dict((k, np.random.uniform(0, 0.1)) for k in feature_index)
    F = DifferentiableFunction(get_likelihood, get_gradient)
    F.method = "LBFGS"
    (fopt, theta, return_status) = F.maximize(init_theta)
    # print chk_grad
    # chk_grad = F.fprime(init_theta)
    # for k in sorted(theta):
    # print theta[k], chk_grad[k], k
    # print return_status
    write_theta = open(options.save, 'w')
    for t in sorted(theta):
        str_t = reduce(lambda a, d: str(a) + '\t' + str(d), t, '')
        write_theta.write(str_t.strip() + '\t' + str(theta[t]) + '' + "\n")
    write_theta.flush()
    write_theta.close()

    write_alignments(theta, options.alignments)
