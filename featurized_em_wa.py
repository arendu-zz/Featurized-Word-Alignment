__author__ = 'arenduchintala'

from optparse import OptionParser
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
global trellis
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


def populate_features():
    # uses the global trellis to populate features
    global trellis, feature_index
    for treli in trellis:
        for t in treli:
            (t_idx, t_tok) = t
            for s in treli[t]:
                (s_idx, s_tok) = s
                ff = FE.get_wa_features_fired(None, decision=(t_idx, t_tok), context=(s_idx, s_tok))
                for f in ff:
                    feature_index[f] = feature_index.get(f, 0) + 1
                    ca2f = conditional_arcs_to_features.get((t, s), set([]))
                    ca2f.add(f)
                    conditional_arcs_to_features[t, s] = ca2f
                    f2ca = features_to_conditional_arcs.get(f, set([]))
                    f2ca.add((t, s))
                    features_to_conditional_arcs[f] = f2ca


def populate_arcs_to_features():
    global features_to_conditional_arcs, conditional_arcs_to_features, feature_index, conditional_arc_index
    for d, c in itertools.product(possible_obs[ALL], possible_states[ALL]):
        if c == BOUNDARY_STATE and d != BOUNDARY_STATE:
            pass
        else:
            fired_features = FE.get_pos_features_fired(E_TYPE, decision=d, context=c)
            fs = conditional_arcs_to_features.get((E_TYPE, d, c), set([]))
            fs = fs.union(fired_features)
            conditional_arcs_to_features[E_TYPE, d, c] = fs
            conditional_arc_index[E_TYPE, d, c] = conditional_arc_index.get((E_TYPE, d, c), len(conditional_arc_index))
            for ff in fired_features:
                feature_index[ff] = feature_index.get(ff, len(feature_index))
                cs = features_to_conditional_arcs.get(ff, set([]))
                cs.add((E_TYPE, d, c))
                features_to_conditional_arcs[ff] = cs

    for d, c in itertools.product(possible_states[ALL], possible_states[ALL]):
        if d == BOUNDARY_STATE and c == BOUNDARY_STATE:
            pass
        else:
            fired_features = FE.get_pos_features_fired(T_TYPE, decision=d, context=c)
            fs = conditional_arcs_to_features.get((T_TYPE, d, c), set([]))
            fs = fs.union(fired_features)
            conditional_arcs_to_features[T_TYPE, d, c] = fs
            conditional_arc_index[T_TYPE, d, c] = conditional_arc_index.get((T_TYPE, d, c), len(conditional_arc_index))
            for ff in fired_features:
                feature_index[ff] = feature_index.get(ff, len(feature_index))
                cs = features_to_conditional_arcs.get(ff, set([]))
                cs.add((T_TYPE, d, c))
                features_to_conditional_arcs[ff] = cs


def populate_normalizing_terms():
    for s in possible_states[ALL]:
        if s == BOUNDARY_STATE:
            normalizing_decision_map[E_TYPE, s] = set([BOUNDARY_STATE])
            normalizing_decision_map[T_TYPE, s] = possible_states[ALL] - set([BOUNDARY_STATE])
        else:
            normalizing_decision_map[E_TYPE, s] = possible_obs[ALL] - set([BOUNDARY_STATE])
            normalizing_decision_map[T_TYPE, s] = possible_states[ALL]


def get_decision_given_context(theta, type, decision, context):
    # print 'finding', type, ' d|c', decision, '|', context
    global normalizing_decision_map, cache_normalizing_decision
    fired_features = FE.get_pos_features_fired(type=type, context=context, decision=decision)
    theta_dot_features = sum([theta[f] for f in fired_features if f in theta])
    # TODO: weights theta are initialized to 0.0
    # TODO: this initialization should be done in a better way
    if (type, context) in cache_normalizing_decision:
        theta_dot_normalizing_features = cache_normalizing_decision[type, context]
    else:
        normalizing_decisions = normalizing_decision_map[type, context]
        theta_dot_normalizing_features = float('-inf')
        for d in normalizing_decisions:
            d_features = FE.get_pos_features_fired(type=type, context=context, decision=d)
            try:
                theta_dot_normalizing_features = utils.logadd(theta_dot_normalizing_features,
                                                              sum([theta[f] for f in d_features if f in theta]))
            except OverflowError:
                pdb.set_trace()
        cache_normalizing_decision[type, context] = theta_dot_normalizing_features
    log_prob = theta_dot_features - theta_dot_normalizing_features
    return log_prob  # log(prob)


def get_possible_states(o):
    if o == BOUNDARY_STATE:
        return [BOUNDARY_STATE]
    # elif o in possible_states:
    # return list(possible_states[o])
    else:
        return list(possible_states[ALL] - set([BOUNDARY_STATE]))


def get_backwards(theta, words, alpha_pi):
    n = len(words) - 1  # index of last word
    beta_pi = {(n, BOUNDARY_STATE): 0.0}
    S = alpha_pi[(n, BOUNDARY_STATE)]  # from line 13 in pseudo code
    for k in range(n, 0, -1):
        for v in get_possible_states(words[k]):
            e = get_decision_given_context(theta, E_TYPE, words[k], v)
            pb = beta_pi[(k, v)]
            accumulate_fc(type=S_TYPE, alpha=alpha_pi[(k, v)], beta=beta_pi[k, v], k=k, S=S, d=v)
            accumulate_fc(type=E_TYPE, alpha=alpha_pi[(k, v)], beta=beta_pi[k, v], S=S, d=words[k], c=v)
            for u in get_possible_states(words[k - 1]):
                q = get_decision_given_context(theta, T_TYPE, v, u)
                p = q + e
                beta_p = pb + p  # The beta includes the emission probability
                new_pi_key = (k - 1, u)
                if new_pi_key not in beta_pi:  # implements lines 16
                    beta_pi[new_pi_key] = beta_p
                else:
                    beta_pi[new_pi_key] = utils.logadd(beta_pi[new_pi_key], beta_p)
                    # print 'beta     ', new_pi_key, '=', beta_pi[new_pi_key], exp(beta_pi[new_pi_key])
                # alpha_pi[(k - 1, u)] + p + beta_pi[(k, v)] - S
                accumulate_fc(type=T_TYPE, alpha=alpha_pi[k - 1, u], beta=beta_pi[k, v], q=q, e=e, d=v, c=u, S=S)
    return S, beta_pi


def get_viterbi_and_forward(theta, words):
    pi = {(0, BOUNDARY_STATE): 0.0}
    alpha_pi = {(0, BOUNDARY_STATE): 0.0}
    arg_pi = {(0, BOUNDARY_STATE): []}
    for k in range(1, len(words)):  # the words are numbered from 1 to n, 0 is special start character
        for v in get_possible_states(words[k]):  # [1]:
            max_prob_to_bt = {}
            sum_prob_to_bt = []
            for u in get_possible_states(words[k - 1]):  # [1]:
                if u == BOUNDARY_STATE and v == BOUNDARY_STATE:
                    print 'hmm'
                q = get_decision_given_context(theta, T_TYPE, v, u)
                e = get_decision_given_context(theta, E_TYPE, words[k], v)
                # print 'q,e', q, e
                # p = pi[(k - 1, u)] * q * e
                # alpha_p = alpha_pi[(k - 1, u)] * q * e
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


def get_likelihood(theta):
    # theta = dict((k, theta_list[v]) for k, v in feature_index.items())
    global trellis
    reset_fractional_counts()
    data_likelihood = 0.0
    for idx, obs in enumerate(trellis[:]):
        max_bt, max_p, alpha_pi = get_viterbi_and_forward(theta, obs)
        """
        if idx == 0:
            t, p, al = get_viterbi_and_forward(theta, obs)
            t.pop(0)
            oo = obs[1:-1]
            pr = [wo + '/' + to for wo, to in zip(oo, t)]
            print ' '.join(pr)
        """

        S, beta_pi = get_backwards(theta, obs, alpha_pi)

        data_likelihood += S

    reg = sum([t ** 2 for k, t in theta.items()])
    print 'accumulating', data_likelihood - (0.5 * reg)  # , ' '.join(obs)
    return data_likelihood - (0.5 * reg)


def get_gradient(theta):
    fractional_count_grad = {}
    for c in possible_states[ALL]:
        for d in possible_obs[c]:
            if c == BOUNDARY_STATE and d != BOUNDARY_STATE:
                pass  # ignore this
            else:
                event = E_TYPE, d, c
                Adc = exp(fractional_counts.get(event, float('-inf')))
                a_dc = exp(get_decision_given_context(theta, type=E_TYPE, decision=d, context=c))
                fractional_count_grad[event] = Adc * (1 - a_dc)

    for d, c in itertools.product(possible_states[ALL], possible_states[ALL]):
        if d == BOUNDARY_STATE and c == BOUNDARY_STATE:
            pass  # ignore
        else:
            event = T_TYPE, d, c
            Adc = exp(fractional_counts.get(event, float('-inf')))
            a_dc = exp(get_decision_given_context(theta, type=T_TYPE, decision=d, context=c))
            fractional_count_grad[event] = Adc * (1 - a_dc)
    grad = {}
    for fcg_event in fractional_count_grad:
        for f in conditional_arcs_to_features[fcg_event]:
            grad[f] = fractional_count_grad[fcg_event] + grad.get(f, 0.0)

    for t in theta:
        if t not in grad:
            grad[t] = 0.0 - theta[t]
        else:
            grad[t] = grad[t] - theta[t]
    # pdb.set_trace()
    return grad


def populate_trellis(source, target):
    global trellis
    span = 1  # creates a span of +/- span centered around current token
    for s_sent, t_sent in zip(source, target):
        current_trellis = {}
        for t_idx, t_tok in enumerate(t_sent):
            start = t_idx - span if t_idx - span >= 0 else 0
            end = t_idx + span + 1
            current_trellis[t_idx, t_tok] = [(s_idx + start, s_tok) for s_idx, s_tok in
                                             enumerate(s_sent[start:end])] + [(NULL, NULL)]
        trellis.append(current_trellis)


if __name__ == "__main__":
    trellis = []
    possible_states = defaultdict(set)
    possible_obs = defaultdict(set)
    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="data/toy2/en")
    opt.add_option("-f", dest="source_corpus", default="data/toy2/fr")
    opt.add_option("-o", dest="save", default="theta.out")
    opt.add_option("-a", dest="alignments", default="alignments.out")
    (options, _) = opt.parse_args()
    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines() if s.strip() != '']
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines() if s.strip() != '']
    populate_trellis(source, target)
    populate_features()
    pprint(feature_index)
    pprint(features_to_conditional_arcs)

    """
    populate_normalizing_terms()
    populate_arcs_to_features()
    init_theta = dict((k, np.random.uniform(-0.1, 0.1)) for k in feature_index)
    F = DifferentiableFunction(get_likelihood, get_gradient)
    (fopt, theta, return_status) = F.maximize(init_theta)
    print return_status
    write_theta = open(options.save, 'w')
    for t in theta:
        write_theta.write(str(t) + "\t" + str(theta[t]) + "\n")
    write_theta.flush()
    write_theta.close()
    """