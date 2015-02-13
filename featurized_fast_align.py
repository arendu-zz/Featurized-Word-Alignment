__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log
from scipy.optimize import minimize
import numpy as np
import FeatureEng as FE
import utils
import copy
import pdb
from math import floor

np.seterr(all='raise')

from const import NULL, BOUNDARY_START, BOUNDARY_END, IBM_MODEL_1, HMM_MODEL, E_TYPE, T_TYPE, EPS, Po
from common import populate_trellis, populate_features, write_alignments, write_alignments_col, \
    write_alignments_col_tok, write_probs, write_weights, initialize_theta


global cache_normalizing_decision, features_to_events, events_to_features, normalizing_decision_map, num_toks
global trellis, max_jump_width, number_of_events, EPS, snippet, max_beam_width, rc
global source, target, data_likelihood, event_grad, feature_index, event_index
global events_per_trellis, event_to_event_index, has_pos, event_counts, du, itercount, N
global diagonal_tension, emp_feat
has_pos = False
event_grad = {}
data_likelihood = 0.0
snippet = ''
EPS = 1e-5
rc = 0.25

max_jump_width = 10
max_beam_width = 20  # creates a span of +/- span centered around current token
trellis = []
cache_normalizing_decision = {}

fractional_counts = {}
number_of_events = 0
events_to_features = {}
features_to_events = {}
feature_index = {}
feature_counts = {}
du = []
event_index = []
event_to_event_index = {}
event_counts = {}

normalizing_decision_map = {}
itercount = 0


def compute_z(i, m, n, dt):
    split = floor(i) * n / m
    flr = floor(split)
    ceil = flr + 1
    ratio = exp(-dt / n)
    num_top = n - flr
    ezt = 0
    ezb = 0
    if num_top != 0.0:
        ezt = unnormalized_prob(i, ceil, m, n, dt) * (1.0 - np.power(ratio, num_top)) / (1.0 - ratio)
    if flr != 0.0:
        ezb = unnormalized_prob(i, flr, m, n, dt) * (1.0 - np.power(ratio, flr)) / (1.0 - ratio)
    return ezb + ezt


def arithmetico_geo_series(a1, g1, r, d, n):
    gnp1 = g1 * np.power(r, n)
    an = d * (n - 1) + a1
    x1 = a1 * g1
    g2 = g1 * r
    rm1 = r - 1
    return (an * gnp1 - x1) / rm1 - d * (gnp1 - g2) / (rm1 * rm1)


def unnormalized_prob(i, j, m, n, dt):
    return exp(h(i, j, m, n) * dt)


def compute_d_log_z(i, m, n, dt):
    z = compute_z(i, n, m, dt)
    split = float(i) * n / m
    flr = floor(split)
    ceil = flr + 1
    ratio = exp(-dt / n)
    d = -1.0 / n
    num_top = n - flr
    pct = 0.0
    pcb = 0.0
    if num_top != 0:
        pct = arithmetico_geo_series(h(i, ceil, m, n), unnormalized_prob(i, ceil, m, n, dt), ratio, d,
                                     num_top)

    if flr != 0:
        pcb = arithmetico_geo_series(h(i, flr, m, n), unnormalized_prob(i, flr, m, n, dt), ratio, d, flr)
    return (pcb + pct) / z


def h(i, j, m, n):
    return - abs((float(i) / float(m)) - (float(j) / float(n)))


def get_decision_given_context(theta, type, decision, context):
    global normalizing_decision_map, cache_normalizing_decision, feature_index
    fired_features = FE.get_wa_features_fired(type=type, context=context, decision=decision)

    theta_dot_features = sum([theta[feature_index[f]] * f_wt for f_wt, f in fired_features])

    if (type, context) in cache_normalizing_decision:
        theta_dot_normalizing_features = cache_normalizing_decision[type, context]
    else:
        normalizing_decisions = normalizing_decision_map[type, context]
        theta_dot_normalizing_features = 0
        for d in normalizing_decisions:
            d_features = FE.get_wa_features_fired(type=type, context=context, decision=d)
            theta_dot_normalizing_features += exp(sum([theta[feature_index[f]] * f_wt for f_wt, f in d_features]))

        theta_dot_normalizing_features = log(theta_dot_normalizing_features)
        cache_normalizing_decision[type, context] = theta_dot_normalizing_features
    log_prob = round(theta_dot_features - theta_dot_normalizing_features, 10)
    if log_prob > 0.0:
        # print "log_prob = ", log_prob, type, decision, context
        # pdb.set_trace()
        if options.algorithm == 'LBFGS':
            raise Exception
        else:
            log_prob = 0.0  # TODO figure out why in the EM algorithm this error happens?
    return log_prob


def get_model1_forward(theta, obs_id, fc=None, ef=None):
    global source, target, trellis, diagonal_tension
    obs = trellis[obs_id]
    m = len(obs)
    max_bt = [-1] * len(obs)
    p_st = 0.0
    for t_idx in obs:
        t_tok = target[obs_id][t_idx]
        sum_e = float('-inf')
        sum_pei = float('-inf')
        max_e = float('-inf')
        max_s_idx = None
        sum_sj = float('-inf')
        sum_q = float('-inf')
        for _, s_idx in obs[t_idx]:
            n = len(obs[t_idx])
            s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            sum_e = utils.logadd(sum_e, e)

            if t_tok == BOUNDARY_START or t_tok == BOUNDARY_END:
                q = 0.0
            elif s_tok == NULL:
                q = np.log(Po)
            else:
                # q = get_fast_align_transition(theta, t_idx, s_idx, m - 2, n - 1)
                az = compute_z(t_idx, m - 2, n - 1, diagonal_tension) / (1 - Po)
                q = np.log(unnormalized_prob(t_idx, s_idx, m - 2, n - 1, diagonal_tension) / az)

            sum_pei = utils.logadd(sum_pei, q + e)
            sum_sj = utils.logadd(sum_sj, e + q)
            sum_q = utils.logadd(sum_q, q)

            if e > max_e:
                max_e = e
                max_s_idx = s_idx
        max_bt[t_idx] = (t_idx, max_s_idx)
        p_st += sum_sj

        if p_st == float('inf'):
            pdb.set_trace()
        # update fractional counts
        if fc is not None:
            for _, s_idx in obs[t_idx]:
                n = len(obs[t_idx])
                s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
                e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)

                if t_tok == BOUNDARY_START or t_tok == BOUNDARY_END:
                    q = 0.0
                    hijmn = 0.0
                elif s_tok == NULL:
                    q = np.log(Po)
                    hijmn = 0.0
                else:
                    az = compute_z(t_idx, m - 2, n - 1, diagonal_tension) / (1 - Po)
                    q = np.log(unnormalized_prob(t_idx, s_idx, m - 2, n - 1, diagonal_tension) / az)
                    hijmn = h(t_idx, s_idx, m - 2, n - 1)

                p_ai = e + q - sum_pei  # TODO: times h(j',i,m,n)
                # p_q = q - sum_q
                event = (E_TYPE, t_tok, s_tok)
                fc[event] = utils.logadd(p_ai, fc.get(event, float('-inf')))
                ef += (exp(p_ai) * hijmn)

    return max_bt[:-1], p_st, fc, ef


def reset_fractional_counts():
    global fractional_counts, cache_normalizing_decision, number_of_events
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}
    number_of_events = 0


def get_likelihood(theta):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc, fractional_counts, feature_index, num_toks, emp_feat, diagonal_tension
    reset_fractional_counts()
    data_likelihood = 0.0
    emp_feat = 0.0
    batch = range(0, len(trellis))

    for idx in batch:
        max_bt, S, fractional_counts, emp_feat = get_model1_forward(theta, idx, fractional_counts, emp_feat)
        # print 'p(t|s) for', idx, ':', S  # , max_bt
        data_likelihood += S

    reg = np.sum(theta ** 2)
    ll = data_likelihood - (rc * reg)

    e1 = get_decision_given_context(theta, E_TYPE, decision='.', context=NULL)
    e2 = get_decision_given_context(theta, E_TYPE, decision='.', context='.')
    # e3 = get_decision_given_context(theta, E_TYPE, decision='pero', context='but')
    print 'log likelihood:', ll, 'p(.|NULL)', e1, 'p(.|.)', e2, 'lf', diagonal_tension
    print 'emp_feat_norm', emp_feat / num_toks

    return -ll


def get_likelihood_with_expected_counts(theta):
    global fractional_counts, cache_normalizing_decision
    sum_likelihood = 0.0
    cache_normalizing_decision = {}
    for event in fractional_counts:
        (t, d, c) = event
        if t == E_TYPE:
            A_dct = exp(fractional_counts[event])
            a_dct = get_decision_given_context(theta=theta, type=t, decision=d, context=c)
            sum_likelihood += A_dct * a_dct
    reg = np.sum(theta ** 2)

    print '\tec log likelihood:', sum_likelihood, reg
    sum_likelihood -= (rc * reg)
    return -sum_likelihood


def get_gradient(theta):
    global fractional_counts, feature_index, event_grad, rc
    assert len(theta) == len(feature_index)
    event_grad = {}
    for event_j in fractional_counts:
        (t, dj, cj) = event_j
        if t == E_TYPE:
            f_val, f = FE.get_wa_features_fired(type=t, context=cj, decision=dj)[0]
            a_dp_ct = exp(get_decision_given_context(theta, decision=dj, context=cj, type=t)) * f_val
            sum_feature_j = 0.0
            norm_events = [(t, dp, cj) for dp in normalizing_decision_map[t, cj]]
            for event_i in norm_events:
                A_dct = exp(fractional_counts.get(event_i, 0.0))
                # print isinstance(event_j, tuple), isinstance(event_i, tuple)
                if event_i == event_j:
                    (ti, di, ci) = event_i
                    fj, f = FE.get_wa_features_fired(type=ti, context=ci, decision=di)[0]
                else:
                    fj = 0.0
                sum_feature_j += A_dct * (fj - a_dp_ct)
            event_grad[event_j] = sum_feature_j  # - abs(theta[event_j])  # this is the regularizing term

    # grad = np.zeros_like(theta)
    grad = -2 * rc * theta  # l2 regularization with lambda 0.5
    for e in event_grad:
        feats = events_to_features[e]
        for f in feats:
            grad[feature_index[f]] += event_grad[e]

    return -grad


def get_diagonal_tension(mnd, dt):
    global trellis, data_likelihood, rc, fractional_counts, feature_index, num_toks, emp_feat
    emp_feat_norm = emp_feat / num_toks
    mod_feat = 0
    for m, n in mnd:
        for j in xrange(1, m):
            mod_feat += mnd[m, n] * compute_d_log_z(j, m, n, dt)

    mod_feat_norm = mod_feat / num_toks
    print emp_feat_norm, mod_feat_norm, dt
    dt += (emp_feat_norm - mod_feat_norm) * 20
    if dt < 0.1:
        dt = 0.1
    elif dt > 14:
        dt = 14
    return dt


def gradient_check_em():
    global EPS, feature_index
    init_theta = initialize_theta(None)
    f_approx = {}
    for f in feature_index:
        theta_plus = copy.deepcopy(init_theta)
        theta_minus = copy.deepcopy(init_theta)
        theta_plus[feature_index[f]] = init_theta[feature_index[f]] + EPS
        get_likelihood(theta_plus)  # updates fractional counts
        val_plus = get_likelihood_with_expected_counts(theta_plus)
        theta_minus[feature_index[f]] = init_theta[feature_index[f]] - EPS
        get_likelihood(theta_minus)  # updates fractional counts
        val_minus = get_likelihood_with_expected_counts(theta_minus)
        f_approx[f] = (val_plus - val_minus) / (2 * EPS)

    my_grad = get_gradient(init_theta)
    diff = []
    for k in sorted(f_approx):
        diff.append(abs(my_grad[feature_index[k]] - f_approx[k]))
        print str(round(my_grad[feature_index[k]] - f_approx[k], 3)).center(10), str(
            round(my_grad[feature_index[k]], 5)).center(10), \
            str(round(f_approx[k], 5)).center(10), k
    f_approx = sorted([(feature_index[k], v) for k, v in f_approx.items()])
    f_approx = np.array([v for k, v in f_approx])

    print 'component difference:', round(sum(diff), 3), \
        'cosine similarity:', utils.cosine_sim(f_approx, my_grad), \
        ' sign difference', utils.sign_difference(f_approx, my_grad)


def gradient_check_lbfgs():
    global EPS, feature_index
    init_theta = initialize_theta(None, feature_index)
    chk_grad = utils.gradient_checking(init_theta, EPS, get_likelihood)
    my_grad = get_gradient(init_theta)
    diff = []
    for f in sorted(feature_index):  # xrange(len(chk_grad)):
        k = feature_index[f]
        diff.append(abs(my_grad[k] - chk_grad[k]))
        print str(round(my_grad[k] - chk_grad[k], 5)).center(10), str(
            round(my_grad[k], 5)).center(10), \
            str(round(chk_grad[k], 5)).center(10), f

    print 'component difference:', round(sum(diff), 3), \
        'cosine similarity:', utils.cosine_sim(chk_grad, my_grad), \
        ' sign difference', utils.sign_difference(chk_grad, my_grad)


def batch_sgd_accumilate(obs_id):
    print obs_id


def write_logs(theta, current_iter):
    global max_beam_width, max_jump_width, trellis, feature_index, fractional_counts
    feature_val_typ = 'bin' if options.feature_values is None else 'real'
    name_prefix = '.'.join(
        ['sp', options.algorithm, str(rc), 'fast-model1', feature_val_typ])
    if current_iter is not None:
        name_prefix += '.' + str(current_iter)
    write_weights(theta, name_prefix + '.' + options.output_weights, feature_index)
    write_probs(theta, name_prefix + '.' + options.output_probs, fractional_counts, get_decision_given_context)

    if options.source_test is not None and options.target_test is not None:
        source_test = [s.strip().split() for s in open(options.source_test, 'r').readlines()]
        target_test = [t.strip().split() for t in open(options.target_test, 'r').readlines()]
        trellis = populate_trellis(source_test, target_test, max_jump_width, max_beam_width)

    write_alignments(theta, name_prefix + '.' + options.output_alignments, trellis, get_model1_forward)
    write_alignments_col(theta, name_prefix + '.' + options.output_alignments, trellis, get_model1_forward)
    write_alignments_col_tok(theta, name_prefix + '.' + options.output_alignments, trellis, source_test, target_test,
                             get_model1_forward)


if __name__ == "__main__":
    trellis = []

    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/dev.small.es")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/dev.small.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/dev.small.es")
    opt.add_option("--ts", dest="source_test", default="experiment/data/dev.small.en")

    opt.add_option("--iw", dest="input_weights", default=None)
    opt.add_option("--fv", dest="feature_values", default=None)
    opt.add_option("--ow", dest="output_weights", default="theta", help="extention of trained weights file")
    opt.add_option("--oa", dest="output_alignments", default="alignments", help="extension of alignments files")
    opt.add_option("--op", dest="output_probs", default="probs", help="extension of probabilities")
    opt.add_option("-g", dest="test_gradient", default="false")
    opt.add_option("-r", dest="regularization_coeff", default="0.0")
    opt.add_option("-a", dest="algorithm", default="EM",
                   help="use 'EM' 'LBFGS' 'SGD'")

    (options, _) = opt.parse_args()
    rc = float(options.regularization_coeff)

    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines()]
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines()]
    num_toks = len(open(options.target_corpus, 'r').read().split())
    diagonal_tension = 4.0
    trellis = populate_trellis(source, target, max_jump_width, max_beam_width)
    events_to_features, features_to_events, feature_index, feature_counts, event_index, event_to_event_index, event_counts, normalizing_decision_map, du = populate_features(
        trellis, source, target, IBM_MODEL_1)
    FE.load_feature_values(options.feature_values)
    snippet = "#" + str(opt.values) + "\n"

    if options.algorithm == "LBFGS":
        if options.test_gradient.lower() == "true":
            gradient_check_lbfgs()
        else:
            print 'skipping gradient check...'
            init_theta = initialize_theta(options.input_weights, feature_index)
            t1 = minimize(get_likelihood, init_theta, method='L-BFGS-B', jac=get_gradient, tol=1e-3,
                          options={'maxiter': 20})
            theta = t1.x

    elif options.algorithm == "EM":
        if options.test_gradient.lower() == "true":
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            theta = initialize_theta(options.input_weights, feature_index)
            new_e = get_likelihood(theta)
            exp_new_e = get_likelihood_with_expected_counts(theta)
            print 'new_e', new_e
            print 'exp_new_e', exp_new_e

            mn_list = [(len(trg) - 2, len(src) - 1) for trg, src in zip(target, source)]
            mn_dict = {}
            for mn in mn_list:
                mn_dict[mn] = mn_dict.get(mn, 0) + 1

            old_e = float('-inf')
            converged = False
            iterations = 0
            num_iterations = 5
            while not converged and iterations < num_iterations:
                t1 = minimize(get_likelihood_with_expected_counts, theta, method='L-BFGS-B', jac=get_gradient, tol=1e-3,
                              options={'maxfun': 5})
                theta = t1.x

                if 0 < iterations < num_iterations - 1:
                    for rep in xrange(8):
                        diagonal_tension = get_diagonal_tension(mn_dict, diagonal_tension)

                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 1) == 0.0
                old_e = new_e
                iterations += 1

    if options.test_gradient.lower() == "true":
        pass
    else:
        write_logs(theta, current_iter=None)
