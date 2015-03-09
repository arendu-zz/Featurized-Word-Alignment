__author__ = 'arenduchintala'

from optparse import OptionParser
from math import exp, log

from scipy.optimize import minimize
import numpy as np
import FeatureEng as FE
import utils
import copy
import pdb
from pprint import pprint
import traceback
import random
import sharedmem
import multiprocessing
from multiprocessing import Pool

from const import NULL, BOUNDARY_START, BOUNDARY_END, HYBRID_MODEL_1, E_TYPE, T_TYPE, EPS
from common import populate_trellis, populate_features, write_alignments, write_alignments_col, \
    write_alignments_col_tok, write_probs, write_weights, initialize_theta


global cache_normalizing_decision, features_to_events, events_to_features, normalizing_decision_map
global trellis, max_jump_width, number_of_events, EPS, snippet, max_beam_width, rc
global source, target, data_likelihood, event_grad, feature_index, event_index
global events_per_trellis, event_to_event_index, has_pos, event_counts, du, itercount, N
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


def populate_events_per_trellis():
    global event_index, trellis, events_per_trellis
    events_per_trellis = []
    for obs_id, obs in enumerate(trellis):
        events_observed = []
        obs = trellis[obs_id]
        src = source[obs_id]
        tar = target[obs_id]
        for k in range(1, len(obs)):  # the words are numbered from 1 to n, 0 is special start character
            for v in obs[k]:  # [1]:
                for u in obs[k - 1]:  # [1]:
                    tk, aj = v
                    tk_1, aj_1 = u
                    t_tok = tar[tk]
                    s_tok = src[aj] if aj is not NULL else NULL

                    ei = event_to_event_index[(E_TYPE, t_tok, s_tok)]
                    events_observed.append(ei)
        events_per_trellis.append(list(set(events_observed)))


def get_decision_given_context(theta, type, decision, context):
    global cache_normalizing_decision, feature_index, source_to_target_firing, model1_probs, ets
    m1_event_prob = model1_probs.get((decision, context), 0.0)
    fired_features = FE.get_wa_features_fired(type=type, decision=decision, context=context, hybrid=True)
    theta_dot_features = sum([theta[feature_index[f]] * f_wt for f_wt, f in fired_features])
    numerator = m1_event_prob * exp(theta_dot_features)
    if (type, context) in cache_normalizing_decision:
        denom = cache_normalizing_decision[type, context]
    else:
        denom = ets[context]
        target_firings = source_to_target_firing.get(context, set([]))
        for tf in target_firings:
            m1_tf_event_prob = model1_probs.get((tf, context), 0.0)
            tf_fired_features = FE.get_wa_features_fired(type=type, decision=tf, context=context, hybrid=True)
            tf_theta_dot_features = sum([theta[feature_index[f]] * f_wt for f_wt, f in tf_fired_features])
            denom += m1_tf_event_prob * exp(tf_theta_dot_features)
        cache_normalizing_decision[type, context] = denom
    try:
        log_prob = log(numerator) - log(denom)
    except ValueError:
        print numerator, denom, decision, context, m1_event_prob, theta_dot_features
        raise BaseException
    return log_prob


def get_best_seq(theta, obs_id):
    global source, target, trellis
    obs = trellis[obs_id]
    max_bt = [-1] * len(obs)
    p_st = 0.0
    for t_idx in obs:
        t_tok = target[obs_id][t_idx]
        sum_e = float('-inf')
        max_e = float('-inf')
        max_s_idx = None
        sum_sj = float('-inf')
        for _, s_idx in obs[t_idx]:
            s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            sum_e = utils.logadd(sum_e, e)
            q = log(1.0 / len(obs[t_idx]))
            sum_sj = utils.logadd(sum_sj, e + q)
            if e > max_e:
                max_e = e
                max_s_idx = s_idx
        max_bt[t_idx] = (t_idx, max_s_idx)
        p_st += sum_sj

    return max_bt[:-1], p_st


def get_model1_forward(theta, obs_id, fc):
    global source, target, trellis
    obs = trellis[obs_id]
    max_bt = [-1] * len(obs)
    p_st = 0.0
    for t_idx in obs:
        t_tok = target[obs_id][t_idx]
        sum_e = float('-inf')
        max_e = float('-inf')
        max_s_idx = None
        sum_sj = float('-inf')
        for _, s_idx in obs[t_idx]:
            s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
            e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
            sum_e = utils.logadd(sum_e, e)
            q = log(1.0 / len(obs[t_idx]))
            sum_sj = utils.logadd(sum_sj, e + q)
            if e > max_e:
                max_e = e
                max_s_idx = s_idx
        max_bt[t_idx] = (t_idx, max_s_idx)
        p_st += sum_sj

        # update fractional counts
        if fc is not None:
            for _, s_idx in obs[t_idx]:
                s_tok = source[obs_id][s_idx] if s_idx is not NULL else NULL
                e = get_decision_given_context(theta, E_TYPE, decision=t_tok, context=s_tok)
                delta = e - sum_e
                event = (E_TYPE, t_tok, s_tok)
                fc[event] = utils.logadd(delta, fc.get(event, float('-inf')))

    return max_bt[:-1], p_st, fc


def reset_fractional_counts():
    global fractional_counts, cache_normalizing_decision, number_of_events
    fractional_counts = {}  # dict((k, float('-inf')) for k in conditional_arc_index)
    cache_normalizing_decision = {}
    number_of_events = 0


def get_likelihood(theta):
    assert isinstance(theta, np.ndarray)
    assert len(theta) == len(feature_index)
    global trellis, data_likelihood, rc, itercount, N
    reset_fractional_counts()
    data_likelihood = 0.0
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)  # uses all available CPUs
    full = range(0, len(trellis))
    batches = np.array_split(full, cpu_count)
    for batch in batches:
        pool.apply_async(batch_likelihood, args=(theta, batch), callback=batch_accumilate_likelihood)
    pool.close()
    pool.join()
    reg = np.sum(theta ** 2)
    ll = (data_likelihood - (rc * reg))
    e1 = get_decision_given_context(theta, E_TYPE, decision='.', context=NULL)
    e2 = get_decision_given_context(theta, E_TYPE, decision='.', context='.')
    e3 = get_decision_given_context(theta, E_TYPE, decision='en', context='in')
    e4 = get_decision_given_context(theta, E_TYPE, decision='en', context='and')
    print itercount, 'log likelihood:', ll, 'p(.|NULL)', e1, 'p(.|.)', e2, 'p(en|in)', e3, 'p(en|and)', e4
    itercount += 1

    return -ll


def batch_likelihood(theta, batch):
    dl = 0.0
    batch_fc = {}
    for idx in batch:
        max_bt, S, batch_fc = get_model1_forward(theta, idx, batch_fc)
        dl += S
    return dl, batch_fc


def batch_accumilate_likelihood(result):
    global data_likelihood, fractional_counts
    data_likelihood += result[0]
    fc = result[1]
    for k in fc:
        fractional_counts[k] = utils.logadd(fc[k], fractional_counts.get(k, float('-inf')))


def get_likelihood_with_expected_counts(theta):
    global fractional_counts
    sum_likelihood = 0.0
    for event in fractional_counts:
        (t, d, c) = event
        A_dct = exp(fractional_counts[event])
        a_dct = get_decision_given_context(theta=theta, type=t, decision=d, context=c)
        sum_likelihood += A_dct * a_dct
    reg = np.sum(theta ** 2)
    sum_likelihood -= (rc * reg)

    print '\tec log likelihood:', sum_likelihood
    return -sum_likelihood


def batch_gradient(theta, batch_fractional_counts):
    global event_index
    eg = {}
    for event_j in batch_fractional_counts:
        try:
            event_j = tuple(event_j)
            (t, dj, cj) = event_j
            f_val, f = FE.get_wa_features_fired(type=t, context=cj, decision=dj)[
                0]  # TODO: this only works in basic feat
            a_dp_ct = exp(get_decision_given_context(theta, decision=dj, context=cj, type=t)) * f_val
            sum_feature_j = 0.0
            norm_events = [(t, dp, cj) for dp in normalizing_decision_map[t, cj]]

            for event_i in norm_events:
                A_dct = exp(fractional_counts.get(event_i, 0.0))
                if event_i == event_j:
                    (ti, di, ci) = event_i
                    fj, f = FE.get_wa_features_fired(type=ti, context=ci, decision=di)[
                        0]  # TODO: this only works in basic
                else:
                    fj = 0.0
                sum_feature_j += A_dct * (fj - a_dp_ct)
            eg[(t, dj, cj)] = sum_feature_j
        except Exception, er:
            print traceback.format_exc()

    return eg


def batch_accumilate_gradient(result):
    global event_grad
    for event_j in result:
        if event_j in event_grad:
            raise 'should this happen?'
        else:
            event_grad[event_j] = result[event_j]


def get_gradient(theta):
    global fractional_counts, event_index, feature_index, event_grad, rc, N
    assert len(theta) == len(feature_index)
    event_grad = {}
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)  # uses all available CPUs
    batches_fractional_counts = np.array_split(range(len(event_index)), cpu_count)
    events_to_split = events_to_features.keys()
    batches_events_to_features = np.array_split(events_to_split, cpu_count)
    # for batch_of_fc in batches_fractional_counts:
    for batch_of_fc in batches_events_to_features:
        pool.apply_async(batch_gradient, args=(theta, batch_of_fc), callback=batch_accumilate_gradient)
    pool.close()
    pool.join()
    # grad = np.zeros_like(theta)
    grad = -2 * rc * theta  # l2 regularization with lambda 0.5
    for e in event_grad:
        feats = events_to_features.get(e, [])
        for f in feats:
            grad[feature_index[f]] += event_grad[e]

    # for s in seen_index:
    # grad[s] += -theta[s]  # l2 regularization with lambda 0.5
    assert len(grad) == len(feature_index)
    return -grad


def batch_likelihood_with_expected_counts(theta, batch):
    global event_index
    batch_sum_likelihood = 0.0
    for idx in batch:
        event = event_index[idx]
        (t, d, c) = event_index[idx]
        A_dct = exp(fractional_counts[event])
        a_dct = get_decision_given_context(theta=theta, type=t, decision=d, context=c)
        batch_sum_likelihood += A_dct * a_dct
    return batch_sum_likelihood


def batch_accumilate_likelihood_with_expected_counts(results):
    global data_likelihood
    data_likelihood += results


def get_likelihood_with_expected_counts(theta, display=True):
    global fractional_counts, data_likelihood, event_index
    data_likelihood = 0.0
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)  # uses all available CPUs
    batches_fractional_counts = np.array_split(range(len(event_index)), cpu_count)
    for batch_of_fc in batches_fractional_counts:
        pool.apply_async(batch_likelihood_with_expected_counts, args=(theta, batch_of_fc),
                         callback=batch_accumilate_likelihood_with_expected_counts)
    pool.close()
    pool.join()

    reg = np.sum(theta ** 2)
    data_likelihood -= (rc * reg)
    if display:
        print '\tec:', data_likelihood
    return -data_likelihood


def batch_sgd(obs_ids, sgd_theta, sum_square_grad):
    # print _, obs_id
    for obs_id in obs_ids:
        eo = events_per_trellis[obs_id]
        eg = batch_gradient(sgd_theta, eo)
        gdu = np.array([float('inf')] * len(sgd_theta))
        grad = np.zeros(np.shape(sgd_theta))  # -2 * rc * theta  # l2 regularization with lambda 0.5
        for e in eg:
            feats = events_to_features[e]
            for f in feats:
                grad[feature_index[f]] += eg[e]
                gdu[feature_index[f]] = du[feature_index[f]]

        grad_du = -2 * rc * np.divide(sgd_theta, gdu)

        grad += grad_du
        sum_square_grad += (grad ** 2)
        eta_t = eta0 / np.sqrt(I + sum_square_grad)
        sgd_theta += np.multiply(eta_t, grad)
    return obs_ids


def batch_sgd_accumilate(obs_ids):
    # print obs_id
    pass


def pre_compute_ets(m1_probs, src_to_tar, t_types, s_types):
    _ets = {}
    c = 0
    for s in s_types:
        sum_s = 0.0
        t1 = t_types - src_to_tar.get(s, set([]))  # TODO: can speed up by only iterating over co-occuring t_types
        for t in t1:
            if t is BOUNDARY_START and s is BOUNDARY_START:
                pass
            elif t is BOUNDARY_END and s is BOUNDARY_END:
                pass
            elif (t, s) in m1_probs:
                c += 1
                sum_s += m1_probs[t, s]
            else:
                # t,s do not co-occur so we are adding 0.0 to sum_s (or -inf to log sum_s)
                pass
        _ets[s] = sum_s
    _ets[BOUNDARY_START] = 1.0
    _ets[BOUNDARY_END] = 1.0
    return _ets


def get_source_to_target_firing(eve_to_feat):
    src_to_tar = {}
    for e in eve_to_feat:
        event_type, decision, context = e
        tar_l = src_to_tar.get(context, set([]))
        tar_l.add(decision)
        src_to_tar[context] = tar_l
    return src_to_tar


def load_model1_probs(path_model1_probs):
    m1_probs = {}
    for line in open(path_model1_probs, 'r').readlines():
        prob_type, tar, src, prob = line.strip().split()
        prob = float(prob)
        m1_probs[tar, src] = prob
    m1_probs[BOUNDARY_END, BOUNDARY_END] = 1.0
    m1_probs[BOUNDARY_START, BOUNDARY_START] = 1.0
    return m1_probs


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
    init_theta = initialize_theta(None)
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


def write_logs(theta, current_iter):
    global max_beam_width, max_jump_width, trellis, feature_index, fractional_counts
    feature_val_typ = 'bin' if options.feature_values is None else 'real'
    name_prefix = '.'.join(
        ['mp', options.algorithm, str(rc), 'hybrid-model1', feature_val_typ])
    if current_iter is not None:
        name_prefix += '.' + str(current_iter)
    write_weights(theta, name_prefix + '.' + options.output_weights, feature_index)
    write_probs(theta, name_prefix + '.' + options.output_probs, fractional_counts, get_decision_given_context)

    if options.source_test is not None and options.target_test is not None:
        source_test = [s.strip().split() for s in open(options.source_test, 'r').readlines()]
        target_test = [t.strip().split() for t in open(options.target_test, 'r').readlines()]
        trellis = populate_trellis(source_test, target_test, max_jump_width, max_beam_width)

    write_alignments(theta, name_prefix + '.' + options.output_alignments, trellis, get_best_seq)
    write_alignments_col(theta, name_prefix + '.' + options.output_alignments, trellis, get_best_seq)
    write_alignments_col_tok(theta, name_prefix + '.' + options.output_alignments, trellis, source_test, target_test,
                             get_best_seq)


if __name__ == "__main__":
    trellis = []

    opt = OptionParser()
    opt.add_option("-t", dest="target_corpus", default="experiment/data/dev.es")
    opt.add_option("-s", dest="source_corpus", default="experiment/data/dev.en")
    opt.add_option("--tt", dest="target_test", default="experiment/data/dev.es")
    opt.add_option("--ts", dest="source_test", default="experiment/data/dev.en")
    opt.add_option("--df", dest="dict_features", default=None)
    opt.add_option("--m1", dest="model1_probs", default="experiment/data/model1.probs")
    opt.add_option("--iw", dest="input_weights", default=None)
    opt.add_option("--fv", dest="feature_values", default=None)
    opt.add_option("--ow", dest="output_weights", default="theta", help="extention of trained weights file")
    opt.add_option("--oa", dest="output_alignments", default="alignments", help="extension of alignments files")
    opt.add_option("--op", dest="output_probs", default="probs", help="extension of probabilities")
    opt.add_option("-g", dest="test_gradient", action="store_true", default=False)
    opt.add_option("-r", dest="regularization_coeff", default="0.0")
    opt.add_option("-a", dest="algorithm", default="LBFGS",
                   help="use 'EM' 'LBFGS' 'SGD'")

    (options, _) = opt.parse_args()
    rc = float(options.regularization_coeff)
    source = [s.strip().split() for s in open(options.source_corpus, 'r').readlines()]
    target = [s.strip().split() for s in open(options.target_corpus, 'r').readlines()]
    target_types = set(open(options.target_corpus, 'r').read().split())
    source_types = set(open(options.source_corpus, 'r').read().split())
    source_types.add(NULL)
    trellis = populate_trellis(source, target, max_jump_width, max_beam_width)
    FE.load_feature_values(options.feature_values)
    FE.load_dictionary_features(options.dict_features)
    model1_probs = load_model1_probs(options.model1_probs)
    events_to_features, features_to_events, feature_index, feature_counts, event_index, event_to_event_index, event_counts, normalizing_decision_map, du = populate_features(
        trellis, source, target, HYBRID_MODEL_1)
    source_to_target_firing = get_source_to_target_firing(events_to_features)
    ets = pre_compute_ets(model1_probs, source_to_target_firing, target_types, source_types)
    print len(feature_index), 'features used...'
    snippet = "#" + str(opt.values) + "\n"
    if options.algorithm == "LBFGS":
        if options.test_gradient:
            init_theta = initialize_theta(options.input_weights, feature_index, rand=True)
            gradient_check_lbfgs()
        else:
            print 'skipping gradient check...'
            theta = initialize_theta(options.input_weights, feature_index)
            t1 = minimize(get_likelihood, theta, method='L-BFGS-B', jac=get_gradient, tol=1e-3,
                          options={'maxiter': 20})

            theta = t1.x

    elif options.algorithm == "EM":
        if options.test_gradient:
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            theta = initialize_theta(options.input_weights, feature_index)
            new_e = get_likelihood(theta)
            exp_new_e = get_likelihood_with_expected_counts(theta)
            old_e = float('-inf')
            converged = False
            iterations = 0
            while not converged and iterations < 5:
                t1 = minimize(get_likelihood_with_expected_counts, theta, method='L-BFGS-B', jac=get_gradient, tol=1e-2,
                              options={'maxiter': 5})
                theta = t1.x
                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 1) == 0.0
                old_e = new_e
                iterations += 1
    elif options.algorithm == "EM-SGD":
        if options.test_gradient.lower() == "true":
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            print 'populating events per trellis...'
            populate_events_per_trellis()
            print 'done...'
            theta = initialize_theta(options.input_weights, feature_index)
            new_e = get_likelihood(theta)
            exp_new_e = get_likelihood_with_expected_counts(theta)
            old_e = float('-inf')
            converged = False
            iterations = 0
            ids = range(len(trellis))

            while not converged and iterations < 10:
                eta0 = 1.0
                sum_square_grad = np.zeros(np.shape(theta))
                I = 1.0
                for _ in range(2):
                    random.shuffle(ids)
                    for obs_id in ids:
                        print _, obs_id
                        event_observed = events_per_trellis[obs_id]
                        eg = batch_gradient(theta, event_observed)
                        grad = -2 * rc * theta  # l2 regularization with lambda 0.5
                        for e in eg:
                            feats = events_to_features[e]
                            for f in feats:
                                grad[feature_index[f]] += eg[e]
                        sum_square_grad += (grad ** 2)
                        eta_t = eta0 / np.sqrt(I + sum_square_grad)
                        theta += np.multiply(eta_t, grad)

                new_e = get_likelihood(theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 2) == 0.0
                old_e = new_e
                iterations += 1

    elif options.algorithm == "EM-SGD-PARALLEL":
        if options.test_gradient:
            gradient_check_em()
        else:
            print 'skipping gradient check...'
            print 'populating events per trellis...'
            populate_events_per_trellis()
            print 'done...'

            init_theta = initialize_theta(options.input_weights, feature_index)
            shared_sgd_theta = sharedmem.zeros(np.shape(init_theta))
            shared_sgd_theta += init_theta
            new_e = get_likelihood(shared_sgd_theta)
            old_e = float('-inf')
            converged = False
            iterations = 0
            ids = range(len(trellis))
            while not converged and iterations < 5:
                eta0 = 1.0
                shared_sum_squared_grad = sharedmem.zeros(np.shape(shared_sgd_theta))
                I = 1.0
                for _ in range(2):
                    random.shuffle(ids)

                    cpu_count = multiprocessing.cpu_count()
                    pool = Pool(processes=cpu_count)
                    batches = np.array_split(ids, cpu_count)
                    for obs_ids in batches:
                        pool.apply_async(batch_sgd, args=(obs_ids, shared_sgd_theta, shared_sum_squared_grad),
                                         callback=batch_sgd_accumilate)
                    pool.close()
                    pool.join()
                    """
                    for obs_id in ids:
                        batch_sgd(obs_id)
                    """
                new_e = get_likelihood(shared_sgd_theta)  # this will also update expected counts
                converged = round(abs(old_e - new_e), 2) == 0.0
                old_e = new_e
                iterations += 1
            theta = shared_sgd_theta
    else:
        print 'wrong option for algorithm...'
        exit()

    if options.test_gradient:
        pass
    else:
        write_logs(theta, current_iter=None)
