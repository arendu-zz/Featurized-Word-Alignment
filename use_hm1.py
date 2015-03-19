__author__ = 'arenduchintala'

from optparse import OptionParser
import HybridModel1
from cyth.cyth_common import initialize_theta, load_corpus_file
from scipy.optimize import minimize

if __name__ == '__main__':
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
    rc = 0.0
    hm1 = HybridModel1.HybridModel1(options.source_corpus,
                                    options.source_test,
                                    options.target_corpus,
                                    options.target_test,
                                    options.model1_probs, rc,
                                    options.dict_features)
    theta = initialize_theta(None, hm1.findex)
    t1 = minimize(hm1.get_likelihood, theta, method='L-BFGS-B', jac=hm1.get_gradient, tol=1e-3,
                  options={'maxiter': 20})

    theta = t1.x
    hm1.write_logs_hm(theta, options.output_weights, options.output_probs, options.output_alignments)
