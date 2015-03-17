__author__ = 'arenduchintala'

import HybridModel1
from cyth.cyth_common import initialize_theta
from scipy.optimize import minimize

if __name__ == '__main__':
    rc = 0.0
    target_file = "experiment/data/train.es"
    source_file = "experiment/data/train.en"
    m1_probs = "experiment/data/model1.probs"
    df_file = "experiment/data/dictionary_features.es-en"
    hm1 = HybridModel1.HybridModel1(source_file, target_file, m1_probs, rc, df_file)
    theta = initialize_theta(None, hm1.findex)
    t1 = minimize(hm1.get_likelihood, theta, method='L-BFGS-B', jac=hm1.get_gradient, tol=1e-3,
                  options={'maxiter': 20})

    theta = t1.x
    theta = hm1.train(theta, 1e-3, 20)
    hm1.write_logs(theta, 'cyth_out_weights', 'cyth_out_probs', 'cyth_out_align')
