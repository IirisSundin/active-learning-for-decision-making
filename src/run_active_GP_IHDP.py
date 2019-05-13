# This is the main script for the IDHP-data experiment (Section 5.2.2)
# Runs one LOOCV fold for the IHDP data: do_run(n_pred, num_feedbacks, acqusition, task_id, save_folder)

import sys
import pickle
import numpy as np
import pandas as pd
import numpy.linalg as la
from scipy.stats import norm

from model import Model
from gpmodel import GPModel

from util import get_IHDP_data

def regret(potential_outcomes, decision):
    y_max = np.max(potential_outcomes)
    y = potential_outcomes[decision]
    return(y_max - y)

def kernel(x1,x2):
    l = 0.001 # kernel length-scale
    ret = norm.pdf(la.norm(np.repeat(x1.reshape(1,-1),x2.shape[0],axis=0)-x2, axis=1)/l)
    return ret

def compute_imbalance(predictors):
    x_1 = predictors[1]
    x_0 = predictors[0]
    e1 = np.mean([kernel(x_1[n,:],x_1[np.arange(x_1.shape[0])!=n,:]) for n in range(x_1.shape[0])])
    e2 = np.mean([kernel(x_1[n,:],x_0) for n in range(x_1.shape[0])])
    e3 = np.mean([kernel(x_0[n,:],x_0[np.arange(x_0.shape[0])!=n]) for n in range(x_0.shape[0])])
    return e1 -2*e2 + e3

def error_rate(preds):
    mu_tau = preds[1][0] - preds[0][0]
    sd_tau = np.sqrt(preds[1][1] + preds[0][1])
    alpha = norm.cdf(-np.abs(mu_tau)/sd_tau)
    return alpha
    
def do_run(n_pred, num_feedbacks, acqusition, task_id, save_folder):
    # Load data
    pft = 0.1335 #(N=100: 0.1335)
    if acqusition == 'newobs':
        percent_for_train = pft
        # get all other training samples
        dat_train_all, dat_test = get_IHDP_data(id_test=n_pred, seed_data=0, seed_train=n_pred, percent_for_train=0.998)
        N = dat_train_all['X'].shape[0] + 1
        print("N = " + str(N))
        Ntrain = int(np.ceil(N*percent_for_train))
        print("Ntrain = " + str(Ntrain))
        dat_train = {'X':dat_train_all['X'][:Ntrain,:], 'A':dat_train_all['A'][:Ntrain], 'Y':dat_train_all['Y'][:Ntrain], 'PO':dat_train_all['PO'][:Ntrain]}
        dat_train_newobs = {'X':dat_train_all['X'][Ntrain:,:], 'A':dat_train_all['A'][Ntrain:], 'Y':dat_train_all['Y'][Ntrain:]}
    else:
        dat_train, dat_test = get_IHDP_data(id_test=n_pred, seed_data=0, seed_train=n_pred, percent_for_train=pft)
    x_train = dat_train['X']
    y_train = dat_train['Y']
    po_train = dat_train['PO']
    po_test = dat_test['PO']

    # Baseline decision, population mean:
    mean_0 = np.mean(y_train[dat_train['A']==0])
    mean_1 = np.mean(y_train[dat_train['A']==1])
    a_baseline = 0 if mean_0 > mean_1 else 1
    baseline_regret = regret(po_test, a_baseline)
    
    # Test points for expert to make the next decision on
    filename = save_folder + str(n_pred).replace('.', '_')+'-'+str(task_id)
    
    print('n pred: '+ str(n_pred))
    x_pred = dat_test['X'].reshape(1,-1)
    # fit model on training data
    model = GPModel(dat_train['A'], x_train, dat_train['Y'])
    preds0 = model.predict(x_pred)
    # Compute decision and regret in current model
    a0 = model._decide(preds0)
    decisions = [a0]
    regrets = [regret(po_test, a0)]
    x_s,a_s,y_s = [], [], []
    # Compute imbalance
    imbalance = [model.imbalance()]
    # Compute the estimated Type S error rate
    typeSerror = [error_rate(preds0)]
    print(imbalance)
    print(typeSerror)
    for it in range(num_feedbacks):
        print('it: ' + str(it))
        if acqusition == 'newobs':
            x_newobs = dat_train_newobs['X']
            a_newobs = dat_train_newobs['A']
            y_newobs = dat_train_newobs['Y']
            inds = np.ones(x_newobs.shape[0])
            try:
                inds[model.prevQ] = 0
            except: AttributeError
            inds = np.flatnonzero(inds)
            # Sample new observation randomly
            n_star = np.random.choice(inds,1)
            print("Taking new observation at ind")
            print(n_star)
            x_star = x_newobs[n_star,:].reshape(1,-1)
            a_star = int(a_newobs[n_star])
            y_star = y_newobs[n_star]
            # Fit new model, compute decisions and regret
            model.update_me(model.update(a_star, x_star, y_star), n_star)
            preds = model.predict(x_pred)
            a1 = model._decide(preds)
            decisions += [a1]
            regrets += [regret(po_test, a1)]
            x_s += [x_newobs[n_star,:]]
            a_s += [a_star]
            y_s += [y_star]
        else:
            # Elicit one feedback with the acquisition criterion
            n_star, a_star = model.select_oracle_query(x_pred, acqusition, nn=False, k=10)
            print("Feedback on n={}".format(n_star))
    
            # Acquire feedback from oracle
            y_star = model.answer_oracle_query(n_star, a_star, dat_train['PO'])
            # Fit new model, compute decisions and regret
            model.update_me(model.update(a_star, x_train[n_star,:].reshape(1,-1), y_star), n_star)

            # Compute imbalance
            imbalance += [model.imbalance()]
        
            preds = model.predict(x_pred)
            a1 = model._decide(preds)
            decisions += [a1]
            regrets += [regret(po_test, a1)]
            x_s += [x_train[n_star,:]]
            a_s += [a_star]
            y_s += [y_star]
            # Compute the estimated Type S error rate
            typeSerror += [error_rate(preds)]
        
    print(filename)
    print(imbalance)
    print(typeSerror)
    dat_save  = {'regrets': regrets,
                 'x_s': x_s,
                 'a_s': a_s,
                 'y_s': y_s,
                 'decisions': decisions,
                 'baseline_regret': baseline_regret,
                 'imbalances': imbalance,
                 'typeSerrors': typeSerror}
    pickle.dump(dat_save, open(filename + ".p", "wb" ))

if __name__ == "__main__":
    print(sys.argv) # {} {} $SLURM_ARRAY_TASK_ID {} {}'.format(target_idx, num_feedbacks, res_path, acquisition)
    n_target = int(sys.argv[1])
    num_feedbacks = int(sys.argv[2])
    task_id = int(sys.argv[3])
    save_folder = sys.argv[4]
    acqusition = sys.argv[5]
    do_run(n_target, num_feedbacks, acqusition, task_id, save_folder)
