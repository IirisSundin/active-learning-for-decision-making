# This is the main script for the comparative experiment (Section 5.2.5)
# Runs one LOOCV fold: do_run(n_train, num_feedbacks, acqusition, task_id, save_folder, stan_folder, n_nearest_neighbours, x_pred=None)

import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib.mlab import PCA

from model import Model
from gpmodel_comparison import GPModelComparison

from util import get_IHDP_data
from scipy.stats import norm

import matplotlib
import pylab as plt

def plot_with_uncertainty(x, y, y1=None, y2=None, color='r', linestyle='-', fill=True, label=''):
    x=x.flatten()
    y=y.flatten()
    plt.plot(x, y, color=color, linestyle=linestyle, label=label)
    if not y1 is None:
        y1=y1.flatten()
        y2=y2.flatten()
        if fill:
            plt.fill_between(x.flatten(), y1, y2, color=color, alpha=0.25, linestyle=linestyle)

def plot_simple(x,a,y,beta,x_pred=None, n_candidates=None, x_comp=None, a_comp=None, savename=None, xpp=None, fpp1=None, fpp2=None):
    if(xpp is not None):
        ft1 = fpp1.mean(axis=1)
        ft11 = np.percentile(fpp1, 17, axis=1)
        ft12 = np.percentile(fpp1, 83, axis=1)
        plot_with_uncertainty(xpp, ft1, y1=ft11, y2=ft12, color='r', linestyle='--', fill=True, label='')
        ft2 = fpp2.mean(axis=1)
        ft21 = np.percentile(fpp2, 17, axis=1)
        ft22 = np.percentile(fpp2, 83, axis=1)
        plot_with_uncertainty(xpp, ft2, y1=ft21, y2=ft22, color='b', linestyle='--', fill=True, label='')
    xt = np.linspace(min(x), max(x), num=100)
    y1 = compute_theta(beta, xt, 0)
    y2 = compute_theta(beta, xt, 1)
    plt.plot(xt,y1, 'r')
    plt.scatter(x[a==0,:], y[a==0], c='r')
    plt.plot(xt,y2, 'b')
    plt.scatter(x[a==1,:], y[a==1], c='b')
    if(x_pred is not None):
        plt.axvline(x=x_pred[0])
    if(n_candidates is not None):
        x_candidates=x[n_candidates,:]
        for i in range(0, x_candidates.shape[0]):
            plt.axvline(x=x_candidates[i,:],alpha=0.5)
    if(x_comp is not None):
        y_comp0 = compute_theta(beta, x_comp[a_comp==0,:], 0)
        y_comp1 = compute_theta(beta, x_comp[a_comp==1,:], 1)
        plt.scatter(x_comp[a_comp==0,:], y_comp0, c='r',marker='X')
        plt.scatter(x_comp[a_comp==1,:], y_comp1, c='b',marker='X')
    plt.savefig(savename,dpi=300)
    plt.close()

def rbf(x,i):
    c = np.array([-3., 0., 3.])
    ret = np.exp(-(x - c[i])**2)
    return(ret)

def compute_theta(beta, x, a):
    num_rbf = 3
    w = np.exp(np.sum([beta[i]*rbf(x,i) for i in np.arange(0,num_rbf)],0) + np.sum([beta[num_rbf + i]*rbf(x,i)*a for i in np.arange(0,num_rbf)],0))
    return w/(1+w)

def get_x(ind):
    x = np.linspace(-4.5,4.5,num=9)[ind]
    return x.reshape((-1,1))

def generate_data(N=200, seed=1234):
    # x     covariates
    # a     actions
    # y     outcomes
    np.random.seed(np.random.randint(0,high=10000000)) #seed)
    x = np.random.random(N)*9-4.5 # training points uniformly distributed across [-4.5,4.5]
    a = np.array(x > -1.5, dtype=int) # a=1 for each x > -1.5
    e1 = 1 #base effect
    e2 = 0.5 # modifier
    beta = np.array([e1-e2, e1+e2, e1+e2,
                     2*e2, -2*e2, -2*(e1+e2)])
    theta = compute_theta(beta, x, a)
    y =  theta + np.random.normal(0,0.05, N)
    return (x,a,y), beta

def predict_with_model(beta, x, a):
    return compute_theta(beta, x, a)

def give_oracle_answer(beta, x):
    theta0 = predict_with_model(beta, x, 0) + np.random.normal(0,0.05, N)
    theta1 = predict_with_model(beta, x, 1) + np.random.normal(0,0.05, N)
    return 0 if theta0 > theta1 else 1

# Cost of outcome 
def cost(y): 
    c1 = 1 # cost of an adverse event
    return c1 if y==1 else 0.

def risk(beta, x, a):
    return compute_theta(beta, x, a)*cost(1)

def regret(beta, x, a, action_space_size=2):
    risks = [risk(beta, x, ai) for ai in range(action_space_size)] 
    return max(risks) - risks[a] #actually the best outcome - chosen outcome

def error_rate(preds):
    mu_a = preds[0][0]
    s_a = preds[0][1]
    mu_b = preds[1][0]
    s_b = preds[1][1]
    alpha = norm.cdf(-np.abs(mu_a-mu_b)/np.sqrt(s_a+s_b))
    return alpha

def do_run(n_train, num_feedbacks, acqusition, task_id, save_folder, stan_folder, n_nearest_neighbours, x_pred=None):
    # Load data
    debug=False
    (x,a,y), true_params = generate_data(n_train, seed=0)
    x= x.reshape(-1,1)
    x_pred = x_pred.reshape((-1,1)) if x_pred is not None else np.array([0.0]).reshape((-1,1))
    oracle_outcomes = np.empty((n_train,2))
    oracle_outcomes[:,0] = predict_with_model(true_params,x, 0)[:,0]
    oracle_outcomes[:,1] = predict_with_model(true_params,x, 1)[:,0]
    
    # Baseline decision, population mean:
    mean_0 = np.mean(y[a==0])
    mean_1 = np.mean(y[a==1])
    a_baseline = 0 if mean_0 > mean_1 else 1
    baseline_regret = regret(true_params, x_pred, a_baseline)
    
    # Test points for expert to make the next decision on
    filename = save_folder + str(1).replace('.', '_')+'-'+str(task_id)
    
    print('n pred: '+ str(1))
    
    # fit model on training data
    model = GPModelComparison(a, x, y, x_pred, stan_folder, task_id)
    
    if(debug): plot_simple(x,a,y,true_params,x_pred=x_pred, xpp=model.xpp1, fpp1=model.fpp1, fpp2=model.fpp2, savename='0.png')
    
    preds0 = model.predict()
    # Compute decision and regret in current model
    a0 = model._decide(preds0)
    decisions = [a0]
    
    po_test = [predict_with_model(true_params,x_pred,0), predict_with_model(true_params,x_pred,0)] 
    regrets = [regret(true_params, x_pred, a0)]
    x_s,a_s,y_s, typeSrates,performance = [], [], [], [], []
    typeSrates = [error_rate(preds0)]
    
    for it in range(num_feedbacks):
        print('it: ' + str(it))

        # Elicit one feedback with the acquisition criterion
        n_star, a_star = model.select_oracle_query(x_pred, acqusition, nn=True, k=n_nearest_neighbours)
        
        # Acquire feedback from oracle (we assume true model)
        y_star = model.answer_oracle_query(n_star, a_star, oracle_outcomes)
        
        # Fit new model, compute decisions and regret
        model.update_me(x[n_star,:].reshape(1,-1), y_star, model.predict(x[n_star,:].reshape(1,-1), y_star), n_star)
        
        preds = model.predict()
        a1 = model._decide(preds)
        decisions += [a1]
        regrets += [regret(true_params, x_pred, a1)]
        x_s += [x[n_star,:]]
        a_s += [a_star]
        y_s += [y_star]
        
        if(debug): plot_simple(x,a,y,true_params,x_pred=x_pred, x_comp=np.array(x_s), a_comp=np.array(a_s), xpp=model.xpp1, fpp1=model.fpp1, fpp2=model.fpp2, savename= str(it+1)+'.png')
        typeSrates += [error_rate(preds)]
        performance += [np.mean(regrets[-1]==0)]
    print(filename)
    dat_save  = {'regrets': regrets,
                 'x_s': x_s,
                 'a_s': a_s,
                 'y_s': y_s,
                 'decisions': decisions,
                 'baseline_regret': baseline_regret,
                 'typeSerrors':typeSrates,
                 'performance':performance}
    pickle.dump(dat_save, open(filename + ".p", "wb" ))

if __name__ == "__main__":
    print(sys.argv) # {} {} {} {} $SLURM_ARRAY_TASK_ID {} {} {}'.format(target_idx, n_train, n_feedbacks, n_nearest_neighbours, res_path, acquisition, stan_folder)
    n_target = int(sys.argv[1])
    size_data = int(sys.argv[2])
    num_feedbacks = int(sys.argv[3])
    num_nearest_neighbours = int(sys.argv[4])
    task_id = int(sys.argv[5])
    np.random.seed(task_id)
    x_pred = get_x(n_target)
    save_folder = sys.argv[6]
    acqusition = sys.argv[7]
    stan_folder = sys.argv[8]
    do_run(size_data, num_feedbacks, acqusition, task_id, save_folder, stan_folder, num_nearest_neighbours, x_pred=x_pred)
