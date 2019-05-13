# This is the main script for the simulated experiment (Section 5.2.1)
# Runs one LOOCV fold: do_run(xpred, num_datapoints, num_feedbacks, acquisition, seed, save_folder, stan_folder)

import pystan

import numpy as np
import pylab as plt
from scipy.stats import norm

import stan_utility

import pickle

import sys
##Help functions:

def rbf(x,i):
    c = np.array([-3., 0., 3.])
    ret = np.exp(-(x - c[i])**2)
    return(ret)

def compute_theta(beta, x, a):
    num_rbf = 3
    w = np.exp(np.sum([beta[i]*rbf(x,i) for i in np.arange(0,num_rbf)],0) + np.sum([beta[num_rbf + i]*rbf(x,i)*a for i in np.arange(0,num_rbf)],0))
    return w/(1+w)

def generate_data(N=200, seed=1234):
    # x     covariates
    # a     actions
    # y     outcomes
    np.random.seed(seed)
    mu = np.array([1.5, -1.5]) # p(x|a)~N(mu[a],s)
    s = 1.0
    x = np.random.random(N)*9-4.5 #np.random.normal(0,s,N)
    a = np.array(x > -1.5, dtype=int) # a=1 for each x > -1.5
    #pa = np.array([0.1, 0.9])
    #a = np.array([np.random.binomial(1, pa[int(x[n]>-1.5)]) for n in range(len(x))]) # create imbalance: p(a=1|x>-1.5) = 0.9, p(a=1|x<-1.5) = 0.1
    e1 = 1 #base effect
    e2 = 0.5 # modifier
    beta = np.array([e1-e2, e1+e2, e1+e2,
                     2*e2, -2*e2, -2*(e1+e2)])
    theta = compute_theta(beta, x, a)
    y = np.random.binomial(1, theta)
    return (x,a,y), beta

def predict_with_model(beta, x, a):
    return np.random.binomial(1, compute_theta(beta, x, a))

# Cost of outcome
def cost(y):
    c1 = 1 # cost of an adverse event
    return c1 if y==1 else 0.

def risk(beta, x, a):
    return compute_theta(beta, x, a)*cost(1)

def regret(beta, x, a, action_space_size=2):
    risks = [risk(beta, x, ai) for ai in range(action_space_size)] 
    return  risks[a] - min(risks)

def kernel(x1,x2):
    l = 1.0 # kernel length-scale
    ret = norm.pdf((x1-x2)/l)
    return ret

def compute_imbalance(x,a):
    x_1 = x[a==1]
    x_0 = x[a==0]
    e1 = np.mean([kernel(x_1[n],x_1[np.arange(len(x_1))!=n]) for n in range(len(x_1))])
    e2 = np.mean([kernel(x_1[n],x_0) for n in range(len(x_1))])
    e3 = np.mean([kernel(x_0[n],x_0[np.arange(len(x_0))!=n]) for n in range(len(x_0))])
    return e1 -2*e2 + e3

def error_rate(samples):
    dec = decide(samples)
    theta = samples['thetapred']
    ret = 1
    if dec == 0: # theta_0 < theta_1
        # p(theta_0 > theta_1)
        ret = np.mean([theta[i,0] > theta[i,1] for i in range(theta.shape[0])])
    if dec == 1: # theta_1 < theta_0
        # p(theta_0 < theta_1)
        ret = np.mean([theta[i,0] < theta[i,1] for i in range(theta.shape[0])])
    if ret > 0.5:
        print("Warning: over 0.5 error rate")
    return ret

### Criteria for active learning ###

## Functions below ##

# Map predicted outcomes (distributions) to decisions
def decide(samples):
    theta = samples['thetapred']
    return 0 if np.mean(theta[:,0]) < np.mean(theta[:,1]) else 1

def error_rate_v2(samples):
    dec = decide(samples)
    ypred = samples['ypred']
    ret = 1
    if dec == 0: # theta_0 < theta_1
        # is y[0]==1 and y[1]==0, i.e. y[0]>y[1] ?
        ret = np.mean([ypred[i,0] > ypred[i,1] for i in range(ypred.shape[0])])
    if dec == 1: # theta_1 < theta_0
        # is y[1]>y[0] ?
        ret = np.mean([ypred[i,0] < ypred[i,1] for i in range(ypred.shape[0])])
    if ret > 0.5:
        print("Warning: over 0.5 error rate")
    return ret

def select_query(model, samples, dat, acquisition):
    umax = -np.inf
    x_star, a_star = None, None
    for x in dat['x']:
        #To treat or not to treat, that is the question
        for a in [0,1]:
            u_a = expected_utility(model, samples, dat, x, a, acquisition)
            if u_a > umax:
                umax = u_a
                x_star, a_star = x,a
    return x_star, a_star
    
### Criteria for active learning ###

def expected_utility(model, samples, dat, xstar, astar, acquisition):
    u = 0
    for ystar in np.arange(0,2):
        beta = samples['beta']
        theta = np.mean([predict_with_model(beta[i,:], xstar, astar) for i in range(beta.shape[0])])
        p_ystar = theta**ystar*(1-theta)**(1-ystar)
        u += p_ystar * acquisition(model, samples, dat, xstar, astar, ystar)
    return(u)

## Population level ##
# Uncertainty sampling
def utility_uncertainty_sampling(model, samples, dat, x, a, y):
    n = np.setdiff1d([range(len(dat['x']))], np.nonzero(dat['x'] - x))[0] # find first indice that equals x
    factual_a = dat['a']
    factual_a = factual_a[n]
    if a == factual_a:
        theta = samples['theta']
    else:
        theta = samples['thetainv']
    th_x = np.mean(theta[:,n])
    if th_x == 0 or th_x == 1:
        u = 0
    else:
        u = -(th_x*np.log(th_x) + (1-th_x)*np.log(1-th_x))
    return(u)

# Information gain
def utility_IG(model, samples, dat, xstar, astar, ystar):
    # train model with feedback
    dat_star = append_dat_stan(dat, xstar, astar, ystar)
    fit_star = model.sampling(data=dat_star, seed=194838, chains=4, iter=2000)
    samples_star = fit_star.extract(permuted=True)
    u = 0
    for n, x in enumerate(dat['x']): # sum expected information gain in each training data point
        # factuals
        theta = samples['theta']
        theta = np.mean(theta[:,n])
        theta_next = samples_star['theta']
        theta_next = np.mean(theta_next[:,n])
        u += KL_Bernoulli(theta_next, theta)
        # counterfactuals
        thetainv = samples['thetainv']
        thetainv = np.mean(thetainv[:,n])
        thetainv_next = samples_star['thetainv']
        thetainv_next = np.mean(thetainv_next[:,n])
        u += KL_Bernoulli(thetainv_next, thetainv)
    return(u)

## Individual-level ##
# Targeted information gain 
def utility_targeted_IG(model, samples, dat, xstar, astar, ystar):
    theta = np.mean(samples['ypred'], axis=0)
    dat_star = append_dat_stan(dat, xstar, astar, ystar)
    fit_star = model.sampling(data=dat_star, seed=194838, chains=4, iter=2000)
    samples_star = fit_star.extract(permuted=True)
    theta_next = np.mean(samples_star['ypred'], axis=0)
    u = 0
    for a in [0,1]: # add up information gains in both outcomes
        th = theta[a]
        th_next = theta_next[a]
        u += KL_Bernoulli(th_next, th)
    return(u)

## Decision-making level ##
# Minimize the estimated Type S error
def utility_decerr(model, samples, dat, xstar, astar, ystar):
    dat_star = append_dat_stan(dat, xstar, astar, ystar)
    fit_star = model.sampling(data=dat_star, seed=194838, chains=4, iter=2000)
    samples_star = fit_star.extract(permuted=True)
    u = 1.0 - error_rate(samples_star) # 1-\hat{\gamma}
    return(u)
# Maximize information gain to the estimated Type S error
def utility_decerrig(model, samples, dat, xstar, astar, ystar):
    gamma = 1.0 - error_rate(samples) # estimated Type S error before feedback
    dat_star = append_dat_stan(dat, xstar, astar, ystar)
    fit_star = model.sampling(data=dat_star, seed=194838, chains=4, iter=2000)
    samples_star = fit_star.extract(permuted=True)
    gamma_next = 1.0 - error_rate(samples_star) # estimated Type S error after feedback
    u = KL_Bernoulli(gamma,gamma_next)
    return(u)

def KL_Bernoulli(p, q):
    return p*(np.log(p/q) if p>0. else 0. ) + (1.-p)*(np.log((1.-p)/(1.-q)) if 1.-p > 0. else 0.)

def append_dat_stan(dat_old, x_star, a_star, y_star):
    dat = {'n':dat_old['n']+1,
        'npred': 2,
        'x':np.append(dat_old['x'], x_star),
        'a':np.append(dat_old['a'], a_star),
        'y':np.append(dat_old['y'], y_star),
        'xpred': dat_old['xpred'],
        'apred': dat_old['apred']}
    return dat

def do_run(xpred, num_datapoints, num_feedbacks, acquisition, seed, save_folder, stan_folder):
    filename = save_folder + str(xpred).replace('.', '_')+'-'+str(seed) 
    tr_data, beta_true = generate_data(N=num_datapoints, seed=seed)
    print('xpred: '+ str(xpred))
    # fit model on training data
    dat = {'n':num_datapoints,
        'npred':2,
        'x':tr_data[0],
        'a':tr_data[1],
        'y':tr_data[2],
        'xpred': np.array([xpred, xpred]),
        'apred': np.array([0, 1])}
    # Compute imbalance
    imbalance = [compute_imbalance(dat['x'],dat['a'])]
    # Fit model
    model = stan_utility.compile_model('logit2.stan', model_name='logit-'+str(xpred).replace('.', '_') +'-'+str(seed), model_path=stan_folder)
    fit = model.sampling(data=dat, seed=194838, chains=4, iter=2000)
    samples = fit.extract(permuted=True)
    # Compute the estimated Type S error rate
    typeSerror = [error_rate(samples)]
    # Compute decision and regret in current model
    a0 = decide(samples)
    decisions = [a0]
    regrets = [regret(beta_true, xpred, a0)]
    x_s,a_s,y_s = [], [], []
    for it in range(num_feedbacks):
        print('it: ' + str(it))

        # Elicit one feedback with different criterion
        x_star, a_star = select_query(model, samples, dat, acquisition)

        # Acquire feedback from oracle (we assume true model)
        y_star = predict_with_model(beta_true, x_star, a_star)
        # Fit new model, compute decisions and regret
        x_s += [x_star]
        a_s += [a_star]
        y_s += [y_star]
        dat = append_dat_stan(dat, x_star, a_star, y_star)
        # Compute imbalance
        imbalance += [compute_imbalance(dat['x'],dat['a'])]
        # Re-fit the model
        fit = model.sampling(data=dat, seed=194838, chains=4, iter=2000)
        samples = fit.extract(permuted=True)

        typeSerror += [error_rate(samples)]
        a1 = decide(samples)
        decisions += [a1]
        regrets += [regret(beta_true, xpred, a1)]
    print(filename)
    dat_save  = {'regrets': regrets,
                 'x_s': x_s,
                 'a_s': a_s,
                 'y_s': y_s,
                 'imbalances': imbalance,
                 'typeSerrors': typeSerror,
                 'decisions': decisions}
    pickle.dump(dat_save, open(filename + ".p", "wb" ))

if __name__ == "__main__":
    print(sys.argv) # {} {} {} $SLURM_ARRAY_TASK_ID {} {}'.format(target_x, n_train, n_feedbacks, res_path, acquisition, stanpath)
    xpred = float(sys.argv[1])
    num_datapoints = int(sys.argv[2])
    num_feedbacks = int(sys.argv[3])
    seed = int(sys.argv[4])
    save_folder = sys.argv[5]
    acq = sys.argv[6]
    stan_folder = sys.argv[7]
    if acq == 'decerr':
        acquisition = utility_decerr
    elif acq == 'decerrig':
        acquisition = utility_decerrig
    elif acq == 'targig':
        acquisition = utility_targeted_IG
    elif acq == 'uncertainty':
        acquisition = utility_uncertainty_sampling
    elif acq == 'ig':
        acquisition = utility_IG
    do_run(xpred, num_datapoints, num_feedbacks, acquisition, seed, save_folder, stan_folder)
