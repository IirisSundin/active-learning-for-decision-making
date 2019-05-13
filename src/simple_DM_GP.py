# This is the main script to produce results in Section 5.1.
# Generates imbalanced data sets and computes correlations between imbalance and
# observed and estimated Type S error rates

import numpy as np
import numpy.linalg as la
from scipy.stats import norm
from scipy import stats

from model import Model
from gpmodel import GPModel

import pylab as plt
import pickle
import sys
import matplotlib
import matplotlib.pylab as plt
matplotlib.rcParams.update({'font.size': 20})

class NonlinearModel():
    def __init__(self, sigma, b0, b1, c0, offset):
        self.sigma = sigma
        self.b0 = b0
        self.b1 = b1
        self.c0 = c0
        self.offset = offset
    def predict(self,x,a):
        # Output y = f(x) + (b0 + b1*x)*a + e, e~N(0,sigma)
        y = self._sigmoid(x) + self._effect(x,a) +  np.random.normal(0,[self.sigma[a[n]==0] for n in range(len(x))])
        return y
    def predict_noiseless(self,x,a):
        y = self._sigmoid(x) + self._effect(x,a)
        return y
    def _sigmoid(self, x):
        ret = self.c0*(1/(1 + np.exp(-x + self.offset))-0.5)
        return ret
    def _effect(self,x,a):
        return (self.b0 + self.b1*x)*a
        

def generate_data(N=200, seed=1234, w=0):
    # x     covariates
    # a     actions
    # y     outcomes
    np.random.seed(seed)
    sigma01 = 0.1
    sigma00 = sigma01
    sigma = (sigma01, sigma00)
    sd_x = 2.0
    x = np.random.normal(0.0,sd_x,N)
    # Create imbalance
    pa = np.array([0.5, 0.5])
    a = np.array([np.random.binomial(1, pa[int(x[n]>0)]) for n in range(len(x))]) # create imbalance: p(a=1|x>-1.5) = 0.9, p(a=1|x<-1.5) = 0.1
    # draw model parameters:
    b0 = np.random.normal(0.,0.5)
    b1 = np.random.normal(0.,0.5)
    c0 = 2.0
    offset = np.random.choice([sd_x, -1*sd_x])
    print(offset)
    model = NonlinearModel(sigma, b0, b1, c0, offset)
    # Output
    y = model.predict(x,a)
    return (x,a,y), model
    

def generate_imbalanced_data(N, model, pa):
    # x     covariates
    # a     actions
    # y     outcomes
    x = np.random.normal(0.0,1.0,N)
    a = np.array([np.random.binomial(1, pa[int(x[n]>0)]) for n in range(len(x))]) # create imbalance: p(a=1|x>-1.5) = 0.9, p(a=1|x<-1.5) = 0.1
    if np.sum(a) == len(a) or np.sum(a) == 0:
        a[np.random.choice(np.arange(len(a)),1)] = 1-a[0]
    y = model.predict(x,a)
    return (x,a,y)

def generate_balanced_data(x, model):
    a = np.concatenate([np.repeat(0,len(x)),np.repeat(1,len(x))])
    x = np.concatenate([x,x])
    y = model.predict(x,a)
    return (x,a,y)

def kernel(x1,x2):
    l = 0.8 # kernel length-scale
    ret = norm.pdf((x1-x2)/l)
    return ret
    

def imbalance(x,a):
    x_1 = x[a==1]
    x_0 = x[a==0]
    e1 = np.mean([kernel(x_1[n],x_1[np.arange(len(x_1))!=n]) for n in range(len(x_1))])
    e2 = np.mean([kernel(x_1[n],x_0) for n in range(len(x_1))])
    e3 = np.mean([kernel(x_0[n],x_0[np.arange(len(x_0))!=n]) for n in range(len(x_0))])
    return e1 -2*e2 + e3

def error_rate(preds):
    mu_tau = preds[1][0] - preds[0][0]
    var_tau = np.sqrt(preds[1][1] + preds[0][1])
    ret = norm.cdf(-np.abs(mu_tau)/var_tau)
    return mu_tau, var_tau, ret

def get_dec(x, model):
    x_ = np.array([x,x])
    a = np.array([0,1])
    y = model.predict_noiseless(x_,a)
    d = y[1]>y[0]
    return d


#################################################
seed = 714
num_trains = np.array([30, 40, 50, 60, 70])
num_test = 500
num_rep = 200

correlations = []
err_true = []
err_estim = []



for j in range(len(num_trains)):
    num_train = num_trains[j]

    RES = []

    for rep in range(num_rep):
        print((j,rep))
        N=num_train+num_test
        data, true_model = generate_data(N=N, seed=(rep+1)*seed)
        # Train and test data
        tr_data = [data[i][0:num_train] for i in range(len(data))]
        te_data = [data[i][num_train:N] for i in range(len(data))]

        I = np.linspace(0.0, 0.5, 6)
        imb = np.zeros(len(I))
        imb_model = np.zeros(len(I))
        result = np.zeros(len(I))
        result_dec = np.zeros(len(I))
        i=0

        for b in I:
            pa = [b, 1-b]
            tr_data = generate_imbalanced_data(num_train, true_model, pa)

            dat = {'x':tr_data[0],
                'a':tr_data[1],
                'y':tr_data[2]
            }

            imb[i] = imbalance(dat['x'],dat['a'])

            model = GPModel(dat['a'], dat['x'].reshape(-1,1), dat['y'])
            imb_model[i] = model.imbalance()

            typeSrate = np.zeros(num_test)
            decisions = np.zeros(num_test)
            tau = np.zeros(num_test)
            y_test = np.zeros((num_test,2))

            for n, x in enumerate(np.sort(te_data[0])):
                preds = model.predict(x.reshape(-1,1))
                mu_tau, var_tau, err = error_rate(preds)
                typeSrate[n] = err
                d = mu_tau > 0
                d_true = get_dec(x, true_model)
                decisions[n] = int(d==d_true) # higher better
                tau[n] = mu_tau
                y_test[n,0] = preds[0][0]
                y_test[n,1] = preds[1][0]

            result[i] = np.nanmean(typeSrate)
            result_dec[i] = np.nanmean(decisions)
            i += 1

        imb = np.array(imb)
        imb_model = np.array(imb_model)
        results = np.array(result)
        result_dec = np.array(result_dec)
        ind = np.argsort(imb)
        imb = imb[ind]
        imb_model = imb_model[ind]
        result = result[ind]
        result_dec = result_dec[ind]
        RES += [(imb, result, result_dec, imb_model)]

    imbalances = np.ravel([RES[n][0] for n in range(len(RES))])
    errors = np.ravel([RES[n][1] for n in range(len(RES))])
    decs = np.ravel([RES[n][2] for n in range(len(RES))])
    imb_model = np.ravel([RES[n][3] for n in range(len(RES))])
    ind = ~np.isnan(imbalances)
    imbalances = imbalances[ind]
    errors = errors[ind]
    decs = decs[ind]
    imb_model = imb_model[ind]

    err_true += [1-decs]
    err_estim += [errors]

    c1 = np.corrcoef(imbalances, errors)[0,1]
    c2 = np.corrcoef(imbalances, decs)[0,1]
    c3 = np.corrcoef(errors, decs)[0,1]
    print("corr(imbalance, type S error rate)")
    print(c1)
    print("corr(imbalance, proportion of correct decisions)")
    print(c2)
    print("corr(type S error rate, proportion of correct decisions)")
    print(c3)
    correlations += [(c1, c2, c3)]

corr_is = np.ravel([correlations[n][0] for n in range(len(correlations))])
corr_id = np.ravel([correlations[n][1] for n in range(len(correlations))])
corr_sd = np.ravel([correlations[n][2] for n in range(len(correlations))])

dat_save  = {'num_trains': num_trains,
             'imbalances': imbalances,
             'errors': err_true,
             'estimates': err_estim,
             'corr_imbalance_typeSerrorrate': corr_is,
             'corr_imbalance_decisions': corr_id,
             'corr_typeSerrorrate_decisions': corr_sd}
pickle.dump(dat_save, open("raw_corr_nonlinear_seed" + str(seed) + "_" + str(num_rep) +"rep" +".p", "wb" ))


plt.plot(num_trains, corr_is, 'b-', label='corr(imbalance, error rate)')
plt.plot(num_trains, -corr_id, 'r--', label='-corr(imbalance, performance)')
plt.plot(num_trains, -corr_sd, 'r-', label='-corr(error rate, performance)')
plt.legend(prop={'size': 12})
plt.xlabel('n')
plt.ylabel('Pearson correlation')
plt.show()

# Plot estimated against observed Type S error
n_per_p = dat_save['num_trains']
err = dat_save['errors']
est = dat_save['estimates']

for ind in np.arange(len(err)):
    p = plt.plot(err[ind], est[ind], '.', label='n={}'.format(n_per_p[ind]))
    slope, intercept, r_value, p_value, std_err = stats.linregress(err[ind],est[ind])
    x = np.array([0,0.89])
    y = slope*x + intercept
    plt.plot(x,y, color=p[-1].get_color())
plt.legend(prop={'size': 12})
plt.xlabel('$\gamma$')
plt.ylabel('$\hat{\gamma}$')
plt.show()
