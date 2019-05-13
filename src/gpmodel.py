# This is the GP model class to fit two potential outcomes from
# observations and direct feedback on counterfactuals

import GPy
import numpy as np
import copy
from model import Model

class GPModel(Model):
    def __init__(self, actions, predictors, outcomes):
        self.dat = {'actions': actions, 'predictors': predictors, 'outcomes': outcomes}
        self.d = predictors.shape[1]
        self.n_actions = np.max(actions) + 1 
        self.predictors = [self.dat['predictors'][actions == act_i,:] for act_i in range(self.n_actions)]
        self.outcomes = [self.dat['outcomes'][actions == act_i].reshape(-1,1) for act_i in range(self.n_actions)]
        for i in range(self.n_actions):
            self.outcomes[i] = self.outcomes[i].reshape((-1,1))
        self.n_init = [np.sum(actions == act_i) for act_i in range(self.n_actions)]
        self.models = [self._train_with_all(self.predictors[i], self.outcomes[i], self.n_init[i]) for i in range(self.n_actions)]

    def update_me(self, models, query_idx):
        try:
            self.prevQ += [query_idx]
        except AttributeError:
            self.prevQ = [query_idx]
        self.models = models
        for i in range(self.n_actions):
            self.predictors[i] = models[i].X
            self.outcomes[i] = models[i].Y
            print("Updated GP model {}".format(i))
            print(self.models[i])
    
    def _train_with_all(self, predictors, outcomes, n_regular):
        n = predictors.shape[0]
        prior = GPy.core.parameterization.priors.Gamma(a=1.5,b=3.0)
        kern = GPy.kern.RBF(input_dim=self.d, ARD=True)
        kern.variance.set_prior(prior)
        kern.lengthscale.set_prior(prior)
        lik1 = GPy.likelihoods.Gaussian()
        lik1.variance.set_prior(prior)
        lik_expert = GPy.likelihoods.Gaussian()
        lik_expert.variance.set_prior(prior)
        lik = GPy.likelihoods.MixedNoise([lik1, lik_expert])
        output_index = np.ones((n),dtype=int)
        output_index[0:n_regular] = 0
        m = GPy.core.GP(X = predictors, Y = outcomes, kernel=kern, likelihood=lik, Y_metadata = {'output_index':output_index} )
        m.optimize()
        print("First training")
        print(m)
        return m

    def _cov(self,x1,x2,lengthscale): 
        return np.exp( -np.sum((x1- x2) * (x1- x2) / (lengthscale * lengthscale)))
    
    def compute_inverse_distance(self, n, new_predictors):
        a = 1 - self.dat['actions'][n]
        if a < 0.5:
            lengthscale = self.models[0].kern.lengthscale
        else:
            lengthscale = self.models[1].kern.lengthscale
        return self._cov(new_predictors, self.dat['predictors'][n], lengthscale)
        #Compute the distance of predictor point in the covariance space of the 

    def predict(self, new_predictors, models=None):
        if models is None:
            models = self.models
        ret = []
        for i in range(self.n_actions):
            mu, sigma = models[i]._raw_predict(new_predictors)
            ret += [(mu, sigma)]
        return ret

    def imbalance(self):
        imb = 0
        for i in range(self.n_actions):
            x_a = self.predictors[i]
            l = self.models[i].kern.lengthscale
            var = self.models[i].kern.variance
            d = self.models[i].kern.input_dim
            def K(x):
                r2 = np.transpose(x) @ np.diag(1/l**2) @  x
                ret = (2*np.pi)**(-d/2.)*np.prod(l)**(-0.5)*np.exp(-0.5 * r2)
                return ret
            imb += np.sum([[K(x_a[n,:] - x_a[m,:]) for n in range(len(x_a))] for m in range(len(x_a))])
            for j in range(self.n_actions):
                if(j!=i):
                    x_b = self.predictors[j]
                    imb -= np.sum([[K(x_a[n,:] - x_b[m,:]) for n in range(len(x_a))] for m in range(len(x_b))])
        return imb
    
    def update(self, action, predictor, oracle_outcome):
        oracle_outcome = oracle_outcome.reshape(-1,1)
        if(predictor.shape[0]!=1):
            predictor = predictor.reshape(1,-1)
        mcopy = [copy.deepcopy(self.models[i]) for i in range(self.n_actions)]
        mcopy[action].Y_metadata['output_index'] = np.r_[mcopy[action].Y_metadata['output_index'], np.array([1])]
        mcopy[action].set_XY(np.r_[mcopy[action].X, predictor], np.r_[mcopy[action].Y, oracle_outcome], )
        mcopy[action].optimize()
        return mcopy
