# This is a parent class of models for active learning and counterfactual elicitation.
# Contains implementations of the different acquisition functions.

import numpy as np
import scipy.linalg as la
import copy
from scipy.stats import norm

class Model(object):
    def predict(self, new_predictors, model=None):
        raise NotImplementedError
    def expected_information_gain(self, n, new_predictors):
        '''
        Uses Gauss-Hermite quadrature to compute expected information gain
        '''
        x_train = self.dat['predictors']
        x_star = self.dat['predictors'][n,:].reshape(1,-1)
        if self.n_actions == 2:
            a_star = 1 - self.dat['actions'][n] # counterfactual action
        else:
            raise NotImplementedError
        
        points, weights = np.polynomial.hermite.hermgauss(32) 
        eig = 0.0
        for ii, yy in enumerate(points):
            preds_star = self.predict(x_star)
            mu_star, S_star = preds_star[a_star]
            y_star = np.sqrt(2)*np.sqrt(S_star)*yy + mu_star #for substitution
            try:
                model_star = self.update(a_star, x_star, y_star)
                IG = 0
                for m in range(x_train.shape[0]):
                    x_m = x_train[m,:].reshape(1,-1)
                    ypreds_0 = self.predict(x_m)
                    ypreds_1 = self.predict(x_m, model_star)
                    for d in range(self.n_actions):
                        y_m_0, S_m_0 = ypreds_0[d]
                        y_m_1, S_m_1 = ypreds_1[d]
                        IG += Model.mvn_KL(y_m_1, S_m_1, y_m_0, S_m_0)
            except np.linalg.linalg.LinAlgError:
                IG = 0
            eig += weights[ii] * 1/np.sqrt(np.pi) * IG
        return eig

    def expected_targeted_information_gain(self, n, new_predictors):
        '''
        Uses Gauss-Hermite quadrature to compute expected information gain in decision
        '''
        preds = self.predict(new_predictors)
        d = self._decide(preds)
        ypred, Spred = preds[d]
        
        x_star = self.dat['predictors'][n,:].reshape(1,-1)
        if self.n_actions == 2:
            a_star = 1 - self.dat['actions'][n] # counterfactual action
        else:
            raise NotImplementedError
        
        points, weights = np.polynomial.hermite.hermgauss(32) 
        targig = 0.0
        for ii, yy in enumerate(points):
            preds_star = self.predict(x_star)
            mu_star, S_star = preds_star[a_star]
            y_star = np.sqrt(2)*np.sqrt(S_star)*yy + mu_star #for substitution
            try:
                model_star = self.update(a_star, x_star, y_star)
                preds_next = self.predict(new_predictors, model_star)
            except np.linalg.linalg.LinAlgError:
                preds_next = self.predict(new_predictors)
            D_KL = 0
            for d in range(self.n_actions):
                ypred_next, Spred_next = preds_next[d]
                D_KL += Model.mvn_KL(ypred_next, Spred_next, ypred, Spred)
            targig += weights[ii] * 1/np.sqrt(np.pi) * D_KL
        return targig

    def uncertainty_sampling(self, n, new_predictors):
        x_train = self.dat['predictors']
        x_star = self.dat['predictors'][n,:].reshape(1,-1)
        if self.n_actions == 2:
            a_star = 1 - self.dat['actions'][n] # counterfactual action
        else:
            raise NotImplementedError
        y_star = self.predict(x_star)
        y_mean, y_S = y_star[a_star]
        ret = - 0.5 * np.log(2*np.pi*np.exp(1)*y_S)
        return ret

    def expected_reliability(self, n, new_predictors): #'decerr'
        '''
        Uses Gauss-Hermite quadrature to compute expected type S error rate
        '''
        def error_rate(preds):
            mu_tau = preds[1][0] - preds[0][0]
            sd_tau = np.sqrt(preds[1][1] + preds[0][1])
            alpha = norm.cdf(-np.abs(mu_tau)/sd_tau)
            return alpha
        x_star = self.dat['predictors'][n,:].reshape(1,-1)
        if self.n_actions == 2:
            a_star = 1 - self.dat['actions'][n] # counterfactual action
        else:
            raise NotImplementedError
        decerr = 0.0
        points, weights = np.polynomial.hermite.hermgauss(32) 
        for ii, yy in enumerate(points):
            preds_star = self.predict(x_star)
            mu_star, S_star = preds_star[a_star]
            y_star = np.sqrt(2)*np.sqrt(S_star)*yy + mu_star #for substitution
            try:
                model_star = self.update(a_star, x_star, y_star)
                preds_next = self.predict(new_predictors, model_star)
            except np.linalg.linalg.LinAlgError:
                preds_next = self.predict(new_predictors)
            util = 1-error_rate(preds_next)
            decerr += weights[ii] * 1/np.sqrt(np.pi) * util
        return decerr

    def expected_reliability_infogain(self, n, new_predictors): #'decerrig'
        '''
        Uses Gauss-Hermite quadrature to compute information gain on the expected type S error rate
        '''
        def error_rate(preds):
            mu_tau = preds[1][0] - preds[0][0]
            sd_tau = np.sqrt(preds[1][1] + preds[0][1])
            alpha = norm.cdf(-np.abs(mu_tau)/sd_tau)
            return alpha
        def KL_Bernoulli(p, q):
            return p*(np.log(p/q) if p>0. else 0. ) + (1.-p)*(np.log((1.-p)/(1.-q)) if 1.-p > 0. else 0.)
        x_star = self.dat['predictors'][n,:].reshape(1,-1)
        if self.n_actions == 2:
            a_star = 1 - self.dat['actions'][n] # counterfactual action
        else:
            raise NotImplementedError
        preds_current = self.predict(new_predictors)
        gamma = 1-error_rate(preds_current)
        decerr = 0.0
        points, weights = np.polynomial.hermite.hermgauss(32) 
        for ii, yy in enumerate(points):
            preds_star = self.predict(x_star)
            mu_star, S_star = preds_star[a_star]
            y_star = np.sqrt(2)*np.sqrt(S_star)*yy + mu_star #for substitution
            try:
                model_star = self.update(a_star, x_star, y_star)
                preds_next = self.predict(new_predictors, model_star)
            except np.linalg.linalg.LinAlgError:
                preds_next = self.predict(new_predictors)
            gamma_next = 1-error_rate(preds_next)
            util = KL_Bernoulli(gamma,gamma_next)
            decerr += weights[ii] * 1/np.sqrt(np.pi) * util
        return decerr
    
    def select_oracle_query(self, new_predictors, acquisition='decerrig', nn=False, k=10): 
        # select acquisition:
        if acquisition == 'targig':
            acq  = self.expected_targeted_information_gain
        elif acquisition == 'decerr':
            acq  = self.expected_reliability
        elif acquisition == 'decerrig':
            acq = self.expected_reliability_infogain
        elif acquisition == 'uncertainty':
            acq = self.uncertainty_sampling
        elif acquisition == 'ig':
            acq = self.expected_information_gain
        elif acquisition == 'random':
            N = self.dat['predictors'].shape[0]
            query_idx = np.random.choice(np.arange(N))
            query_a = 1 - self.dat['actions'][query_idx]
            print(query_idx)
            return([query_idx, query_a])

        if(self.n_actions>2):
            raise NotImplementedError
        try:
            previous_queries = copy.deepcopy(self.prevQ)
        except AttributeError:
            previous_queries = []
        N = self.dat['predictors'].shape[0]
        if(nn is False): #Not using nearest neighbour approximation
            utility = np.zeros(N)
            for n in range(N):
                if n in previous_queries:
                    utility[n] = - np.Infinity
                else:
                    u = acq(n, new_predictors)
                    utility[n] = u
        else:
            dist = np.zeros(N)
            for n in range(N): #Compute scaled inverse distance to the test point
                dist[n] = -self.compute_inverse_distance(n, new_predictors)
            ind = np.argsort(dist)
            utility = - np.Infinity * np.ones(N)
            j = 0
            for i in range(N):
                n = ind[i]
                if n in previous_queries:
                    pass
                else:
                    if j < k:
                        u = acq(n, new_predictors)
                        utility[n] = u
                        j += 1
                    else:
                        break
        query_idx = np.random.choice(np.flatnonzero(utility[:] == np.nanmax(utility[:])))
        query_a = 1 - self.dat['actions'][query_idx]
        print(query_idx)
        return([query_idx, query_a])
    
    def update(self, action, predictor, oracle_outcome):
        raise NotImplementedError
    
    def _decide(self, predictions):
        '''
        Choose action with highest expected utility (assume higher outcome is better; u(y)=y)
        '''
        means = np.array([float(predictions[k][0]) for k in range(len(predictions))])
        d = np.random.choice(np.flatnonzero(means[:] == np.nanmax(means[:])))
        return(d)
    
    @staticmethod
    def mvn_KL(mu1, S1, mu2, S2):
        '''
        KL divergence between two multivariate normals
        '''
        if type(mu1)!=np.ndarray:
            d = 1
        else:
            d = mu1.shape[0]
        KL = 0.5*(np.log(la.det(S2)/la.det(S1)) + np.trace(la.inv(S2).dot(S1)) + (mu1 - mu2).T.dot(la.inv(S2).dot((mu1 - mu2))) - d)
        return KL
    
    @staticmethod
    def answer_oracle_query(query_idx, action, oracle_outcomes):
        '''
        Answer a query by adding Gaussian noise to the true value.
        '''
        sigma_expert = 0.1
        bias = 0.0
        ret = oracle_outcomes[query_idx,action] + bias + np.random.normal(0.0,sigma_expert)
        return(np.asarray([ret]))
