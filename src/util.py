import numpy as np
import pickle
import pandas as pd
import numpy.random as rndm
import pylab as plt


def normalize(data):
    '''
    Transforms the data to zero mean unit variance
    '''
    return (data- np.mean(data))/np.std(data)

def categorical2indicator(data, name, categorical_max=4):
    '''
    Transforma categorical variable with name 'name' form a data frame to indicator variables
    '''
    values = data[name].values
    values[values>= categorical_max] = categorical_max
    uni = np.unique(values)
    for value in uni:
        data[name+'.'+str(value)] = np.array((values==value), dtype=int)
    data.drop(name, axis=1)
    return data


def preprocess(data, categorical_max=2):
    '''
    This function preprocesses the hill data 
    '''
    #Normalization is enough here
    data['bw'] = normalize(data['bw'])
    data['nnhealth'] = normalize(data['nnhealth'])
    data['preterm'] = normalize(data['preterm'])
    #Taking logarithm does not harm here before normalizing, but might be unneccessary
    data['b.head'] = normalize(np.log(data['b.head']))
    data['momage'] = normalize(np.log(data['momage']))
    
    #Categorigal variables are made to indicators, could also be just normalized:
    #Birth order is between 1
    data = categorical2indicator(data, 'birth.o', categorical_max=categorical_max)
    
    #Everything else is binary, so processing doesn't really help, makes only understanding the results harder.
    
    #For some reason, indicator variable "first" is either 1 or 2, that is why we subtract 1 from it
    data['first'] = data['first'] -1 
    
    return data

def bboot(N, B=10000):
    '''
    gives bayesian bootstrapping weights
    '''
    g = rndm.gamma(np.ones((N,B)))
    g /= np.sum(g, 0)[None,:]
    return g

def bootstrap_results(dat, prctile =[5, 95]):
    '''
    Input: Data of one elicitation (N x Nq)
    Returns returns means and errorbounds.
    Bayesian bootstrapping is used for the errorbounds.
    Assuming we have N observations and Nq queries,
    the function returns 3 x Nq numpy array with mean (idx 0)
    and percentiles (idxs 1 and 2 )
    '''
    N, Nq = dat.shape
    w = bboot(N)
    ret = np.empty((3,Nq))
    ret[0,:] = np.mean(dat, axis=0)
    means = np.dot(w.T, dat)
    ret[1,:] = np.percentile(means, prctile[0], axis=0)
    ret[2,:] = np.percentile(means, prctile[1], axis=0)
    return ret

def normalize(x):
    return (x - x.mean(axis=0))/x.std(axis=0)
    

def get_IHDP_data(id_test, seed_data=0, seed_train=0, percent_for_train=0.75):
    response_surface = 'c'
    inputs = pd.read_csv('../data/hill_non_overlap/inputs.csv', sep=',')
    inputs = preprocess(inputs, categorical_max=2)
    N = inputs.shape[0]
    Ntrain = int(np.ceil(N*percent_for_train))

    D = inputs.columns.shape[0]
    outcomes = pd.read_csv('../data/hill_non_overlap/observed_outcomes.csv', sep=',')
    outcomes = outcomes['outcome_{}'.format(response_surface)]
    potential_outcomes = pd.read_csv('../data/hill_non_overlap/counterfactual_outcomes.csv', sep=',')
    potential_outcomes = potential_outcomes[['outcome_{}0'.format(response_surface),'outcome_{}1'.format(response_surface)]]
    actions = inputs['treat']
    pred_names = inputs.columns[1:D]
    predictors = inputs[pred_names]
    
    outcomes = outcomes.values
    potential_outcomes = potential_outcomes.values
    actions = actions.values
    predictors = predictors.values
    
    outcomes = np.zeros(actions.shape)
    for i in range(actions.shape[0]):
        outcomes[i] = potential_outcomes[i, actions[i]]
    
    #Shuffle data
    np.random.seed(seed_data)
    ind = np.arange(N)
    np.random.shuffle(ind)
    outcomes = outcomes[ind]
    potential_outcomes = potential_outcomes[ind,:]
    actions = actions[ind]
    predictors = predictors[ind,:]

    x_test = {'X':np.copy(predictors[id_test,:]) , 'A':np.copy(actions[id_test]), 'Y':np.copy(outcomes[id_test]), 'PO':np.copy(potential_outcomes[id_test])}

    #shuffle the rest again (not test sample)
    np.random.seed(seed_train)
    
    ind = np.arange(N-1)
    np.random.shuffle(ind)
    outcomes = outcomes[np.arange(N)!=id_test][ind]
    potential_outcomes = potential_outcomes[np.arange(N)!=id_test][ind,:]
    actions = actions[np.arange(N)!=id_test][ind]
    predictors = predictors[np.arange(N)!=id_test,:][ind,:]    
    
    x_train = {'X':predictors[:Ntrain,:], 'A':actions[:Ntrain], 'Y':outcomes[:Ntrain], 'PO':potential_outcomes[:Ntrain]}
    return x_train, x_test

def shadedplot(x, y, fill=True, label='', color='b'):
    # y[0,:] mean, median etc; in the middle
    # y[1,:] lower
    # y[2,:] upper
    p = plt.plot(x, y[0,:], label=label, color=color)
    c = p[-1].get_color()
    #plt.plot(x, y[1,:], color=c, alpha=0.25)
    #plt.plot(x, y[2,:], color=c, alpha=0.25)
    if fill:
        plt.fill_between(x, y[1,:], y[2,:], color=c, alpha=0.25)

