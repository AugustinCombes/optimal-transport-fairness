import numpy as np
from scipy.stats import norm

def h(x):
    return x[0] * x[3] / (x[1] * x[2])

def grad_h(x):
    return np.array([x[3] / (x[1] * x[2]),
                     -x[0] * x[3] / (x[1] * x[1] * x[2]),
                     -x[0] * x[3] / (x[2] * x[2] * x[1]),
                     x[0] / (x[2] * x[1])])

def disparate(X, y, S_idx, alpha=0.05):
    '''
    X: Data matrix, X[i,j] contains the value of the j-th variable for the i-th sample. (array-like of shape (n_sample, n_variables))
    y: Binary decision taken by the algorithm fed with the X data input. (array-like of shape (n_sample))
    S_idx: Indice of the (binary) sensitive column of X. (int)
    alpha:IC (float)
    '''
    X,y = np.array(X), np.array(y)
    S_idx = int(S_idx)
    try :
        sensitive_column = X[:, S_idx]
    except :
        raise ValueError("S_idx must be the indice of the sensitive column of X")

    if np.unique(sensitive_column).shape[0] != 2:
        raise ValueError("The sensitive column of X must be binary")

    n,dim = X.shape

    # Estimated P(S=1)
    pi_1 = sensitive_column.mean()
    
    # Estimated P(S=0)
    pi_0 = 1 - pi_1 

    # Estimated P(g(X)=1, S=1)
    p_1 = (sensitive_column*y).mean()
    
    # Estimated P(g(X)=1, S=0)
    p_0 = ((1 - sensitive_column)*y).mean()
    
    # Statistic of disparate impact
    Tn = (p_0*pi_1)/(p_1*pi_0)

    grad = grad_h(np.array([p_0, p_1, pi_0, pi_1]))

    cov = np.array([
                    [p_0*(1-p_0), -p_0*p_1,    pi_1*p_0,   -pi_1*p_0],
                    [-p_0*p_1,    p_1*(1-p_1), -pi_0*p_1,  pi_0*p_1],
                    [pi_1*p_0,    -pi_0*p_1,   pi_0*pi_1,  -pi_0*pi_1],
                    [-pi_1*p_0,   pi_0*p_1,    -pi_0*pi_1, pi_0*pi_1]
                    ])
                    
    #print(np.matmul(np.matmul(grad,cov),grad.T))

    sigma = np.sqrt(np.matmul(np.matmul(grad,cov),grad.T))

    lower_lim = Tn - (sigma * norm.ppf(1 - alpha / 2)) / np.sqrt(n)
    upper_lim = Tn + (sigma * norm.ppf(1 - alpha / 2)) / np.sqrt(n)

    # BER
    BER = 0.5 * (p_0 / pi_0 + 1 - p_1 / pi_1)
    return (lower_lim, Tn, upper_lim, BER)
