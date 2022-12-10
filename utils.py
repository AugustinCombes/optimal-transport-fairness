import numpy as np
import matplotlib.pylab as plt
import ot

def simulate_dataset(n0, n1, mu0, mu1, sigma, beta0, beta1):
  # Sample examples from the two multivariate normal distributions
  X0 = np.random.multivariate_normal(mu0, sigma, size=n0)
  X1 = np.random.multivariate_normal(mu1, sigma, size=n1)
  
  # Compute the logit model for each group
  logit0 = np.exp(X0.dot(beta0)) / (1 + np.exp(X0.dot(beta0)))
  logit1 = np.exp(X1.dot(beta1)) / (1 + np.exp(X1.dot(beta1)))
  
  # Compute the classification labels
  Y0 = (logit0 > 0.5).astype(int)
  Y1 = (logit1 > 0.5).astype(int)
  
  return X0, X1, Y0, Y1

def format_dataset(X0, X1, Y0, Y1):

    X= np.concatenate([X0,X1])
    Y= np.concatenate([Y0, Y1])
    S= np.concatenate([np.repeat(0, X0.shape[0]),np.repeat(1, X1.shape[0])])
    S= np.expand_dims(S, 1)
    X= np.concatenate([S, X], axis=1)
    return X, Y