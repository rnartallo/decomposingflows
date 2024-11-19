import numpy as np
from scipy.linalg import expm


def isotropic_OU2(theta,eps,x0,tau,samples):
    B = np.array([[2,-theta],[+theta,2]])
    D = np.array([[eps,0],[0,eps]])
    S = np.array([[eps/2,0],[0,eps/2]])
    Sinv = np.array([[2/eps,0],[0,2/eps]])
    expb = expm(-tau*B)
    Q = B@S-D
    C = np.array([[np.exp(2*tau)*eps*np.sinh(2*tau),0],[0,np.exp(2*tau)*eps*np.sinh(2*tau)]])
    expirrev = expm(-tau*Q@Sinv)
    exprev = expm(-tau*D@Sinv)

    N = x0.shape[0]
    X = np.zeros((N,samples))
    X_rev = np.zeros((N,samples))
    X_irrev = np.zeros((N,samples))
    X_check = np.zeros((N,samples))
    
    X[:,0] = x0
    X_rev[:,0] = x0
    X_irrev[:,0] = x0
    X_check[:,0] = x0

    for s in range(1,samples):
        xi = np.random.multivariate_normal(np.reshape(np.zeros((N,1)),N),C)
        X[:,s] = expb@X[:,s-1] + xi
        X_rev[:,s] = exprev@X_rev[:,s-1] + xi
        X_irrev[:,s] = expirrev@X_irrev[:,s-1]
        X_check[:,s] = expirrev@(exprev@X_check[:,s-1] + xi)
    
    return(X,X_rev,X_irrev,X_check)