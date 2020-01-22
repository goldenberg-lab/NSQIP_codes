import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root_scalar

from support.support_funs import stopifnot

# x=Z.copy();y=latent[:,0];standardize=True; add_intercept=True;lam=0
def least_squares(y, x, lam=0, standardize=True, add_intercept=True):
    p = x.shape[1]
    if standardize:
        mu_x = x.mean(axis=0).reshape([1, p])
        sd_x = x.std(axis=0).reshape([1, p])
        x = (x - mu_x) / sd_x
    else:
        mu_x = np.repeat(0, p).reshape([1, p])
        sd_x = mu_x + 1
    if add_intercept:
        x = np.insert(arr=x, obj=0, values=1, axis=1)
    # Run optimize
    b = minimize(fun=loss_l2, x0=np.append(np.mean(y), np.repeat(0, p)),
                   args=(y, x, lam), jac=grad_l2,method='L-BFGS-B').x
    # re-normalize
    b[1:] = b[1:] / sd_x.flatten()
    b[0] = b[0] - sum(b[1:]*mu_x.flatten())
    return(b)

# Fast logistic regression fit
def fast_logit(y,eta, lam):
    n = y.shape[0]
    eta = np.c_[np.ones([n,1]),eta]
    mu = np.mean(y)
    x0 = [np.log(mu/(1-mu)),1]
    b = minimize(fun=loss_logit, x0=x0, args=(y, eta, lam), method='L-BFGS-B', jac=grad_logit).x
    return(b)

class multitask_logistic():
    # Initialize base parameters
    def __init__(self, add_intercept=True, standardize=True):
        self.add_intercept = add_intercept
        self.standardize = standardize

    def fit(self, X, Y, lam1=0, lam2=0):
        n, p, k = X.shape + (Y.shape[1],)
        stopifnot(n == Y.shape[0])
        # Fit l2-regularized least squares along columns of X
        Bhat = np.apply_along_axis(least_squares, 0, Y,
                        *(X, lam1, self.standardize, self.add_intercept))
        Eta = X.dot(Bhat[1:]) + Bhat[0].reshape([1, Bhat.shape[1]])
        What = Bhat.copy()
        for jj in range(k):
            w_jj = fast_logit(Y[:, jj], Eta[:, jj], lam2)
            What[0, jj] = (w_jj[0] + w_jj[1] * Bhat[0, jj])
            What[1:, jj] = w_jj[1] * Bhat[1:, jj]
        # Zeta = X.dot(What[1:]) + What[0].reshape([1, What.shape[1]])
        self.weights = What[1:]
        self.intercept = What[0].reshape([1,What.shape[1]])
        self.p = p
        self.Bhat = Bhat

    def predict(self, Xnew):
        stopifnot(Xnew.shape[1] == self.weights.shape[0])
        eta = Xnew.dot(self.weights) + self.intercept
        return (eta)

def dgp_yX(n,p,ss=1234):
    X = np.random.randn(n,p)
    eta = X.dot(np.repeat(1,p))
    py = sigmoid(eta)
    y = np.random.binomial(n=1,p=py,size=n)
    return (y, X)

def sigmoid(x):
    return(1 / (1 + np.exp(-x)))

def loss_l2(b,y,X,lam):
    eta = X.dot(b)
    mse = np.mean((y - eta) ** 2)
    reg = lam*np.sum(b[1:]**2) / 2
    return( mse + reg)

def grad_l2(b,y,X,lam):
    res = y - X.dot(b)
    gnll = -X.T.dot(res)/X.shape[0]
    greg = lam * np.append(0, b[1:])
    return( gnll + greg )

def loss_logit(b,y,X,lam):
    eta = X.dot(b)
    nll = -1*np.mean( y*eta - np.log(1+np.exp(eta)) )
    reg = lam*np.sum(b[1:]**2) / 2
    return( nll + reg )

def grad_logit(b,y,X,lam):
    py = sigmoid(X.dot(b))
    gnll = -X.T.dot(y - py)/X.shape[0]
    greg = lam * np.append(0, b[1:])
    return( gnll + greg )