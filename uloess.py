import numpy as np
from scipy.stats import t as tdist

debug = False
def debug_print(i):
    return debug and i==5

# weighted least squares fit with confidence interval
# i = point to predict
# x = x values
# y = y values
# e = statistical errors on y
# w = tricubic weights
# deg = degree of polynomial (1 or 2)
# alpha = confidence interval
def wls(i, x, y, e, w, deg):
    weights = w/(e**2)
    if debug_print(i): print("e",e)
    if debug_print(i): print("w",w)
    if debug_print(i): print("weights",weights)
    # polyfit convention: coef is sorted with highest power first
    # form A matrix (which corresponds to gradient of parameters for least squares fit) following that convention
    # include weights following weighted least squares convention
    # Because we don't care about the x values, just the y value when evaluating fit results, we scale the x values used 
    # as the input to the fit algorithm to be between -1 and 1 to avoid numerical instability
    X = np.vander((x-np.mean(x[w>0]))/(2*(np.max(x)-np.min(x))), deg+1)
    W = np.diag(weights)
    L = X.dot(np.linalg.inv(X.T.dot(W).dot(X))).dot(X.T.dot(W)) # smoothing matrix
    Y = L.dot(y) # predictions
    y_pred = Y[i]
#    if debug_print(i): np.set_printoptions(threshold=np.inf)
    if debug_print(i): print("L",L)
    if debug_print(i): print("y[i]",y[i])
    if debug_print(i): print("y_pred",y_pred)
    return y_pred, L[i]

# second-derivative (curvature) operator
# equates to: f[i] - 2 f[i+1] + f[i+2]
def d2op(x):
    n = len(x)
    if n<3: raise ValueError("d2op needs at least 3 x values")
    if np.any(np.diff(x)<=0): raise ValueError("d2op needs strictly increasing x")

    D = np.zeros((n - 2, n), dtype=float)

    for i in range(n - 2):
        h1 = x[i + 1] - x[i]
        h2 = x[i + 2] - x[i + 1]

        D[i, i] = 2.0 / (h1 * (h1 + h2))
        D[i, i + 1] = -2.0 / (h1 * h2)
        D[i, i + 2] = 2.0 / (h2 * (h1 + h2))

    return D

# fraction of MC second-derivative roughness expected to survive in smoothed template
# q = tr(D L e L^T D^T) / tr(D e D^T)
def leak_rough(L, e, D):
    DL = D @ L
    numer = np.sum((DL * e[None, :])**2)
    denom = np.sum((D * e[None, :])**2)
    if denom <= 0:
        return np.nan
    return numer/denom

# fraction of MC variance expected to survive in smoothed template
def leak_var(L, e):
    numer = np.sum((L * e[None, :])**2)
    denom = np.sum(e**2)
    if denom <= 0:
        return np.nan
    return numer/denom

# this follows "Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting", W. Cleveland, S. Devlin
def ci(x, y, e, y_pred, L, alpha):
    R = y-y_pred # residuals
    IL = np.identity(L.shape[0]) - L
    d1 = np.trace(IL.T.dot(IL)) # effective degrees of freedom
    d2 = np.trace((IL.T.dot(IL))**2)
    rho = d1**2/d2 # dof for t distribution
    S = np.sum(R**2)/d1 # estimator of variance scale
    V = S*L.dot(L.T) # fit variance
    C = np.sqrt(np.diag(V))
#    if debug: np.set_printoptions(threshold=np.inf)
    if debug: print("L",L)
    if debug: print("R",R)
    if debug: print("d1",d1)
    if debug: print("S",S)
    if debug: print("V",V)
    if debug: print("C",C)
    cl_factor = tdist.ppf(1-alpha/2,df=rho)
    if debug: print("d2",d2)
    if debug: print("rho",rho)
    if debug: print("cl_factor",cl_factor)
    y_dn = y_pred - cl_factor*C
    y_up = y_pred + cl_factor*C
    # generalized cross validation (for span optimization)
    gcv = S/d1
    if debug: print("gcv",gcv)
    # noise leakage
    D = d2op(x)
    if debug: print("D",D)
    q_rough = leak_rough(L,e,D)
    if debug: print("q_rough",q_rough)
    return y_dn, y_up, gcv, q_rough

# span = fraction of points to include in fit
def loess(x, y, e, deg, alpha, span):
    denom = span*(x.max()-x.min())
    tricube = lambda d: np.clip((1 - np.abs(d/denom)**3)**3, 0, 1)
    N = len(x)
    y_pred = np.zeros_like(y)
    L_final = np.diag(np.zeros_like(y))
    for i in range(N):
        w = tricube(np.abs(x[i] - x))
        y_pred[i], L_final[i] = wls(i,x,y,e,w,deg)
#        if debug_print(i): print(y_pred[i], L_final[i].dot(y))

    y_dn, y_up, gcv, q_rough = ci(x, y, e, y_pred, L_final, alpha)
    if debug:
        i = 5
        print("stderr",y_pred[i]-y_dn[i])
        print(y_pred[i], (y_dn[i], y_up[i]))
        import sys
        sys.exit(0)

    return y_pred, (y_dn, y_up), gcv, q_rough
