import numpy as np
from scipy.linalg import solve
from sim_utils import *
from lifelines import CoxTimeVaryingFitter

import sys
sys.path.insert(0, './BoXHED2.0') #Cannot use normal relative import due to '.' in directory name

from boxhed import boxhed
from model_selection import cv

def estimate_Pi(X,Z,L2_pen=0.0001):
    n_s,n_T = Z.shape
    Pihat = np.zeros((n_s,n_T))
    for t in range(1,n_T):
        Xt = X[:,t]
        Zt = Z[:,:t]
        ols = solve(Zt.transpose()@Zt+L2_pen*np.identity(t),Zt.transpose()@Xt,assume_a='pos')
        Pihat[:,t] = Zt@ols
    return Pihat


def estimate_Pi_on_test(Z1,X,Z2,L2_pen=0.0001):
    n_s,n_T = Z2.shape
    Pihat = np.zeros((n_s,n_T))
    for t in range(1,n_T):
        Zt = Z1[:,:t]
        Zp = Z2[:,:t]
        Xt = X[:,t]
        ols = solve(Zt.transpose()@Zt+L2_pen*np.identity(t),Zt.transpose()@Xt,assume_a='pos')
        Pihat[:,t] = Zp@ols
    return Pihat


def fit_boxhed(data,box_params,cross_validate=False):
    boxhed_ = boxhed(**box_params)
    subjects, X0, w, delta = boxhed_.preprocess(
        data             = data,
        num_quantiles    = 256,
        is_cat           = [],
        weighted         = False,
        nthreads         = 1
    )
    if cross_validate:
        param_grid = {'max_depth':    [1, 2, 3, 4],
              'n_estimators': [50, 100, 150, 200, 250, 300],
              'eta':          [0.1]}

        cv_rslts, best_params = cv(
            param_grid, X0, w, delta, subjects, 
            5, [-1], 6  # num_folds, gpu_list -> cpu, batchsize 
        )
        boxhed_.set_params(**best_params)
        boxhed_.fit(X0, delta, w)
        return boxhed_, best_params
    else:
        boxhed_.fit(X0, delta, w)
        return boxhed_



def compute_gamma(tau,X,Z,box_params,quantiles,L2_pen=0.0001,cross_validate=False):
    data = format_data(tau,X)

    if cross_validate:
        boxhed_, best_params = fit_boxhed(data,box_params,cross_validate)
    else:
        boxhed_ = fit_boxhed(data,box_params,cross_validate)
    
    X_pos = X[tau>0,...]
    Z_pos = Z[tau>0,...]

    Pihat = estimate_Pi(X_pos,Z_pos,L2_pen)
    preds = boxhed_.predict(data[["t_start","X_0"]])

    hazards,dN = np.zeros(Z_pos.shape),np.zeros(Z_pos.shape)
    tmp = 0
    for i,tau_i in enumerate(tau[tau>0]):
        hazards[i,:tau_i] = preds[tmp:tmp+tau_i]
        if tau_i < quantiles:
            dN[i,tau_i] = 1
        tmp += tau_i
    hazards[:,quantiles-1] = 0

    dM = dN - hazards/quantiles
    g_p = np.cumsum(Z_pos * dM, axis = 1).mean(0)
    g_c = np.cumsum((Z_pos-Pihat) * dM, axis = 1).mean(0)
    sig = np.cumsum((Z_pos-Pihat)**2 * dN, axis = 1).mean(0)

    if cross_validate:
        return g_p, g_c, sig, best_params

    return g_p, g_c, sig



def fit_and_test(X_train,Z_train,tau_train,X_test,Z_test,tau_test,quantiles,box_params,L2_pen=0.0001):
    data_train = format_data(tau_train,X_train)
    data_test = format_data(tau_test,X_test)
    boxhed_ = fit_boxhed(data_train,box_params)

    X_pos = X_test[tau_test>0,...]
    Z_pos = Z_test[tau_test>0,...]

    preds = boxhed_.predict(data_test[["t_start","X_0"]])
    Pihat = estimate_Pi_on_test(X_train,Z_train,X_pos,L2_pen)

    hazards,dN = np.zeros(Z_pos.shape),np.zeros(Z_pos.shape)
    tmp = 0
    for i,tau_i in enumerate(tau_test[tau_test>0]):
        hazards[i,:tau_i] = preds[tmp:tmp+tau_i]
        if tau_i < quantiles:
            dN[i,tau_i] = 1
        tmp += tau_i
    hazards[:,quantiles-1] = 0

    dM = dN - hazards/quantiles
    gamma = np.cumsum((Z_pos-Pihat) * dM, axis = 1)
    sig = np.cumsum((Z_pos-Pihat)**2 * dN, axis = 1)

    return gamma.mean(0), sig.mean(0)



def compute_gamma_double(tau,X,Z,box_params, quantiles, n_splits=2,L2_pen=0.0001):
    fitted_gammas = []
    fitted_sigmas = []

    split_size = len(tau)//n_splits

    for k in range(n_splits):
        I_k = np.zeros_like(tau).astype(int)
        if k != n_splits-1:
            I_k[k*split_size:(k+1)*split_size] = np.ones(split_size)
        else:
            I_k[k*split_size:] = np.ones(len(tau)-k*split_size)
        
        X1, X2 = X[I_k==0,...],X[I_k>0,:]
        Z1, Z2 = Z[I_k==0,...],Z[I_k>0,:]
        tau1,tau2 = tau[I_k==0],tau[I_k>0]

        gamma,sigma = fit_and_test(X1,Z1,tau1,X2,Z2,tau2,quantiles,box_params,L2_pen)
        
        fitted_gammas.append(gamma)
        fitted_sigmas.append(sigma)
    
    return np.mean(fitted_gammas,0), np.mean(fitted_sigmas,0)


pi = np.pi
def BM_supnorm_cdf(x,T_max=None,N=1000):
    if T_max!=None:
        x /= np.sqrt(T_max)
    
    S = np.arange(1,N+1)
    return 0 if x<=0 else (np.exp(-(2*S-1)**2*pi**2/(8*x**2))*np.power((-1),S+1)*4/(pi*(2*S-1))).sum()


def cox_test(X,Z,tau):
    df = format_data_with_X(X,Z,tau)
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(df, id_col="subject", event_col="delta", start_col="t_start", stop_col="t_end", show_progress=False,robust=False)
    return ctv.summary['p'][1]