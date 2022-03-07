import numpy as np
import pandas as pd
from itertools import product
from sim_utils import cox_sampler
import tst_utils
import argparse


from tqdm.auto import tqdm
import pickle


def simulate(n_reps = 100, n_sample=500, n_quant = 128,
                kernel_X = 'constant',sig_X=1,
                kernel_Y = 'constant',sig_Y=1,
                sig_Z=1,beta1=1,
                box_params = {'max_depth':1,'n_estimators':200,'eta':0.1},
                L2_pen=0.0001,n_splits=5,
                dependency = 0,savepaths=False):

    if savepaths:
        (gam_plug,gam_corrected,gam_double) = ([] for _ in range(3))
    (T_plug,T_corrected,T_double,sig_list,sigd_list,p_cox) = ([] for _ in range(6))

    data_sampler = cox_sampler(sig_X,sig_Y,sig_Z,dependency,beta1,kernel_X,kernel_Y,n_quant)
    data_sampler.scale_and_set_baseline()

    for _ in tqdm(range(n_reps), leave=False):
        ## Sample data
        X,Y,Z,tau = data_sampler.sample_all(n_sample)

        # Fit tests
        g_p, g_c, sig, best_params = tst_utils.compute_gamma(tau,Z,X,box_params,n_quant,L2_pen=L2_pen,cross_validate=True)
        g_d, sig_d = tst_utils.compute_gamma_double(tau,Z,X,best_params, n_quant, n_splits=n_splits,L2_pen=L2_pen)

        ## save test results
        if savepaths:
            gam_plug.append(g_p)
            gam_corrected.append(g_c)
            gam_double.append(g_d)
        T_plug.append(np.linalg.norm(g_p,ord=np.inf))
        T_corrected.append(np.linalg.norm(g_c,ord=np.inf))
        T_double.append(np.linalg.norm(g_d,ord=np.inf))
        sig_list.append(sig if savepaths else sig[-1])
        sigd_list.append(sig_d if savepaths else sig_d[-1])
        p_cox.append(tst_utils.cox_test(X,Z,tau))
    
    df = pd.DataFrame({
        "T_plug":T_plug,
        "T_corrected":T_corrected,
        "T_double":T_double,
        "sigma":sig_list,
        "sigma_double":sigd_list,
        "p_cox":p_cox,
        "n_sample": n_sample*np.ones(n_reps),
        "beta1": beta1*np.ones(n_reps),
        "kernel_X": [kernel_X]*n_reps,
        "kernel_Y": [kernel_Y]*n_reps,
        "alt_param": [dependency]*n_reps
    })
    if savepaths:
        df["gam_plug"] = gam_plug,
        df["gam_corrected"] = gam_corrected,
        df["gam_double"] = gam_double,

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    filename = '/Users/bwq666/Documents/GitHub/nonparametric-cli-test/sim_data/main_simulation.pkl'
    parser.add_argument('--file_save_path',default=filename,type=str,
        help="specify the path to save simulation results.")
    parser.add_argument("--sample_sizes", nargs="+", default=[100,500,1000,2000])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--same_kernels", type=bool, default=True)
    parser.add_argument("--kernels", nargs="+", default=['constant','gaussian','sine'])
    parser.add_argument("--betas", nargs="+", default=[-1,1])
    parser.add_argument("--alternatives", nargs="+", default=[0,5,10])
    parser.add_argument("--store_sample_paths", type=bool, default=False)
    
    settings = parser.parse_args()

    ## Simulation settings
    if settings.same_kernels:
        kernels = [(k,k) for k in settings.kernels]
    else:
        kernels = list(product(settings.kernels,settings.kernels))
    betas = settings.betas
    sample_sizes = settings.sample_sizes
    alternatives = settings.alternatives
    n_sim = len(kernels)*len(betas)*len(sample_sizes)*len(alternatives)
    param_grid = product(kernels,betas,sample_sizes,alternatives)

    # Run simulations over parameter grid
    simulation_data = []
    for (k_X,k_Y),beta_1,sample_size,dependency in tqdm(param_grid, position = 0, leave=True, total=n_sim):
        sim_data = simulate(
            n_reps = settings.repetitions, n_sample=sample_size,
            kernel_X=k_X, kernel_Y=k_Y,
            beta1=beta_1,
            dependency=dependency, 
            savepaths=settings.store_sample_paths
        )
        simulation_data.append(sim_data)
    full_data = pd.concat(simulation_data)

    with open(settings.file_save_path, 'wb') as f:
        pickle.dump(full_data, f)