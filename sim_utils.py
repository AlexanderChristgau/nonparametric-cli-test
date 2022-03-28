import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

rng = np.random.default_rng()

'''
Functions for simulating the Cox example.
Here tau is T in the paper.
'''

class cox_sampler:
    def __init__(self,sig_X=1,sig_Y=1,sig_Z=1,dependency=0,beta1=1,kernel_X='constant',kernel_Y='constant',n_quant=128,seed=1):
        # Set parameters
        self.n_quant = n_quant
        self.sig_X = sig_X
        self.sig_Y = sig_Y
        self.sig_Z = sig_Z
        self.dependency = np.array(dependency)
        self.beta1 = beta1

        # Create kernels
        k_X = self.create_kernel(kernel_X)
        k_Y = self.create_kernel(kernel_Y)
        self.surface_X = self.kernel_surface(k_X,n_quant)
        self.surface_Y = self.kernel_surface(k_Y,n_quant)
        self.rng = np.random.default_rng(seed)


    def create_kernel(self,kernel_name):
        if kernel_name == 'zero':
            return lambda grid: np.zeros_like(grid[0])
        if kernel_name == 'constant':
            return lambda grid: np.ones_like(grid[0])
        if kernel_name == 'gaussian':
            return lambda grid: np.exp(-2*((grid[1]-grid[0])**2))
        if kernel_name == 'sine':
            return lambda grid: np.sin(4*grid[1]-20*grid[0])


    def kernel_surface(self,kernel,n_quant=128):
        interval =  np.linspace(0, 1, n_quant)
        kernel_grid = kernel(np.meshgrid(interval,interval))

        return np.tril(kernel_grid).transpose() #(s,t)


    def sample_Z_cont(self,n_sample=100,n_quant=128):
        gamma = self.rng.standard_normal(size=(4,n_sample))
        grid =  np.linspace(0,1,n_quant).reshape((n_quant,1)) @ np.ones((1,n_sample))
        z = gamma[0] + grid*gamma[1] + np.sin(6.5*gamma[3]*grid)
        noise = np.cumsum(self.rng.normal(scale=self.sig_Z/n_quant,size=(n_sample,n_quant)),axis=1)    

        return z.transpose() + noise # (n_sample, n_quant)


    def sample_historic(self,Z,surface,sig=1,drift=0):
        n_sample, n_quant = Z.shape
        integral = Z @ surface
        noise = np.cumsum(self.rng.normal(loc=drift,scale=sig/np.sqrt(n_quant), size=(n_sample,n_quant)),axis=1)

        return integral/n_quant + noise


    def sample_ZXY(self,n_sample=100):
        Z = self.sample_Z_cont(n_sample,self.n_quant)
        X = self.sample_historic(Z,self.surface_X,self.sig_X)
        Y = self.sample_historic(Z,self.surface_Y,self.sig_Y)
        
        if (self.dependency>0).any():
            Y += self.dependency*X / np.sqrt(n_sample)

        return Z,X,Y

    def _solve_eq(self,eq):
        return np.searchsorted(eq,0)

    def _sample_tau(self,Z,Y,baseline=lambda t: t**2,link=np.exp):
        n_sample, n_quant = Z.shape
        T = np.linspace(0,1,n_quant)
        intensity = baseline(T)*link(self.beta1*Z+Y)

        E = np.random.exponential(size = (n_sample,1))@np.ones((1,n_quant))
        equations = np.cumsum(intensity,axis=1)/n_quant - E        
        tau = np.apply_along_axis(func1d=self._solve_eq,axis=1,arr=equations)

        return tau

    def scale_and_set_baseline(self,baseline=lambda t: t**2,link=np.exp,print_scale=False,initial=1):
        Z,X,Y = self.sample_ZXY(1000)

        scale_coeff = initial
        scaled_baseline = lambda t: scale_coeff * baseline(t)
        tau = self._sample_tau(Z,Y,scaled_baseline,link)

        for _ in range(500):
            if np.mean(tau==self.n_quant)*self.n_quant < 1:
                break
            scale_coeff += 1
            tau = self._sample_tau(Z,Y,scaled_baseline,link)
        if print_scale:
            print("Scaled baseline coefficient:", scale_coeff)
        self.baseline = scaled_baseline

    def sample_all(self,n_sample=1000):
        Z,X,Y = self.sample_ZXY(n_sample)

        if not hasattr(self, 'baseline'):
            self.scale_and_set_baseline()
        
        tau = self._sample_tau(Z,Y,self.baseline)
        
        return X,Y,Z,tau

    def _plot_sample(self):
        X,Y,Z,tau = self.sample_all()

        plt.subplot(221); plt.plot(Z[0:10].transpose())
        plt.subplot(222); plt.plot(X[0:10].transpose())
        plt.subplot(223); plt.plot(Y[0:10].transpose())
        plt.subplot(224); plt.hist(tau)
        plt.show()



def format_data(tau,Z): # ,include_hazard=False,hazard=None):
    subject,Z_flat,t_start,t_end,delta= [],[],[],[],[]
    # true_hazard = []
    n_quant = Z.shape[1]
    tau_pos = tau[tau>0]

    for ii, death in enumerate(tau_pos):
        subject += death * [ii+1]
        delta += (death - 1) * [0] + [1] if death<n_quant else n_quant * [0]
        Z_flat.append(Z[ii,:death])
        t_start.append(np.arange(death)/n_quant)
        t_end.append(np.arange(1,death+1)/n_quant)

    data = pd.DataFrame({"subject":subject,
                         "t_start":np.concatenate(t_start),
                         "t_end":np.concatenate(t_end),
                         "X_0":np.concatenate(Z_flat), # should be called Z_0, but boxhed had issues with that
                         "delta":delta})
    return data

def format_data_with_X(X,Z,tau):
    subject,Z_flat,X_flat,t_start,t_end,delta= [],[],[],[],[],[]
    n_quant = Z.shape[1]
    tau_pos = tau[tau>0]

    for ii,death in enumerate(tau_pos):
        subject += death * [ii+1]
        delta += (death - 1) * [0] + [1] if death<n_quant else n_quant * [0]
        Z_flat.append(Z[ii,:death])
        X_flat.append(X[ii,:death])
        t_start.append(np.arange(death)/n_quant)
        t_end.append(np.arange(1,death+1)/n_quant)

    data = pd.DataFrame({"subject":subject,
                         "t_start":np.concatenate(t_start),
                         "t_end":np.concatenate(t_end),
                         "Z":np.concatenate(Z_flat),
                         "X":np.concatenate(X_flat),
                         "delta":delta})
    return data