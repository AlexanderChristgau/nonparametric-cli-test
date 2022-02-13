import numpy as np
import pandas as pd

rng = np.random.default_rng()

'''
Functions for simulating the Cox example
'''

# # Sample noisy sinusoidal Zt (time-continuous)
# def sample_Z_cont(n_sample=100,n_quant=128):
#     gamma = rng.standard_normal(size=(4,n_sample))
#     grid =  np.linspace(0,1,n_quant).reshape((n_quant,1)) @ np.ones((1,n_sample))
#     z = gamma[0] + grid*gamma[1] + np.sin(6.5*gamma[3]*grid) # 2pi approx 6.5
#     noise = np.cumsum(rng.normal(scale=4/n_quant,size=(n_sample,n_quant)),axis=1)    

#     return z.transpose() #+ noise # (n_sample, n_quant)



# #Sample a jump process Zt
# def sample_Z_jump(n_sample=100,n_quant=128,max_jumps=10):
#     n_jumps = rng.integers(1,max_jumps, size=n_sample)
#     z_vals = np.cumsum(rng.uniform(-1,1,size=(n_sample,max_jumps)),1)    
#     z = np.zeros((n_sample,n_quant))
    
#     T = np.ones((n_sample,1)) @ np.arange(n_quant-2).reshape(1,n_quant-2);
#     T_perm = rng.permuted(T,axis=1).astype(int)
#     splits = [np.concatenate(([0],np.sort(T_perm[i,0:n_jump]),[n_quant])) for i,n_jump in enumerate(n_jumps)]
    
#     for i,split in enumerate(splits):
#         for j,(s1,s2) in enumerate(zip(split[:-1],split[1:])):
#             z[i,s1:s2] = z_vals[i,j]
    
#     return z


# # create kernels for the historical linear model
# def create_kernel(kernel_name = 'constant'):
#     if kernel_name == 'constant':
#         return lambda grid: np.ones_like(grid[0])
#     if kernel_name == 'exp':
#         return lambda grid: np.exp((grid[1]-grid[0])**2)
#     if kernel_name == 'sine':
#         return lambda grid: np.sin(4*grid[1]-20*grid[0])


# # Kernel evaluated in timegrid (s,t): 0<=s<=t.
# def kernel_surface(kernel,n_quant=128):
#     interval =  np.linspace(0, 1, n_quant)
#     beta_grid = kernel(np.meshgrid(interval,interval))
#     return np.tril(beta_grid).transpose() #(s,t)



# def sample_historic(Z,kernel,sig=1,drift=0):
#     n_sample, n_quant = Z.shape
#     integral = Z @ kernel_surface(kernel,n_quant) 
#     noise = np.cumsum(rng.normal(loc=drift,scale=sig, size=(n_sample,n_quant)),axis=1)
#     return (integral + noise)/n_quant



# def sample_ZXY(kernel_X,kernel_Y,sig_X=5,sig_Y=5,n_sample=100, n_quant=128,Z_jump=False):
#     Z = sample_Z_jump(n_sample,n_quant) if Z_jump else sample_Z_cont(n_sample,n_quant)
#     X = sample_historic(Z,kernel_X,sig_X)
#     Y = sample_historic(Z,kernel_Y,sig_Y)
#     return Z,X,Y



class cox_sampler:
    def __init__(self,sig_X=5,sig_Y=5,sig_Z=0,dependency=0,beta1=1,kernel_X='constant',kernel_Y='constant',n_quant=128,seed=1):
        # Set parameters
        self.n_quant = n_quant
        self.sig_X = sig_X
        self.sig_Y = sig_Y
        self.sig_Z = sig_Z
        self.dependency = dependency
        self.beta1 = beta1

        # Create kernels
        k_X = self.create_kernel(kernel_X)
        k_Y = self.create_kernel(kernel_Y)
        self.surface_X = self.kernel_surface(k_X,n_quant)
        self.surface_Y = self.kernel_surface(k_Y,n_quant)
        self.rng = np.random.default_rng(seed)


    def create_kernel(self,kernel_name):
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
        noise = np.cumsum(self.rng.normal(loc=drift,scale=sig, size=(n_sample,n_quant)),axis=1)

        return (integral + noise)/n_quant


    def sample_ZXY(self,n_sample=100):
        Z = self.sample_Z_cont(n_sample,self.n_quant)
        X = self.sample_historic(Z,self.surface_X,self.sig_X)
        Y = self.sample_historic(Z,self.surface_Y,self.sig_Y)
        
        if self.dependency>0:
            Y += self.dependency*X / np.sqrt(n_sample)

        return Z,X,Y

    def _solve_eq(self,eq):
        return np.searchsorted(eq,0)

    def _sample_tau(self,Z,Y,baseline=lambda t: t**2,link=np.exp):
        n_sample, n_quant = Z.shape
        T = np.linspace(0,1,n_quant)
        intensity = link(baseline(T)*np.exp(self.beta1*Z+Y))

        E = np.random.exponential(size = (n_sample,1))@np.ones((1,n_quant))
        equations = np.cumsum(intensity,axis=1)/n_quant - E        
        tau = np.apply_along_axis(func1d=self._solve_eq,axis=1,arr=equations)

        return tau

    def scale_and_set_baseline(self,baseline=lambda t: t**2,link=np.exp,print_scale=False):
        Z,X,Y = self.sample_ZXY(1000)

        scale_coeff = 1
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




### Sample the survival time from the cumulated hazard
# def solve_eq(eq): return np.searchsorted(eq,0)

# def sample_tau(Z,Y,baseline=lambda t: 15*(t**2),
#                 link=lambda z: np.exp(np.minimum(z, 1) - 1)*(z<1) + z*(z>=1),
#                 beta_1 = 1):
#     n_sample, n_quant = Z.shape
#     T = np.linspace(0,1,n_quant)
#     intensity = link(np.exp(beta_1*Z+Y) * baseline(T))
    
#     E = np.random.exponential(size = (n_sample,1))@np.ones((1,n_quant))
#     equations = np.cumsum(intensity,axis=1)/n_quant - E
    
#     tau = np.apply_along_axis(func1d=solve_eq,axis=1,arr=equations)
    
#     # if return_hazard:
#     #     for i,t in enumerate(tau):
#     #         intensity[i,t:] = np.zeros(n_quant-t)
#     #     return tau, intensity
#     return tau


### format training data for boxhed2.0
# def format_data_subsample(tau,Z,splits):
#     subject,Z_flat,t_start,t_end,delta, = [],[],[],[],[]
#     n_quant = Z.shape[1]
#     tau_pos = tau[tau>0]
#     splits_pos = np.array(splits)[tau>0,...]

#     for ii,(death,split) in enumerate(zip(tau_pos,splits_pos)):
#         for s1,s2 in zip(split[:-1],split[1:]):
#             subject.append(ii+1)
#             Z_flat.append(Z[ii,s1])
#             t_start.append(s1/n_quant)
#             t_end.append(s2/n_quant)
#             if s1 <= death < s2:
#                 delta.append(1)
#                 break
#             else:
#                 delta.append(0)

#     data = pd.DataFrame({"subject":subject,
#                             "t_start":t_start,
#                             "t_end":t_end,
#                             "X_0":Z_flat,
#                             "delta":delta})
#     return data



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