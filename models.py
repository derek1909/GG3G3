from inference import *
import numpy as np
import numpy.random as npr
import scipy.stats as stats
import matplotlib.pyplot as plt

LOG_EPS = 1e-16


def lo_histogram(x, bins):
    """
    Left-open version of np.histogram with left-open bins covering the interval (left_edge, right_edge]
    (np.histogram does the opposite and treats bins as right-open.)
    Input & output behaviour is exactly the same as np.histogram
    """
    out = np.histogram(-x, -bins[::-1])
    return out[0][::-1], out[1:]


def gamma_isi_point_process(rate, shape):
    """
    Simulates (1 trial of) a sub-poisson point process (with underdispersed inter-spike intervals relative to Poisson)
    :param rate: time-series giving the mean spike count (firing rate * dt) in different time bins (= time steps)
    :param shape: shape parameter of the gamma distribution of ISI's
    :return: vector of spike counts with same shape as "rate".
    """
    sum_r_t = np.hstack((0, np.cumsum(rate)))
    gs = np.zeros(2)
    while gs[-1] < sum_r_t[-1]:
        gs = np.cumsum( npr.gamma(shape, 1 / shape, size=(2 + int(2 * sum_r_t[-1]),)) )
    y, _ = lo_histogram(gs, sum_r_t)

    return y

def emit(dt, rate, GammaShape=None):
    """
    emit spikes based on rates
    :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
    :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
             trials and time bins.
    """
    if GammaShape is None:
        # poisson spike emissions
        y = npr.poisson(rate * dt)
    else:
        # sub-poisson/underdispersed spike emissions
        y = gamma_isi_point_process(rate * dt, GammaShape)

    return y


class StepModel():
    """
    Simulator of the Stepping Model of Latimer et al. Science 2015.
    """
    def __init__(self, m=50, r=10, x0=0.2, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        """
        Simulator of the Stepping Model of Latimer et al. Science 2015.
        :param m: mean jump time (in # of time-steps). This is the mean parameter of the Negative Binomial distribution
                  of jump (stepping) time
        :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time
                  (Note that it is more customary to parametrise the NB distribution by its parameter p and r,
                  instead of m and r, where p is so-called "probability of success" (see Wikipedia). The two
                  parametrisations are equivalent and one can go back-and-forth via: m = r (1-p)/p and p = r / (m + r).)
        :param x0: determines the pre-jump firing rate, via  R_pre = x0 * Rh (see below for Rh)
        :param Rh: firing rate of the "up" state (the same as the post-jump state in most of the project tasks)
        :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                            see https://en.wikipedia.org/wiki/Gamma_distribution
        :param Rl: firing rate of the post-jump "down" state (rarely used)
        :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
        """
        self.m = m
        self.r = r
        self.x0 = x0

        self.p = r / (m + r)

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt


    @property
    def params(self):
        return self.m, self.r, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl


    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                 trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y


    def simulate(self, Ntrials=1, T=100, get_rate=True, GammaShape = None):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        if GammaShape != None:
            self.isi_gamma_shape = GammaShape
            
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        ts = np.arange(T)

        spikes, jumps, rates = [], [], []
        for tr in range(Ntrials):
            # sample jump time
            jump = npr.negative_binomial(self.r, self.p)
            jumps.append(jump) # (unit: 1/T s )

            # first set rate at all times to pre-step rate
            rate = np.ones(T) * self.x0 * self.Rh #=R0
            # then set rates after jump to self.Rh
            rate[ts >= jump] = self.Rh
            rates.append(rate)

            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates)
        else:
            return np.array(spikes), np.array(jumps)

        
        
        
        
    def simulate_HMM_inhomo(self, Ntrials=1, T=100, get_rate=True, GammaShape = None):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        if GammaShape != None:
            self.isi_gamma_shape = GammaShape
            
            
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        ts = np.arange(T)
        # Inhomogeneous markov chain
        PMF_jump = stats.nbinom.pmf(ts, self.r, self.p)
        CMF_jump = stats.nbinom.cdf(ts, self.r, self.p)
        
        trans_matrices = np.empty((T-1,2,2)) # inhomogeneous transition matrix

        for t in range(0,T-1):
            if CMF_jump[t]==1:
                trans_matrices[t] = np.array([[0, 1], [0,1]])
            else:
                trans_matrices[t] = np.array([[ (1-CMF_jump[t+1]) / (1-CMF_jump[t]), PMF_jump[t+1]/(1-CMF_jump[t]) ], [0,1]])
            if np.any(trans_matrices[t] < 0):
                print("invalid probability: negative value!")
                print(trans_matrices[t])
                print(f"t={t},m={self.m}, r={self.r}")
        # sample the first state (t=0)
        p0 = np.array([1-PMF_jump[0],PMF_jump[0]]) 
        # logProb of t=0
        log_pi0 = np.log(p0)
        #log trans matrix
        log_trans_matrices = np.log(trans_matrices)
        
        states = np.zeros(T, dtype=int) 
        spikes, jumps, rates = [], [], []
        for tr in range(Ntrials):
            # sample jump time
 
            jump = 0
            # Simulate the chain
            states[0] = np.random.choice(2, size=None, replace=True, p=p0)
            
            # sample all other (T-1) states
            for t in range(1, T):
                # The transition probabilities depend on the current state
                states[t] = np.random.choice(2, size=None, replace=True, p=trans_matrices[t-1, states[t-1]])
                jump = t
                if states[t]==1:
                    jumps.append(jump) # (unit: 1/T s )
                    break
                elif t == T-1:
                    jumps.append(jump+1) # Not jumped during 0:T-1

            # first set rate at all times to pre-step rate
            rate = np.ones(T) * self.x0 * self.Rh #=R0
            # then set rates after jump to self.Rh
            rate[ts >= jump] = self.Rh
            rates.append(rate)

            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates), log_trans_matrices, log_pi0
        else:
            return np.array(spikes), np.array(jumps), log_trans_matrices, log_pi0
  
        
#     def simulate_HMM_homo(self, Ntrials=1, T=100, get_rate=True):
#         """
#         :param Ntrials: (int) number of trials
#         :param T: (int) duration of each trial in number of time-steps.
#         :param get_rate: whether or not to return the rate time-series
#         :return:
#         spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
#                 an array of spike counts in each time-bin (= time step)
#         jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
#         rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
#         """
#         # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
#         dt = 1 / T
#         self.dt = dt

#         ts = np.arange(T)
        
#         spikes, jumps, rates = [], [], []
#         for tr in range(Ntrials):
#             # sample jump time
#             # homogeneous markov chain
           
#             # Find transition matrix for r and p
#             trans_matrix = np.zeros((self.r+1,self.r+1))
#             trans_matrix[0, 0:2] = [1-self.p, self.p] # Set the first row
#             for i in range(1, self.r): # Create the remaining rows
#                 trans_matrix[i, i:i+2] = [1-self.p, self.p]
#             trans_matrix[self.r , self.r ] = 1 # Set the last row
#             # Find initial distribution of latent state 
#             pi = np.zeros(self.r+1)
#             pi[0] = 1
#             p0 = np.matmul(pi, trans_matrix)
#             # 1xT matrix to record states. Initial state = 0
#             states = np.zeros(self.r+T, dtype=int) 
#             states[0] = np.random.choice((self.r+1), size=None, replace=True, p=p0)

#             # sample all other (T-1) states
#             jump=0
#             for t in range(1, T+self.r):
#                 # The transition probabilities depend on the current state
#                 states[t] = np.random.choice((self.r+1), size=None, replace=True, p=trans_matrix[states[t-1]])
#                 jump = t-self.r
#                 if states[t]==self.r:
#                     jumps.append(jump) #this is the jump time (ms)
#                     states[t:] = self.r
#                     break
#                 elif t == T+self.r-1:
#                     jumps.append(jump+T) # Not jumped during 0:T-1

                    
           
#             # first set rate at all times to pre-step rate
#             rate = np.ones(T) * self.x0 * self.Rh #=R0
#             # then set rates after jump to self.Rh
#             rate[ts >= jump] = self.Rh
#             rates.append(rate)

#             spikes.append(self.emit(rate))
            
#             # Prob of t=0
#             log_pi0 = np.log(p0+LOG_EPS)
#             # log tran matrix
#             log_trans_matrix = np.log(trans_matrix+LOG_EPS)

            
#         if get_rate:
#             return np.array(spikes), np.array(jumps), np.array(rates), log_trans_matrix, log_pi0, states
#         else:
#             return np.array(spikes), np.array(jumps), log_trans_matrix, log_pi0, states
   
    def simulate_HMM_homo(self, Ntrials=1, T=100, get_rate=True, GammaShape = None):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        if GammaShape != None:
            self.isi_gamma_shape = GammaShape
            
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        ts = np.arange(T)
        p = self.r/(self.m+self.r)
        
        # homogeneous markov chain

        # Find transition matrix for r and p
        trans_matrix = np.zeros((self.r+1,self.r+1))
        for i in range(self.r+1): # axis=0
            for j in range(i, self.r+1): # axis=1
                trans_matrix[i, j] = (p**(j-i)) * (1-p) if j < self.r else p**(self.r-i)

        # Find initial distribution of latent state 
        pi = np.zeros(self.r+1)
        pi[0] = 1
        p0 = np.matmul(pi, trans_matrix)

        # Prob of t=0
        log_pi0 = np.log(p0+LOG_EPS)
        # log tran matrix
        log_trans_matrix = np.log(trans_matrix)
            
        states = np.zeros(T, dtype=int) 
        spikes, jumps, rates = [], [], []
        for tr in range(Ntrials):
            # sample jump time

            # 1xT matrix to record states. Initial state = 0
            states[0] = np.random.choice((self.r+1), size=None, replace=True, p=p0)

            # sample all other (T-1) states
            jump=0
            for t in range(1, T):
                # The transition probabilities depend on the current state
                states[t] = np.random.choice((self.r+1), size=None, replace=True, p=trans_matrix[states[t-1]])
                jump = t
                if states[t]==self.r:
                    jumps.append(jump) #this is the jump time (ms)
                    states[t:] = self.r
                    break
                elif t == T-1:
                    jumps.append(jump+T) # Not jumped during 0:T-1

                    
           
            # first set rate at all times to pre-step rate
            rate = np.ones(T) * self.x0 * self.Rh #=R0
            # then set rates after jump to self.Rh
            rate[ts >= jump] = self.Rh
            rates.append(rate)

            spikes.append(self.emit(rate))
            

            
        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates), log_trans_matrix, log_pi0, states
        else:
            return np.array(spikes), np.array(jumps), log_trans_matrix, log_pi0, states
        

    def simulate_HMM_2states(self, Ntrials=1, T=100, get_rate=True, GammaShape = None):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        
        if GammaShape != None:
            self.isi_gamma_shape = GammaShape
            
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        ts = np.arange(T)
        p = self.r/(self.m+self.r)
        
        # homogeneous markov chain

        # Find transition matrix for r and p
        trans_matrix = np.array([[1- 1/self.m, 1/self.m],[0,1]])
       


        # Find initial distribution of latent state 
        pi = [1,0] # initial state

        p0 = np.matmul(pi, trans_matrix)

        # Prob of t=0
        log_pi0 = np.log(p0+LOG_EPS)
        # log tran matrix
        log_trans_matrix = np.log(trans_matrix)
            
        states = np.zeros(T, dtype=int) 
        spikes, jumps, rates = [], [], []
        for tr in range(Ntrials):
            # sample jump time

            # 1xT matrix to record states. Initial state = 0
            states[0] = np.random.choice(2, size=None, replace=True, p=p0)

            # sample all other (T-1) states
            jump=0
            for t in range(1, T):
                # The transition probabilities depend on the current state
                states[t] = np.random.choice(2, size=None, replace=True, p=trans_matrix[states[t-1]])
                jump = t
                if states[t]==1:
                    jumps.append(jump) #this is the jump time (ms)
                    states[t:] = 1
                    break
                elif t == T-1:
                    jumps.append(jump+T) # Not jumped during 0:T-1

                    
           
            # first set rate at all times to pre-step rate
            rate = np.ones(T) * self.x0 * self.Rh #=R0
            # then set rates after jump to self.Rh
            rate[ts >= jump] = self.Rh
            rates.append(rate)

            spikes.append(self.emit(rate))
            

            
        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates), log_trans_matrix, log_pi0, states
        else:
            return np.array(spikes), np.array(jumps), log_trans_matrix, log_pi0, states
         
        
        
        
class RampModel():
    """
    Simulator of the Ramping Model (aka Drift-Diffusion Model) of Latimer et al., Science (2015).
    """
    def __init__(self, beta=0.5, sigma=0.2, x0=.2, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        """
        Simulator of the Ramping Model of Latimer et al. Science 2015.
        :param beta: drift rate of the drift-diffusion process
        :param sigma: diffusion strength of the drift-diffusion process.
        :param x0: average initial value of latent variable x[0]
        :param Rh: the maximal firing rate obtained when x_t reaches 1 (corresponding to the same as the post-step
                   state in most of the project tasks)
        :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                            see https://en.wikipedia.org/wiki/Gamma_distribution
        :param Rl: Not implemented. Ignore.
        :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
        """
        self.beta = beta
        self.sigma = sigma
        self.x0 = x0

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt


    @property
    def params(self):
        return self.mu, self.sigma, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl


    def f_io(self, xs, b=None):
        if b is None:
            return self.Rh * np.maximum(0, xs)
        else:
            return self.Rh * b * np.log(1 + np.exp(xs / b))


    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                 trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y


    def simulate(self, Ntrials=1, T=100, get_rate=True, GammaShape = None):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        
        if GammaShape != None:
            self.isi_gamma_shape = GammaShape
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

       # simulate all trials in parallel (using numpy arrays and broadcasting)

        # first, directly integrate/sum the drift-diffusion updates
        # x[t+1] = x[t] + β dt + σ √dt * randn (with initial condition x[0] = x0 + σ √dt * randn)
        # to get xs in shape (Ntrials, T):
        ts = np.arange(T)
        xs = self.x0 + self.beta * dt * ts + self.sigma * np.sqrt(dt) * np.cumsum(npr.randn(Ntrials, T), axis=1)
        # in each trial set x to 1 after 1st passage through 1; padding xs w 1 assures passage does happen, possibly at T+1
        taus = np.argmax(np.hstack((xs, np.ones((xs.shape[0],1)))) >= 1., axis=-1)
        xs = np.where(ts[None,:] >= taus[:,None], 1., xs)
        # # the above 2 lines are equivalent to:
        # for x in xs:
        #     if np.sum(x >= 1) > 0:
        #         tau = np.nonzero(x >= 1)[0][0]
        #         x[tau:] = 1

        rates = self.f_io(xs) # shape = (Ntrials, T)

        spikes = np.array([self.emit(rate) for rate in rates]) # shape = (Ntrial, T)

        if get_rate:
            return spikes, xs, rates
        else:
            return spikes, xs

    def simulate_HMM(self, Ntrials=1, T=100, K=100, get_rate=True, GammaShape = None):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param K: (int) number of states of the HMM
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        
        if GammaShape != None:
            self.isi_gamma_shape = GammaShape
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt
        ts = np.arange(T)
        
        
        ## transition matrix
        # calculate difference between states, forming a K x K matrix.
        st = np.arange(K) # states
        s_grid = st - st.reshape(-1,1) 
        mu = self.beta * dt * (K-1)
        std = self.sigma * np.sqrt(dt) * (K-1)  + LOG_EPS # To avoid error when sigma = 0

        # Suppose trans_matrix is in log space
        log_trans_matrix = np.log(stats.norm.cdf(s_grid+0.5, mu, std) - stats.norm.cdf(s_grid-0.5, mu, std))
        # First column
        log_trans_matrix[:,0] = stats.norm.logcdf(s_grid[:,0]+0.5, mu, std)
        # Last column
        log_trans_matrix[:,-1] = stats.norm.logsf(s_grid[:,-1]-0.5, mu, std)
        # Last row
        log_trans_matrix[-1] = -np.inf
        log_trans_matrix[-1,-1] = 0

        # Normalization is not necessary
        #Compute row sums in log space using logsumexp
        log_row_sums = np.zeros(log_trans_matrix.shape[0]) 
        for i in range(log_trans_matrix.shape[0]):
            log_row_sums[i] = logsumexp(log_trans_matrix[i, :])
            
        #Subtract row sums from each element in log space to normalize
        normalized_log_trans_matrix = log_trans_matrix - log_row_sums[:, np.newaxis]
        trans_matrix = np.exp(normalized_log_trans_matrix)
        
#             # normalise each row
#             row_sums = trans_matrix.sum(axis=1)
#             trans_matrix = trans_matrix / row_sums[:, np.newaxis]

        mu = self.x0 * (K-1)
        std = self.sigma * np.sqrt(dt) * (K-1)  + LOG_EPS # To avoid error when sigma = 0
        #Suppose trans_matrix is in log space
        log_pi = np.log(stats.norm.cdf(st+0.5, mu, std) - stats.norm.cdf(st-0.5, mu, std))
        log_pi[0] = stats.norm.logcdf(st[0]+0.5, mu, std)
        log_pi[-1] = stats.norm.logsf(st[-1]-0.5, mu, std) # sf = 1-cdf
        #Compute row sums in log space using logsumexp
        log_pi_sums = logsumexp(log_pi)
        #Subtract row sums from each element in log space to normalize
        normalized_log_pi = log_pi - log_pi_sums
        # recover the pi
        pi = np.exp(normalized_log_pi)

        normalized_log_pi0 = np.log(np.matmul(pi,trans_matrix))
        # NxT matrix to record states
        states = np.zeros((Ntrials, T), dtype=int)
        for n in range(Ntrials):
            # Draw the initial state from the initial distribution
            # K states, pi initial distribution. => return a scaler with initial values (discrete)
            states[n,0] = np.random.choice(K, p=pi)

            # Simulate the chain
            for t in range(1, T):
                # The transition probabilities depend on the current state
                current_state = states[n,t-1]
                states[n,t] = np.random.choice(K, p=trans_matrix[current_state])

        xs = states / (K-1)
        
        # in each trial set x to 1 after 1st passage through 1; padding xs w 1 assures passage does happen, possibly at T+1
#         taus = np.argmax(np.hstack((xs, np.ones((xs.shape[0],1)))) >= 1., axis=-1)
#         xs = np.where(ts[None,:] >= taus[:,None], 1., xs)

        rates = self.f_io(xs) # shape = (Ntrials, T)

        spikes = np.array([self.emit(rate) for rate in rates]) # shape = (Ntrial, T)

        if get_rate:
            return spikes, xs, rates, normalized_log_trans_matrix, normalized_log_pi0
        else:
            return spikes, xs, normalized_log_trans_matrix, normalized_log_pi0


    def simulate_HMM_ns(self, Ntrials=1, num_ns=30, T=100, K=100, get_rate=True, GammaShape = None):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param K: (int) number of states of the HMM
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        
        if GammaShape != None:
            self.isi_gamma_shape = GammaShape
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt
        ts = np.arange(T)
        
 #        num_ns = 30 number of negative states = num_ns * K
        ## transition matrix
        # calculate difference between states, forming a K x K matrix.
        st = np.arange(-num_ns*K,K) # states
        s_grid = st - st.reshape(-1,1) 
        mu = self.beta * dt * (K-1)
        std = self.sigma * np.sqrt(dt) * (K-1)  + LOG_EPS # To avoid error when sigma = 0
        
        # Suppose trans_matrix is in log space
        log_trans_matrix = np.log(stats.norm.cdf(s_grid+0.5, mu, std) - stats.norm.cdf(s_grid-0.5, mu, std))
        # First column
        log_trans_matrix[:,0] = stats.norm.logcdf(s_grid[:,0]+0.5, mu, std)
        # Last column
        log_trans_matrix[:,-1] = stats.norm.logsf(s_grid[:,-1]-0.5, mu, std)
        # Last row
        log_trans_matrix[-1] = -np.inf
        log_trans_matrix[-1,-1] = 0

        # Normalization is not necessary
        #Compute row sums in log space using logsumexp
        log_row_sums = np.zeros(log_trans_matrix.shape[0]) 
        for i in range(log_trans_matrix.shape[0]):
            log_row_sums[i] = logsumexp(log_trans_matrix[i, :])
            
        #Subtract row sums from each element in log space to normalize
        normalized_log_trans_matrix = log_trans_matrix - log_row_sums[:, np.newaxis]
        trans_matrix = np.exp(normalized_log_trans_matrix)
        
#             # normalise each row
#             row_sums = trans_matrix.sum(axis=1)
#             trans_matrix = trans_matrix / row_sums[:, np.newaxis]

        mu = self.x0 * (K-1)
        std = self.sigma * np.sqrt(dt) * (K-1)  + LOG_EPS # To avoid error when sigma = 0
        #Suppose trans_matrix is in log space
        log_pi = np.log(stats.norm.cdf(st+0.5, mu, std) - stats.norm.cdf(st-0.5, mu, std))
        log_pi[0] = stats.norm.logcdf(st[0]+0.5, mu, std)
        log_pi[-1] = stats.norm.logsf(st[-1]-0.5, mu, std) # sf = 1-cdf

        #Compute normalized_log_pi in log space using logsumexp
        normalized_log_pi = log_pi - logsumexp(log_pi)
        # recover the pi
        pi = np.exp(normalized_log_pi)
        
        normalized_log_pi0 = np.log(np.matmul(pi,trans_matrix))


        # NxT matrix to record states
        states = np.zeros((Ntrials, T), dtype=int)
        for n in range(Ntrials):
            # Draw the initial state from the initial distribution
            # K states, pi initial distribution. => return a scaler with initial values (discrete)
            states[n,0] = np.random.choice(st, p=pi)
            # Simulate the chain
            for t in range(1, T):
                # The transition probabilities depend on the current state
                current_state = states[n,t-1]
                states[n,t] = np.random.choice(st, p=trans_matrix[current_state+num_ns*K])

        xs = states / (K-1)
        
        # in each trial set x to 1 after 1st passage through 1; padding xs w 1 assures passage does happen, possibly at T+1
#         taus = np.argmax(np.hstack((xs, np.ones((xs.shape[0],1)))) >= 1., axis=-1)
#         xs = np.where(ts[None,:] >= taus[:,None], 1., xs)

        rates = self.f_io(xs) # shape = (Ntrials, T)

        spikes = np.array([self.emit(rate) for rate in rates]) # shape = (Ntrial, T)

        if get_rate:
            return spikes, xs, rates, normalized_log_trans_matrix, normalized_log_pi0, states
        else:
            return spikes, xs, normalized_log_trans_matrix, normalized_log_pi0, states
        
