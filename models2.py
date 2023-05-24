import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from models import *

def classifier_tester(M=100, N=400, T=100, classifier="var", thresholds=3):
    
    accuracies = np.zeros((2,3,3)) # accuracies[0,:] is for first sigma range
    r_list = [-1, 0, 1]
    s_list = [-1, 0, 1]
    
    
    
    for ii in [0, 1, 2]: ## for sigma
        for jj in [0, 1, 2]: ## for r
            smin = s_list[ii]
            rmin = r_list[jj]

            data_points = generate_test_spike_trains(M=M, N=N, T=T, rmin=rmin, rmax=rmin+1, bmin=-1, bmax=1, smin=smin, smax=smin+1, mmin=10, mmax=90, xmin=0.1, xmax=0.8)
            # r, b, s are controled by exponent
            # m, x are generated linearily
    
            if classifier == "var":
                predictions, _, _= var_classifier(data_points, thresholds)
                
            print(f"{ii*3+jj+1}/9 parameter group")
            
            accuracy = (np.sum(1-predictions[0]) + np.sum(predictions[1]))/(2*M)
            accuracies[0,ii,jj] = np.sum(1-predictions[0])/M # Detection accuracy for step models
            accuracies[1,ii,jj] = np.sum(predictions[1])/M # Detection accuracy for step models    
    return accuracies

def generate_test_spike_trains(M=20, N=400, T=100, rmin=0, rmax=1, bmin=0.5, bmax=2, smin=0.1, smax=1, mmin=10, mmax=90, xmin=0, xmax=1):
    """
    Generate M data points for both ramp model and step model
    :param M: number of data points for each model
    :param N: number of trls per data point
    :param T: duration of each trial in number of time-steps
    :param m: mean jump time (in # of time-steps) for StepModel
    :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time for StepModel
    :param sigma: diffusion strength of the drift-diffusion process for RampModel
    :param beta: drift rate of the drift-diffusion process for RampModel
    :return: A matrix with dim (2, M, N, T)
    """
    
       
    # Generate M data points
    data_points = np.empty((2, M, N, T))  # for an n x m array
    for MM in range(M):
  # Initialize random model parameters
        m = npr.uniform(mmin, mmax)
        
        b = npr.uniform(bmin,bmax)
        s = npr.uniform(smin,smax)
        xr = npr.uniform(xmin,xmax)
        xs = npr.uniform(xmin,xmax)
        
        r_exponent = npr.uniform(rmin,rmax)
        r = 10**r_exponent
        b_exponent = npr.uniform(bmin,bmax)
        b = 10**b_exponent
        s_exponent = npr.uniform(bmin,bmax)
        s = 10**s_exponent
        
        #initialise models
        step_model = StepModel(m=m, r=r, x0=0.2, Rh=50);
        ramp_model = RampModel(beta=b, sigma=s);

        # Generate spike trains
        step_spikes, _, _ = step_model.simulate(Ntrials=N, T=T)
        ramp_spikes, _, _ = ramp_model.simulate(Ntrials=N, T=T)

        # Add spike trains to data points
        data_points[0,MM]=step_spikes
        data_points[1,MM]=ramp_spikes

  
    # Convert data_points to integer type to save memory
    data_points = data_points.astype(int)
    return data_points

def generate_spike_trains(M=20, N=400, T=100, m=50, r=10, sigma=0.2, beta=0.5):
    """
    Generate M data points for both ramp model and step model
    :param M: number of data points for each model
    :param N: number of trls per data point
    :param T: duration of each trial in number of time-steps
    :param m: mean jump time (in # of time-steps) for StepModel
    :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time for StepModel
    :param sigma: diffusion strength of the drift-diffusion process for RampModel
    :param beta: drift rate of the drift-diffusion process for RampModel
    :return: A matrix with dim (2, M, N, T)
    """
    # Initialize models
    step_model = StepModel(m=m, r=r, x0=0.2, Rh=50);
    ramp_model = RampModel(beta=beta, sigma=sigma);

    # Generate M data points
    data_points = np.empty((2, M, N, T))  # for an n x m array
    for MM in range(M):
        # Generate spike trains
        step_spikes, _, _ = step_model.simulate(Ntrials=N, T=T)
        ramp_spikes, _, _ = ramp_model.simulate(Ntrials=N, T=T)

        # Add spike trains to data points
        data_points[0,MM]=step_spikes
        data_points[1,MM]=ramp_spikes


    # Convert data_points to integer type to save memory
    data_points = data_points.astype(int)

    return data_points


def generate_raster_and_timestamps(spike_trains, plot=False):
    """
    Generate a raster plot and timestamps of the given spike trains.
    :param spike_trains: spike trains to plot (N by T matrix)
    :param plot: whether to plot the raster
    :return: spike trains timestamps
    """
    
    
    T = len(spike_trains[0])
    # Record time of spikes in milliseconds
    spike_trains_timestamp = []
    for spike_train in spike_trains:  # for each trial
        timestamp = []
#         print(spike_train)
        for ii in range(len(spike_train)):  # for each time point
#             print(spike_train[ii])
            for jj in range(spike_train[ii]):  # handle multiple spikes in a time stamp
                timestamp.append(ii*1e3/T)
        spike_trains_timestamp.append(timestamp)

    if plot:
        fig, ax = plt.subplots()
        fig.suptitle("Spike Raster Plot")
        colors = ['C{}'.format(i) for i in range(len(spike_trains))]  # different color for each set of neural data
        ax.eventplot(spike_trains_timestamp, colors=colors, linelengths=0.2)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xlabel("time from motion onset (ms)")
        ax.set_ylabel("spike trains")
        plt.show()

    return spike_trains_timestamp


def generate_psth(spike_trains, bin_size=20, bin_size_2=50, plot=False):
    """
    Generate a Peri-Stimulus Time Histogram (PSTH) from given timestamps.
    :param spike_trains: spike trains to plot (N by T matrix)
    :param bin_size: bin size for the PSTH (in milliseconds)
    :param bin_size2: a larger size to calculate the variance of psth (in milliseconds)
    :param plot: whether to plot the PSTH
    :return: averaged PSTH, smoothed_psth, variance, Fano factor
    """
    
    N = spike_trains.shape[0]; # number of trials
    #print(N)
    spike_trains_timestamp = generate_raster_and_timestamps(spike_trains); # timestamps of spike trains
    
    # Calculate the PSTH
    bin_edges = np.arange(0, 1e3 + bin_size, bin_size)
    psth, _ = np.histogram(np.concatenate(spike_trains_timestamp), bins=bin_edges)

    averaged_psth = (psth / bin_size * 1e3) / N # spikes per sec per trail

    # Apply Gaussian smoothing
    sigma = 1.5  # Standard deviation of the Gaussian filter
    gaussian_smoothed_psth = gaussian_filter(averaged_psth, sigma)

    # Calculate the PSTH for larger bins
    bin_edges_2 = np.arange(0, 1e3+bin_size_2, bin_size_2)
    psth_2, _ = np.histogram(np.concatenate(spike_trains_timestamp), bins=bin_edges_2)
    averaged_psth_2 = psth_2 / N # spikes per trail

    var_s = np.zeros_like(averaged_psth_2)

    # psth_matrix is a 2D numpy array where each row is a PSTH vector
    psth_matrix = np.zeros((len(spike_trains_timestamp), len(averaged_psth_2))); # (N x T)

    # Calculate the PSTH for each trail
    for ii in range(len(spike_trains_timestamp)):
        psth_matrix[ii], _ = np.histogram(np.array(spike_trains_timestamp[ii]), bins=bin_edges_2)

#     print(psth_matrix.shape)
    # Find the variance across trials (i.e., along the rows)
    var_s = np.var(psth_matrix, axis=0);

    ## Calculate Fano Factor ##

    fano_factors = var_s / averaged_psth_2


    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle("PSTH Diagram")

        # Plot the PSTH
        ax1.plot(bin_edges[:-1], averaged_psth,  label='Original')
        ax1.plot(bin_edges[:-1], gaussian_smoothed_psth,  label='Smoothed')
        ax2.plot(bin_edges_2[:-1], var_s,  label='Variance')
        ax3.plot(bin_edges_2[:-1], fano_factors,  label='Fano factor')

        ax1.set_ylabel("spike rate (sp/s)")
        ax2.set_ylabel("Variance")
        ax3.set_ylabel("Fano factor")
        ax3.set_xlabel("time from motion onset (ms)")
        ax1.legend()
        ax2.legend()
        plt.show()

    return averaged_psth, gaussian_smoothed_psth, var_s, fano_factors




def var_classifier(data_points, thresholds):
    """
    Classify spike trains as being generated by the step model (return 0) or the ramp model (return 1).
    :param data_points: 2*M data points, M data points for each model, where each data point is a (N by T) matrix
    :param m: mean jump time (in # of time-steps) for StepModel
    :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time for StepModel
    :param sigma: diffusion strength of the drift-diffusion process for RampModel
    :param beta: drift rate of the drift-diffusion process for RampModel
    :param threshold: threshold for variance
    :return: M predictions, each being 0 (step model) or 1 (ramp model)
    """
    predictions = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
    var_s = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
    fano_factors = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
    
    for ii in [0,1]:
        # ii = 0 -> STEP spike trains
        # ii = 1 -> RAMP spike trains
        for jj in range(data_points[ii].shape[0]): 
            spike_trains = data_points[ii, jj]; # (N by T) spike train matrix
            # Calculate the PSTH
            _,psth,_,_ = generate_psth(spike_trains, bin_size=20, bin_size_2=50)

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Scale the PSTH 
    #         psth_step_scaled = psth_step * 2 * m / len(psth_step)
    #         psth_ramp_scaled = psth_ramp * 2 * m / len(psth_ramp)

            # Find the gradient of the PSTH
            grad_psth = np.gradient(psth)

            # Find the variance and the Fano factor of the gradient
            var = np.var(grad_psth)
            fano_factor = var / np.mean(grad_psth)
            
            # Print the variance and the Fano factor
            # print(f"variance = {var}, Fano factor = {fano_factor}")
            var_s[ii,jj] = var
            fano_factors[ii,jj] = fano_factor

            # Classify the spike trains based on the variance 
            if var > thresholds:
                predictions[ii, jj] = 0  # step model
            else:
                predictions[ii, jj] = 1  # ramp model

#             if var_ramp > threshold:
#                 predictions.append(0)  # step model
#             else:
#                 predictions.append(1)  # ramp model

    return predictions, var_s, fano_factors

def higher_order_classifier(data_points, thresholds):
    """
    Classify spike trains as being generated by the step model (return 0) or the ramp model (return 1).
    :param data_points: 2*M data points, M data points for each model, where each data point is a (N by T) matrix
    :param m: mean jump time (in # of time-steps) for StepModel
    :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time for StepModel
    :param sigma: diffusion strength of the drift-diffusion process for RampModel
    :param beta: drift rate of the drift-diffusion process for RampModel
    :param threshold: threshold for variance
    :return: M predictions, each being 0 (step model) or 1 (ramp model)
    """
    predictions = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
    var_s = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
    fano_factors = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
    
    for ii in [0,1]:
        # ii = 0 -> STEP spike trains
        # ii = 1 -> RAMP spike trains
        for jj in range(data_points[ii].shape[0]): 
            spike_trains = data_points[ii, jj]; # (N by T) spike train matrix
            # Calculate the PSTH
            _,psth,_,_ = generate_psth(spike_trains, bin_size=20, bin_size_2=50)
            
            
            ### find the vilid region
            
            
            
            ### Gradient and average gradient

            # Find the gradient of the PSTH
            grad_psth = np.gradient(psth)
            average_grad = (psth[-1] - psth[1]) / len(psth)
            # Find the variance and the Fano factor of the gradient
            var = np.var(grad_psth)
            fano_factor = var / np.mean(grad_psth)
            
            # Print the variance and the Fano factor
            # print(f"variance = {var}, Fano factor = {fano_factor}")
            var_s[ii,jj] = var
            fano_factors[ii,jj] = fano_factor

            # Classify the spike trains based on the variance 
            if var > thresholds:
                predictions[ii, jj] = 0  # step model
            else:
                predictions[ii, jj] = 1  # ramp model

#             if var_ramp > threshold:
#                 predictions.append(0)  # step model
#             else:
#                 predictions.append(1)  # ramp model

    return predictions, var_s, fano_factors


       

# def normalised_var_classifier(data_points, m, r, sigma, beta, threshold):
#     """
#     Classify spike trains as being generated by the step model (return 0) or the ramp model (return 1).
#     :param data_points: 2*M data points, M data points for each model, where each data point is a (N by T) matrix
#     :param m: mean jump time (in # of time-steps) for StepModel
#     :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time for StepModel
#     :param sigma: diffusion strength of the drift-diffusion process for RampModel
#     :param beta: drift rate of the drift-diffusion process for RampModel
#     :param threshold: threshold for variance
#     :return: M predictions, each being 0 (step model) or 1 (ramp model)
#     """
#     predictions = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
#     var_s = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
#     fano_factors = np.empty((data_points.shape[0], data_points.shape[1])) # 2 x M
    
#     for ii in [0,1]:
#         # ii = 0 -> STEP spike trains
#         # ii = 1 -> RAMP spike trains
#         for jj in range(data_points[ii].shape[0]): 
#             spike_trains = data_points[ii, jj]; # (N by T) spike train matrix
#             # Calculate the PSTH
#             psth,_,_ = generate_psth(spike_trains, bin_size=20, bin_size_2=50)

#             #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#             # Scale the PSTH 
#     #         psth_step_scaled = psth_step * 2 * m / len(psth_step)
#     #         psth_ramp_scaled = psth_ramp * 2 * m / len(psth_ramp)

#             # Find the gradient of the PSTH
#             grad_psth = np.gradient(psth)
            
#             # Normalize the gradient so that the area under it is equal to 1
#             grad_psth_normalized = grad_psth / np.sum(grad_psth)

#             # Find the variance and the Fano factor of the gradient
#             var = np.var(grad_psth_normalized)
#             fano_factor = var / np.mean(grad_psth_normalized)
            
            
#             # Print the variance and the Fano factor
#             # print(f"variance = {var}, Fano factor = {fano_factor}")
#             var_s[ii,jj] = var
#             fano_factors[ii,jj] = fano_factor
#             #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#             # Classify the spike trains based on the variance 
# #             if var > threshold:
# #                 predictions[ii,].append(0)  # step model
# #             else:
# #                 predictions[ii].append(1)  # ramp model

# #             if var_ramp > threshold:
# #                 predictions.append(0)  # step model
# #             else:
# #                 predictions.append(1)  # ramp model

#     return predictions, var_s, fano_factors
