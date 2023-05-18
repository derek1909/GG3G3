import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from models import *



def generate_spike_trains(M=20, N=400, T=100, m=50, r=10, sigma=0.2, beta=0.5):
    """
    Generate M data points, each containing N ramp spike trains and N step spike trains.
    :param M: number of data points
    :param N: number of trls per data point
    :param T: duration of each trial in number of time-steps
    :param m: mean jump time (in # of time-steps) for StepModel
    :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time for StepModel
    :param sigma: diffusion strength of the drift-diffusion process for RampModel
    :param beta: drift rate of the drift-diffusion process for RampModel
    :return: M data points, each containing spike trains for both models
    """
    # Initialize models
    step_model = StepModel(m=50, r=10, x0=0.2, Rh=50);
    ramp_model = RampModel(beta=beta, sigma=sigma);

    # Generate M data points
    data_points = np.empty((M, 2, N, T))  # for an n x m array
    for MM in range(M):
        # Generate spike trains
        step_spikes, _, _ = step_model.simulate(Ntrials=N, T=T)
        ramp_spikes, _, _ = ramp_model.simulate(Ntrials=N, T=T)

        # Add spike trains to data points
        data_points[MM]=[step_spikes, ramp_spikes]

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
    :return: averaged PSTH, variance, Fano factor
    """
    
    N = len(spike_trains[1]); # number of trials
    spike_trains_timestamp = generate_raster_and_timestamps(spike_trains); # timestamps of spike trains
    
    # Calculate the PSTH
    bin_edges = np.arange(0, 1e3 + bin_size, bin_size)
    psth, _ = np.histogram(np.concatenate(spike_trains_timestamp), bins=bin_edges)

    averaged_psth = (psth / bin_size * 1e3) / N # spikes per sec per trail

    # Apply Gaussian smoothing
    sigma = 1.5  # Standard deviation of the Gaussian filter
    gaussian_smoothed_psth = gaussian_filter(averaged_psth, sigma)

    # Calculate the variance
    bin_edges_2 = np.arange(0, 1e3, bin_size_2)
    psth_2, _ = np.histogram(np.concatenate(spike_trains_timestamp), bins=bin_edges_2)
    averaged_psth_2 = (psth_2 / bin_size_2 * 1e3) / N # spikes per sec per trail

    var = np.zeros_like(averaged_psth_2)
    for ii in range(len(spike_trains_timestamp)):
        psth_p_trail, _ = np.histogram(np.array(spike_trains_timestamp[ii]), bins=bin_edges_2) 
        var += (psth_p_trail / bin_size_2 * 1e3) ** 2 / N
    var -= averaged_psth_2**2

    # Calculate Fano factor
    fano = var / averaged_psth_2

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle("PSTH Diagram")

        # Plot the PSTH
        ax1.plot(bin_edges[:-1], averaged_psth,  label='Original')
        ax1.plot(bin_edges[:-1], gaussian_smoothed_psth,  label='Smoothed')
        ax2.plot(bin_edges_2[:-1], var,  label='Variance')
        ax3.plot(bin_edges_2[:-1], fano,  label='Fano factor')

        ax1.set_ylabel("spike rate (sp/s)")
        ax2.set_ylabel("spike rate (sp/s)")
        ax3.set_ylabel("Fano factor")
        ax3.set_xlabel("time from motion onset (ms)")
        ax1.legend()
        ax2.legend()
        plt.show()

    return gaussian_smoothed_psth, var, fano




def fano_classifier(data_points, m, r, sigma, beta, threshold):
    """
    Classify spike trains as being generated by the step model (return 0) or the ramp model (return 1).
    :param data_points: M data points, each containing spike trains for both models
    :param m: mean jump time (in # of time-steps) for StepModel
    :param r: parameter r ("# of successes") of the Negative Binomial (NB) distribution of jump (stepping) time for StepModel
    :param sigma: diffusion strength of the drift-diffusion process for RampModel
    :param beta: drift rate of the drift-diffusion process for RampModel
    :param threshold: threshold for variance
    :return: M predictions, each being 0 (step model) or 1 (ramp model)
    """
    predictions = []
    for step_spikes, ramp_spikes in data_points:
        # Calculate the PSTH
        psth_step = np.mean(step_spikes, axis=0)
        psth_ramp = np.mean(ramp_spikes, axis=0)

        # Scale the PSTH
        psth_step_scaled = psth_step * 2 * m / len(psth_step)
        psth_ramp_scaled = psth_ramp * 2 * m / len(psth_ramp)

        # Find the gradient of the PSTH
        grad_psth_step = np.gradient(psth_step_scaled)
        grad_psth_ramp = np.gradient(psth_ramp_scaled)

        # Find the variance and the Fano factor of the gradient
        var_step = np.var(grad_psth_step)
        var_ramp = np.var(grad_psth_ramp)
        fano_step = var_step / np.mean(grad_psth_step)
        fano_ramp = var_ramp / np.mean(grad_psth_ramp)

        # Print the variance and the Fano factor
        print(f"Step model: variance = {var_step}, Fano factor = {fano_step}")
        print(f"Ramp model: variance = {var_ramp}, Fano factor = {fano_ramp}")

        # Classify the spike trains based on the variance
        if var_step < threshold:
            predictions.append(0)  # step model
        else:
            predictions.append(1)  # ramp model

    return predictions

