import numpy as np
import pandas as pd
from scipy.stats import linregress

from . import get_frequency_data

def get_frequency_names(*args, **kwargs):
    """
    Calls get_frequency_data's get_frequency_names. This function is only here, so that the module
    doesn't have to be imported by other modules
    """
    
    return get_frequency_data.get_frequency_names(*args, **kwargs)

def get_cc_parameters():
    """
    Get the electrophysiological parameters used for current clamp
    """
    # define order of current clamp calculations
    parameters = ['Resting membrane potential (mV)',
                  'Sag potential',
                  'Approximate resistance (MOhm)',
                  'AP halfwidth (ms)',
                  'AP symmetricity',
                  'AP peak amplitude (mV)',
                  'Latency (ms)',
                  'Attenuation',
                 ]
    
    # add frequency definitions
    parameters += get_frequency_names(append_endings=True)
    
    return parameters

def drop_noise(df, df_spike, row):
    """
    df_spike can contain spikes where there shouldn't be spikes. This is important, both for purposes of 
    quality control algorithms, and so people can manually check the spike positions if they are interested.
    However, for further analysis we don't need these, so we drop them.
    For similar reasons, we drop spiking from traces that failed qc
    """
    
    # drop unwanted trace
    df_spike = df_spike[df_spike.index.isin(df.columns)]
    df_spike = df_spike[df_spike.index>0.]
    
    # drop spikes outside of spiking range
    start_time = row['Start CC (ms)']
    end_time = row['End CC (ms)']
    df_spike = df_spike[np.logical_and(df_spike.Start_Time > start_time,
                                       df_spike.Peak_Time < end_time)]
    
    df_spike = df_spike.copy()
    
    return df_spike

def calculate_peak_shape_parameters(df_spike):
    """
    Calculate parameters related to the shape of a spike; the amplitude, half-width, and
    symmetricity. We calculate these on the first trace with at least 3 spikes
    """
    # get the first trace with at least 3 spikes
    # return np.NaN if no such trace exists
    spike_counts = df_spike.index.value_counts()
    spike_counts = spike_counts[spike_counts>=3]
    if spike_counts.size == 0:
        return np.NaN, np.NaN, np.NaN
    trace = spike_counts.sort_index().index[0]
    
    # take the average of the trace's values
    data = df_spike.loc[trace,:].mean(axis=0)
    
    # get the values
    amplitude = data['Amplitude']
    halfwidth = data['HalfWidth']
    symmetricity = data['Symmetricity']
    
    return amplitude, halfwidth, symmetricity

def calculate_latency(df_spike, start_time):
    """
    Calculate the latency; the time between the current injection, and the first spike of the first
    trace to have a spike
    """
    # get the first trace
    df_spike = df_spike.loc[df_spike.index.min(),:]
    
    # get the first spike's time
    peak_time = df_spike.Peak_Time.min()
    
    return peak_time - start_time

def calculate_attenuation(df_spike):
    """
    Calcuation attenuation; the ratio of the size of the first peaks to the last peaks
    We grab the first trace with at least 9 spikes, and use the average of the first and last 3 spikes to 
    calculate the amplitudes
    """
    
    # get the first trace with at least 9 spikes
    # return np.NaN if no such trace exists
    spike_counts = df_spike.index.value_counts()
    spike_counts = spike_counts[spike_counts>=9]
    if spike_counts.size == 0:
        return np.NaN
    trace = spike_counts.sort_index().index[0]
    
    # get the spikes, sorted by time
    df_spike = df_spike.loc[trace,:].sort_values('Peak_Time')
    
    # get the starting and ending averages
    amplitudes = df_spike.Amplitude.values
    start_mean = amplitudes[:3].mean()
    end_mean = amplitudes[-3:].mean()
    
    return start_mean / end_mean

def calculate_peak_parameters(df_spike, row):
    """
    Calculate the parameters that we can get from just the spiking information
    """
    # get trace shape values
    amplitude, halfwidth, symmetricity = calculate_peak_shape_parameters(df_spike)
    
    # get latency
    latency = calculate_latency(df_spike, row['Start CC (ms)'])
    
    # get attenuation
    attenuation = calculate_attenuation(df_spike)
    
    # save results to a series
    parameters = ['AP halfwidth (ms)',
                  'AP symmetricity',
                  'AP peak amplitude (mV)',
                  'Latency (ms)',
                  'Attenuation',
                 ]
    results = pd.Series(np.NaN, index=parameters)
    results['AP halfwidth (ms)'] = halfwidth
    results['AP symmetricity'] = symmetricity
    results['AP peak amplitude (mV)'] = amplitude
    results['Latency (ms)'] = latency
    results['Attenuation'] = attenuation
    
    return results

def calculate_approximate_resistance(df, start_time, end_time):
    """
    for each trace in a current clamp dataset, calculate the membrane potential before
    current injection, and during current injection. The difference is the step size, and
    a linear fit of it against the current injection is the approximate resistance of the cell
    """
    
    # get the step size
    v_rest = np.median(df.loc[:start_time,:], axis=0)
    time_adjust = 0.1 * (end_time - start_time)
    v_inject = np.median(df.loc[(start_time+time_adjust):(end_time-time_adjust),:],axis=0)
    step_size = v_inject - v_rest
    
    # get the currents
    currents = df.columns.values
    
    # get the fit
    resistance = linregress(currents, step_size)[0] * 1000
    
    return resistance

def calculate_trace_parameters(df, df_spike, row):
    """
    Calculate the parameters that we can get from the traces without concern for spikes;
    resting membrane potential, sag potentail, and approximate resistance
    """
    
    # get current injection start and end times
    start_time = row['Start CC (ms)']
    end_time = row['End CC (ms)']
    inject_time = end_time - start_time
    
    # for the resting membrane potential, we take the first trace with non-zero current injection
    # and calculate the median voltage before current injection
    trace = df.loc[:,df.columns.values>=0.].iloc[:,0]
    membrane_potential = np.median(trace[:start_time].values)
    
    # for the sag potential, we take the trace with -150pA current injection, or the next smallest
    # if it doesn't exist, and calculate the difference between the drop, and the stable level
    # during current injection
    trace = df.loc[:,df.columns.values>=-150.].iloc[:,0]
    low_point = trace[start_time:(start_time + inject_time / 4.)].min()
    stable_value = np.median(trace[(end_time - inject_time / 4.):end_time])
    sag_potential = stable_value - low_point
    
    # for the approximate resistance, we restrict ourselves up to the first trace before spikes
    first_spike = df_spike.index.min()
    df = df.loc[:,df.columns.values < first_spike]
    resistance = calculate_approximate_resistance(df, start_time, end_time)
    
    # save results to a series
    parameters = ['Resting membrane potential (mV)',
                  'Sag potential',
                  'Approximate Resistance (MOhm)'
                 ]
    results = pd.Series(np.NaN, index=parameters)
    results['Resting membrane potential (mV)'] = membrane_potential
    results['Sag potential'] = sag_potential
    results['Approximate resistance (MOhm)'] = resistance
    
    return results

def get_cc_data(df, df_spike, row, custom_start=800., custom_end=1000.):
    """
    Calculate current clamp electrophysiological data for a dataframe of trace data
    and the corresponding spike locations
    """
    
    # drop the noisy data
    df_spike = drop_noise(df, df_spike, row)
    
    # initialize a results variable
    parameters = get_cc_parameters()
    results = pd.Series(np.NaN, index=parameters)
    
    # calculate peak parameters
    peak_params = calculate_peak_parameters(df_spike, row)
    results.update(peak_params)
    
    # calculate trace shape parameters
    trace_params = calculate_trace_parameters(df, df_spike, row)
    results.update(trace_params)
    
    # calcualte frequency based parameters
    args = (df, df_spike, row['Start CC (ms)'], row['End CC (ms)'], custom_start, custom_end)
    frequency_params = get_frequency_data.calculate_frequency_data(*args)
    results.update(frequency_params)
    
    return results