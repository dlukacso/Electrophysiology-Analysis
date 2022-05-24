import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit

def get_frequency_endings():
    """
    Return the endings that we can use for frequencies
    """
    
    endings = ['Maximum (Hz)',
               'Slope (Hz/pA)',
               'Threshold (pA)'
              ]
    
    return endings

def get_frequency_names(append_endings=False):
    """
    Get all of the names for the different types of frequencies that we measure. Yes, we have so many
    different frequencies, that there is a function, whose sole purpose is to pass the list of the names
    to the user, so that they don't have to be defined in multiple notebooks
    """
    
    names = ['Average Full Frequency',
             'Average Custom Frequency',
             'Average Firing Frequency',
             'Average Firing Instantaneous Frequency',
             'Median Firing Instantaneous Frequency',
             'Average Custom Instantaneous Frequency',
             'Median Custom Instantaneous Frequency'
            ]
    
    if append_endings:
        endings = get_frequency_endings()
        names = [f'{name} {ending}' for name in names for ending in endings]
    
    return names

def calculate_spike_count(df_spike, start_time, end_time):
    """
    Calculate the number of spikes in a given temporal range
    """
    
    # restrict spiking to range of interest
    df_spike = df_spike.loc[np.logical_and(df_spike.Start_Time>start_time, df_spike.Peak_Time<end_time),:]
    
    # count the number of spikes per trace
    spike_count = df_spike.index.value_counts().astype(float)
    
    return spike_count

def calculate_average_frequency(df_spike, start_time, end_time):
    """
    Calculate the average frequency of each trace between start and end times
    """
    
    # count the number of spikes per trace
    spike_count = calculate_spike_count(df_spike, start_time, end_time)
    
    # divide by time_span, return results
    return spike_count / (end_time - start_time) * 1000.

def calculate_peak_average_frequency(df_spike, start_time, end_time):
    """
    Calculate the average frequency from the first to last spike within the given time interval
    for each trace
    """
    
    # restrict spiking to range of interest
    df_spike = df_spike.loc[np.logical_and(df_spike.Start_Time>start_time, df_spike.Peak_Time<end_time),:]
    
    # count the number of spikes per trace
    spike_count = df_spike.index.value_counts().astype(float)
    
    # we remove 1 from the count, as a normalization factor
    spike_count = spike_count - 1
    
    # calculate time intervals
    df_time = pd.DataFrame(np.NaN, index=spike_count.index, columns=['Start', 'End', 'Interval'])
    for index in df_time.index:
        df_sub = df_spike.loc[index,:]
        df_time.loc[index, 'Start'] = df_sub.Peak_Time.min()
        df_time.loc[index, 'End'] = df_sub.Peak_Time.max()
    df_time.Interval = df_time.End - df_time.Start
    
    # return frequency
    return spike_count / df_time.Interval * 1000.

def calculate_instant_frequency(df_spike, start_time, end_time, summary_function):
    """
    Calculate the instantaneous frequency for each trace over the given range
    summary_function is used to compile it into a single value
    """
    # restrict spiking to range of interest
    df_spike = df_spike.loc[np.logical_and(df_spike.Start_Time>start_time, df_spike.Peak_Time<end_time),:]
    df_spike = df_spike.sort_values('Peak_Time')
    
    # restrict to traces with at least 1 spikes
    spike_count = df_spike.index.value_counts()
    spike_count = spike_count[spike_count > 1]
    
    # initialize result series
    results = pd.Series(np.NaN, index=spike_count.index)
    
    # iterate over each trace
    for index in results.index:
        df_sub = df_spike.loc[index,:]
        
        # get time differences, and instant frequences
        times = np.diff(df_sub.Peak_Time)
        freqs = 1000. / times
        
        # compile values
        results[index] = summary_function(freqs)
    
    return results

def get_frequency_slope(frequencies, shift=20, current_range=250):
    """
    We calculate the slope of a linear fit for frequencies vs current injections. Since this slope
    eventually falls off, we don't calculate over the entire range. instead, we start shift above the
    first injection with spiking, and only calculate it over a range of current_range
    """
    
    # get range of interest
    start = frequencies[frequencies>0.].index.min() + shift
    start = frequencies.index[frequencies.index>=start].min()
    end = start + current_range
    frequencies = frequencies.loc[start:end+0.001]
    
    # calculate the slope
    slope = linregress(frequencies.index.values, frequencies.values)[0]
    
    return slope

def sigmoid(x, L, x0, k):
    """
    Defines a sigmoid function with amplitude L, centered on x0, and with a rate of k
    """
    
    return L / (1 + np.exp(-k*(x-x0)))

def reverse_sigmoid(y, L, x0, k):
    """
    Defined the inverse of a sigmoid function with amplitude L, centered on x0, and with a rate of k
    """
    
    esp = 1e-6
    
    return x0 + (np.log(y+esp) - np.log(L-y+esp)) / k

def get_curve_fit(frequencies):
    """
    Given a pandas series of frequencies with corresponding injection currents as their index,
    calculate a sigmoid fit of the data
    """
    
    # check if we have enough data for an assessment
    if frequencies.size < 5 or (frequencies > 0).sum() < 2:
        return False, {'max_frequency':np.NaN, 'threshold':np.NaN}
    
    # we amke an initial guess for the sigmoid parameters
    # If the fit is frequently failing, look into fine-tuning these values
    
    # we estimate that we haven't seen the highest value yet, but have seen most of the data
    L = frequencies.max() / 0.75
    freq_above = frequencies[frequencies >= L/2.]
    x0 = freq_above.index.min()
    
    # having too large of a k will cause problems in the sigmoid, as exp(-k*(x-x0)) causes overflow errors
    # for this reason we do 2 things:
    # 1. set all injection currents < -100pA to -100pA; all frequencies at or below 0pA should be 0, so this
    # shouldn't get any information, but will avoid overflow errors
    # we initialize k so that abs(k*(x-x0)) <= 10. for all values
    index = frequencies.index.values.copy()
    index[index<-100.] = -100.
    frequencies = pd.Series(frequencies.values, index=index)
    k = 10. / np.abs(frequencies.index - x0).max()
    guess = [L, x0, k]
    
    # run sigmoid fit
    try:
        popt, pcov = curve_fit(sigmoid, frequencies.index.values, frequencies.values, p0=guess, maxfev=10000)
    except RuntimeError:
        # if it failed, we give a best guess about maximum frequency and firing threshold
        max_freq = frequencies.max()
        if max_freq > 0.:
            threshold = max(frequencies.index[frequencies>0.].min(), 0.)
        else:
            threshold = np.NaN
        return False, {'max_frequency':max_freq, 'threshold':threshold}
    
    # if successful, we return the fit results instead
    # those can be used to calculate the parameters, adjusting for edge cases
    max_freq, x0, rate = popt
    
    return True, {'max_frequency':max_freq, 'x0':x0, 'rate':rate}
    
def get_key_frequencies(frequencies):
    """
    Given a pandas series of frequencies with corresponding injection currents as their index,
    we fit the data to a sigmoid curve, and use the results to get approximations for both the maximum
    firing frequency and the firing threshold.
    """
    
    # we perform the fit
    can_fit, fit_results = get_curve_fit(frequencies)
    
    if not can_fit:
        # if the fit failed, we return the given values, there isn't anything that we can do
        return fit_results['max_frequency'], fit_results['threshold']
    
    # if successful, we take the fit parameters, and start estimating
    max_freq = fit_results['max_frequency']
    x0 = fit_results['x0']
    rate = fit_results['rate']
    
    # to estimate the firing threshold, first we fit a line to the fit curve between 20% to 80% of its maximum value
    x1 = reverse_sigmoid(max_freq*.2, max_freq, x0, rate)
    x2 = reverse_sigmoid(max_freq*.8, max_freq, x0, rate)
    xvals = np.linspace(x1,x2,21)
    yvals = sigmoid(xvals, max_freq, x0, rate)
    slope, intercept = linregress(xvals,yvals)[:2]
    
    # the firing threshold is where the slope intersects the x-axis
    threshold = -intercept / slope
    
    # we put 3 thresholds on our prediction; it must be above 0, it must be above the last injection before firing,
    # and it must be lesss than the first firing injection
    thresh_max = max(frequencies.index.values[frequencies.values>0.].min(), 0.)
    thresh_min = max(frequencies.index.values[frequencies.index.values<thresh_max].max(), 0.)
    threshold = np.clip(threshold, thresh_min, thresh_max)
    
    # we hard code it, so that we can't predict a maximum frequency more than 20% above the highest observed frequency
    max_freq = min(1.2 * np.max(frequencies), max_freq)

    return max_freq, threshold

def calculate_frequency_parameters(frequencies):
    """
    For a given series of measured frequencies vs current injections, we calculate a series of parameters;
    maximum firing frequency, frequency slope, and firing threshold
    """
    
    # get the frequency slope
    slope = get_frequency_slope(frequencies, shift=20, current_range=250)
    
    # get firing threshold and maximum frequency
    max_freq, threshold = get_key_frequencies(frequencies)
    
    # save the results to a series
    parameters = get_frequency_endings()
    results = pd.Series(np.NaN, index=parameters)
    results['Maximum (Hz)'] = max_freq
    results['Slope (Hz/pA)'] = slope
    results['Threshold (pA)'] = threshold
    
    return results

def calculate_trace_frequency_data(df, df_spike, inject_start, inject_end, custom_start, custom_end):
    """
    Calculate 7 different frequencies for each trace in a current clamp dataframe
    """
    # initialize dataframe to calculate frequency for each trace
    columns = get_frequency_names(append_endings=False)
    index = df.columns
    df_trace_frequency = pd.DataFrame(np.NaN, index=index, columns=columns)
    
    # calculate defined range frequencies
    name = 'Average Full Frequency'
    df_trace_frequency[name] = calculate_average_frequency(df_spike, inject_start, inject_end)
    name = 'Average Custom Frequency'
    df_trace_frequency[name] = calculate_average_frequency(df_spike, custom_start, custom_end)
    name = 'Average Firing Frequency'
    df_trace_frequency[name] = calculate_peak_average_frequency(df_spike, inject_start, inject_end)
    
    # calculate instantaneous frequencies
    name = 'Average Firing Instantaneous Frequency'
    df_trace_frequency[name] = calculate_instant_frequency(df_spike, inject_start, inject_end, np.mean)
    name = 'Median Firing Instantaneous Frequency'
    df_trace_frequency[name] = calculate_instant_frequency(df_spike, inject_start, inject_end, np.median)
    name = 'Average Custom Instantaneous Frequency'
    df_trace_frequency[name] = calculate_instant_frequency(df_spike, custom_start, custom_end, np.mean)
    name = 'Median Custom Instantaneous Frequency'
    df_trace_frequency[name] = calculate_instant_frequency(df_spike, custom_start, custom_end, np.median)
    
    # missing values means not enough spikes to get frequency -> 0 frequency
    df_trace_frequency.fillna(0., inplace=True)
    
    return df_trace_frequency

def calculate_frequency_data(df, df_spike, inject_start, inject_end, custom_start, custom_end):
    """
    Calculate 7 different frequencies, and 3 different parameters for each frequency
    """
    # first we calculate the frequencies for each trace
    df_trace_frequency = calculate_trace_frequency_data(df, df_spike, inject_start, inject_end, custom_start, custom_end)
    
    # create a dataframe holding the frequency metric vs key parameters
    columns = get_frequency_endings()
    df_params = pd.DataFrame(np.NaN, index=df_trace_frequency.columns, columns=columns)
    for index, frequencies in df_trace_frequency.iteritems():
        df_params.loc[index,:] = calculate_frequency_parameters(frequencies)
    
    # unflatten the results into a series, and then return it
    pairs = [(index, column) for index in df_params.index for column in df_params.columns]
    values = [df_params.loc[index, column] for index, column in pairs]
    metrics = [f'{index} {column}' for index, column in pairs]
    results = pd.Series(values, index=metrics)
    
    return results