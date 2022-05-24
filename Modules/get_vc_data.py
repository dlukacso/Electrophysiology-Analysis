import numpy as np
import pandas as pd

def get_vc_parameters():
    """
    Get the electrophysiological parameters used for voltage clamp
    """
    parameters = ['Input resistance (MOhm)', 'Series resistance (MOhm)',
                  'Capacitance (pF)', 'Current Drop (pA)']
    return parameters

def get_q(df, flat_point, ind1, ind2):
    # calculate charge stored from input current
    diff = flat_point - df.values[ind1:ind2,0]
    diff[diff<0] = 0
    steps = np.diff(df.index.values)
    return np.dot(steps[ind1:ind2], diff)

def get_stable(data, num=10):
    """
    Given a region of data, get the median value over a region with the lowest variation
    """
    
    # break down data into equally sized segments that we will explore
    length = data.size-num+1
    datalist = np.zeros((length,num),dtype=int)
    for column in range(datalist.shape[1]):
        datalist[:,column] = np.arange(length) + column
    datalist = data[datalist]
    
    # find the variation across each region
    # pick the first point right before the variation greatly increases
    stds = np.std(datalist,axis=1)
    for i in reversed(range(stds.size-1)):
        if stds[i]*.9 < stds[i-1]:
            return np.median(datalist[i])
    
    return

def get_charge(trace):
    """
    Calculate the charge collected due to a voltage change, between the current drop time
    and the end of the voltage change
    """
    
    # we calculate the high point of the region, and use it as a baseline
    # we use a bit of local normalization to minimize the effect of noise
    count = 5
    values = trace.values
    datalist = np.zeros((values.size-count+1, count),dtype=int)
    for column in range(datalist.shape[1]):
        datalist[:,column] = np.arange(datalist.shape[0]) + column
    datalist = values[datalist]
    peak = datalist.mean(axis=1).max()
    
    # calculate the area under the peak. To allow for inconsistent time-steps
    # we multiply the average of 2 consecutive points by their time step
    time_step = np.diff(trace.index.values)
    ave_value = (values[:-1] + values[1:]) / 2.
    charge = np.dot(ave_value - peak, time_step)
    
    return charge, peak

def get_vc_data(df, row):
    """
    For voltage clamp data, with a voltage change of delta_volt, calculated capacitance and resistances
    """
    
    # get the values that we work with
    delta_volt = row['Signal (mV)']
    start_time = row['Start VC (ms)']
    end_time = row['End VC (ms)']
    trace = df.iloc[:,0]
    
    # initialize a results variable
    parameters = get_vc_parameters()
    results = pd.Series(np.NaN, index=parameters)
    
    # get resting current flow, and locations of the 2 extrema after voltage change
    current_rest = np.median(trace[:start_time])
    min_time = trace[start_time:].idxmin()
    max_time = trace[start_time:].idxmax()
    
    # if the max comes before the min, invert the trace
    if max_time < min_time:
        min_time, max_time = max_time, min_time
        trace = -(trace - current_rest) + current_rest
        delta_volt = -delta_volt
    
    # make sure that the spikes occur after the voltage changes
    if min_time < start_time:
        return results
    if max_time < end_time:
        return results
    
    # calculate the charge stored
    charge, stable_rest = get_charge(trace[min_time:end_time])
    
    # calculate key parameters
    capacitance = charge / delta_volt
    resistance_input = delta_volt / (stable_rest - current_rest) * 1000
    resistance_series = delta_volt / (trace[min_time] - current_rest) * 1000
    
    # see if after voltage drop, it stabalizes to below stable current
    current_drop = stable_rest - current_rest
    
    # save values to results
    results['Input resistance (MOhm)'] = resistance_input
    results['Series resistance (MOhm)'] = resistance_series
    results['Capacitance (pF)'] = capacitance
    results['Current Drop (pA)'] = current_drop
    
    return results