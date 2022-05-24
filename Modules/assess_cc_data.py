import numpy as np
import pandas as pd
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
from scipy.stats import linregress
from scipy.optimize import curve_fit

from . import get_spike_data

def qc_trace(injection, trace, df_spike, row):
    """
    Perform quality control on a dataset trace. We need to determine if it is spiking too much
    where it shouldn't, and whether it is too noisy during the steady state
    """
    
    # first we identify the current injection start and end times
    start_time = row['Start CC (ms)']
    end_time = row['End CC (ms)']
    end_time = end_time + (trace.index.max() - end_time) / 3.
    
    # we do a lowess fit on the start and end regions, and find how noisy they are
    # since it takes some time for the signal to die, we only start 50ms after the end time
    start_trace = trace[:start_time]
    end_trace = trace[end_time+50.:]
    params = {'frac':.04, 'delta':20, 'is_sorted':True, 'return_sorted':False}
    if start_trace.size > 50:
        fit = lowess(start_trace.values, start_trace.index.values, **params)
        start_noise = np.std(fit)
    else:
        start_noise = 0.
    if end_trace.size > 50:
        fit = lowess(end_trace.values, end_trace.index.values, **params)
        end_noise = np.std(fit)
    else:
        end_noise = 0.
    
    # if either is too high, we are done; fails QC
    if max(start_noise, end_noise) > 2.5:
        return False
    
    # if there are no spikes, it can't fail qc, otherwise get the current injection
    if df_spike.shape[0] == 0:
        return True
    injection = np.unique(df_spike.index)
    
    # if there are 4 or more spikes total in the before and after ranges
    # or 4 or more spikes during a negative injection, qc fails
    if np.logical_or(df_spike.Peak_Time <= start_time, df_spike.Peak_Time > end_time).sum() >= 4:
        return False
    if injection > 0:
        return True
    return np.logical_and(df_spike.Peak_Time < end_time, df_spike.Peak_Time > start_time).sum() < 4

def qc_dataset(df, df_spike, row):
    """
    Run quality control on every trace in a dataset
    """
    
    qc = np.array([qc_trace(injection, trace, df_spike.loc[df_spike.index==injection,:], row)
                   for injection, trace in df.iteritems()])
    
    return qc

def adjust_dataframe(df, start, step):
    """
    Adjust a current clamp dataframe, so that the columns contain information
    on voltage injection levels
    """
    
    injections = start + step * np.arange(df.shape[1])
    df.columns = injections
    df.columns.name = 'Injection (pA)'
    
    return
    
def assess_cc_data(df, row, max_fail=2, drop_fail=True):
    """
    We assess a current clamp dataset. This involves dropping traces that fail quality control,
    determining if the entire cell failed quality control, and getting a dataframe of action potentials,
    even for traces that failed qc
    """
    
    # adjust column labels
    adjust_dataframe(df, row['Start (pA)'], row['Step (pA)'])
    
    # identify the locations of action potentials
    df_spike = get_spike_data.get_atf_spikes(df, min_width=0.2, min_height=2.)
    
    # run quality control
    qc = qc_dataset(df, df_spike, row)
    failed = (~qc).sum()
    if drop_fail:
        df = df.loc[:,qc]
    
        # return the results
        return df, df_spike, failed <= max_fail
    
    return df, df_spike, qc