import numpy as np
import pandas as pd

def potential_spike_start(slope, time, min_count, min_slope):
    """
    Evaluates if a starting point is a potential spike start point
    """
    # get the numerical index location
    index = slope.index.get_loc(time)
    
    # makes sure that the first min_count points have slope above min_slope
    if not np.all(slope.iloc[index:index+min_count] > min_slope):
        return False
    
    # make sure that for 0.1ms after this point, slope is positive
    return np.all(slope[time:time+0.1001] > 0.)

def trim_wide_spike(trace, start_time, min_count, min_slope, max_width):
    """
    Due to the spike measurement, it is possible for the voltage step from the current injection
    to be detected as a peak. Other, anomalous situations might happen as well in various datasets
    As such, if a peak is too wide, we try to see if there are any sub-peak positions within it 
    that set it to less than max_width
    """
    
    # calculate the slope
    slope = pd.Series(np.diff(trace.values) / np.diff(trace.index.values), index=trace.index[:-1])
    
    # we only need to look at points immediately after the start time, until we encounter 1 below min_slope
    # any points after that would have been picked up as candidates points already. We also drop the min_count
    # points before that, since it needs that many with the right slope after it
    candidates = slope[start_time:] > min_slope
    to_drop = (np.arange(candidates.size)[~candidates])[0]
    candidates = candidates.index[1:to_drop-min_count]
    
    # if we have no candidate time points, we are done
    if candidates.size == 0:
        return start_time, np.NaN, start_time-1., np.NaN
    
    # try each point until we find a match
    for current_time in candidates:
        if potential_spike_start(slope, current_time, min_count, min_slope):
            current_pos = trace.loc[current_time]
            end_time, end_pos = get_end_time(trace, current_time, current_pos)
            if np.isfinite(end_time) and (end_time - current_time) <= max_width:
                return current_time, current_pos, end_time, end_pos
    
    return start_time, np.NaN, start_time-1., np.NaN

def trim_shallow_spike(trace, start_time, min_count, min_slope, min_ratio):
    """
    We shrink any spikes who aren't tall enough for their girth until we either run into non-candidates
    or find one with a desired ratio
    """
    
    # calculate the slope
    slope = pd.Series(np.diff(trace.values) / np.diff(trace.index.values), index=trace.index[:-1])
    
    # we only need to look at points immediately after the start time, until we encounter 1 below min_slope
    # any points after that would have been picked up as candidates points already. We also drop the min_count
    # points before that, since it needs that many with the right slope after it
    candidates = slope[start_time:] > min_slope
    to_drop = (np.arange(candidates.size)[~candidates])[0]
    candidates = candidates.index[1:to_drop-min_count]
    
    # if we have no candidate time points, we are done
    if candidates.size == 0:
        return start_time, np.NaN, start_time-1., np.NaN
    
    # try each point until we find a match
    for current_time in candidates:
        if potential_spike_start(slope, current_time, min_count, min_slope):
            current_pos = trace.loc[current_time]
            end_time, end_pos = get_end_time(trace, current_time, current_pos)
            if np.isfinite(end_time):
                peak_pos = trace[start_time:end_time].max()
                if (peak_pos - (current_pos+end_pos) / 2.) / (end_time - current_time) >= min_ratio:
                    return current_time, current_pos, end_time, end_pos
    
    return start_time, np.NaN, start_time-1., np.NaN

def trim_overlapping_spike(trace, df_candidates, spike, sub_spikes, min_count, min_slope):
    """
    If spike contains 1 or more sub_spikes within it, we check those to see which should be kept or altered
    """
    # if there is only 1 spike inside, we keep the larger spike
    if sub_spikes.size == 1:
        df_candidates.drop(index=sub_spikes, inplace=True)
        
        return
    
    # we look at the spike sizes
    main_size = df_candidates.loc[spike, 'Amplitude']
    sub_sizes = df_candidates.loc[sub_spikes, 'Amplitude']
    
    # if there are at least 2 sub-spikes more than half the size of the main spike, keep the sub-spikes
    if (sub_sizes > 0.5*main_size).sum() >= 2:
        df_candidates.drop(index=[spike], inplace=True)
    
    # otherwise, we keep the main spike
    df_candidates.drop(index=sub_spikes, inplace=True)
    
    return

def trim_wide_spikes(trace, df_candidates, min_count, min_slope, max_width):
    """
    We shrink or elimiate any spikes that are too wide as needed
    """
    # find all traces that are too wide
    too_wide = df_candidates.index[df_candidates.BaseWidth > max_width]
    
    # if there is at least 1, we iteratively trim them
    for wide in too_wide:
        args = trace, df_candidates.loc[wide, 'Start_Time'], min_count, min_slope, max_width
        df_candidates.loc[wide, ['Start_Time', 'Start_Pos', 'End_Time', 'End_Pos']] = trim_wide_spike(*args)
    
    # remove any whose width have become 0 or less
    df_candidates = df_candidates[~(df_candidates.End_Time.isna())]
    df_candidates = df_candidates[df_candidates.Start_Time < df_candidates.Peak_Time]
    df_candidates = df_candidates[df_candidates.Peak_Time < df_candidates.End_Time].copy()
    
    # drop duplicates
    df_candidates.drop_duplicates(subset='Start_Time', inplace=True)
    
    return df_candidates

def trim_shallow_spikes(trace, df_candidates, min_count, min_slope, min_ratio):
    """
    Sometimes regions where spiking isn't occuring can meet the requirement to be registered as spikes
    If there happens to be an actual spike between them, and the next time that the trace is at their height,
    they can overwrite the real spike. As such, we eliminate these with a cutoff, that spikes can't be too
    shallow
    """
    
    # find all traces that are too shallow
    too_shallow = df_candidates.index[(df_candidates.Amplitude / df_candidates.BaseWidth) < min_ratio]
    
    # if there is at least 1, we iteratively trim them
    for shallow in too_shallow:
        args = trace, df_candidates.loc[shallow, 'Start_Time'], min_count, min_slope, min_ratio
        df_candidates.loc[shallow, ['Start_Time', 'Start_Pos', 'End_Time', 'End_Pos']] = trim_shallow_spike(*args)
        
    # remove any whose width have become 0 or less
    df_candidates = df_candidates[~(df_candidates.End_Time.isna())]
    df_candidates = df_candidates[df_candidates.Start_Time < df_candidates.Peak_Time]
    df_candidates = df_candidates[df_candidates.Peak_Time < df_candidates.End_Time].copy()
    
    # drop duplicates
    df_candidates.drop_duplicates(subset='Start_Time', inplace=True)
    
    return df_candidates

def drop_repeating_spikes(df_candidates):
    """
    If multiple start points lead to the same peak, keep the widest one that doesn't contain any other
    candidate peaks
    """
    
    # get all spikes that contain only one peak
    df_spike = df_candidates.loc[:,['Start_Time', 'End_Time', 'BaseWidth', 'Peak_Time']].copy()
    peak_times = np.unique(df_candidates.Peak_Time)
    df_spike['Count'] = 0
    for ind, row in df_spike.iterrows():
        count = np.logical_and(row.Start_Time < peak_times,
                               row.End_Time > peak_times).sum()
        df_spike.loc[ind, 'Count'] = count
    
    df_spike = df_spike[df_spike.Count == 1]
    
    # Get all peaks that appear at least twice
    peaks = df_spike.Peak_Time.value_counts()
    peaks = peaks[peaks>1]
    
    # drop all but the widest spike for each
    # since they are already sorted by increasign BaseWidth, this means all but hte last
    for peak_time in peaks.index:
        to_drop = df_spike.index[df_spike.Peak_Time == peak_time][:-1]
        df_candidates.drop(index=to_drop, inplace=True)
    
    return

def trim_overlapping_spikes(trace, df_candidates, min_count, min_slope):
    """
    Sometimes we get a candidate spike within another candidate spike. This can happen for a number of reasons
    Sometimes the sub-spike is just noise, and sometimes it is a legitimate signal, and the main spike is too wide
    We try to determine which the case is, and fix the spikes
    """
    
    # we sort the data by sizes, so we go from smallest to largest
    df_candidates.sort_values('BaseWidth', inplace=True)
    
    # iteratively check a spike for contained spikes, and if so fix them
    has_adjusted = True
    while has_adjusted and df_candidates.shape[0] > 0:
        has_adjusted = False
        
        start_size = df_candidates.shape[0]
        drop_repeating_spikes(df_candidates)
        
        for index, row in df_candidates.iterrows():
            df_sub = df_candidates.loc[:index].iloc[:-1]
            if df_sub.shape[0] == 0:
                continue
            sub_inds = np.logical_and(row.End_Time >= df_sub.End_Time,
                                      row.Start_Time <= df_sub.Start_Time
                                     )
            if sub_inds.sum() > 0:
                sub_spikes = df_sub.index[sub_inds]
                trim_overlapping_spike(trace, df_candidates, index, sub_spikes, min_count, min_slope)
                has_adjusted = True
                continue
    
    # drop duplicates
    df_candidates.drop_duplicates(subset='Start_Time', inplace=True)
    
    return df_candidates

def get_half_parameters(trace, df_candidates):
    """
    Get half-width and height related parameters
    """
    # initialize columns
    df_candidates['Half_Pos'] = (df_candidates.Peak_Pos + df_candidates.Base_Pos) / 2.
    df_candidates['Half_Start_Time'] = np.NaN
    df_candidates['Half_End_Time'] = np.NaN
    
    # calculate start and end points by row
    for ind, row in df_candidates.iterrows():
        start_trace = trace[row.Start_Time:row.Peak_Time]
        end_trace = trace[row.Peak_Time:row.End_Time]
        try:
            time1 = start_trace.index[:-1][np.logical_and(start_trace.values[:-1]<=row.Half_Pos,
                                                          start_trace.values[1:]>row.Half_Pos)][0]
            time2 = end_trace.index[:-1][np.logical_and(end_trace.values[:-1]>=row.Half_Pos,
                                                        end_trace.values[1:]<row.Half_Pos)][0]
            df_candidates.loc[ind, 'Half_Start_Time'] = time1
            df_candidates.loc[ind, 'Half_End_Time'] = time2
        except IndexError:
            # these are rows where the half-time could not be calculated
            pass
    
    # remove candidates where we couldn't get either time
    df_candidates.drop(index=df_candidates.index[df_candidates.Half_Start_Time.isna()], inplace=True)
    df_candidates.drop(index=df_candidates.index[df_candidates.Half_End_Time.isna()], inplace=True)
    
    # finally we get the halfwidth
    df_candidates['HalfWidth'] = df_candidates.Half_End_Time - df_candidates.Half_Start_Time
    
    return

def get_peak_positions(trace, df_candidates):
    """
    Calculate the positions of spike peaks, given spike ranges are define
    """
    # iteratively run over each row
    for ind, row in df_candidates.iterrows():
        start_time, end_time = row.Start_Time, row.End_Time
        
        # it is the point with the highest value in the given range
        peak_time = trace[row.Start_Time:row.End_Time].idxmax()
        peak_pos = trace[peak_time]
        df_candidates.loc[ind, ['Peak_Time', 'Peak_Pos']] = peak_time, peak_pos
    
    return

def get_key_parameters(trace, df_candidates):
    """
    Calculate key parameters for spikes
    """
    # calculate peak values
    get_peak_positions(trace, df_candidates)
    
    # we can direct get positional values
    df_candidates['Base_Pos'] = (df_candidates.Start_Pos + df_candidates.End_Pos) / 2.
    df_candidates['Amplitude'] = df_candidates.Peak_Pos - df_candidates.Base_Pos
    df_candidates['BaseWidth'] = df_candidates.End_Time - df_candidates.Start_Time
    df_candidates['Symmetricity'] = (df_candidates.Peak_Time - df_candidates.Start_Time) / df_candidates.BaseWidth
    
    return

def get_end_time(trace, start_time, start_pos):
    """
    Given a trace, and a trace with a starting time and position, we find the first time point after it
    whose value is greater than or equal to start_pos, and the following value is less than it
    """
    
    # get all positions that match the criteria
    candidates = trace.index[:-1][np.logical_and(trace.values[:-1]>=start_pos, trace.values[1:]<start_pos)]
    
    # keep ones that happen after start_time
    candidates = candidates[candidates>start_time]
    
    if candidates.size > 0:
        end_time = candidates[0]
        end_pos = trace.loc[end_time]
        
        return end_time, end_pos
    
    return np.NaN, np.NaN

def get_spike_start_candidates(trace, min_count, min_slope):
    """
    We use the slope to identify potential candidates in a trace where spiking might start
    """
    # get the slope at all points
    slope = pd.Series(np.diff(trace.values) / np.diff(trace.index.values), index=trace.index[:-1])
    
    # we first identify candidate regions as places where the slope is >10V/s for 3 points in a row
    # with the immediately prior point having <10V/s slope (else the candidate would be the prior point)
    datalist = np.zeros((trace.size-min_count-1, min_count+1), dtype=int)
    for col in range(datalist.shape[1]):
        datalist[:,col] = col + np.arange(datalist.shape[0])
    candidates = slope.values[datalist] > min_slope
    candidates[:,0] = ~candidates[:,0]
    
    start_candidates = slope.index.values[1:-min_count+1][np.all(candidates, axis=1)]
    
    # to cut down on potential noise, we restrict to regions where, for at least 0.1ms, the slope is positive
    keep = [potential_spike_start(slope, start, min_count, min_slope) for start in start_candidates]
    start_candidates = start_candidates[keep]
    
    return start_candidates

def get_spike_dataframe(trace, start_candidates):
    """
    Create a dataframe containing start, peak, and end time and position information for each
    spike in start_candidates. We drop those that physically can't be a candidate due to a lack
    of matching opposite position
    """
    
    # we create a dataframe where we can store the spike information
    columns = ['Injection',
               'Start_Time', 'Start_Pos',
               'End_Time', 'End_Pos',
               'Peak_Time', 'Peak_Pos'
              ]
    df_candidates = pd.DataFrame(np.NaN, index=np.arange(start_candidates.size), columns=columns)
    if start_candidates.size == 0:
        get_key_parameters(trace, df_candidates)
        return df_candidates
    df_candidates.Start_Time = start_candidates
    df_candidates.Start_Pos = trace[start_candidates].values
    
    # for each dandidate, find the next time point with equal value to it. This is the end position
    for ind, row in df_candidates.iterrows():
        start_time, start_pos = row.Start_Time, row.Start_Pos
        df_candidates.loc[ind, ['End_Time', 'End_Pos']] = get_end_time(trace, start_time, start_pos)
    
    # drop missing end points
    df_candidates = df_candidates[~df_candidates.End_Time.isna()].copy()
    
    get_key_parameters(trace, df_candidates)
    
    return df_candidates

def get_trace_spikes(trace, min_width=0., min_height=0.):
    """
    Calculate the action potential spike locations in a trace
    """
    
    # create a dataframe containing information on candidate spikes
    start_candidates = get_spike_start_candidates(trace, 3, 10.)
    df_candidates = get_spike_dataframe(trace, start_candidates)
    
    # if min_width and min_height are above 0, we can use those to drop some spikes
    df_candidates = df_candidates[df_candidates.BaseWidth >= min_width]
    df_candidates = df_candidates[df_candidates.Amplitude >= min_height].copy()
    
    # we sanity check the spikes
    # we need to re-calculate the key parameters after each check, since they might change
    spike_count = df_candidates.shape[0]
    if spike_count > 0:
        df_candidates = trim_wide_spikes(trace, df_candidates, 3, 10., 100.)
        spike_count = df_candidates.shape[0]
        if spike_count > 0:
            get_key_parameters(trace, df_candidates)
    if spike_count > 0:
        df_candidates = trim_shallow_spikes(trace, df_candidates, 3, 10., 5.)
        spike_count = df_candidates.shape[0]
        if spike_count > 0:
            get_key_parameters(trace, df_candidates)
    if spike_count > 0:
        df_candidates = trim_overlapping_spikes(trace, df_candidates, 3, 10.)
        get_key_parameters(trace, df_candidates)
        pass
    
    # we get half-width related parameters
    get_half_parameters(trace, df_candidates)
    
    # resort data
    df_candidates.sort_values('Peak_Time', inplace=True)
    
    return df_candidates

def get_atf_spikes(df_atf, min_width=0., min_height=0):
    """
    Calculate the action potentials for every trace in a pandas dataframe
    """
    
    # initialize save variable
    dfs = []
    
    # iterate by trace
    for column, trace in df_atf.iteritems():
        df_spikes = get_trace_spikes(trace, min_width=min_width, min_height=min_height)
        df_spikes.Injection = column
        dfs.append(df_spikes)
    
    # merge results
    df = pd.concat(dfs, axis=0)
    df.set_index('Injection', inplace=True)
    
    return df

def read_spike_data(cell, project):
    """
    Read in a cell's spiking data
    """
    
    fname = f'Calculated/Action_Potentials/{project}/{cell}.tsv'
    df = pd.read_csv(fname, sep='\t', header=0, index_col=None, dtype=float)
    df.set_index('Injection', inplace=True)
    
    return df