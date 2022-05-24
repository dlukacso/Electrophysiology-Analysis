import pyabf
import numpy as np
import pandas as pd
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
import warnings

def read_header(abf):
    """
    Read the header of an abf file
    We use this function, because normally pyabf's header reading throws multiple a warning.
    This gets noisy with a lot of files, and refuses to be suppressed, so we try to ignore the warnings
    It doesn't work on all, unfortunately
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        header = abf.headerText.split('\n')
    
    return header

def read_file_header(fname):
    """
    Read the header of an abf file. Different from read_header in that here we pass the path to an abf file
    instead of an already opened abf file
    """
    # read file 
    abf = pyabf.ABF(fname) 
    
    # read header
    header = read_header(abf)
    
    return header

def read_abf_traces(path, trace_num=5):
    """
    Read in the first trace_num traces of an abf file for all channels
    """
    # read file
    abf = pyabf.ABF(path)
    
    # get channel names
    header = read_header(abf)
    adcnames = [line.split(' = ')[1] for line in header if line.startswith('adcNames ')][0]
    adcnames = [data.strip("'") for data in adcnames.strip("][").split(', ')]
    
    # initialize save variable, and get dataset sizes
    dfs = []
    xvals = np.around(abf.sweepX * 1000, 2)
    
    # read in channel by channel
    
    for channel in range(len(adcnames)):
        try:
            # we save the data in lists, because sometimes trace sizes are inconsistent
            # we want to be able to spot this error
            data = [xvals]
            for col in range(trace_num):
                abf.setSweep(sweepNumber = col, channel=channel)
                data.append(abf.sweepY)

            dfs.append(data)
        except:
            continue
    
    return dfs, adcnames

def read_header_basics(abf, channel):
    """
    Read in the basic stuff that we need from the headers for all abf files
    """
    # get header
    header = read_header(abf)
    
    # get channel
    adcnames = [line.split(' = ')[1] for line in header if line.startswith('adcNames')][0]
    adcnames = [data.strip("'") for data in adcnames.strip("][").split(', ')]
    text = f"The channel {channel} doesn't exist. Options are {', '.join(adcnames)}"
    assert channel in adcnames, text
    channel_index = adcnames.index(channel)
    
    # read number and size of sweeps
    sweepCount = [int(line.split()[2]) for line in header if line.startswith('sweepCount')][0]
    sweepPointCount = [int(line.split()[2]) for line in header if line.startswith('sweepPointCount')][0]
    
    return header, sweepCount, sweepPointCount, channel_index

def read_ac_header(abf, channel):
    """
    Read in the header for an abf file where we don't know the exact formatting, and return basic information
    """
    
    # read the stuff
    header, sweepCount, sweepPointCount, channel_index = read_header_basics(abf, channel)
    
    return sweepCount, sweepPointCount, channel_index

def read_vc_header(abf, channel):
    """
    Read in the header for a Voltage Clamp abf file. We return metadata parameters
    """
    # get basics
    header, sweepCount, sweepPointCount, channel_index = read_header_basics(abf, channel)
    
    # get time step size
    rate = [float(line.split(' = ')[1]) for line in header if line.startswith('dataRate = ')][0]
    time_step = 1000. / rate
    
    # get target epoch
    epochs = [line.split('Step')[1:] for line in header if line.startswith('sweepEpochs = ')][0]
    epochs = [epoch.strip(', ').split() for epoch in epochs]
    epochs = [epoch for epoch in epochs if float(epoch[0]) != 0.][0]
    
    # get signal strength
    signal = float(epochs[0])
    
    # get signal time points
    times = [int(epoch) for epoch in epochs[1].strip('[]').split(':')]
    voltageStart = np.around(times[0] * time_step,2)
    voltageEnd = np.around(times[1] * time_step,2)
    
    # define metadata
    vc_meta = pd.Series([voltageStart, voltageEnd, signal],
                        index=['Start VC (ms)', 'End VC (ms)', 'Signal (mV)'])
    
    return sweepCount, sweepPointCount, channel_index, vc_meta

def read_cc_header(abf, channel):
    """
    Read in the header for a Current Clamp abf file. We return metadata parameters
    """
    # get basics
    header, sweepCount, sweepPointCount, channel_index = read_header_basics(abf, channel)
    
    # get start and step size
    sweepStart = [line.split(' = ')[1].strip('][') for line in header if line.startswith('fEpochInitLevel')][0]
    sweepStart = [float(data) for data in sweepStart.strip("][").split(', ')][2*channel_index+1]
    sweepStep = [line.split(' = ')[1].strip('][') for line in header if line.startswith('fEpochLevelInc')][0]
    sweepStep = [float(data) for data in sweepStep.strip("][").split(', ')][2*channel_index+1]
    
    # adjust by multiplier
    scale = [line.split(' = ')[1] for line in header if line.startswith('fDACScaleFactor')][0]
    scale = [float(data) for data in scale.strip("][").split(', ')][channel_index]
    scalefactor = 400. / scale
    sweepStart = sweepStart * scalefactor
    sweepStep = sweepStep * scalefactor
    
    # calculate time points
    rate = [float(line.split(' = ')[1]) for line in header if line.startswith('dataRate = ')][0]
    time_step = 1000. / rate
    epochs = [line.split('Step')[1:] for line in header if line.startswith('sweepEpochs = ')][0]
    epochs = [epoch.strip(', ').split() for epoch in epochs]
    epochs = [epoch[1].strip('[]') for epoch in epochs if float(epoch[0]) != 0.][0]
    epochs = [int(epoch) for epoch in epochs.split(':')]
    injectStart = epochs[0] * time_step
    injectEnd = epochs[1] * time_step
    
    # define metadata
    cc_meta = pd.Series([sweepStart, sweepStep, injectStart, injectEnd],
                        index=['Start (pA)', 'Step (pA)', 'Start CC (ms)', 'End CC (ms)'])
    
    return sweepCount, sweepPointCount, channel_index, cc_meta

def read_ac_data(fname, channel):
    """
    Read in an abf file of unknown format, and return a basic dataframe of the measured values
    """
    # read in the data
    abf = pyabf.ABF(fname)
    sweepCount, sweepPointCount, channel = read_ac_header(abf, channel)
    
    # initialize dataframe
    xvals = np.around(abf.sweepX * 1000, 2)
    yvals = np.arange(1, dtype=int)
    df = pd.DataFrame(0., index=xvals, columns=yvals)
    df.index.name = 'Time (ms)'
    
    # add column values
    abf.setSweep(sweepNumber = 0, channel = channel)
    df[0] = abf.sweepY
    
    df.columns = ['Trace #0']
    
    return df

def read_vc_data(fname, channel):
    """
    Read in a Voltage Clamp abf file, and return a basic dataframe of the measured values
    """
    # read in the data
    abf = pyabf.ABF(fname)
    sweepCount, sweepPointCount, channel, vc_meta = read_vc_header(abf, channel)
    
    # initialize results dataframe
    xvals = np.around(abf.sweepX * 1000, 2)
    yvals = np.arange(sweepCount, dtype = int)
    df = pd.DataFrame(0., index=xvals, columns=yvals)
    df.index.name = 'Time (ms)'
        
    # read in values column by column
    for col in df.columns:
        abf.setSweep(sweepNumber = col, channel=channel)
        df[col] = abf.sweepY
        
    # take the mean
    data = df.mean(axis=1)
    df = pd.DataFrame(data.values, index=data.index, columns=['Trace #1'])
    
    return df, vc_meta

def read_cc_data(fname, channel):
    """
    Read in a Current Clamp abf file, and return a basic dataframe of the measured values
    """
    # read in the data
    abf = pyabf.ABF(fname)
    sweepCount, sweepPointCount, channel, cc_meta = read_cc_header(abf, channel)
    
    # initialize results dataframe
    xvals = np.around(abf.sweepX * 1000, 2)
    yvals = np.arange(sweepCount, dtype = int)
    df = pd.DataFrame(0., index=xvals, columns=yvals)
    df.index.name = 'Time (ms)'
    
    # read in values for each column
    for col in df.columns:
        abf.setSweep(sweepNumber = col, channel = channel)
        df[col] = abf.sweepY
        
    df.columns = ['Trace #%d' % (col+1) for col in df.columns]
    
    return df, cc_meta