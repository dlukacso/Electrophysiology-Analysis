import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import read_abf
from . import read_atf
from . import get_paths
from . import manage_reference_data
from . import assess_cc_data
from . import get_frequency_data

def trim_trace(trace, peak_times, n=10):
    """
    For a trace, trim the number of points that we use by a factor of n, so that the plot uses less space
    We also make sure to keep all of the peaks and the minima between them
    """
    
    # keep every nth point
    base_time = trace.index[0:trace.size:10]
    if peak_times.size == 0:
        return trace[base_time]
    
    # keep minimum values in-between peaks
    times = sorted(set([0] + list(peak_times) + [trace.index[-1]]))
    lows = [trace[times[i]:times[i+1]].idxmin() for i in range(len(times)-1)]
    
    times = base_time.tolist() + times + lows
    times = np.unique(times)
    
    trace = trace[times]
    
    return trace

def add_sigmoid_axes(cell, fig, row, col):
    """
    Create an axes for sigmoid plots
    """
    # create the axes
    ax = fig.add_axes([0.078 + 0.18*col, 0.73 - 0.17*row, 0.132, 0.102])
    
    # label axes
    ax.tick_params(size=1, labelsize=6)
    ax.set_title(cell, fontsize=8)
    ax.set_xlabel('Current (pA)', fontsize=8)
    if col == 0:
        ax.set_ylabel('Frequency (Hz)', fontsize=8)
    
    return ax

def check_atf_file(filepath):
    """
    Check an atf file to see if it exists, and if so, whether it is in a tab-separated matrix format
    """
    # check existence
    if not os.path.isfile(filepath):
        return 'Missing'
    
    # check if pandas can read it
    try:
        df = read_atf.read_atf_file(filepath)
        if df.shape[0] == 0:
            return 'No Time Points'
        if df.shape[1] == 0:
            return 'No Traces'
        return 'Normal'
    except:
        pass
    
    # check if we can read it with open file
    try:
        with open(filepath) as f:
            data = [line.split('\t') for line in f]
        if len(data) < 4:
            return 'No Time Points'
        sizes = {len(line) for line in data[2:]}
        if len(sizes) > 1:
            return 'Inconsistent Line Sizes'
        return "Pandas Can't Open"
    except:
        return "Can't Open"
    
    return "New Error"

def check_dataset_formatting(projects, celltypes):
    """
    For all cells in a dataset - defined by a list of projects and cell types - we assess their atf
    files to see if they exist, and if so, whether they are in a tab-separated matrix format of
    consistant size
    """
    # get target cells
    df_ref = read_atf.get_dataset_targets(projects, celltypes, add_metadata=False)
    df_ref.index = df_ref.index.get_level_values('Cell')
    df_ref.sort_index(inplace=True)
    
    # strip to data of interest
    df = df_ref.loc[:,['CCName', 'VCName', 'ACName']].copy()
    
    # evaluate each cell by row
    for cell in df.index:
        for column in df.columns:
            path = df.loc[cell, column]
            if len(path) == 0:
                continue
            df.loc[cell, column] = check_atf_file(path)
    
    # get good cells
    okay = np.logical_or(df=='', df=='Normal')
    good = okay.sum(axis=1) == okay.shape[1]
    
    # print problem information
    problems = (~good).sum()
    if problems == 0:
        print(f"All {df.shape[0]} cells had no problems.")
        return
    
    print(f"Out of {df.shape[0]} cells, {problems} had problems with their atf files")
    df = df.loc[~good,:]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    
    return

def plot_vc_trace_data(df_vc, save_name):
    """
    For a given list of cells and their voltage clamp atf files, plot their traces
    """
    
    # initialize save figure
    pp = PdfPages(save_name)
    
    # iterate through each cell
    
    for cellnum, (cell, path) in enumerate(zip(df_vc.index, df_vc.VCName)):
        # read atf file
        df = read_atf.read_atf_file(path)
        
        # determine cell positioning
        axnum = cellnum % 12
        row = axnum // 3
        col = axnum % 3
        
        # if needed, create a new page
        if axnum == 0:
            fig = plt.figure(figsize=(8.5,11))
        
        # create axes
        ax = fig.add_axes([0.16 + 0.26*col, 0.7 - 0.2*row, 0.22, 0.17])
        
        # plot data
        ax.plot(df.index.values, df.iloc[:,0], linewidth=1)
        ax.set_title(cell, fontsize=8)
        ax.set_xlim(0,20)
        ax.tick_params(size=1, labelsize=5, pad=1)
        
        # if end of page, save the page
        if axnum == 11:
            pp.savefig(fig)
            plt.close()
    
    # if we didn't reach end of page, save the page
    if axnum < 11:
        pp.savefig(fig)
        plt.close()
    
    # close and save figure
    pp.close()
    plt.close()
    
    return

def plot_cc_trace(cell, path, row, save_name):
    """
    For a given cell, plot their current clamp traces
    """
    # read in plotting data
    df_atf = read_atf.read_atf_file(path)
    df_atf, df_spike, pass_qc = assess_cc_data.assess_cc_data(df_atf, row, drop_fail=False)
    
    # initialize plotting pdf
    pp = PdfPages(save_name)
    low = df_atf.values.min() - 10
    high = 100
    
    # iteratively plot each trace
    for tracenum, (current, qc) in enumerate(zip(df_atf.columns, pass_qc)):
        trace = df_atf[current]
        # determine trace positioning
        axnum = tracenum % 30
        row = axnum // 5
        col = axnum % 5
        
        # if needed, create a new page
        if axnum == 0:
            fig = plt.figure(figsize=(8.5,11))
        
        # create axes
        ax = fig.add_axes([0.139 + 0.160*col, 0.775 - 0.125*row, 0.121, 0.0935])
        
        # get peak data and trim trace
        df_trace_spike = df_spike.loc[df_spike.index==current,:]
        trace = trim_trace(trace, df_trace_spike.Peak_Time.values, n=50)
        color = 'black' if qc else 'orange'
        
        # plot data
        ax.plot(trace.index.values, trace.values, linewidth=0.25, color=color, zorder=0)
        ax.scatter(df_trace_spike.Peak_Time.values, df_trace_spike.Peak_Pos.values, s=1, color='red', zorder=1)
        ax.set_title('%.1f pA' % current, fontsize=8)
        ax.axis([0, 3000, low, high])
        ax.tick_params(size=1, labelsize=5, pad=1)
        
        # if end of page, save the page
        if axnum == 29:
            pp.savefig(fig)
            plt.close()
    
    # if we didn't reach end of page, save the page
    if axnum < 29:
        pp.savefig(fig)
        plt.close()
    
    # close and save figure
    pp.close()
    plt.close()
    
    return

def plot_cc_trace_data(df_cc, save_dir):
    """
    For a given list of cells and their current clamp atf files, plot their traces
    """
    
    # plot cell by cell
    for cell, row in df_cc.iterrows():
        path = row.CCName
        save_name = f'{save_dir}/{cell}.pdf'
        plot_cc_trace(cell, path, row, save_name)
        
    return

def make_trace_plot(xvals, data, title, fig, row, col, width):
    """
    Plot out a single trace's data
    """
    
    # create axes
    ax = fig.add_axes([col, 0.76 - 0.16*row, width, 0.1275])
    
    # adjust x-axis data size if needed
    if xvals.size != data.size:
        diff = np.diff(xvals).mean()
        xvals = np.arange(data.size) * diff
    
    # plot the data
    ax.plot(xvals, data, c='k')
    
    # adjust axes
    ax.set_title(title, fontsize=8)
    ax.set_xlim(xvals.min(), xvals.max())
    ax.tick_params(size=1, labelsize=5, pad=1)
    
    return

def create_trace_figure(count):
    """
    Create a figure that is wide enough to fit all of the traces from the different channels
    """
    
    # initialize variables
    borders = 1.
    width = 1.4025
    shift = 0.2975
    
    total_width = 2*borders + count*width + (count-1)*shift
    cols = np.arange(count) * (width+shift) + borders + shift
    
    fig = plt.figure(figsize=(total_width,11))
    cols = cols / total_width
    width = width / total_width
    
    return fig, cols, width

def plot_abf_trace_data(series, save_dir, label=''):
    """
    For a given list of cells and corresponding abf files, plot their traces
    """
    
    # get file locatoin
    ephys_dir = get_paths.get_ephys_dir(abf=True)
    
    # iterate through each cell to plot it
    for cellnum, ((cell, project), abf) in enumerate(series.items()):
        # read abf file
        if len(abf) == 0:
            continue
        read_path = f'{ephys_dir}/{abf}.abf'
        save_path = f'{save_dir}/{project}_{cell}.pdf'
        try:
            dfs, titles = read_abf.read_abf_traces(read_path, trace_num=5)

            # create a figure page
            fig, cols, width = create_trace_figure(len(dfs))

            # plot 1 row and column at a time
            for col, dataset, title in zip(cols, dfs, titles):
                xvals = dataset[0]
                for row, data in enumerate(dataset[1:]):
                    if row > 0:
                        title = ''
                    make_trace_plot(xvals, data, title, fig, row, col, width)

            fig.savefig(save_path)
            plt.close()
        except:
            print(f"Couldn't read {label} abf files for {cell} ({project})")
    
    return

def plot_trace_data(dataset, projects, celltypes):
    """
    For all cells, we plot both voltage clamp and current clamp traces, to see if their shapes make sense
    """
    # get target cells
    df_ref = read_atf.get_dataset_targets(projects, celltypes, add_metadata=True)
    df_ref.index = df_ref.index.get_level_values('Cell')
    df_ref.sort_index(inplace=True)
    
    # get save path
    save_dir = get_paths.make_assess_dirs(dataset)[0]
    
    # plot voltage clamp traces
    df_vc = df_ref.loc[df_ref.VCName.str.len() > 0,:]
    save_name = f'{save_dir}/Traces_VC.pdf'
    plot_vc_trace_data(df_vc, save_name)
    
    # plot current clamp traces
    df_cc = df_ref.loc[df_ref.CCName.str.len() > 0,:]
    save_dir_cc = f'{save_dir}/Traces_CC'
    plot_cc_trace_data(df_cc, save_dir_cc)
    
    return
    
def plot_sigmoid_data(dataset, projects, celltypes, custom_start=800., custom_end=1000.):
    """
    For all cells, we try to make a spike frequency vs current injection plot on current clamp data
    By an assumption, these should produce approximately sigmoid plots. Since there are 7 distinct
    frequencies, we save 7 distinct pdf files
    """
    
    # get target cells
    df_ref = read_atf.get_dataset_targets(projects, celltypes, add_metadata=True)
    df_ref.index = df_ref.index.get_level_values('Cell')
    df_ref.sort_index(inplace=True)
    
    # remove cells without current clamp files
    df_ref = df_ref[df_ref.CCName.str.len() > 0]
    
    # get save path
    save_dir = get_paths.make_assess_dirs(dataset)[4]
    frequency_names = get_frequency_data.get_frequency_names(append_endings=False)
    save_names = {frequency_name:f'{save_dir}/{frequency_name}.pdf'
                  for frequency_name in frequency_names}
    
    # create pdf files
    pps = {frequency_name:PdfPages(save_name) for frequency_name, save_name in save_names.items()}
    
    # plot 1 cell at a time
    for cellenum, (cell, row) in enumerate(df_ref.iterrows()):
        # determine cell positioning
        axnum = cellenum % 25
        ax_row = axnum // 5
        ax_col = axnum % 5
        
        # if needed, create a new page
        if axnum == 0:
            figures = {frequency_name:plt.figure(figsize=(8.5,11)) for frequency_name in frequency_names}
        
        # read in plotting data
        df_atf = read_atf.read_atf_file(row.CCName)
        df_atf, df_spike, pass_qc = assess_cc_data.assess_cc_data(df_atf, row, drop_fail=False)
        args = (df_atf, df_spike, row['Start CC (ms)'], row['End CC (ms)'], custom_start, custom_end)
        df_freqs = get_frequency_data.calculate_trace_frequency_data(*args)
        
        # plot data
        for frequency_name, frequencies in df_freqs.iteritems():
            fig = figures[frequency_name]
            ax = add_sigmoid_axes(cell, fig, ax_row, ax_col)
            ax.scatter(frequencies.index.values, frequencies.values, s=4, c='k')
        
        # if end of page, save the page
        if axnum == 24:
            for frequency_name in frequency_names:
                pps[frequency_name].savefig(figures[frequency_name])
        plt.close()
    
    # if we didn't reach end of page, save the page
    if axnum < 24:
        for frequency_name in frequency_names:
            pps[frequency_name].savefig(figures[frequency_name])
        plt.close()
    
    # close and save figure
    for frequency_name in frequency_names:
        pps[frequency_name].close()
    
    return

def plot_abf_data(dataset, reference_file):
    """
    For all cells, we read in their abf files, and plot the first 5 traces of each of their channels
    """
    
    # get target files
    df_ref = manage_reference_data.get_abf_paths(reference_file, [], [])
    
    # get save path
    save_cc_dir, save_vc_dir = get_paths.make_assess_dirs(dataset)[2:4]
    
    # plot voltage clamp traces
    df_vc = df_ref.loc[df_ref.VCName.str.len() > 0,'VCName']
    plot_abf_trace_data(df_vc, save_vc_dir, label='VC')
    
    # plot current clamp traces
    df_cc = df_ref.loc[df_ref.CCName.str.len() > 0,'CCName']
    plot_abf_trace_data(df_cc, save_cc_dir, label='CC')
    
    return

def save_abf_headers(reference_file):
    """
    For all cells, we read in their abf files, and save their headers as text files for reading
    """
   
    # get target files
    df_targets = manage_reference_data.get_abf_paths(reference_file, [], [])
    
    # create target directory
    projects = df_targets.index.get_level_values('Project')
    for project in np.unique(projects):
        get_paths.make_header_dirs(project)
    
    # rename columns to directory reference names
    df_targets.columns = [f'{column[0]}clamp' for column in df_targets.columns]
    ephys_dir = get_paths.get_ephys_dir(abf=True)
    
    # create headers
    for (cell, project), row in df_targets.iterrows():
        save_dirs = get_paths.get_header_dirs(project)
        for key, abf in row.items():
            try:
                if len(abf) > 0:
                    header = read_abf.read_file_header(f'{ephys_dir}/{abf}.abf')
                    save_dir = save_dirs[key]
                    save_name = f'{save_dir}/{cell}.txt'
                    with open(save_name, 'w') as w:
                        w.write('\n'.join(header))
            except:
                print(f'Could not reader {key} header for {cell} ({project})')
    
    return    