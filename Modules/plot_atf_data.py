import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import read_atf
from . import get_atf_data
from . import get_spike_data
from . import plot_ax_violins
from . import get_frequency_data

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams["xtick.major.size"] = 2
plt.rcParams["ytick.major.size"] = 2
plt.rcParams['xtick.major.width'] = .5
plt.rcParams['ytick.major.width'] = .5

def generate_random_color():
    """
    Generate a random hexadecial color
    """
    
    # get triplet of integer values
    values = np.random.randint(0,256,3)
    
    # convert them into hexadecimal values
    hex_values = [hex(value) for value in values]
    
    # fix their formatting
    hex_values = ['%02s' % value[2:] for value in hex_values]
    
    # merge them, replace empty spaces with zeros, and use upper case numbers
    color = '#' + ''.join(hex_values).replace(' ', '0').upper()
    
    return color

def adjust_plotting_parameters(df, plot_inds, plot_range, color_dict, name_converter):
    """
    Make some slightly adjustments to the variables used for ephys plotting to be more compatible with the
    rest of the program
    """
    
    # trim to columns that we want to plot
    if len(plot_inds) > 0:
        converter = {ind:column for ind, column in enumerate(df.columns, start=1)}
        plot_inds = [converter.get(index, index) for index in plot_inds]
        for plot_ind in plot_inds:
            text = (f'The column {plot_ind} does not exist in the dataframe.'\
                    ' Options are:\n{', '.join(df.columns)}')
            assert plot_ind in df.columns.values, text
        df = df.loc[:,plot_inds].copy()
    
    # get plot ranges
    converter = {ind:column for ind, column in enumerate(df.columns, start=1)}
    plot_range = {converter.get(label,label):value for label, value in plot_range.items()}
    plot_range = [plot_range.get(column, (None, None)) for column in df.columns]
    
    # fill out color_dict for any missing celltypes
    color_dict = {CellType:color_dict.get(CellType, generate_random_color()) for CellType in df.index}
    
    # adjust any multi-line columns
    df.columns = [name_converter.get(column,column) for column in df.columns]
    
    return df, plot_range, color_dict

def read_compiled_data(dataset, celltypes=None):
    """
    Read in already compiled and evaluated data
    """
    
    # read in the data
    try:
        df = pd.read_csv(f'Calculated/Compiled/{dataset}.tsv', sep='\t', header=0, index_col=0)
    except FileNotFoundError:
        print(f"There is no summary file for {dataset}. Did you produce it? Check in Calculated for {dataset}.tsv")
        
        return
    
    # trim to cell types of interest
    if celltypes is None or len(celltypes) == 0:
        celltypes = np.unique(df.CellType)
        return df, celltypes
    df = df[df.CellType.isin(celltypes)]
    celltypes = np.unique(df.CellType)
    
    return df, celltypes

def plot_ephys_values(dataset, plot_inds=[], celltypes=[], plot_range={}, color_dict={}, violin_args={}, name_converter={}):
    """
    Plot the electrophysiological valueus as violin plots, with scatters of the values shown
    """
    
    # read in the data
    try:
        df, celltypes = read_compiled_data(dataset, celltypes=celltypes)
    except TypeError:
        # indicates that the file wasn't read in, so there were no values
        return
    df.set_index('CellType', inplace=True)
    
    # adjust the dataframe, plot_range, and color_dict
    df, plot_range, color_dict = adjust_plotting_parameters(df, plot_inds, plot_range, color_dict, name_converter)
    
    # define keyword argument variables
    kwargs = {'left':.1,
              'right':.9,
              'row_count':5,
              'rotation':45,
              'ticklabels':celltypes,
              'show_violin':True,
              'show_error':True,
              'show_scatter':False,
              'show_box':False,
              'height':.07,
              'color_dict':color_dict,
              'limits':plot_range,
              'dh':.06}
    kwargs.update(violin_args)
    
    # plot the data in chunks of 30 properties
    save_name = f'Plots/{dataset}/Distributions.pdf'
    save_dir = os.path.split(save_name)[0]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    pp = PdfPages(save_name)
    
    count = 30
    for start_col in range(0,df.shape[1],count):
        end_col = min(start_col+count, df.shape[1])
        fig = plt.figure(figsize=(8.5,11))
        args = (fig, .9, df.iloc[:,start_col:end_col], celltypes)
        plot_ax_violins.plot_generated_electrophys(*args, **kwargs)
        pp.savefig(fig)
        plt.close()
    pp.close()
    
    return

def plot_action_potential_count(dataset, savename, projects=None, celltypes=None, color_dict=None, thresholds=None, min_count=3):
    """
    Plot the action potential count vs current injection of current clamp data
    """
    
    # adjust input variables
    if projects is None:
        projects = []
    if celltypes is None:
        celltypes = []
    if color_dict is None:
        color_dict = {}
    if thresholds is None:
        thresholds = {}
    
    # get target files
    df_ref = read_atf.get_dataset_targets(projects, celltypes, add_metadata=True)
    try:
        df, celltypes = read_compiled_data(dataset, celltypes=celltypes)
    except TypeError:
        # indicates that the file wasn't read in, so there were no values
        return
    
    # restrict to data that matches all of the provided parameters
    for parameter, (lower_bound, upper_bound) in thresholds.items():
        if not (lower_bound is None):
            df = df[df[parameter] >= lower_bound]
        if not (upper_bound is None):
            df = df[df[parameter] <= upper_bound]
    df_ref = df_ref.loc[df_ref.index.get_level_values('Cell').isin(df.index),:].copy()
    df = df.loc[df_ref.index.get_level_values('Cell'),:].copy()
    df.index = df_ref.index
    
    # get the spike counts for all cells
    df_count = []
    for (cell, project), row in df_ref.iterrows():
        try:
            start_time = row['Start CC (ms)']
            end_time = row['End CC (ms)']
            df_spike = get_spike_data.read_spike_data(cell, project)
            df_count.append(get_frequency_data.calculate_spike_count(df_spike, start_time, end_time))
        except:
            print(f"Couldn't get spike counts for cell {cell} ({project}).")
    df_count = pd.concat(df_count, join='outer', sort=True, axis=1).interpolate(method='index', limit_area='inside')
    df_count.columns = df_ref.index
    
    # fill in all np.NaN values before the first detected spike with 0.
    for column, data in df_count.iteritems():
        indices = data.index[~(data.isna())]
        if indices.size == 0:
            continue
        index = indices[0]
        df_count.loc[df_count.index < index, column] = 0.
    
    # plot the data
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
    df_count.columns = df.CellType
    for celltype in celltypes:
        # get plotting paramters
        df_plot = df_count.loc[:,df_count.columns==celltype]
        df_plot = df_plot.loc[(~(df_plot.isna())).sum(axis=1)>=min_count,:]
        data_plot = df_plot.mean(axis=1)
        error_plot = df_plot.std(axis=1) / np.sqrt((~(df_plot.isna())).sum(axis=1))
        color = color_dict.get(celltype, generate_random_color())
        
        # plot data
        kwargs = {'color':color, 'capsize':2, 'linewidth':0.5, 'elinewidth':0.5}
        ax.errorbar(data_plot.index.values, data_plot.values, yerr=error_plot.values, **kwargs)
    
    # adjust axes
    ax.tick_params(size=2, labelsize=7)
    ax.set_xlabel('Current injection (pA)', fontsize=9)
    ax.set_ylabel('AP count', fontsize=9)
    ax.set_xlim(0,None)
    ax.set_ylim(0,None)
    
    fig.savefig(f'Plots/{dataset}/{savename}')
    
    plt.close()
    
    return df_count