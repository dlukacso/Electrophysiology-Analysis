import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from sklearn.neighbors.kde import KernelDensity

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams["xtick.major.size"] = 2
plt.rcParams["ytick.major.size"] = 2
plt.rcParams['xtick.major.width'] = .5
plt.rcParams['ytick.major.width'] = .5

def define_axes(fig, top, num, left=.1, right=.9, row_count=6, height=.12, dw=.045, dh=.02):
    """
    generate a series of uniformly spaces axes to do plotting on
    Inputs:
        fig - matplotlib.pyplot figure
        top - the top of the axes, axs a fraction of figure size
        num - number of axes to be generated
        left - the left edge of the axes, as a fraction of figure size. Default 0.1
        right - the right edge of the axes, as a fraction of girue size. Default 0.9
        row_count - the number of axes that should be on a single row. Default 6
        height - the height of each axis, as a fraction of figure size. Default 0.12
        dw - the horizontal spacing between axes, as a fraction of figure size. Default 0.045
        dh - the vertical spacing between axes, as a fraction of figure size. Default 0.02
    Outputs:
        axes - the axes generated, stored as a list going from left to right, top to bottom
        bot - bottom edge where the axes end, as a fraction of figure size
    """
    dw = dw
    width = (right-left) / row_count - dw
    bot = top - height
    axes = []
    
    for i in range(num):
        row = i // row_count
        col = i % row_count
        ax = fig.add_axes([left+dw+(dw+width)*col, bot-(dh+height)*row, width, height])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        axes.append(ax)

    return axes, bot-(dh+height)*((num-1)//row_count)

def label_axis(ax, label, ticklabels, rotation=0):
    """
    do labels and tick adjustments for axis
    Inputs:
        ax - the axis
        ylabel - y-axis label
        ticklabels - x-axis tick labels
        rotation - how much the ticks should be rotated. Default is 0
    """
    
    positions = np.arange(len(ticklabels))+.5
    ax.set_ylabel(label, fontsize=7, labelpad=1)
    ax.set_xlim(0, len(ticklabels))
    ax.tick_params(labelsize=6, pad=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(ticklabels, rotation=rotation)
    
    return

def plot_ax_violin(ax, data, xval, color, marker='o', width=.5, show_violin=True, show_error=False, show_box=False, show_scatter=True):
    """
    plot a series of points. Can be a violin plot (default) or box and whiskers plot
    Inputs:
        ax - axis to plot on
        data - data to plot
        xval - x-value on axis on which to plot data
        color - color of the plot
        marker - marker shape to plot points with. Default 'o'
        width - width of the plot. Defautl is 0.5
        show_violin - whether to do a violin plot. Default is True
        show_box - whether to do a box plot. Default is False
        show_error - whether to plot mean and error. Default is False
    """
    # remove non finite values    
    data = data[np.isfinite(data)]
    if data.size==0:
        return
    
    # scatter points
    if show_scatter:
        params = {'s':16,
                  'facecolor':'none',
                  'edgecolor':'black',
                  'linewidth':.25,
                  'zorder':2,
                  'marker':marker
                 }
        ax.scatter([xval]*data.size, data, **params)
    
    # plot errors
    if show_error:
        mean, error = data.mean(), data.std() / np.sqrt(data.size)
        ax.errorbar(xval, mean, yerr=error, color='k', capsize=4, zorder=1, elinewidth=.5, capthick=.5)
        ax.errorbar(xval, mean, yerr=0, color='k', capsize=6, zorder=1, elinewidth=.5, capthick=.5)
    
    # plot box plot
    if show_box:
        # do plot
        boxy = ax.boxplot(data, positions=[xval], widths=width, notch=True, whis=2, showfliers=False, zorder=0)
        # adjust color
        #for key, vals in boxy.items():
        #    for val in vals:
        #        val.set_color(color)
    
    # plot violin plot
    if show_violin:        
        # calculate step size for KDE
        low, high = np.min(data), np.max(data)
        if high == 0:
            high = 1
        if low == high:
            low = high*.9
        diff = high-low
        step = diff/15
        yvals = np.linspace(low,high,100)
        
        # generate and plot KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=step).fit(data[:,np.newaxis])
        log_dens = np.exp(kde.score_samples(yvals[:,np.newaxis]))
        log_dens = log_dens / np.max(log_dens)*width / 2
        ax.fill_betweenx(yvals, log_dens+xval, xval-log_dens, alpha=.5, color=color, linewidth=0, edgecolor='face', zorder=0)
    
    return

def plot_sequencing(fig, top, celltypes, celltype_labels, colors=['blue', 'red'], tissue=False, left=.1, right=.9, use_cols='All', row_count=6, rotation=0, ticklabels=False, limits=None, show_violin=True, show_error=False, plot_box=False, dh=.02):
    # do plotting of sequencing on figure
    
    if len(colors) < len(celltypes):
        colors += [np.random.rand(3) for i in range(len(celltypes)-len(colors))]
    
    # read in data
    df = get_sequencing_data(celltypes)
    
    # create sequencing data
    label_order = ['Reads', 'Aligned', 'Mapped', 'GeneCount', 'AlignRate', 'MapRate']
    label_names = ['Reads (M)', 'Aligned (M)', 'Mapped (M)', 'Gene Count (K)', 'Align Rate (%)', 'Mapping Rate (%)']
    normalization = [1/1000000, 1/1000000, 1/1000000, 1/1000, 1., 1.]
    if not limits:
        if not tissue:
            limits = [(0,25), (0, 25), (0, 25), (0, 12), (0, 100), (0, 100)]
        else:
            limits = [(0,40), (0, 30), (0, 20), (0, 18), (0, 100), (0, 100)]
    
    # normalize data
    for label, norm in zip(label_order, normalization):
        df[label] = df[label]*norm
        
    # pick only categories we want to plot
    if use_cols == 'All':
        use_cols = [i for i in range(len(limits))]
    plot_count = len(use_cols)
    
    label_order = [label_order[col] for col in use_cols]
    label_names = [label_names[col] for col in use_cols]
    normalization = [normalization[col] for col in use_cols]
    limits = [limits[col] for col in use_cols]
    kept = ['CellType'] + label_order
    df = df[kept].set_index('CellType')
    
    positions = np.arange(len(celltypes))+.5
    
    # do plotting
    axes, bot = define_axes(fig, top, plot_count, left=left, right=right, row_count=row_count, dh=dh)
    #ticklabels = [category.split('-')[0] for category in categories]
    if not ticklabels:
        ticklabels = [celltype_label.replace('-','-\n') for celltype_label in celltype_labels]
    for ax, label_name, label_ord, limit in zip(axes, label_names, label_order, limits):
        for celltype, position, color in zip(celltypes, positions, colors):
            data = df.loc[celltype, label_ord]
            plot_ax_violin(ax, data, position, color, width=.4, show_violin=show_violin, show_error=show_error, plot_box=plot_box)
        ax.set_ylabel(label_name, fontsize=8, labelpad=1, **hfont)
        ax.axis([0,len(celltypes),limit[0],limit[1]])
        ax.tick_params(size=1, labelsize=6, pad=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels, rotation=rotation, **hfont)

    return bot

def add_pval(ax, series, categories):
    
    assert len(categories) == 2
    
    series = series[~series.isna()]
    
    data1 = series.loc[categories[0]]
    data2 = series.loc[categories[1]]
    pval = ttest_ind(data1, data2, equal_var=False)[1]
    
    ax.text(.5, .99, 'p = %.2e' % pval, fontsize=5, transform=ax.transAxes, ha='center', va='top')
    
    return

def plot_generated_electrophys(fig, top, df, categories, color_dict={}, marker_dict={}, left=.1, right=.9, row_count=6, rotation=0, ticklabels=False, limits=[], show_violin=True, show_error=False, show_box=False, show_scatter=True, height=.12, dh=.02, calc_pval=False):
    """
    
    """
    
    # initialize variables
    axes, bot = define_axes(fig, top, df.shape[1], left=left, right=right, row_count=row_count, height=height, dh=dh)
    positions = np.arange(len(categories))+.5
    if type(ticklabels) == str:
        ticklabels = [ticklabels] * len(categories)
    elif type(ticklabels) == bool:
        ticklabels = ['\n'.join(category.split('.')) for category in categories]    
    if len(limits) < df.shape[1]:
        limits = limits + [(None,None)] * (df.shape[1] - len(limits))
    
    # do plotting
    params = {'width':.5,
              'show_violin':show_violin,
              'show_error':show_error,
              'show_box':show_box,
              'show_scatter':show_scatter
             }
    
    for ax, label, limit in zip(axes, df.columns, limits):
        ax.set_xlim(0, len(categories))
        low = df.loc[:,label].min()
        high = df.loc[:,label].max()
        #ax.set_ylim(low-.2, high+.2)
        for category, position in zip(categories, positions):
            color = color_dict.get(category, 'black')
            marker = marker_dict.get(category, 'o')
            try:
                data = df.loc[[category], label]
            except:
                print(df.shape, np.unique(df.index))
                assert False
            plot_ax_violin(ax, data, position, color, marker=marker, **params)
        
        label_axis(ax, label, ticklabels, rotation=rotation)
        if calc_pval and len(categories)==2:
            add_pval(ax, df.loc[df.index.isin(categories), label], categories)
        ax.set_ylim(limit[0], limit[1])
    
    return bot

def plot_electrophys(fig, top, categories, refs, colors=['red', 'blue'], left=.1, right=.9, use_cols='All', row_count=6, height=.06, show_violin=True, show_error=False, plot_box=False):
    # do plotting of electrophys on figure

    # read in data
    celltypes = {cell for category in categories for cell in refs[category]}
    dataframes = get_electrophys_data(categories)
    
    # create sequencing data
    label_order = ['Voltage Resting', 'Maximum Frequency', 'Firing Threshold', 'Resistance Input', 'Resistance Series', 'Capacitance', 'Basewidth', 'Halfwidth', 'Symmetricity', 'Peak', 'Saq Potential', 'Attenuation', 'Delay']
    label_names = ['V resting (mV)', 'Max. AP firing (Hz)', 'Firing thresh (pA)', 'Input res. (MΩ)', 'Series res. (MΩ)', 'Capacitance (pF)', 'AP base-width (ms)', 'AP half-width (ms)', 'AP symmetricity', 'AP peak (mV)', 'Sag Potential (mV)', 'Attenuation', 'Delay (ms)']
    
    limits = [(-85,-40), (0, 125), (0, 180), (80, 500), (5, 30), (20, 120),
             (.4,1.4), (.2,.8), (.28,.46), (60, 105), (0,15), (1.,1.7),(None,None)]
    
    # pick only categories we want to plot
    if use_cols == 'All':
        use_cols = [i for i in range(len(limits))]
    plot_count = len(use_cols)
    
    label_order = [label_order[col] for col in use_cols]
    label_names = [label_names[col] for col in use_cols]
    limits = [limits[col] for col in use_cols]
    
    plot_data = [[dataframe[label] for dataframe in dataframes] for label in label_order]
    positions = np.arange(len(categories))+.5
    
    # do plotting
    axes, bot = define_axes(fig, top, plot_count, left=left, right=right, row_count=row_count, height=height, dw=0.045)
    ticklabels = [category.replace('-','-\n') for category in categories]
    for ax, label, data, limit in zip(axes, label_names, plot_data, limits):
        #ax.violinplot(data, positions=positions, showextrema=False)
        for cat_data, position, color in zip(data, positions, colors):
            plot_ax_violin(ax, cat_data, position, color, width=.25, show_violin=show_violin, show_error=show_error, plot_box=plot_box)
        ax.set_ylabel(label, fontsize=8, labelpad=1)
        ax.axis([0,len(categories),limit[0],limit[1]])
        ax.tick_params(size=1, labelsize=7, pad=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels, **hfont)
    
    return bot
