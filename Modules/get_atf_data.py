import os
import numpy as np
import pandas as pd

from . import read_atf
from . import get_paths
from . import get_vc_data
from . import get_cc_data
from . import assess_cc_data

def get_parameters_lists():
    """
    Get the list of calculated electrophysiological parameters, and the order that they are in
    """
    
    # define parameters for each method separately
    cc_parameters = get_cc_data.get_cc_parameters()
    vc_parameters = get_vc_data.get_vc_parameters()
    
    return cc_parameters, vc_parameters

def update_with_recording_values(df):
    """
    Update a list of calculated electrophysiological values, using the reference table of
    replacement values for individual cells that people could add to for whatever reason
    """
    
    # read in prerecorded values
    df_rec = read_atf.get_recording_values(df.index)
    df_rec = df_rec.loc[:,df_rec.columns.isin(df.columns)].copy()
    
    # fill in missing values
    df_sub = df.loc[df_rec.index,df_rec.columns]
    for cell, data in df_rec.iterrows():
        missing = data.isna()
        if missing.sum() == 0:
            continue
        df_rec.loc[cell, missing] = df_sub.loc[cell, missing]
    
    # replace values
    df.loc[df_rec.index,df_rec.columns] = df_rec
    
    return

def calculate_dataset_data(df_ref, spike_dirs, custom_start=900., custom_end=1300.):
    """
    Calculate the electophsiological properties for a list of cells, and their corresponding 
    atf file paths. As there existing atf files that throw errors in every project appears to
    be inevitable, the code is set up to merely output which cells cause problems and to skip 
    those. It is up to the researcher to decide if they wish to dig into why those particular
    cells are problematic, or to toss them out.
    """
    # create result dataframe
    cc_parameters, vc_parameters = get_parameters_lists()
    columns = ['CellType'] + vc_parameters + cc_parameters
    df = pd.DataFrame(np.NaN, index=df_ref.index, columns=columns)
    
    # iteratively get data for each cell
    for (cell, project), row in df_ref.iterrows():
        cc_path = row.CCName
        vc_path = row.VCName
        
        # calculate voltage clamp data
        if len(vc_path) > 0:
            try:
                df_atf = read_atf.read_atf_file(vc_path)
                df.loc[(cell, project), vc_parameters] = get_vc_data.get_vc_data(df_atf, row)
            except:
                print(f'There were problems with the VC file for {cell} ({project})')
        # calculate current clamp data
        if len(cc_path) > 0:
            try:
                df_atf = read_atf.read_atf_file(cc_path)
                df_atf, df_spike, pass_qc = assess_cc_data.assess_cc_data(df_atf, row)
                save_dir = spike_dirs[project]
                df_spike.to_csv(f'{save_dir}/{cell}.tsv', sep='\t')
                if pass_qc:
                    args = (df_atf, df_spike, row)
                    kwargs = {'custom_start':custom_start, 'custom_end':custom_end}
                    df.loc[(cell, project), cc_parameters] = get_cc_data.get_cc_data(*args, **kwargs)
                else:
                    print(f'{cell} ({project}) failed QC')
            except:
                print(f'There were problems with the CC file for {cell} ({project})')
    
    # add other parameters
    df.CellType = df_ref.CellType
    
    return df

def get_ephys_values(dataset, celltypes=[], projects=[], custom_start=800., custom_end=1500.):
    """
    Calculate the electrophysiological properties for a list of celltypes and projects
    And save it under the name of dataset
    """
    
    # get target files
    df_ref = read_atf.get_dataset_targets(projects, celltypes, add_metadata=True)
    spike_dirs = get_paths.make_spike_dirs(projects)
    
    # calculate the data
    df = calculate_dataset_data(df_ref, spike_dirs, custom_start=custom_start, custom_end=custom_end)
    update_with_recording_values(df)
    
    # drop project names. We do it this late, so that if people reuse the same cell name
    # the code can still run. The results will just be confusing with repeated names
    df.index = df.index.get_level_values('Cell')
    df.sort_index(inplace=True)
    
    # save the results
    df.to_csv(f'Calculated/Compiled/{dataset}.tsv', sep='\t')
    
    return