import numpy as np
import pandas as pd

from . import get_paths

def read_atf_file(filename):
    """
    Read in an atf file
    """
    # read the data
    df = pd.read_csv(filename, sep='\t', skiprows=3, header=None, index_col=0)
    
    # adjust columns and index
    df.index.name = 'Time'
    df.index = df.index.astype(float)
    df.columns = np.arange(df.shape[1])
    df.columns.name = 'Trace'
    
    return df

def get_recording_values(cells):
    """
    Get the precorded values for an array-like of cell-project pairs
    """
    
    # read in the data
    filename = 'References/Metadata/Recording_Values.tsv'
    kwargs = {'sep':'\t', 'header':0, 'index_col':[0,1]}
    df_rec = pd.read_csv(filename, **kwargs)
    
    # reduce to data of interest
    df_rec = df_rec.loc[df_rec.index.isin(cells),:]
    
    return df_rec

def add_dataset_metadata(df):
    """
    For each cell-project pair, get the corresponding current and voltage clamp metadata values
    """
    
    # read in values
    kwargs = {'sep':'\t', 'header':0, 'index_col':[0,1]}
    df_cc = pd.read_csv('References/Metadata/Recording_Configurations_CC.tsv', **kwargs)
    df_vc = pd.read_csv('References/Metadata/Recording_Configurations_VC.tsv', **kwargs)
    
    # index data to match dataset
    df_cc = df_cc.reindex(df.index)
    df_vc = df_vc.reindex(df.index)
    
    # merge datasets
    df = pd.concat((df, df_vc, df_cc), axis=1)
    
    return df

def get_dataset_targets(projects, celltypes, add_metadata=False, allow_all=False):
    """
    For a list of projects and celltypes, return a dataframe of the cells contained within,
    their cell types, and the paths to all of their .atf files
    """
    
    # check if projects or celltypes are specified
    # if allow_all is False, not specifying at least 1 will return an error
    if not allow_all:
        error_text = ("You haven't specified any projects or cell types."\
                      " Please specify at least one of the two.")
        assert max(len(celltypes), len(projects)) > 0, error_text
    
    # read in the basic metadata
    kwargs = {'sep':'\t', 'header':0, 'index_col':None, 'dtype':str}
    df = pd.read_csv('References/Metadata/Cell_Parameters.tsv', **kwargs)
    df.fillna('', inplace=True)
    
    # restrict our search space to projects and celltypes of interest
    if len(projects) > 0:
        df = df.loc[df.Project.isin(projects),:]
    if len(celltypes) > 0:
        df = df.loc[df.CellType.isin(celltypes),:]
    df = df.copy()
    
    # add paths
    ephys_dir = get_paths.get_ephys_dir(abf=False)
    df['CCName'] = ephys_dir + '/Cclamp/' + df.Project + '/' + df.Cell + df.CC_ending
    df['VCName'] = ephys_dir + '/Vclamp/' + df.Project + '/' + df.Cell + df.VC_ending
    df['ACName'] = ephys_dir + '/Aclamp/' + df.Project + '/' + df.Cell + df.AC_ending
    
    # remove paths for missing cells
    df.loc[df.CC_ending == '', 'CCName'] = ''
    df.loc[df.VC_ending == '', 'VCName'] = ''
    df.loc[df.AC_ending == '', 'ACName'] = ''
    
    # adjust index and columns
    df.set_index(['Cell', 'Project'], inplace=True)
    df = df.loc[:,['CellType', 'CCName', 'VCName', 'ACName']]
    
    # add current clamp start and step if needed
    if add_metadata:
        df = add_dataset_metadata(df)
    
    return df