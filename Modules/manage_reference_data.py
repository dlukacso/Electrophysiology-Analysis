import os
import numpy as np
import pandas as pd

from . import get_paths

def read_reference_file(fname):
    """
    Reads in a reference file for converting .abf files to .atf files. It removes unnecessary
    columns, and tries to handle different file types
    """
    # adjust reference file path
    fname = f'References/ConversionFiles/{fname}'
    # read in the reference file
    kwargs = {'header':0, 'index_col':None, 'dtype':str}
    if fname.endswith('.xls') or fname.endswith('.xlsx'):
        df = pd.read_excel(fname, engine="openpyxl", **kwargs)
    else:
        sep = ',' if fname.endswith('.csv') else '\t'
        df = pd.read_csv(fname, sep=sep, **kwargs)

    # trim columns
    if not ('ACName') in df.columns.values:
        df['ACName'] = ''
    if not ('ACChannel') in df.columns.values:
        df['ACChannel'] = ''
    columns = ['CellName', 'CellType', 'Project',
               'CCName', 'CCChannel',
               'VCName', 'VCChannel',
               'ACName', 'ACChannel']
    for column in columns:
        text = f'The column "{column}" is not in the reference file. Please check for any typos'
        assert column in df.columns.values, text
    df = df.loc[:,columns].copy()
    
    # adjust data columns to account for missing values and numerical indices
    df.fillna('', inplace=True)
    df.set_index(['CellName', 'Project'], inplace=True)
    
    return df

def get_abf_targets(refname, projects, celltypes):
    """
    Given either a reference file, project, or cell types, get a dataframe of the corresponding abf files
    """
    
    # if the reference file exists, we use that
    try:
        # read in reference file
        df_ref = read_reference_file(refname)
        
        # trim data
        if len(projects) > 0:
            df_ref = df_ref.loc[df_ref.index.get_level_values('Project').isin(projects)]
        if len(celltypes) > 0:
            df_ref = df_ref.loc[df_ref.CellType.isin(celltypes)]
        
        return df_ref.loc[:,['CCName', 'VCName', 'ACName']]
    except:
        # otherwise, we stick to already converted data
        print('Could not read reference file, trying already converted abfs only')
        assert (len(projects) + len(celltypes)) > 0, "Either 'projects' or 'celltypes' needs to be non-empty"
        
        # read in the metadata files
        filename = 'References/Metadata/ABF_Matches.tsv'
        df_abf = pd.read_csv(filename, sep='\t', header=0, index_col=[0,1], dtype=str)
        df_abf.fillna('', inplace=True)
        filename = 'References/Metadata/Cell_Parameters.tsv',
        df_params = pd.read_csv(filename, sep='\t', header=0, index_col=[0,1], dtype=str)
        df_params.fillna('', inplace=True)
        
        # trim down
        if len(projects) > 0:
            df_params = df_params.loc[df_params.index.get_level_values('Project').isin(projects)]
        if len(celltypes) > 0:
            df_params = df_params.loc[df_params.CellType.isin(celltypes)]
        df_abf = df_abf.loc[df_abf.index.isin(df_params.index)].copy()
        
        return df_abf
    
    return

def get_abf_paths(refname, projects, celltypes):
    """
    Given either a reference file, project, or cell types, get a dataframe of paths to the corresponding abf files
    """
    
    # get the list of abf files
    df_abf = get_abf_targets(refname, projects, celltypes)
    
    # remove missing cells
    remove_missing(df_abf)
    
    # make sure that cells exist
    assert df_abf.shape[0] > 0, "There were no cells belonging to both the provided projects and celltypes"
    
    return df_abf

def evaluate_cell_existence(cell, project, row):
    """
    evaluate if the .abf files for a cell are present or not
    """
    # initialize variables
    ephys_dir = get_paths.get_ephys_dir(abf=True)
    ccname, vcname, acname = row.CCName, row.VCName, row.ACName
    
    # check if it has a project that it belongs to
    if len(project.strip()) == 0:
        return ['Dropped', 'No Project']
    
    # check if it has any files at all
    if max(len(ccname), len(vcname), len(acname)) == 0:
        return ['Dropped', 'No Files']
    
    # iteratively check each file
    # we only keep it if at least 1 file is present
    to_keep = False
    titles = ('CC File', 'VC File', 'AC File')
    names = (ccname, vcname, acname)
    dropped = []
    for name, title in zip(names, titles):
        if len(name) == 0:
            continue
        fname = f'{ephys_dir}/{name}.abf'
        if not os.path.isfile(fname):
            dropped.append(title)
        else:
            to_keep = True
    
    kept = 'Kept' if to_keep else 'Dropped'
    
    return [kept] + dropped

def evaluate_channel_existence(cell, project, row):
    """
    evalute if the specified .abf files have corresponding, specified channels
    """
    # initialize variables
    ccname, vcname, acname = row.CCName, row.VCName, row.ACName
    ccchannel, vcchannel, acchannel = row.CCChannel, row.VCChannel, row.ACChannel
    
    # check if it has any files at all
    has_cc = np.logical_and(len(ccname) > 0, len(ccchannel) > 0)
    has_vc = np.logical_and(len(vcname) > 0, len(vcchannel) > 0)
    has_ac = np.logical_and(len(acname) > 0, len(acchannel) > 0)
    if not np.any([has_cc, has_vc, has_ac]):
        return ['Dropped', 'No Channels']
    
    # iteratively check each file to see if any are dropped
    # due to missing channels
    titles = ('CC File', 'VC File', 'AC File')
    names = (ccname, vcname, acname)
    conditions = (has_cc, has_vc, has_ac)
    dropped = []
    for name, title, condition in zip(names, titles, conditions):
        if len(name) == 0:
            continue
        if not condition:
            dropped.append(title)
    
    return ['Kept'] + dropped

def print_missing_results(missing):
    """
    Print out the results of which cells have missing files, or were completely removed
    from the conversion process
    """
    # start with how many have problems and how many are kept
    kept = [cell for cell, cond in missing.items() if cond[0] == 'Kept']
    correct = [cell for cell, cond in missing.items() if len(cond) == 1]
    
    if len(correct) == len(missing):
        print(f"All {len(missing)} cells had all abf files")
        
        return
    
    print(f"Out of {len(missing)} cells, {len(correct)} cells had existing abf files.")
    print(f"Overall, {len(kept)} out of {len(missing)} cells had files to convert.")
    
    # get a sub-dictionary that only contains cells with errors
    # this creates a copy of the dictionary, so that we can use pop to remove future cells
    missing = {cell:cond for cell, cond in missing.items() if len(cond) > 1}
    
    # print cells without projects
    no_projects = [cell[0] for cell, cond in missing.items() if cond[1] == 'No Project']
    if len(no_projects) > 0:
        print(f"The following {len(no_projects)} cells were dropped due to having no corresponding projects:")
        print(', '.join(no_projects))
        for cell in no_projects:
            missing.pop(cell)
    
    # print cells that had no files
    no_files = [cell for cell, cond in missing.items() if cond[1] == 'No Files']
    if len(no_files) > 0:
        print(f"The following {len(no_files)} cells were dropped due to having no abf files:")
        print(', '.join([f'{cell} ({project})' for cell, project in no_files]))
        for cell in no_files:
            missing.pop(cell)
    
    # print cells that had all missing files
    miss_files = [cell for cell, cond in missing.items() if cond[0] == 'Dropped']
    if len(miss_files) > 0:
        print(f"The following {len(miss_files)} cells were dropped due to all"\
              " of their abf files missing:"
             )
        print(', '.join([f'{cell} ({project})' for cell, project in miss_files]))
        for cell in miss_files:
            missing.pop(cell)
    
    # for the remainder, print out which of their files were missing
    if len(missing) > 0:
        print(f"The following {len(missing)} cells had at least 1 abf file missing."\
              " The remainder will be converted:"
             )
        for (cell, project), cond in missing.items():
            print(f"Cell: {cell}, Project: {project}, ABF Files: {', '.join(cond[1:])}")
    
    return

def print_missing_channels(missing):
    """
    Print out the results of which cells have missing channels, or were completely removed
    for missing all channels
    """
    # start with how many have problems and how many are kept
    kept = [cell for cell, cond in missing.items() if cond[0] == 'Kept']
    correct = [cell for cell, cond in missing.items() if len(cond) == 1]
    
    if len(correct) == len(missing):
        print(f"All {len(missing)} cells had no problems with their channels")
        
        return
    
    print(f"Out of {len(missing)} cells, {len(correct)} cells had no problems with their channels.")
    print(f"Overall, {len(kept)} out of {len(missing)} cells had some channels.")
    
    # get a sub-dictionary that only contains cells with errors
    # this creates a copy of the dictionary, so that we can use pop to remove future cells
    missing = {cell:cond for cell, cond in missing.items() if len(cond) > 1}
    
    # print cells that had no channels
    no_channel = [cell for cell, cond in missing.items() if cond[1] == 'No Channels']
    if len(no_channel) > 0:
        print(f"The following {len(no_channel)} cells were dropped due to having no abf channels:")
        print(', '.join([f'{cell} ({project})' for cell, project in no_channel]))
        for cell in no_channel:
            missing.pop(cell)
    
    # for the remainder, print out which of their files were missing
    if len(missing) > 0:
        print(f"The following {len(missing)} cells had at least unspecified channel."\
              " The remainder will be converted:"
             )
        for (cell, project), cond in missing.items():
            print(f"Cell: {cell}, Project: {project}, Channels: {', '.join(cond[1:])}")
    
    return

def remove_missing(df):
    """
    Remove cells for whom at least one of the listed .abf files are missing
    """
    
    # iteratively check for each cell
    missing = {(cell, project):evaluate_cell_existence(cell, project, row)
               for (cell, project), row in df.iterrows()}
    
    # print out the results
    print_missing_results(missing)
    
    # remove dropped cells
    kept = [cell for cell, cond in missing.items() if cond[0] == 'Kept']
    df = df.loc[kept,:]
    missing = {cell:cond[1:] for cell, cond in missing.items() if cell in kept and len(cond) > 1}
    
    # remove missing abf files
    converter = {'CC File':'CCName', 'VC File':'VCName', 'AC File':'ACName'}
    for cell, abfs in missing.items():
        for abf in abfs:
            df.loc[cell, converter[abf]] = ''
    
    df = df.copy()
    
    return df

def remove_no_channels(df):
    """
    Remove abf files that don't have channels specified
    """
    
    # iteratively check for each cell
    missing = {(cell, project):evaluate_channel_existence(cell, project, row)
               for (cell, project), row in df.iterrows()}
    
    # print out the results
    print_missing_channels(missing)
    
    # remove dropped cells
    kept = [cell for cell, cond in missing.items() if cond[0] == 'Kept']
    df = df.loc[kept,:]
    missing = {cell:cond[1:] for cell, cond in missing.items() if cell in kept and len(cond) > 1}
    
    # remove missing abf files
    converter = {'CC File':'CCName', 'VC File':'VCName', 'AC File':'ACName'}
    for cell, abfs in missing.items():
        for abf in abfs:
            df.loc[cell, converter[abf]] = ''
    
    df = df.copy()
    
    return df
    

def initialize_metadata(df):
    """
    Given a dataframe produced from reading in a reference file, initialize a dataframe
    to store the necessary metadata
    """
    
    # define columns
    measurements = ['Start (pA)', 'Step (pA)',
                    'Start CC (ms)', 'End CC (ms)',
                    'Start VC (ms)', 'End VC (ms)',
                    'Signal (mV)',
                   ]
    fileinfo = ['CellType', 'CC_ending',
                'VC_ending', 'AC_ending']
    
    # create dataframe
    df_meta = pd.DataFrame('', index=df.index, columns=fileinfo)
    df_meta.CellType = df.CellType
    df_measure = pd.DataFrame(np.NaN, index=df.index, columns=measurements)
    df_meta = pd.concat((df_meta, df_measure), axis=1)
    
    return df_meta

def fillout_metadata(df_meta, result):
    """
    Fill out the values of a metadata dataframe based on the conversion results
    """
    
    # convert results from a dict of dicts to a dataframe
    df_result = pd.DataFrame(result).T
    df_meta = df_meta.loc[df_meta.index.isin(df_result.index)]
    df_result = df_result.loc[df_meta.index,:]
    
    # add endings
    df_meta.CC_ending = df_result.CC_ending
    df_meta.VC_ending = df_result.VC_ending
    df_meta.AC_ending = df_result.AC_ending
    
    # add start and step
    df_meta['Start (pA)'] = df_result.Start
    df_meta['Step (pA)'] = df_result.Step
    
    return df_meta

def update_metadata_file(filename, df):
    """
    Update a single metadata file with either new cells, or modifications to existing cell information
    """
    
    # read in the old data
    df_meta = pd.read_csv(filename, sep='\t', header=0, index_col=[0,1], dtype=str)
    df_meta.fillna('', inplace=True)
    
    # reduce to key columns
    columns = df_meta.columns
    missing = columns[~(columns.isin(df.columns))]
    assertion_text = (f"The following columns are missing from the new metadata: {', '.join(missing)}\n"\
                      f"Metadata columns: {', '.join(df.columns)}")
    assert missing.size == 0, assertion_text
    df = df.loc[:,columns].copy()
    
    # overwrite modified data
    df_both = df.loc[df.index.isin(df_meta.index),:].copy()
    df_both_meta = df_meta.loc[df_both.index]
    # start by making sure we don't delete data
    for row, data in df_both.iterrows():
        replace = data == ''
        if replace.sum() > 0:
            df_both.loc[row, replace] = df_both_meta.loc[row, replace]
    #df_both[df_both==''] = df_both_meta[df_both==''].values
    df_meta.loc[df_both.index] = df_both
    
    # add new data
    df_new = df.loc[~(df.index.isin(df_meta.index)),:]
    df_meta = pd.concat((df_meta, df_new), axis=0)
    
    # save the result
    df_meta.to_csv(filename, sep='\t')
    
    return

def update_metadata(df_meta, df):
    """
    Update metadata files with new cells
    """
    
    # make sure that we only have the same cells in both dataframes
    # we also want to drop any duplicate columns (CellType is in both) before merging them
    df_meta = df_meta.loc[df_meta.index.isin(df.index),:]
    df = df.loc[df_meta.index,~df.columns.isin(df_meta.columns)]
    df = pd.concat((df, df_meta), axis=1)
    
    # update files one at a time
    filenames = ('References/Metadata/ABF_Matches.tsv',
                 'References/Metadata/Cell_Parameters.tsv',
                 'References/Metadata/Recording_Configurations_CC.tsv',
                 'References/Metadata/Recording_Configurations_VC.tsv'
                )
    for filename in filenames:
        update_metadata_file(filename, df)
    
    return