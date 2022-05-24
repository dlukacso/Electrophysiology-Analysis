import numpy as np
import pandas as pd

from . import get_paths
from . import create_atf
from . import manage_reference_data

def print_unconverted(result):
    """
    Print out cells whose abf files were only paritially or not at all converted
    """
    # print out cells that had no conversions, and rmeove them
    non_conv = [cell for cell, values in result.items() if not values['Convert']]
    if len(non_conv) > 0:
        print(f"Out of {len(result)} cells, the following {len(non_conv)} cells couldn't get"
              f" any of their .abf files converted:"
             )
        print(', '.join([f'{cell} ({project})' for cell, project in non_conv]))
    for cell in non_conv:
        result.pop(cell)
    
    # print out cells with partial conversions
    part_conv = []
    for cell, value in result.items():
        failures = []
        for name in ('CC', 'VC', 'AC'):
            if not value[f'{name}_success']:
                failures.append(f'{name} File')
        if len(failures) > 0:
            part_conv.append((cell, failures))
    if len(part_conv) > 0:
        print(f"The following {len(part_conv)} cells had at least 1 unconverted abf file:")
        for (cell, project), failures in part_conv:
            failures = ', '.join(failures)
            print(f'{cell} ({project})\t{failures}')
    
    return

def convert_files(fname):
    """
    Read in a reference file that contains basic metadata - cell type, project name, and 
    corresponding abf files - for a cell, and convert them to atf files, while adding their
    information to the metadata files
    """
    
    # read in the reference file
    df = manage_reference_data.read_reference_file(fname)
    df = manage_reference_data.remove_missing(df)
    df = manage_reference_data.remove_no_channels(df)
    
    text = (f"There are no .abf files for conversion. Please check that the"\
            " storage drive is mounted, that you have transferred the files,"\
            " and that they have channels specified.")
    assert df.shape[0] > 0, text
    
    # create a metadata dataframe
    df_meta = manage_reference_data.initialize_metadata(df)
    
    # create missing folders
    projects = np.unique(df.index.get_level_values('Project'))
    for project in projects:
        get_paths.make_project_dirs(project)
    
    # iteratively run each cell
    result = {}
    for (cell, project), row in df.iterrows():
        result[(cell, project)] = create_atf.create_cell(cell, project, row, df_meta)
    
    # print out problems, and remove unconverted
    print_unconverted(result)
    df_meta = df_meta.loc[df_meta.index.isin(result.keys())]
    
    # update metadata files
    #df_meta = manage_reference_data.fillout_metadata(df_meta, result)
    manage_reference_data.update_metadata(df_meta, df)
    
    return