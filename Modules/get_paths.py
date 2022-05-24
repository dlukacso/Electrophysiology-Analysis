import os

def get_ephys_dir(abf=False):
    """
    Get the directory where ephy files are stores
    """
    
    ephys_dir = '/media/foldy_lab/Storage_Analysis/Electrophysiology'
    if abf:
        ephys_dir = f'{ephys_dir}/ABF'
    
    return ephys_dir

def get_project_dirs(project):
    """
    Get the directories where a project's atf files are stored
    """
    
    # get the directory where all things are stored
    ephys_dir = get_ephys_dir(abf=False)
    
    # define the full paths
    paths = {'Cclamp':f'{ephys_dir}/Cclamp/{project}',
             'Vclamp':f'{ephys_dir}/Vclamp/{project}',
             'Aclamp':f'{ephys_dir}/Aclamp/{project}'
            }
    
    return paths

def get_header_dirs(project):
    """
    Get the directories where a project's abf file headers are stored
    """
    
    # get the directory where all things are stored
    head_dir = f'Headers/{project}'
    
    # define the full paths
    paths = {'Cclamp':f'{head_dir}/Cclamp',
             'Vclamp':f'{head_dir}/Vclamp',
             'Aclamp':f'{head_dir}/Aclamp'
            }
    
    return paths

def get_assess_dirs(dataset):
    """
    Get the directories where a project's assessment figures are stored
    """
    
    # define the base directory
    base_dir = f'Plots/{dataset}'
    
    # define subdirectories
    sub_dirs = [f'{base_dir}/Traces_CC',
                f'{base_dir}/ABF_CC',
                f'{base_dir}/ABF_VC',
                f'{base_dir}/Sigmoid',
               ]
    
    return [base_dir] + sub_dirs

def get_spike_dirs(projects):
    """
    Get the directories where we save a project's spiking data
    """
    
    return {project:f'Calculated/Action_Potentials/{project}' for project in projects}

def make_project_dirs(project):
    """
    Checks if the save folders for a project exist, and if not, makes them
    """
    # define project paths
    paths = get_project_dirs(project)
    
    # iteratively check each
    # we use makedirs instead of mkdir so that nested projects are possible
    for path in paths.values():
        if not os.path.isdir(path):
            os.makedirs(path)
            
    return paths

def make_header_dirs(project):
    """
    Checks if the save folder for a project's headers exists, and if not, makes them
    """
    # define project paths
    paths = get_header_dirs(project)
    
    # iteratively check each
    # we use makedirs instead of mkdir so that nested projects are possible
    for path in paths.values():
        if not os.path.isdir(path):
            os.makedirs(path)
            
    return paths

def make_assess_dirs(dataset):
    """
    Create the directory and sub-directories where we store the figures for assessing a dataset
    """
    
    # define dataset paths
    paths = get_assess_dirs(dataset)
    
    # iteratively check each
    # we use makedirs instead of mkdir so that nested projects are possible
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
            
    return paths

def make_spike_dirs(projects):
    """
    Create directories where we can save the spiking data for projects
    """
    
    # define directory paths
    paths = get_spike_dirs(projects)
    
    # iteratively check each
    # we use makedirs instead of mkdir so that nested projects are possible
    for path in paths.values():
        if not os.path.isdir(path):
            os.makedirs(path)
            
    return paths