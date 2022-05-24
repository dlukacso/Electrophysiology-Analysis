import os
import pandas as pd
from shutil import move

from . import get_paths

def get_move_directory(start_file, start_dir, allow_nesting=True):
    """
    Get the directory where to move a file
    """
    
    # make sure that the files are located in start_dir
    assert start_file.startswith(start_dir), "One of the files isn't at the correct location"
    
    # make sure that the files are stored within a directory within start_dir
    assert len(start_file.split('/')) > 2, "Please put the files within a directory"
    
    # if we allow nesting, only drop start_dir
    if allow_nesting:
        return start_file[len(start_dir)+1:]
    
    # otherwise, we only keep the first directory after start_dir
    
    return os.path.join(start_file.split('/')[1], start_file.split('/')[-1])

def get_ending_pairs(ending='.atf', allow_nesting=True):
    """
    get all files with the given ending, and the location that they should be moved to
    """
    
    # initialize variables
    start_dir = 'EphysData'
    end_dir = get_paths.get_ephys_dir(abf=False)
    start_files = []
    
    # get all target files
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(ending):
                start_files.append(os.path.join(root, file))
    
    # get paired end locations
    end_files = [get_move_directory(start_file, start_dir, allow_nesting=allow_nesting) for start_file in start_files]
    
    file_pairs = [(start_file, os.path.join(end_dir, end_file)) for start_file, end_file in zip(start_files, end_files)]
    
    return file_pairs

def get_file_pairs():
    """
    Get all .abf and .atf files that need to be moved
    Then, for all of them get the paths to their targets
    Return results as a list of tuple pairs
    """
    
    # get pairs for atf files
    atf_pairs = get_ending_pairs(ending='.atf', allow_nesting=True)
    
    # get pairs for abf files
    abf_pairs = get_ending_pairs(ending='.abf', allow_nesting=False)
    
    return atf_pairs + abf_pairs

def move_files():
    """
    Move .abf and .atf files to a different internal storage drive
    """
    # get pair of current and future file locations for all files to be moved
    file_pairs = get_file_pairs()
    
    # get list of all target directories, and create missing ones
    directories = {'/'.join(target.split('/')[:-1]) for current, target in file_pairs}
    for directory in directories:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    
    # move files
    for current_path, target_path in file_pairs:
        move(current_path, target_path)
    
    return