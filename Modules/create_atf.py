import numpy as np

from . import read_abf
from . import get_paths

def write_atf(df, out_name):
    """
    Write a pandas dataframe as an atf file
    """
    # define header
    header = 'ATF\t1.0\n0\t8\n'
    
    # write the data
    with open(out_name, 'w') as ict:
        ict.write(header)
        
        df.to_csv(ict, sep='\t')
    
    return

def create_ac_file(in_name, out_name, channel):
    """
    Create a corresponding atf file for an abf file whose structure is unknown
    """
    # read in the data
    df = read_abf.read_ac_data(in_name, channel)
    
    # write the data
    write_atf(df, out_name)
    
    return

def create_vc_file(in_name, out_name, channel):
    """
    Create a corresponding atf file for a Voltage Clamp abf file
    """
    # read in the data
    df, vc_meta = read_abf.read_vc_data(in_name, channel)
    
    # write the data
    write_atf(df, out_name)
        
    return vc_meta

def create_cc_file(in_name, out_name, channel):
    # read in the data
    df, cc_meta, = read_abf.read_cc_data(in_name, channel)
    
    # write the data
    write_atf(df, out_name)
    
    return cc_meta

def create_cell(cell, project, row, df_meta):
    """
    For a cell, convert all .abf files into .atf files. Also return information on which
    conversions failed
    """
    
    # initialize variables
    result = {'Convert':False, 'CC_success':True,
              'VC_success':True, 'AC_success':True
             }
    
    # define directories
    abf_dir = get_paths.get_ephys_dir(abf=True)
    project_dirs = get_paths.get_project_dirs(project)
    
    # convert ac data
    acname = row.ACName
    acchannel = row.ACChannel
    if (len(acname) > 0) and (len(acchannel) > 0):
        abf_in = f'{abf_dir}/{acname}.abf'
        ac_dir = project_dirs['Aclamp']
        atf_out = f'{ac_dir}/{cell}_AC.atf'
        try:
            create_ac_file(abf_in, atf_out, acchannel)
            df_meta.loc[(cell, project), 'AC_ending'] = '_AC.atf'
            result['Convert'] = True
        except:
            result['AC_success'] = False
    
    # convert vc data
    vcname = row.VCName
    vcchannel = row.VCChannel
    if (len(vcname) > 0) and (len(vcchannel) > 0):
        abf_in = f'{abf_dir}/{vcname}.abf'
        vc_dir = project_dirs['Vclamp']
        atf_out = f'{vc_dir}/{cell}_VC.atf'
        #try:
        vc_meta = create_vc_file(abf_in, atf_out, vcchannel)
        df_meta.loc[(cell, project), 'VC_ending'] = '_VC.atf'
        df_meta.loc[(cell, project), vc_meta.index] = vc_meta.values
        result['Convert'] = True
        #except:
        #    result['VC_success'] = False
    
    # convert cc data
    ccname = row.CCName
    ccchannel = row.CCChannel
    if (len(ccname) > 0) and (len(ccchannel) > 0):
        abf_in = f'{abf_dir}/{ccname}.abf'
        cc_dir = project_dirs['Cclamp']
        atf_out = f'{cc_dir}/{cell}_CC.atf'
        #try:
        cc_meta = create_cc_file(abf_in, atf_out, ccchannel)
        df_meta.loc[(cell, project), 'CC_ending'] = '_CC.atf'
        df_meta.loc[(cell, project), cc_meta.index] = cc_meta.values
        result['Convert'] = True
        #except:
        #    result['CC_success'] = False
    
    return result