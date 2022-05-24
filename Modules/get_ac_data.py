import numpy as np
import pandas as pd
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

electro_dir = '/media/foldy_lab/Storage_Analysis/Electrophysiology'

def get_ac_data(df):
    # calculate lowess fit
    xvals = df.index
    yvals = df.iloc[:,0]
    params = {'frac':.040, 'delta':0.2, 'is_sorted':True, 'return_sorted':False}
    fit = lowess(yvals, xvals, **params)
    
    # get average resting state
    v_rest = np.median(fit)
    
    return [v_rest]