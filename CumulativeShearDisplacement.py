import numpy as np
import pandas as pd

def set_zero_disp(disp):
    initial_ts = min(disp['ts'].values)
    disp['xz'] = disp['xz'].values - disp[disp.ts == initial_ts]['xz'].values[0]
    disp['xy'] = disp['xy'].values - disp[disp.ts == initial_ts]['xy'].values[0]
    
    return disp

def ComputeCumShear(disp_ts):
    cumsheardf = pd.DataFrame(columns = ['ts','cumshear'])
    cumsheardf.ix['ts'] = disp_ts.ts.values[0]
    
    cum_xz = np.sum(disp_ts.xz.values)
    cum_xy = np.sum(disp_ts.xy.values)
    sum_cum = np.sqrt(np.square(cum_xz) + np.square(cum_xy))
    cumsheardf['cumshear'] = sum_cum
    
    return cumsheardf

def GetCumShearDF(disp,nodes):
    #####################################################################################  
    #### INPUT: disp -> data frame containing the timestamp (ts),                       #
    ####            node id (id), xz and xy displacement of site & event of interest    #
    ####        nodes -> selected nodes to include in the vector sum                    #
    #### OUTPUT: Returns the magnitude of the resultant of the vector sum of            #  
    ####         xz and xy displacement,                                                #
    #####################################################################################

    #### Select only relevant nodes
    mask = np.zeros(len(disp.id.values))
    for values in nodes:
        mask = np.logical_or(mask,disp.id.values == values)
    disp = disp[mask]
    
    #### Set initial displacements to zero
    disp_id = disp.groupby('id',as_index = False)
    disp = disp_id.apply(set_zero_disp)    
    
    #### Compute Shear Displacements
    disp_ts = disp.groupby('ts',as_index = True)
    return disp_ts.apply(ComputeCumShear).reset_index(drop = True).set_index('ts')
    
