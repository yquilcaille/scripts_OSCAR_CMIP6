import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as scp
import datetime
import os
from scipy.optimize import fmin
import csv

import sys
sys.path.append("H:/MyDocuments/Repositories/OSCARv31_CMIP6") ## line required for run on server ebro
from core_fct.fct_loadD import load_all_hist
from core_fct.fct_process import OSCAR,OSCAR_landC
from core_fct.fct_misc import aggreg_region




list_setMC = range(20)
#path_runs = 'H:/MyDocuments/Repositories/OSCARv31_CMIP6/results/CMIP6_v3.1'               # folder where will look for OSCAR runs:: 'C:/Users/quilcail/Documents/Repositories/OSCARv3_CMIP6/results/CMIP6_v3.0'  |  'E:/OSCARv3_CMIP6/results/CMIP6_v3.0'
path_runs = '/net/exo/landclim/yquilcaille/OSCARv31_CMIP6/results/CMIP6_v3.1'
folder_extra = 'results/CMIP6_v3.1_extra/'
option_plots_treatment = False


## PRINCIPLE OF EXCLUSION:
## * Experiments: 'abrupt-4xCO2','abrupt-2xCO2','abrupt-0p5xCO2','G1','esm-abrupt4xCO2': --> full exclusion
##      - C ocean sink, value over last 50years: nan or aberrant (20 PgC/yr)
##      - C land sink, any year: aberrant (1.e4 PgC/yr)
##      - CO2 emissions from permafrost, any year: aberrant (1.e4 PgC/yr)
##      - CO2 emissions from LUC, any year: aberrant (1.e4 PgC/yr)
## * Land experiments:
##      - CO2 emissions from LUC, any year: aberrant (1.e4 PgC/yr)
## * Experiments derivated from '1pctCO2': use the most restrictive mask for all of them. Includes G2.
## * Experiments derivated from 'esm-1pctCO2': like '1pctCO2'. Includes 'esm-1pct-brchXXXXPgC'.
## * Other experiments:
##      - C ocean sink, any year: aberrant (20 PgC/yr)
##      - C land sink, any year: aberrant (20 PgC/yr)
##      - CO2 emissions from permafrost, any year: aberrant (1.e4 PgC/yr)
##      - CO2 emissions from LUC, any year: aberrant (1.e4 PgC/yr)
## NB: BEFORE, on the Family '1pctCO2' experiments, had only '1pctCO2-cdr' using exclusions from '1pctCO2' and extending them over cdr period.
##
##  !!!! WARNING !!!!
## * Once a threshold of exclusion is met in year=Y, there is exclusion from the year Y-N.
## * For historical and scenarios, number of years removed: min( 0 , max( int(Delta_DV * (year-2015.) / (2100-2015.)) , Delta_DV ) )  --> no removal on historical because of scenarios, allows for consistency of historical to scenarios.
## * For other scenarios, number of years removed: Delta_DV



###################################
## PREPARING FUNCTIONS
###################################

def N_YR_REMOVE(YR,xp):
    year_cut = 2015 # 2050.
    if xp in ['1pctCO2','1pctCO2-4xext','1pctCO2-bgc','1pctCO2-cdr','1pctCO2-rad','G2'] + ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC','esm-1pctCO2']:
        Delta_DV = 80 ## number of years before criteria matched for the removal of the run: if 100 or more, removes more than the extra 100 years in from 1pctCO2
    else:
        Delta_DV = 120 ## number of years before criteria matched for the removal of the run: for scenarios, do not go under 100, bottom limit.
    if (xp[:len('hist')]=='hist') or (xp[:len('esm-hist')]=='esm-hist') or (xp[:len('rcp')]=='rcp') or (xp[:len('esm-rcp')]=='esm-rcp') or (xp[:len('ssp')]=='ssp') or (xp[:len('esm-ssp')]=='esm-ssp') or (xp[:len('esm-yr2010')]=='esm-yr2010') or (xp in ['G6solar','yr2010CO2']):
        ## linear from year_cut to year_cut+Delta_DV
        out = max( 0 , min( int(Delta_DV * 0.5 * (YR-year_cut) / Delta_DV) , Delta_DV ) )
        ## quadratic from year_cut to year_cut+Delta_DV
        # year_cut = max(2015 , year_cut-int(np.sqrt(Delta_DV)-1)+1) ## shifting year_cut to guarantee that year_cut+1 is one year removal
        # out = max( 0 , min( int(Delta_DV * (max(0,YR-year_cut) / Delta_DV)**3 ) , Delta_DV ) )
    else:
        out = Delta_DV
    return out

if False:
    tt = np.arange(2000,2200)
    nn = [N_YR_REMOVE(yr,'ssp585') for yr in tt]
    plt.plot(tt, nn )

def exclude_backwards( ind_yr, force_Delta_DV=np.nan ):
    global list_noDV
    year_cut_DV = {}
    for cfg in np.arange(out_TMP.config.size):
        aa = [ind_yr[0][ii] for ii in np.arange(len(ind_yr[0])) if ind_yr[1][ii]==cfg]
        if len(aa)>0:
            if np.isnan(force_Delta_DV):
                to_rem = N_YR_REMOVE(out_TMP.year.values[np.min(aa)],name_experiment)
            else:
                to_rem = force_Delta_DV
            year_cut_DV[cfg] = np.min(aa)
            ## looping on preceding experiments to apply mask/np.nan
            maxyr2rem,rem,xp = year_cut_DV[cfg],0,name_experiment
            while rem<to_rem:
                ny2rem = min([maxyr2rem,(to_rem-rem)])
                list_noDV[xp][setMC][maxyr2rem-ny2rem:,cfg] = np.nan
                rem += ny2rem
                xp = dico_experiments_before[xp]
                if xp != None: maxyr2rem = int(list_noDV[xp][setMC].shape[0])
                else: rem = to_rem
    return

def fig_exclusions(out_TMP, name_fig, name_experiment, setMC, var_force=None):
    if type(var_force)==str:
        var = var_force
        if var_force == 'D_Epf_CO2':
            to_plot = out_TMP[var_force].sum('reg_pf')
        else:
            to_plot = out_TMP[var_force]
    else:
        if name_experiment[:len('land-')]=='land-':
            var = 'D_Fland'
        else:
            var = 'D_Focean'
        to_plot = out_TMP[var]
    fig = plt.figure(figsize=(20,10))
    plt.title(var)
    plt.plot( out_TMP.year, to_plot , color='orange',lw=1 ) ## 'all'
    plt.plot( out_TMP.year, np.nan*to_plot.isel(config=0) , color='orange',lw=1 ,label='All runs on '+name_experiment+', set '+str(setMC)) ## ghost plot for 'all'
    plt.plot( out_TMP.year, (to_plot*list_noDV[name_experiment][setMC]) , color='gray',lw=2 )## 'kept'
    plt.plot( out_TMP.year, np.nan*to_plot.isel(config=0) , color='gray',lw=2 ,label='Non diverging on '+name_experiment+', set '+str(setMC)) ## ghost plot for 'kept'
    plt.plot( out_TMP.year, (to_plot*list_noDV[name_experiment][setMC]).mean('config') , color='k',lw=4,label='mean of non-DV' ) ## average
    mm,MM = np.min(to_plot*list_noDV[name_experiment][setMC]) , np.max(to_plot*list_noDV[name_experiment][setMC])
    plt.ylim( mm-0.1*(MM-mm) , MM+0.1*(MM-mm)  )
    fig.savefig(path_runs+'/treated/masks/figures/'+name_fig+'.png')
    plt.close(fig)
    return

def fig_exclusions_allsetMC(name_fig,name_experiment):
    if name_experiment[:len('land-')]=='land-':
        var = 'D_Fland'
    else:
        var = 'D_Focean'

    fig = plt.figure(figsize=(20,10))
    plt.title(var)

    out,fullout = [],[]
    for setMC in list_setMC:
        print("Preparing "+name_experiment+", set "+str(setMC)+".")
        ## loading everything related to the experiment
        with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
        out_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' )
        for_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_For-'+str(setMC)+'.nc' )
        if name_experiment in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc', 'ssp585-bgcExt','ssp534-over-bgcExt']:
            with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
            Par['D_CO2_rad'] = for_runs_hist.D_CO2.sel(year=1850)
        elif name_experiment in ['1pctCO2-rad']:
            with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
            Par['D_CO2_bgc'] = for_runs_hist.D_CO2.sel(year=1850)
        ## calculating additionals variables
        out_TMP[var] = OSCAR[var](out_TMP, Par, for_TMP,recursive=True)
        ## preparing
        fullout.append( out_TMP[var] )
        out.append( out_TMP[var]*list_noDV[name_experiment][setMC] )
        del out_TMP
    ## plot
    for ii in np.arange(len(out)):
        plt.plot( out[ii].year, fullout[ii] , color='orange',lw=1 ) ## 'all'
    plt.plot( out[ii].year, np.nan*fullout[ii].isel(config=0) , color='orange',lw=1 ,label='All runs on '+name_experiment+', all sets') ## ghost plot for 'all'
    for ii in np.arange(len(out)):
        plt.plot( out[ii].year, out[ii] , color='gray',lw=2 )## 'kept'
    plt.plot( out[ii].year, np.nan*out[ii].isel(config=0) , color='gray',lw=2 ,label='Non diverging on '+name_experiment+', all sets') ## ghost plot for 'kept'
    plt.ylim( np.nanmin(out) , np.nanmax(out) )
    val = np.nansum(np.array(out),axis=(0,2)) / np.nansum( np.array([list_noDV[name_experiment][setMC] for setMC in list_setMC]) , axis=(0,2) )
    plt.plot( out[ii].year, val , color='k',lw=4,label='mean of non-DV' ) ## average
    plt.ylim( mm-0.1*(MM-mm) , MM+0.1*(MM-mm)  )

    fig.savefig(path_runs+'/treated/masks/figures/'+name_fig+'.png')
    plt.close(fig)
    return

def func_restrict_mask( list_xp_cut , year_cut,force_NoFig=True ):
    for setMC in list_setMC:
        print("Merging masks for set "+str(setMC))
        if type(list_xp_cut[0])==list:
            if len(list_xp_cut[0])==1:
                raise Exception("To simplify script, if scenarios with extensions in 'list_xp_cut', the first one in list MUST have an extension.")
            tmp = []
            for xp in list_xp_cut:
                if len(xp)==2:
                    ## taking scenario and extension
                    tmp.append( np.vstack( [list_noDV[xp[0]][setMC],list_noDV[xp[1]][setMC][1:]] ) )
                else:
                    ## taking scenario and extending with last values up to match required year_cut
                    tmp.append( np.vstack( [list_noDV[xp[0]][setMC] , np.repeat(list_noDV[xp[0]][setMC][-1,np.newaxis,:],year_cut+1-int(list_noDV[xp[0]][setMC].shape[0]),axis=0)] ) )
        else:
            tmp = [ list_noDV[name_experiment][setMC] for name_experiment in list_xp_cut]
        ## creating cut mask
        cut_mask = np.ones( (year_cut+1,int(tmp[0].shape[1])) )
        for ii in range(len(tmp)):cut_mask = np.multiply( cut_mask , tmp[ii][:year_cut+1,:] )
        ## applying to all experiments
        for ii in range(len(tmp)):
            if type(list_xp_cut[0])==list:
                mm = np.vstack( [cut_mask , cut_mask[-1,np.newaxis,:] * tmp[ii][year_cut+1:,:]] )
                list_noDV[list_xp_cut[ii][0]][setMC] = mm[:int(list_noDV[xp[0]][setMC].shape[0]),:]
                if len(list_xp_cut[ii])==2:list_noDV[list_xp_cut[ii][1]][setMC] = mm[int(list_noDV[xp[0]][setMC].shape[0])-1:,:]
            else:
                list_noDV[ list_xp_cut[ii] ][setMC] = np.vstack( [cut_mask , cut_mask[-1,np.newaxis,:] * tmp[ii][year_cut+1:,:]] )
            
        ## editing figures
        if option_plots_treatment and force_NoFig==False:
            for xp  in list_xp_cut:
                if type(xp)!=list:
                    xp = [xp]
                for name_experiment in xp:
                    ## loading everything related to the experiment
                    with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
                    out_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' )
                    for_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_For-'+str(setMC)+'.nc' )
                    if name_experiment in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc', 'ssp585-bgcExt','ssp534-over-bgcExt']:
                        with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
                        Par['D_CO2_rad'] = for_runs_hist.D_CO2.sel(year=1850)
                    elif name_experiment in ['1pctCO2-rad']:
                        with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
                        Par['D_CO2_bgc'] = for_runs_hist.D_CO2.sel(year=1850)
                    ## calculating additionals variables
                    for VAR in (name_experiment[:len('land-')]!='land-')*['D_Focean'] + ['D_Fland']:
                        out_TMP[VAR] = OSCAR[VAR](out_TMP, Par, for_TMP,recursive=True)

                    # figure
                    fig_exclusions( out_TMP , name_fig='masknoDV_'+name_experiment+'_'+str(setMC)+'-cut',name_experiment=name_experiment,setMC=setMC )
                    # cleaning
                    Par.close()
                    out_TMP.close()
                    for_TMP.close()
                    del Par,out_TMP,for_TMP
    return

def func_start_scen( xp ):
    if xp[-4:] in ['-ext','-Ext']: start = 2100 ## extensions of ssp
    elif (xp[-3:] in ['ext','Ext']) and (xp not in ['1pctCO2-4xext']): start = 2100 ## extensions of ssp
    elif xp in ['ssp245-GHG', 'ssp245-CO2'] + ['ssp245-aer'] + ['ssp245-nat', 'ssp245-sol', 'ssp245-volc']  + ['ssp245-stratO3']: start = 2020
    elif xp in ['ssp534-over-bgc','ssp585-bgc']: start = 2014
    elif xp[:3]=='ssp': start = 2014 ## ssp
    elif xp[:7]=='esm-ssp': start = 2014 ## esm-ssp
    elif xp[:3]=='rcp': start = 2000 ## rcp
    elif xp[:7]=='esm-rcp': start = 2000 ## esm-rcp
    elif xp=='land-hist-altStartYear': start = 1700
    elif xp=='G6solar': start = 2014
    else: start = 1850
    return start



###################################
## PREPARING LIST OF EXPERIENCES
###################################

## Dictionary for information on experiments (used for list of them, and which ones used in different exercices)
with open('dico_experiments_MIPs.csv','r',newline='') as ff:
    dico_experiments_MIPs = np.array([line for line in csv.reader(ff)])[:-1,:]
aa = list(set(list(dico_experiments_MIPs[1:,1])))
aa.remove('')
## All experiments used here
list_xp = list(dico_experiments_MIPs[1:,0]) + aa

## making sure to have the correct order in experiments. Crucial to have '1pctCO2' before '1pctCO2-cdr'.
## for figures only, need to do mask of extension before mask of scenario
list_xp.sort(reverse=True)
## start with 1pctCO2 and esm-1pctCO2
for xp in ['1pctCO2', 'esm-1pctCO2']:
    list_xp.remove(xp)
    list_xp.insert(0,xp)

## Experiments preceding
dico_experiments_before = {}
for xp in list_xp:
    if xp[-4:] in ['-ext','-Ext']: xp2 = xp[:-4]
    elif (xp[-3:] in ['ext','Ext']) and (xp not in ['1pctCO2-4xext']): xp2 = xp[:-3]
    elif xp in ['ssp245-GHG', 'ssp245-CO2'] + ['ssp245-aer'] + ['ssp245-nat', 'ssp245-sol', 'ssp245-volc']  + ['ssp245-stratO3']:xp2='hist-'+str.split(xp,'-')[1]
    elif xp in ['ssp534-over-bgc','ssp585-bgc']:xp2 = 'hist-bgc'
    elif xp[:3]=='ssp': xp2 = 'historical'
    elif xp[:7]=='esm-ssp': xp2 = 'esm-hist'
    elif xp[:3]=='rcp': xp2 = 'historical-CMIP5'
    elif xp[:7]=='esm-rcp': xp2 = 'esm-histcmip5'
    elif xp=='G6solar': xp2 = 'historical'
    else: xp2 = None
    if (xp2 in list_xp) or xp2==None:dico_experiments_before[xp] = xp2
    else: raise Exception("Correct the name of this experiment")





###################################
## PREPARING MASKS
###################################
## must prepare BEFORE the sizes of masks for transmission in-between experiments
list_noDV = {}

for name_experiment in list_xp:
    print("preparing masks for "+name_experiment)#+'/'+str(setMC)
    list_noDV[name_experiment] = {}
    for setMC in list_setMC:
        out_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' )
        ## preparing shape of list_noDV
        list_noDV[name_experiment][setMC] = np.ones( (out_TMP.year.size,out_TMP.config.size) )
        out_TMP.close()




###################################
## RUNNING EXCLUSION
###################################

## looping over all experiments
for name_experiment in list_xp:
    ## calculating masks
    for setMC in list_setMC:
        print("Identifying divergent configurations in "+name_experiment+", set "+str(setMC)+".")
        ## LOADING ALL INPUTS
        with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
        out_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' )
        for_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_For-'+str(setMC)+'.nc' )
        ## preparing input 1% CO2 for future additional exclusions
        if name_experiment == '1pctCO2':
            CO2_1pct = for_TMP['D_CO2'].copy(deep=True)
        elif name_experiment == 'esm-1pctCO2':
            CO2_esm1pct = out_TMP['D_CO2'].copy(deep=True)
        ## fixing parameters if need be
        if name_experiment in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc', 'ssp585-bgcExt','ssp534-over-bgcExt']:
            with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
            Par['D_CO2_rad'] = for_runs_hist.D_CO2.sel(year=1850)
        elif name_experiment in ['1pctCO2-rad']:
            with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
            Par['D_CO2_bgc'] = for_runs_hist.D_CO2.sel(year=1850)
        ## calculating variables that may be used
        for VAR in ['D_Eluc'] + (name_experiment[:len('land-')]!='land-')*['D_Focean','D_Epf_CO2'] + ['D_Fland']:#,'D_Cosurf','dic_0'
            out_TMP[VAR] = OSCAR[VAR](out_TMP, Par, for_TMP,recursive=True)

        ## EXCLUSION
        if name_experiment in ['abrupt-4xCO2','abrupt-2xCO2','abrupt-0p5xCO2','G1','esm-abrupt-4xCO2']:
            ## Exclusion depends on final ocean sink of C: >20Pgc/yr over the last 50 years of the run
            ## Exclusions depends as well on presence of aberrant CO2 emissions from permafrost: >20.  PgC/yr at any point
            ## Exclusions depends as well on presence of aberrant CO2 emissions from LUC: >20. PgC/yr at any point (eg config=(0?-)166 on spp585ext)
            ## Exclusions depends as well on presence of aberrant CO2 land sink: >20. PgC/yr at any point (case in G1, 6-195, other criteria not enough somehow)
            ind_yr = np.where(  (np.abs(out_TMP['D_Focean'].isel(year=np.arange(-50,-1+1))) > 20.).any('year')  |  (np.abs(out_TMP['D_Epf_CO2'].sum('reg_pf')) > 20.).any('year')  |  (np.abs(out_TMP['D_Eluc']) > 20.).any('year')  |  (np.isnan(out_TMP['D_Focean'].isel(year=np.arange(-50,-1+1)))).any('year')  |  (np.abs(out_TMP['D_Fland']) > 20.).any('year')  )[0]
            list_noDV[name_experiment][setMC][:,ind_yr] = np.nan

        elif name_experiment in ['1pctCO2', 'esm-1pctCO2']:
                                ## '1pctCO2-cdr','1pctCO2-4xext','1pctCO2-bgc','1pctCO2-rad','esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC' removed
            ## Exclusion depends on final ocean sink of C: >20PgC/yr or <0PgC/yr
            ## Exclusions depends as well on presence of aberrant CO2 emissions from permafrost: >20. PgC/yr
            ## Exclusions depends as well on presence of aberrant CO2 emissions from LUC: >20. PgC/yr
            ## additional criteria! Derivative of D_Focean must be greater to -0.01 PgC/yr
            ind_yr = np.where(  (out_TMP['D_Focean'].diff('year') < -0.01)  |  (out_TMP['D_Focean'] > 20.)  |  (out_TMP['D_Focean'] < 0.)  |  (np.abs(out_TMP['D_Fland']) > 20.)  |  (np.abs(out_TMP['D_Epf_CO2'].sum('reg_pf')) > 20.)  |  (np.abs(out_TMP['D_Eluc']) > 20.)  )
            exclude_backwards( ind_yr )

        elif name_experiment[:len('land-')] == 'land-':
            ## Exclusions depends as well on presence of aberrant CO2 emissions from LUC: >20. PgC/yr
            ind_yr = np.where(  (np.abs(out_TMP['D_Eluc']) > 20.)  )# |  (np.isnan(out_TMP['D_Focean']))  )
            exclude_backwards( ind_yr )

        elif name_experiment in ['ssp585','ssp585ext'] + ['ssp585-ssp126Lu'] + ['ssp585-bgc','ssp585-bgcExt'] + ['G6solar'] + \
                                ['ssp370','ssp370ext'] + ['ssp370-lowNTCF','ssp370-lowNTCFext'] + ['ssp370-lowNTCF-gidden','ssp370-lowNTCFext-gidden'] + \
                                ['esm-ssp370','esm-ssp370ext'] + ['esm-ssp370-lowNTCF','esm-ssp370-lowNTCFext'] + ['esm-ssp370-lowNTCF-gidden','esm-ssp370-lowNTCFext-gidden'] + \
                                ['esm-ssp585','esm-ssp585ext'] + ['esm-ssp585-ssp126Lu','esm-ssp585-ssp126Lu-ext']:
            ## Exclusion depends on final ocean sink of C: >20PgC/yr or <0PgC/yr
            ## Exclusions depends as well on presence of aberrant CO2 emissions from permafrost: >20. PgC/yr
            ## Exclusions depends as well on presence of aberrant CO2 emissions from LUC: >20. PgC/yr
            ind_yr = np.where(  (out_TMP['D_Focean'] > 20.)  |  (out_TMP['D_Focean'] < 0.)  |  (np.abs(out_TMP['D_Fland']) > 20.)  |  (np.abs(out_TMP['D_Epf_CO2'].sum('reg_pf')) > 20.)  |  (np.abs(out_TMP['D_Eluc']) > 20.)  )# |  (np.isnan(out_TMP['D_Focean']))  )
            exclude_backwards( ind_yr )

        else:
            ## Exclusion depends on final ocean sink of C: >20PgC/yr or <-10PgC/yr
            ## Exclusions depends as well on presence of aberrant CO2 emissions from permafrost: >20. PgC/yr
            ## Exclusions depends as well on presence of aberrant CO2 emissions from LUC: >20. PgC/yr
            ind_yr = np.where(  (out_TMP['D_Focean'] > 20.)  |  (out_TMP['D_Focean'] < -10.)  |  (np.abs(out_TMP['D_Fland']) > 20.)  |  (np.abs(out_TMP['D_Epf_CO2'].sum('reg_pf')) > 20.)  |  (np.abs(out_TMP['D_Eluc']) > 20.)  )# |  (np.isnan(out_TMP['D_Focean']))  )
            exclude_backwards( ind_yr )

        if False:
            ## REMOVING ADDITIONAL CONFIGURATION USING 1%
            ## method: for each experiment, for every year, taking CO2 of the experiment, looking at what date the same CO2 is obtained in 1pctCO2, and applying the mask of 1pctCO2 to the experiment.
            ## for emissions-driven experiment, the mask of esm-1pctCO2 is used instead, but this search is made for configuration-dependent.
            if name_experiment not in ['1pctCO2','esm-1pctCO2'] + ['abrupt-4xCO2','abrupt-2xCO2','abrupt-0p5xCO2','G1','esm-abrupt-4xCO2']:
                diff_1pct = 0 ## option, checked that low impact, eg only around 2100 for ssp585 and ssp370
                ## loading required input
                if name_experiment[:len('esm-')] == 'esm-':
                    co2_xp = out_TMP['D_CO2'].values
                    for ind_yr in range(for_TMP.year.size):
                        for cfg in range(for_TMP.config.size):
                            if co2_xp[ind_yr,cfg] <= CO2_esm1pct.values[0,cfg]: ## does the CO2 decrease under preindustrial?
                                ind_yr_1pct = np.array([0])
                            elif CO2_esm1pct.values[CO2_esm1pct.year.size-1,cfg] <= co2_xp[ind_yr,cfg]: ## does the CO2 increase beyond last value of 1%?
                                ind_yr_1pct = np.array([CO2_esm1pct.year.size-1])
                            else:
                                ind_yr_1pct = np.where( (CO2_esm1pct.values[:-1,cfg]<co2_xp[ind_yr,cfg])  &  (co2_xp[ind_yr,cfg]<CO2_esm1pct.values[1:,cfg]) )[0]
                            if len(ind_yr_1pct)==0:
                                ## cant find correct CO2 in esm-1pctCO2, the run has diverged: removing
                                list_noDV[name_experiment][setMC][ind_yr,cfg] = np.nan
                            else:
                                ## removing from the identified year, (first value found, because of prescribed increasing CO2)
                                list_noDV[name_experiment][setMC][ind_yr,cfg] *= list_noDV['esm-1pctCO2'][setMC][ np.max([0,ind_yr_1pct[0]-diff_1pct]) , cfg]
                else:
                    co2_xp = for_TMP['D_CO2'].values
                    for ind_yr in range(for_TMP.year.size):
                        if co2_xp[ind_yr] <= CO2_1pct.values[0]: ## does the CO2 decrease under preindustrial?
                            ind_yr_1pct = np.array([0])
                        elif CO2_1pct.values[CO2_esm1pct.year.size-1] <= co2_xp[ind_yr]: ## does the CO2 increase beyond last value of 1%?
                            ind_yr_1pct = np.array([CO2_esm1pct.year.size-1])
                        else:
                            ind_yr_1pct = np.where( (CO2_1pct.values[:-1]<=co2_xp[ind_yr])  &  (co2_xp[ind_yr]<CO2_1pct.values[1:]) )[0]
                        if len(ind_yr_1pct)==0:raise Exception("Normally, the extended 1% CO2 should cover the full range of CO2 forcing. Check that out.")
                        ## removing from the identified year
                        list_noDV[name_experiment][setMC][ind_yr,:] *= list_noDV['1pctCO2'][setMC][ np.max([0,ind_yr_1pct[0]-diff_1pct]) ,:]
                ## once excluded, doesnt come back: cumprod
                list_noDV[name_experiment][setMC] = np.cumprod(list_noDV[name_experiment][setMC],axis=0)
                ## passing to extensions additional exclusions performed over scenarios:
                if type(dico_experiments_before[name_experiment]) == str: list_noDV[name_experiment][setMC] *= list_noDV[ dico_experiments_before[name_experiment] ][setMC][-1,:]

        ## CONTROL PLOT
        if option_plots_treatment: fig_exclusions( out_TMP , name_fig='masknoDV_'+name_experiment+'_'+str(setMC),name_experiment=name_experiment,setMC=setMC )

        Par.close()
        out_TMP.close()
        for_TMP.close()
        del Par,out_TMP,for_TMP
    print("In average over "+name_experiment+", get: "+str(np.round(np.sum([np.nansum(list_noDV[name_experiment][setMC]) / int(list_noDV[name_experiment][setMC].shape[0]) for setMC in list_setMC]),1))+" configurations.")







###################################
## IDENTIFICATION OF CONFIGURATIONS THAT LEAD TO NaN or np.inf Eff_comp
###################################
# not fully: 49,56
for exp in list_xp:
    for nset in list_setMC:
        if os.path.isfile( folder_extra + exp + '_Out2-' + str(nset) + '.nc' )==False:
            print('Missing extra variables on '+exp + ' | ' + str(nset))
        else:
            init = np.nansum( list_noDV[exp][nset] )
            ## loading
            with xr.open_dataset(folder_extra + exp + '_Out2-' + str(nset) + '.nc') as TMP: Out2 = TMP.load().copy(deep=True)
            ## checking
            idNaIn = np.cumprod( xr.where( (np.isnan(Out2['Eff_comp'])) | (np.abs(Out2['Eff_comp']) == np.inf), np.nan, 1. ).values , axis=0)
            list_noDV[exp][nset] *= idNaIn
            Out2.close()
            print ('Checking on '+exp + ' | ' + str(nset)+': removed '+str(np.round(100.*(init-np.nansum( list_noDV[exp][nset] ))/init,2))+'% with nan or inf values.' )




###################################
## MERGING MASKS OF COMMON FAMILIES
###################################
func_restrict_mask( ['1pctCO2','1pctCO2-4xext','1pctCO2-bgc','1pctCO2-cdr','1pctCO2-rad','G2'] , year_cut=140 , force_NoFig=False ) ## Experiments derivated from '1pctCO2': use the most restrictive mask for all of them. Includes G2.
func_restrict_mask( ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC', 'esm-1pctCO2'] , year_cut=140 , force_NoFig=False  ) ## Experiments derivated from 'esm-1pctCO2': like '1pctCO2'. Includes 'esm-1pct-brchXXXXPgC'.
func_restrict_mask( ['esm-bell-750PgC','esm-bell-1000PgC','esm-bell-2000PgC'] , year_cut=100+1000 , force_NoFig=False  )
func_restrict_mask( ['abrupt-0p5xCO2','abrupt-2xCO2','abrupt-4xCO2'] + ['G1'] , year_cut=1000 , force_NoFig=False  )

## variant esm-lowNTCF
tmp_xp = [ ['esm-ssp370','esm-ssp370ext'] , ['esm-ssp370-lowNTCF','esm-ssp370-lowNTCFext'] ]
if 'esm-ssp370-lowNTCF-gidden' in list_xp: tmp_xp.append( ['esm-ssp370-lowNTCF-gidden','esm-ssp370-lowNTCFext-gidden'] )
func_restrict_mask( tmp_xp , year_cut=2500-2014 , force_NoFig=False )
## variant esm-ssp585-ssp126Lu, esm-ssp585-bgc, esm-ssp585-ssp126Lu
func_restrict_mask( [ ['esm-ssp585','esm-ssp585ext'] , ['esm-ssp585-ssp126Lu','esm-ssp585-ssp126Lu-ext'] ] , year_cut=2500-2014 , force_NoFig=False  )

## variant lowNTCF
tmp_xp = [ ['ssp370','ssp370ext'] , ['ssp370-lowNTCF','ssp370-lowNTCFext'] ]
if 'ssp370-lowNTCF-gidden' in list_xp: tmp_xp.append( ['ssp370-lowNTCF-gidden','ssp370-lowNTCFext-gidden'] )
func_restrict_mask( tmp_xp , year_cut=2500-2014 , force_NoFig=False )
## variant ssp585-ssp126Lu, ssp585-bgc, ssp585-ssp126Lu
func_restrict_mask( [ ['ssp585','ssp585ext'] , ['ssp585-ssp126Lu'] , ['ssp585-bgc','ssp585-bgcExt'] , ['G6solar'] ] , year_cut=2500-2014 , force_NoFig=False  )
## variant ssp126-ssp370Lu
func_restrict_mask( [ ['ssp126','ssp126ext'] , ['ssp126-ssp370Lu'] ] , year_cut=2500-2014 , force_NoFig=False  )
## variant ssp534-over-bgc
func_restrict_mask( [ ['ssp534-over','ssp534-over-ext'] , ['ssp534-over-bgc','ssp534-over-bgcExt'] ] , year_cut=2500-2014 , force_NoFig=False  )

## nothing on variant ssp245







###################################
## SAVING INDIVIDUAL MASKS
###################################
for name_experiment in list_xp:
    print('saving mask for '+name_experiment)
    for setMC in list_setMC:
        with open(path_runs+'/treated/masks/masknoDV_'+name_experiment+'_'+str(setMC)+'.csv','w',newline='') as ff:
            csv.writer(ff).writerows( np.array(list_noDV[name_experiment][setMC]) )




###################################
## CREATING UNIFORM MASK
###################################
mask_all = {}
for setMC in list_setMC:
    tmp = []
    for exp in list_xp:
        if exp in ['1pctCO2','1pctCO2-bgc','1pctCO2-cdr','1pctCO2-rad','G2'] + ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC', 'esm-1pctCO2']:
            tmp.append( list_noDV[exp][setMC][150,:] )
        elif exp in ['abrupt-0p5xCO2','abrupt-2xCO2','abrupt-4xCO2'] + ['G1'] + ['esm-abrupt-4xCO2']:
            pass ## these experiments dont work well with the rest of them, creating restricted masks. checking if still correct figures
        else:
            tmp.append( list_noDV[exp][setMC][-1,:] )
    mask_all[setMC] = np.prod( tmp , axis=0)
print('Uniform and unique mask: '+str(np.nansum([mask_all[setMC] for setMC in list_setMC]))+' members.' )
## 1121 members if exclusion part only
## 1118 members after removing those with nan/inf values in Eff_comp
## 1118 members after merging masks

if False:## checking if not adding abrupt experiments to the uniform mask doesnt hamper identification of divergences in these runs
    for name_experiment in ['abrupt-0p5xCO2','abrupt-2xCO2','abrupt-4xCO2'] + ['G1'] + ['esm-abrupt-4xCO2']:
        ## calculating masks
        for setMC in list_setMC:
            print("Checking divergent configurations in "+name_experiment+", set "+str(setMC)+" with uniform mask.")
            ## LOADING ALL INPUTS
            with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
            out_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' )
            for_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_For-'+str(setMC)+'.nc' )
            ## calculating variables that may be used
            for VAR in ['D_Focean']: out_TMP[VAR] = OSCAR[VAR](out_TMP, Par, for_TMP,recursive=True)
            ## updating configurations with uniform mask
            list_noDV[name_experiment][setMC] = np.repeat( mask_all[setMC][np.newaxis,:], int(list_noDV[name_experiment][setMC].shape[0]), axis=0)
            fig_exclusions( out_TMP , name_fig='masknoDV_'+name_experiment+'_'+str(setMC)+'_CheckUniform',name_experiment=name_experiment,setMC=setMC )


## saving
for setMC in list_setMC:
    with open(path_runs+'/treated/masks/mask_all_exp_'+str(setMC)+'.csv','w',newline='') as ff:
        csv.writer(ff).writerows(  mask_all[setMC][:,np.newaxis] )



## move masks to correct folder and run next.

