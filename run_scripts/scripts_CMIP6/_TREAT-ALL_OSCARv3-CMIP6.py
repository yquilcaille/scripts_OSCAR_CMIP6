#!usr/bin/env python
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
sys.path.append("/home/yquilcaille/oscar-cmip6/core_fct")

# update for ETH-exo, quick fix of a plot
os.chdir( '/home/yquilcaille/oscar-cmip6' )

from core_fct.fct_loadD import load_all_hist
from core_fct.fct_process import OSCAR,OSCAR_landC
from core_fct.fct_misc import aggreg_region







##################################################
## 1. OPTIONS
##################################################
list_setMC = np.arange(0,19+1)
path_runs = 'H:/MyDocuments/Repositories/OSCARv31_CMIP6/results/CMIP6_v3.1'               # folder where will look for OSCAR runs:: 'C:/Users/quilcail/Documents/Repositories/OSCARv3_CMIP6/results/CMIP6_v3.0'  |  'E:/OSCARv3_CMIP6/results/CMIP6_v3.0'
path_extra= 'H:/MyDocuments/Repositories/OSCARv31_CMIP6/results/CMIP6_v3.1_extra'
path_save = 'H:/MyDocuments/Repositories/OSCARv31_CMIP6/results/CMIP6_v3.1'               # folder where treated outputs will be saved::  'C:/Users/quilcail/Documents/Repositories/OSCARv3_CMIP6/results/CMIP6_v3.0'  |  'E:/OSCARv3_CMIP6/results/CMIP6_v3.0'

option_select_MIP = 'ZECMIP'        # ALL  |  CDRMIP  |  ZECMIP  |  LUMIP  |  RCMIP  |  CMIP6
model = 'OSCARv3.1'

option_maskDV = 'LOAD_unique'    ## LOAD_indiv  |  LOAD_unique          ## affect only section 2


option_weights = 'LOAD_assessed_ranges'             # CREATE_constraints_4  |  LOAD_constraints_4  |  LOAD_assessed_ranges

# option_removeDV = True      ## True | False --> used only if option_maskDV=='LOAD_unique', because keep only non diverging members. Used to lighten calculations
## !!!!!
## I did not implement this option yet: although it would lighten calculation by about 80%, this code is so complicated that it would be used only once, and the cost to update the full code offsets this benefit... Sorry :/
## !!!!!

option_OVERWRITE = False        # True | False          ## affect only section 4
option_plots_treatment = False                          ## only for figures in sections 2 and 3
option_AddingVarToCMIP6 = False
##################################################
##################################################

k_subsetXP = 7
print(" ")
print(k_subsetXP)
print(" ")






##################################################
## 2. PREPARING EXPERIMENTS
##################################################
#########################
## 2.1. INFO ON EXPERIMENTS
#########################
## preparing folders
#for fold in ['treated','treated/masks','treated/masks/figures','treated/weights','treated/temporary']:
#    if os.path.isdir(path_runs+'/'+fold)==False:
#        os.mkdir( path_runs+'/'+fold )


## Dictionary for information on experiments (used for list of them, and which ones used in different exercices)
with open('dico_experiments_MIPs.csv','r',newline='') as ff:
    dico_experiments_MIPs = np.array([line for line in csv.reader(ff)])[:-1,:]
aa = list(set(list(dico_experiments_MIPs[1:,1])))
aa.remove('')
## All experiments used in this exercice
list_experiments = list(dico_experiments_MIPs[1:,0]) + aa

## Experiments preceding
dico_experiments_before = {}
for xp in list_experiments:
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
    if (xp2 in list_experiments) or xp2==None:dico_experiments_before[xp] = xp2
    else: raise Exception("Correct the name of this experiment")

## Control for each experiment
## spinups = ['esm-spinup-CMIP5', 'esm-spinup', 'land-spinup-altLu1', 'land-spinup-altLu2', 'land-spinup-altStartYear', 'land-spinup', 'spinup-CMIP5', 'spinup']
dico_Xp_Control = { 'piControl':['1pctCO2-4xext', '1pctCO2-bgc', '1pctCO2-cdr', '1pctCO2-rad', '1pctCO2', 'abrupt-0p5xCO2', 'abrupt-2xCO2', 'abrupt-4xCO2', 'G1', 'G2', 'G6solar', 'hist-1950HC', 'hist-aer', 'hist-bgc', 'hist-CO2', 'hist-GHG', 'hist-nat', 'hist-piAer', 'hist-piNTCF', 'hist-sol', 'hist-stratO3', 'hist-volc', 'historical', 'hist-noLu', 'ssp119', 'ssp119ext', 'ssp126-ssp370Lu', 'ssp126', 'ssp126ext', 'ssp245-aer', 'ssp245-CO2', 'ssp245-GHG', 'ssp245-nat', 'ssp245-sol', 'ssp245-stratO3', 'ssp245-volc', 'ssp245', 'ssp245ext', 'ssp370-lowNTCF', 'ssp370-lowNTCFext', 'ssp370-lowNTCF-gidden', 'ssp370-lowNTCFext-gidden', 'ssp370-ssp126Lu', 'ssp370', 'ssp370ext', 'ssp434', 'ssp434ext', 'ssp460', 'ssp460ext', 'ssp534-over-bgc', 'ssp534-over-bgcExt', 'ssp534-over-ext', 'ssp534-over', 'ssp585-bgc', 'ssp585-bgcExt', 'ssp585-ssp126Lu', 'ssp585', 'ssp585ext', 'yr2010CO2'] ,
                    'esm-piControl':['esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-1pct-brch-750PgC', 'esm-1pctCO2', 'esm-abrupt-4xCO2', 'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC', 'esm-hist', 'esm-pi-cdr-pulse', 'esm-pi-CO2pulse', 'esm-ssp119', 'esm-ssp119ext', 'esm-ssp126', 'esm-ssp126ext', 'esm-ssp245', 'esm-ssp245ext', 'esm-ssp370-lowNTCF', 'esm-ssp370-lowNTCFext', 'esm-ssp370-lowNTCF-gidden', 'esm-ssp370-lowNTCFext-gidden', 'esm-ssp370', 'esm-ssp370ext', 'esm-ssp460', 'esm-ssp460ext', 'esm-ssp434', 'esm-ssp434ext', 'esm-ssp534-over-ext', 'esm-ssp534-over', 'esm-ssp585-ssp126Lu-ext', 'esm-ssp585-ssp126Lu', 'esm-ssp585', 'esm-ssp585ext', 'esm-yr2010CO2-cdr-pulse', 'esm-yr2010CO2-CO2pulse', 'esm-yr2010CO2-control', 'esm-yr2010CO2-noemit'],
                    'esm-piControl-CMIP5':['esm-histcmip5','esm-rcp26', 'esm-rcp45', 'esm-rcp60', 'esm-rcp85'],
                    'piControl-CMIP5':['historical-CMIP5', 'rcp26', 'rcp45', 'rcp60', 'rcp85'],
                    'land-piControl':['land-cClim', 'land-cCO2', 'land-crop-grass', 'land-hist', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv'],
                    'land-piControl-altLu1':['land-hist-altLu1'],
                    'land-piControl-altLu2':['land-hist-altLu2'],
                    'land-piControl-altStartYear':['land-hist-altStartYear'],
                    'spinup':[],'esm-spinup':[],'esm-spinup-CMIP5':[],'spinup-CMIP5':[],'land-spinup':[],'land-spinup-altLu1':[],'land-spinup-altLu2':[],'land-spinup-altStartYear':[] }

## checking that all runs are here
#dico_miss = {}
#for xp in list_experiments:
#    for setMC in list_setMC:
#        if os.path.isfile( path_runs+'/'+xp+'_Out-'+str(setMC)+'.nc' )==False:
#            if xp not in dico_miss:
#                dico_miss[xp] = []
#            dico_miss[xp].append( setMC )
#if len(dico_miss)>0:raise Exception("OSCAR runs are missing, please check 'dico_miss'.")
#########################
#########################




#########################
## 2.2. DIVERGING RUNS
#########################
## preparing the list of experiments
if option_select_MIP=='ALL':
    list_xp = list_experiments
else:
    list_xp =   [ dico_experiments_MIPs[ii,0] for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-'+option_select_MIP)]=='1')[0] ]  +  \
                [ dico_experiments_MIPs[ii,1] for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-'+option_select_MIP)]=='1')[0] if dico_experiments_MIPs[ii,1]!='']
    list_xp = list(set(list_xp + ['historical','ssp460','piControl','abrupt-4xCO2','1pctCO2'] + list(dico_Xp_Control.keys())))
list_xp.sort() ## making sure to have the correct order in experiments. Crucial to have '1pctCO2' before '1pctCO2-cdr'.


if option_maskDV=='LOAD_indiv':
    for name_experiment in list_xp:
        print('Loading masks for '+name_experiment)
        list_noDV[name_experiment] = {}
        for setMC in list_setMC:
            with open(path_runs+'/treated/masks/masknoDV_'+name_experiment+'_'+str(setMC)+'.csv','r',newline='') as ff:
                list_noDV[name_experiment][setMC] = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)


elif option_maskDV=='LOAD_unique':
    mask_all,list_noDV = {},{}
    for setMC in list_setMC:
        print('Loading unique mask for set '+str(setMC))
        ## loading unique mask for this set
        with open(path_runs+'/treated/masks/mask_all_exp_'+str(setMC)+'.csv','r',newline='') as ff:
            mask_all[setMC] = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)
        for name_experiment in list_xp:
            if name_experiment not in list_noDV.keys():list_noDV[name_experiment] = {}
            ## checking the length of the run to conform to the previous code
            out_TMP = xr.open_dataset(path_runs+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' )
            list_noDV[name_experiment][setMC] = np.repeat( mask_all[setMC][np.newaxis,:,0], out_TMP.year.size, axis=0 )
            out_TMP.close()
print(" ")

#########################
#########################
##################################################
##################################################












##################################################
## 3. CONSTRAINTS
##################################################
#########################
## 3.1. PREPARATIONS
#########################
## getting sizes of setMC
dico_sizesMC = {}
for setMC in list_setMC:
    out_TMP = xr.open_dataset(path_runs+'/'+'historical'+'_Out-'+str(setMC)+'.nc' )
    dico_sizesMC[setMC] = out_TMP.config.size
    out_TMP.close()


if option_select_MIP in ['ALL','ZECMIP']  or  option_weights == 'CREATE_constraints_4':
    #########################
    ## CONSTRAINTS
    #########################
    Constraints = xr.Dataset()
    Constraints.coords['type_val'] = ['mean','std_dev']

    ## Global Temperature: 2006-2015 with reference to 1850-1900 (source: Average in IPCC SR1.5 Ch1, Table 1.1)
    # Constraints['D_Tg'] = xr.DataArray( np.array([0.87-0.23*1./(0.5*(2006+2015)-0.5*(1986+2005))/(2015-2006+1) ,0.12/0.955]) , dims=('type_val') ) ## shifting one year using 1986-2005  to  2006-2015
    Constraints['D_Tg'] = xr.DataArray( np.array([0.87 * 0.99/0.86 , np.sqrt((0.12/0.955 * (1.37-0.65) / (1.18-0.54))**2. + 0.1467**2.)]) , dims=('type_val') )

    ## Land and Ocean carbon sinks and FF emissions: average over 2000-2009 (source: table 6 of GCB2018, doi: 10.5194/essd-10-2141-2018)
    Constraints['D_Fland-D_Eluc'] = xr.DataArray( np.array([1.6,0.7]) , dims=('type_val') )
    Constraints['D_Focean'] = xr.DataArray( np.array([2.1,0.5]) , dims=('type_val') ) ## 2.2 +/- 0.4 in Friedlingstein et al, 2019
    Constraints['Eff'] = xr.DataArray( np.array([7.8,0.4]) , dims=('type_val') ) ## idem in Friedlingstein et al, 2019

    ## Cumulated Land and Ocean Carbon sinks and FF emissions: 1959-2017 (source: table 8 of GCB2018, doi: 10.5194/essd-10-2141-2018)
    # Constraints['Cum D_Fland-D_Eluc'] = xr.DataArray( np.array([130-80, np.sqrt(30.**2+40**2.)]) , dims=('type_val') )
    Constraints['Cum D_Fland-D_Eluc'] = xr.DataArray( np.array([350-190-100, np.sqrt(5.**2+20**2.+20**2.)]) , dims=('type_val') )## 365-200-105 +/- sqrt(20**2.+20**2.+25**2.) over 1959-2018 in Friendlingstein et al, 2018
    Constraints['Cum D_Focean'] = xr.DataArray( np.array([100,20.]) , dims=('type_val') ) ## 105 +/- 20 over 1959-2018 in Friendlingstein et al, 2018
    Constraints['CumEff'] = xr.DataArray( np.array([350.,20.]) , dims=('type_val') ) ## 365 +/- 20 over 1959-2018 in Friendlingstein et al, 2018



    #########################
    ## Special case ECS
    #########################
    ## From Roe et al, 2007 (https://doi.org/10.1126/science.1144735)
    Delta_T0 = 1.2 ## K, reference climate sensitivity (no feedbacks accounted)
    pdf_ECS = lambda Delta_T, mean_f, std_dev_f, Delta_T0: 1/(std_dev_f * np.sqrt(2.*np.pi)) * Delta_T0/Delta_T**2. * np.exp( -0.5*(1-mean_f-Delta_T0/Delta_T)**2./std_dev_f**2. )

    def percentile_ECS( qq , val ,dT=1.e-4):
        ## qq: percentile (%)
        TT = np.arange( dT,50.,dT )
        test = pdf_ECS(Delta_T=TT , mean_f=val[0], std_dev_f=val[1], Delta_T0=Delta_T0)
        cum = np.cumsum( test*dT )
        return TT[np.argmin(np.abs(cum-qq/100.))]

    def main_values_ECS( val , dT=1.e-4, rng = 68.27):
        TT = np.arange( dT,20.,dT )
        test = pdf_ECS(Delta_T=TT , mean_f=val[0], std_dev_f=val[1], Delta_T0=Delta_T0)
        mm = np.average( TT[~np.isnan(test)],weights=test[~np.isnan(test)] )
        ss = np.sqrt(np.average( (TT[~np.isnan(test)] - mm)**2. ,axis=0, weights=test[~np.isnan(test)] ))
        return mm, ss, percentile_ECS(qq=50-rng/2.,val=val,dT=dT) , percentile_ECS(qq=50+rng/2.,val=val,dT=dT)

    ## ECS
    distrib_ECS = 'Roe2007'     ## Roe2007  |  normal
    if distrib_ECS=='Roe2007':
        Constraints.coords['type_val_ECS'] = ['mean_f','std_dev_f']
        Constraints['ECS_charney'] = xr.DataArray( np.array([0.49179582, 0.24155785]) , dims=('type_val_ECS') )
        Constraints['ECS_tot'] = xr.DataArray( np.array([0.49179582, 0.24155785]) , dims=('type_val_ECS') )
    else:
        Constraints['ECS_charney'] = xr.DataArray( np.array([3., 1.5]) , dims=('type_val') )
        Constraints['ECS_tot'] = xr.DataArray( np.array([3., 1.5]) , dims=('type_val') )



    #########################
    ## additional functions
    #########################
    def func_calc_val( VAR , out_tmp,for_tmp,out_scen,for_scen,out_2x,for_2x,Par , setMC,out_pi=None,for_pi=None , option_minus_pi = False ):
        ## this function calculates the outputs of OSCAR for the required variables observed. It is used to produce weights and to plot histograms.

        if 'D_Tg' == VAR:
            val = xr.concat([out_tmp['D_Tg'].sel(year=np.arange(2006,2014+1)),out_scen['D_Tg'].sel(year=2015)],dim='year').mean('year') - out_tmp['D_Tg'].sel(year=np.arange(1850,1900+1)).mean('year')
            val *= np.prod(list_noDV['historical'][setMC][2006-1850:2014+1-1850,:],axis=0) ## mask for diverging

        if 'OHC' == VAR:
            out_tmp['D_OHC'] = OSCAR[VAR](out_tmp, Par, for_tmp,recursive=True)
            raise Exception("not prepared yet")
            val *= np.prod(list_noDV['historical'][setMC][2006-1850:2014+1-1850,:],axis=0) ## mask for diverging

        if 'ECS_charney' == VAR:
            val = Par.lambda_0 * OSCAR['RF_CO2'](Var=xr.Dataset({'D_CO2':2.*(Par.CO2_0+out_tmp.D_CO2.isel(year=0))-Par.CO2_0}),Par=Par)

        if 'ECS_tot' == VAR:
            if option_minus_pi:
                vv = out_2x.D_Tg.isel(year=-1) - out_pi.D_Tg.isel(year=-1)
            else:
                vv = out_2x.D_Tg.isel(year=-1)
            val = vv #* OSCAR['RF_CO2'](Var=xr.Dataset({'D_CO2':2.*(Par.CO2_0+out_4x.D_CO2.isel(year=0))-Par.CO2_0}),Par=Par) / OSCAR['RF_CO2'](Var=xr.Dataset({'D_CO2':out_4x.D_CO2.isel(year=-1)}),Par=Par)
            val *= list_noDV['abrupt-4xCO2'][setMC][-1,:] ## mask for diverging

        if 'Eff' == VAR:
            for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4']:
                if VAR not in out_tmp:
                    out_tmp[VAR] = OSCAR[VAR](out_tmp, Par, for_tmp,recursive=True)
            tmp = - out_tmp.D_Eluc  -  out_tmp.D_Epf_CO2.sum('reg_pf',min_count=1)  +  out_tmp.D_Fland  +  out_tmp.D_Focean  -  out_tmp.D_Foxi_CH4
            val = ( Par.a_CO2 * out_tmp.D_CO2.diff(dim='year') + 0.5*(tmp + tmp.shift(year=1)) ).sel(year=np.arange(2000,2009+1)).mean('year')
            val *= np.prod(list_noDV['historical'][setMC][2000-1850:2009+1-1850,:],axis=0) ## mask for diverging

        if 'D_Fland-D_Eluc' == VAR:
            for VAR in ['D_Eluc','D_Fland']:
                if VAR not in out_tmp:
                    out_tmp[VAR] = OSCAR[VAR](out_tmp, Par, for_tmp,recursive=True)
            val = ( out_tmp.D_Fland.sel(year=np.arange(2000,2009+1)) - out_tmp.D_Eluc.sel(year=np.arange(2000,2009+1)) ).mean('year')
            val *= np.prod(list_noDV['historical'][setMC][2000-1850:2009+1-1850,:],axis=0) ## mask for diverging

        if 'D_Focean' == VAR:
            if 'D_Focean' not in out_tmp:
                out_tmp['D_Focean'] = OSCAR['D_Focean'](out_tmp, Par, for_tmp,recursive=True)
            val = out_tmp.D_Focean.sel(year=np.arange(2000,2009+1)).mean('year')
            val *= np.prod(list_noDV['historical'][setMC][2000-1850:2009+1-1850,:],axis=0) ## mask for diverging

        if 'CumEff' == VAR:
            for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4']:
                if VAR not in out_tmp:
                    out_tmp[VAR] = OSCAR[VAR](out_tmp, Par, for_tmp,recursive=True)
            tmp = - out_tmp.D_Eluc  -  out_tmp.D_Epf_CO2.sum('reg_pf',min_count=1)  +  out_tmp.D_Fland  +  out_tmp.D_Focean  -  out_tmp.D_Foxi_CH4
            val = ( Par.a_CO2 * out_tmp.D_CO2.diff(dim='year') + 0.5*(tmp + tmp.shift(year=1)) ).sel(year=np.arange(1959,2014+1)).sum('year')
            for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4']:
                if VAR not in out_scen:
                    out_scen[VAR] = OSCAR[VAR](out_scen, Par, for_scen,recursive=True)
            tmp = - out_scen.D_Eluc  -  out_scen.D_Epf_CO2.sum('reg_pf',min_count=1)  +  out_scen.D_Fland  +  out_scen.D_Focean  -  out_scen.D_Foxi_CH4
            val += ( Par.a_CO2 * out_scen.D_CO2.diff(dim='year') + 0.5*(tmp + tmp.shift(year=1)) ).sel(year=np.arange(2014+1,2017+1)).sum('year')
            val *= np.prod(list_noDV['historical'][setMC][1959-1850:2014+1-1850,:],axis=0) ## mask for diverging

        if 'Cum D_Fland-D_Eluc' == VAR:
            for VAR in ['D_Eluc','D_Fland']:
                if VAR not in out_tmp:
                    out_tmp[VAR] = OSCAR[VAR](out_tmp, Par, for_tmp,recursive=True)
                if VAR not in out_scen:
                    out_scen[VAR] = OSCAR[VAR](out_scen, Par, for_scen,recursive=True)
            val = ( out_tmp.D_Fland.sel(year=np.arange(1959,2014+1)) - out_tmp.D_Eluc.sel(year=np.arange(1959,2014+1)) ).sum('year')
            val += ( out_scen.D_Fland.sel(year=np.arange(2014,2017+1)) - out_scen.D_Eluc.sel(year=np.arange(2014,2017+1)) ).sum('year')
            val *= np.prod(list_noDV['historical'][setMC][1959-1850:2014+1-1850,:],axis=0) * np.prod(list_noDV['ssp460'][setMC][2014-1850:2017+1-1850,:],axis=0) ## mask for diverging

        if 'Cum D_Focean' == VAR:
            if 'D_Focean' not in out_tmp:
                out_tmp['D_Focean'] = OSCAR['D_Focean'](out_tmp, Par, for_tmp,recursive=True)
            if 'D_Focean' not in out_scen:
                out_scen['D_Focean'] = OSCAR['D_Focean'](out_scen, Par, for_scen,recursive=True)
            val = out_tmp.D_Focean.sel(year=np.arange(1959,2014+1)).sum('year')   +   out_scen.D_Focean.sel(year=np.arange(2014+1,2017+1)).sum('year')
            val *= np.prod(list_noDV['historical'][setMC][1959-1850:2014+1-1850,:],axis=0) * np.prod(list_noDV['ssp460'][setMC][2014-1850:2017+1-1850,:],axis=0) ## mask for diverging

        return val

    def func_weights( VAR_CONST , strength_CONST , provided_val=None , option_minus_pi=False):
        ## initializing weights
        WEIGHTS = xr.Dataset()
        WEIGHTS.coords['all_config'] = [str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        WEIGHTS['weights'] = xr.DataArray(  np.ones((WEIGHTS.all_config.size)) , dims=('all_config')  )

        for setMC in list_setMC:
            print("Preparing weights for set "+str(setMC))

            if provided_val == None:
                ## loading everything related to the experiment
                with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
                out_tmp = xr.open_dataset(path_runs+'/'+'historical'+'_Out-'+str(setMC)+'.nc' )
                for_tmp = xr.open_dataset(path_runs+'/'+'historical'+'_For-'+str(setMC)+'.nc' )
                ## change in atm C stock over 1959-2017 from historical & ssp460: 190.5 PgC vs 190 +/- 5 for GCP2018
                out_scen = xr.open_dataset(path_runs+'/'+'ssp460'+'_Out-'+str(setMC)+'.nc' )
                for_scen = xr.open_dataset(path_runs+'/'+'ssp460'+'_For-'+str(setMC)+'.nc' )
                ## piControl
                if option_minus_pi:
                    out_pi = xr.open_dataset( path_runs+'/'+'piControl'+'_Out-'+str(setMC)+'.nc' )
                    for_pi = xr.open_dataset( path_runs+'/'+'piControl'+'_For-'+str(setMC)+'.nc' )
                else:
                    out_pi,for_pi = None,None

            ## calculating weights
            for VAR in VAR_CONST:
                if provided_val == None:
                    val = func_calc_val( VAR , out_tmp,for_tmp,out_scen,for_scen,out_pi,for_pi,Par , setMC )
                else:
                    val = provided_val[setMC][VAR]
                if (distrib_ECS=='Roe2007')   and   (VAR in ['ECS_charney' , 'ECS_tot']):
                    WEIGHTS['weights'].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] }] *= pdf_ECS( val.values, mean_f=Constraints[VAR].sel(type_val_ECS='mean_f').values,std_dev_f=Constraints[VAR].sel(type_val_ECS='std_dev_f').values , Delta_T0=Delta_T0) ** strength_CONST[VAR_CONST.index(VAR)]
                else:
                    WEIGHTS['weights'].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] }] *= scp.norm.pdf( val, loc=Constraints[VAR].sel(type_val='mean'),scale=Constraints[VAR].sel(type_val='std_dev') ) ** strength_CONST[VAR_CONST.index(VAR)] ## loc: center, scale: std dev
            if provided_val == None:
                out_tmp.close()
                out_scen.close()
                for_tmp.close()
                for_scen.close()
                Par.close()
                if option_minus_pi:
                    out_pi.close()
                    for_pi.close()
                del out_tmp,out_scen, for_tmp,for_scen, Par, out_pi,for_pi

        ## normalization
        # WEIGHTS['weights'] #/= WEIGHTS.weights.sum()
        return WEIGHTS


    #########################
    ## compute all once.
    #########################
    option_minus_pi = True # !!!
    list_var_plot = ['D_Tg','Cum D_Fland-D_Eluc','D_Fland-D_Eluc','Cum D_Focean','D_Focean' , 'CumEff','Eff' , 'ECS_charney','ECS_tot']
    prov_val = {}
    print("Preparing values used for weights AND histograms")
    for setMC in list_setMC:
        print("set "+str(setMC))
        prov_val[setMC] = xr.Dataset()
        prov_val[setMC].coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]

        ## loading everything related to the experiment
        with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
        out_tmp = xr.open_dataset(path_runs+'/'+'historical'+'_Out-'+str(setMC)+'.nc' )
        for_tmp = xr.open_dataset(path_runs+'/'+'historical'+'_For-'+str(setMC)+'.nc' )
        ## change in atm C stock over 1959-2017 from historical & ssp460: 190.5 PgC vs 190 +/- 5 for GCP2018
        out_scen = xr.open_dataset(path_runs+'/'+'ssp460'+'_Out-'+str(setMC)+'.nc' )
        for_scen = xr.open_dataset(path_runs+'/'+'ssp460'+'_For-'+str(setMC)+'.nc' )
        ## need abrupt-4xCO2 for ECS
        out_2x = xr.open_dataset(path_runs+'/'+'abrupt-2xCO2'+'_Out-'+str(setMC)+'.nc' )
        for_2x = xr.open_dataset(path_runs+'/'+'abrupt-2xCO2'+'_For-'+str(setMC)+'.nc' )
        if option_minus_pi:
            ## piControl
            out_pi = xr.open_dataset( path_runs+'/'+'piControl'+'_Out-'+str(setMC)+'.nc' )
            for_pi = xr.open_dataset( path_runs+'/'+'piControl'+'_For-'+str(setMC)+'.nc' )
        else:
            out_pi,for_pi = None,None

        ## calculating weights
        for VAR in list_var_plot:
            prov_val[setMC][VAR] = func_calc_val( VAR=VAR , out_tmp=out_tmp,for_tmp=for_tmp,out_scen=out_scen,for_scen=for_scen,out_pi=out_pi,for_pi=for_pi,out_2x=out_2x,for_2x=for_2x,Par=Par , setMC=setMC , option_minus_pi=option_minus_pi)
        out_tmp.close()
        out_scen.close()
        for_tmp.close()
        for_scen.close()
        Par.close()
        if option_minus_pi:
            out_pi.close()
            for_pi.close()
        del out_tmp,out_scen, for_tmp,for_scen, Par, out_pi,for_pi

#########################
#########################




#########################
## 3.2. WEIGHTS
#########################

if option_weights == 'CREATE_constraints_4':
    ## Actually producing weights
    list_var_const = ['D_Tg', 'CumEff','Cum D_Focean', 'Eff']
    weights_CMIP6 = func_weights(VAR_CONST=list_var_const , strength_CONST=np.ones(len(list_var_const)),provided_val=prov_val,option_minus_pi=option_minus_pi )
    weights_RCMIP = func_weights(VAR_CONST=list_var_const+['ECS_charney'] , strength_CONST=np.hstack([np.ones(len(list_var_const)),1]) , provided_val=prov_val,option_minus_pi=option_minus_pi )
    w_empty = func_weights(VAR_CONST=[] , strength_CONST=np.ones(len(list_var_const)),provided_val=prov_val,option_minus_pi=option_minus_pi )
    weights_CMIP6.to_netcdf(path_runs+'/treated/weights/weights_CMIP6.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in weights_CMIP6})
    weights_RCMIP.to_netcdf(path_runs+'/treated/weights/weights_RCMIP.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in weights_RCMIP})
    w_empty.to_netcdf(path_runs+'/treated/weights/weights_unconstrained.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in w_empty})
    print(" ")

    prov_val_masked = {ii:prov_val[ii]*list_noDV['1pctCO2'][ii][140,:] for ii in prov_val.keys()}
    weights_CMIP6_masked = func_weights(VAR_CONST=list_var_const , strength_CONST=np.ones(len(list_var_const)),provided_val=prov_val_masked,option_minus_pi=option_minus_pi )
    w_empty_masked = func_weights(VAR_CONST=[] , strength_CONST=np.ones(len(list_var_const)),provided_val=prov_val_masked,option_minus_pi=option_minus_pi )


    ## plot to check distributions
    if option_plots_treatment:
        plt.figure( figsize=(30,20) )
        plt.suptitle( 'Constraints used: '+', '.join( list_var_const ) )
        ## unconstrained
        for ii in np.arange( len(list_var_plot) ):
            ax = plt.subplot( 3,len(list_var_plot),1+ii )
            func_plot_distrib( ax=ax,VAR=list_var_plot[ii] , weights=w_empty , provided_val=prov_val,option_minus_pi=option_minus_pi )
            if ii==0:
                plt.ylabel( 'Unconstrained' )
        ## constrained
        for ii in np.arange( len(list_var_plot) ):
            ax = plt.subplot( 3,len(list_var_plot),len(list_var_plot)+1+ii )
            func_plot_distrib( ax=ax,VAR=list_var_plot[ii] , weights=weights_CMIP6 , provided_val=prov_val,option_minus_pi=option_minus_pi )
            if ii==0:
                plt.ylabel( 'Constrained CMIP6' )
        ## constrained
        for ii in np.arange( len(list_var_plot) ):
            ax = plt.subplot( 3,len(list_var_plot),2*len(list_var_plot)+1+ii )
            func_plot_distrib( ax=ax,VAR=list_var_plot[ii] , weights=weights_RCMIP , provided_val=prov_val,option_minus_pi=option_minus_pi )
            if ii==0:
                plt.ylabel( 'Constrained RCMIP' )


elif option_weights == 'LOAD_constraints_4':
    weights_CMIP6 = xr.open_dataset(path_runs+'/treated/weights/weights_CMIP6.nc')
    weights_RCMIP = xr.open_dataset(path_runs+'/treated/weights/weights_RCMIP.nc')


elif option_weights == 'LOAD_assessed_ranges':
    ## Loading indicators
    folder_rcmip = 'results/RCMIP_phase2/'
    if option_maskDV == 'LOAD_indiv':
        indic = xr.load_dataset( folder_rcmip + 'oscar_indicators_full-configs_mask_indiv.nc' )
    elif option_maskDV == 'LOAD_unique':
        indic = xr.load_dataset( folder_rcmip + 'oscar_indicators_full-configs_mask_unique.nc' )
    ## correction, some coordinates become variables....
    for var in ['RCMIP variable', 'RCMIP region', 'RCMIP scenario']:
        indic.coords[var] = indic[var]
    indic = indic.drop('distrib')

    ## indicators to use for weighting OSCAR by Yann Quilcaille for RCMIP-phase 2
    ind_list = ['Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011',
                'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019',
                'Cumulative compatible emissions CMIP5 historical-CMIP5',
                'Cumulative compatible emissions CMIP5 RCP2.6',
                'Cumulative compatible emissions CMIP5 RCP4.5',
                'Cumulative compatible emissions CMIP5 RCP6.0',
                'Cumulative compatible emissions CMIP5 RCP8.5',
                ]
    ## checking that all ind_list are within
    if len( [ind for ind in ind_list if ind not in indic.indicator] )>0: raise Exception('Warning, missing indicators.')

    ## calculating products of weights
    WEIGHTS = xr.Dataset()
    WEIGHTS.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]#mask_all.config.sel(config=configs)
    val = indic['w'].sel(index=[i for i in indic.index if str(indic.indicator.sel(index=i).values) in ind_list]).prod('index')
    WEIGHTS['weights'] = xr.DataArray( data=val.values , dims=('all_config') )

    weights_CMIP6 = WEIGHTS.copy()
    weights_RCMIP = WEIGHTS.copy()
    del WEIGHTS


if False: ## creating list of configurations for project OSCAR-SSP
    ## loading CMIP6 weights
    weights_CMIP6 = xr.open_dataset(path_runs+'/treated/weights/weights_CMIP6.nc')
    date_removal = 2500
    if False:
        ## taking masks into account, using all ssps and their esm- counterpart
        if date_removal<=2100:
            weights_CMIP6['mask'] = xr.DataArray(  np.hstack( [ np.prod([list_noDV[esm+xp][setMC][date_removal-2014,:] for xp in ['ssp119','ssp126','ssp534-over','ssp434','ssp245','ssp460','ssp370','ssp585'] for esm in ['','esm-']],axis=0) for setMC in list_setMC] )  , dims=('all_config')  )
        else:
            weights_CMIP6['mask'] = xr.DataArray(  np.hstack( [ np.prod([list_noDV[esm+xp][setMC][date_removal-2100,:] for xp in ['ssp119ext','ssp126ext','ssp534-over-ext','ssp434ext','ssp245ext','ssp460ext','ssp370ext','ssp585ext'] for esm in ['','esm-']],axis=0) for setMC in list_setMC] )  , dims=('all_config')  )
    else:
        ## taking ALL maks into account
        tmp,tmp_xp_as_last = [],[]
        for setMC in list_setMC:
            vect = np.ones( 500 )
            for xp in list_xp:
                if xp in ['ssp119ext','ssp126ext','ssp534-over-ext','ssp434ext','ssp245ext','ssp460ext','ssp370ext','ssp585ext'] + ['esm-ssp119ext','esm-ssp126ext','esm-ssp534-over-ext','esm-ssp434ext','esm-ssp245ext','esm-ssp460ext','esm-ssp370ext','esm-ssp585ext']:
                    vect *= list_noDV[xp][setMC][date_removal-2100,:]
                elif xp in ['ssp119','ssp126','ssp534-over','ssp434','ssp245','ssp460','ssp370','ssp585'] + ['esm-ssp119','esm-ssp126','esm-ssp534-over','esm-ssp434','esm-ssp245','esm-ssp460','esm-ssp370','esm-ssp585']:
                    pass
                elif xp in ['1pctCO2','1pctCO2-bgc','1pctCO2-cdr','1pctCO2-rad','G2'] + ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC', 'esm-1pctCO2']:
                    vect *= list_noDV[xp][setMC][150,:]
                else:
                    vect *= list_noDV[xp][setMC][-1,:]
                    if xp not in tmp_xp_as_last:tmp_xp_as_last.append(xp)
            tmp.append(vect)
        weights_CMIP6['mask'] = xr.DataArray(  np.hstack(tmp)  , dims=('all_config')  )

    ## initialization
    weights_select,Par_select = xr.Dataset() , []
    if False:## for plot and comparison
        weights_select['weights'] = weights_CMIP6['weights'] * weights_CMIP6['mask']
        ## selecting further
        # n_best = 600
        # weights_select['weights'] = np.nan * xr.zeros_like( weights_CMIP6.weights )
        # xx = (weights_CMIP6['weights'] * weights_CMIP6['mask']).sortby( weights_CMIP6['weights'] * weights_CMIP6['mask'] , ascending=False )
        # ind = np.where( xx>0. )[0]
        # yy = xx.isel(all_config=ind).isel(all_config=np.arange(n_best))
        # weights_select['weights'].loc[{'all_config':yy.all_config}] = yy

    else:## for save
        ## creating sorted listed of weights
        weights_select['weights'] = (weights_CMIP6['weights'] * weights_CMIP6['mask']).sortby( weights_CMIP6['weights'] * weights_CMIP6['mask'], ascending=False )
        ## creating list of parameters
        for setMC in list_setMC:
            print(setMC)
            with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP:
                TMP = TMP.rename({'config':'all_config'})
                TMP.coords['all_config'] = [str(setMC)+'-'+str(cfg) for cfg in np.arange(500)]
                Par_select.append( TMP.copy(deep=True) )
                del TMP
        Par_select = xr.concat( Par_select , dim='all_config')
        ## selecting relevant list of parameters
        ind = np.where( weights_select['weights']>0. )[0]
        weights_select = weights_select.isel( all_config=ind )
        ## selecting further
        # n_best = 600
        weights_select,xx = xr.Dataset() , weights_select['weights'].sortby(weights_select['weights'] , ascending=False).isel(all_config=np.arange(n_best))
        weights_select['weights'] = xx
        ## plot for check
        if False:
            setMC,xp = 17,'ssp370'
            mem = {}
            for setMC in list_setMC:
                with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par_tmp = TMP.load()
                out_tmp = xr.open_dataset(path_runs+'/'+xp+'_Out-'+str(setMC)+'.nc' )
                for_tmp = xr.open_dataset(path_runs+'/'+xp+'_For-'+str(setMC)+'.nc' )
                out_tmp['D_Focean'] = OSCAR['D_Focean'](out_tmp, Par_tmp, for_tmp,recursive=True)
                plt.plot( out_tmp.year,  out_tmp.D_Focean.sel(  config=[cfg for cfg in np.arange(500) if (str(setMC)+'-'+str(cfg) in weights_select.all_config)]   ) )
                plt.plot( out_tmp.year,  out_tmp.D_Focean.sel( config=192 ) )
                plt.plot( out_tmp.year,  out_tmp.D_Tg.sel( config=192 ) )
                # out_me,for_me,par_me = out_tmp.sel(config=192),for_tmp.sel(config=192),Par_tmp.sel(config=192)

                # ## comparing to Iris:
                # sc,a = 'SSP3-7.0',''
                # for_iris = xr.open_dataset('P:\imbalancep\Earth_system_modeling\OSCAR-SSP\SSP run full data\Forcings\For_c_{}{}_full_2500.nc'.format(sc,a))
                # out_iris = xr.open_dataset('P:\imbalancep\Earth_system_modeling\OSCAR-SSP\SSP run full data\OUT\OUT_c_{}{}_full_2500.nc'.format(sc,a))
                # # for_iris = xr.open_dataset('P:\imbalancep\Earth_system_modeling\OSCAR-SSP\SSP run full data\Forcings\For_c_{}{}_2500.nc'.format(sc,a))
                # # out_iris = xr.open_dataset('P:\imbalancep\Earth_system_modeling\OSCAR-SSP\SSP run full data\OUT\OUT_c_{}{}_2500.nc'.format(sc,a))
                # par_iris = xr.open_dataset("P:\imbalancep\Earth_system_modeling\OSCAR-SSP\parameters_OSCAR-SSP_2500.nc")
                # out_iris['D_Focean'] = OSCAR['D_Focean'](out_iris, par_iris, for_iris,recursive=True)
                # # plt.plot( out_iris.year , out_iris['D_Focean'].sel(all_config=2) )
                # out_ir,for_ir,par_ir = out_iris.sel(all_config=2).copy(deep=True),for_iris.copy(deep=True),par_iris.sel(all_config=2).copy(deep=True)
                # for_iris.close()
                # out_iris.close()
                # del for_iris,out_iris

                # a,dico_folder = '',{'hist':'historical run','SSP5-8.5':'SSP run full data','SSP3-7.0':'SSP run full data'}
                # for sc in ['hist','SSP3-7.0']:
                #     for_iris = xr.open_dataset('P:\imbalancep\Earth_system_modeling\OSCAR-SSP\{}\Forcings\For_c_{}{}_2500.nc'.format(dico_folder[sc],sc,a))
                #     out_iris = xr.open_dataset('P:\imbalancep\Earth_system_modeling\OSCAR-SSP\{}\OUT\OUT_c_{}{}_2500.nc'.format(dico_folder[sc],sc,a))
                #     par_iris = xr.open_dataset("P:\imbalancep\Earth_system_modeling\OSCAR-SSP\parameters_OSCAR-SSP_2500.nc")
                #     # out_iris['D_Focean'] = OSCAR_lite['D_Focean'](out_iris, par_iris, for_iris,recursive=True)
                #     # plt.plot( out_iris.year , out_iris['D_Focean'].sel(all_config=2) , label=sc )
                #     plt.plot( out_iris.year , out_iris['D_Tg'].sel(all_config=2) , label=sc )
                #     for_iris.close()
                #     out_iris.close()
                #     del for_iris,out_iris
                
                # for var in for_me:
                #     if var not in for_ir:
                #         print(var+' missing in for_ir')
                #     else:
                #         if var in ['Eff','E_CH4','E_N2O','RF_volc','RF_solar']:## normal
                #             pass
                #         elif var in ['d_Ashift','d_Acover','d_Hwood']:## strange
                #             pass
                #         elif 'spc_halo' not in for_me[var].dims:
                #             if len( np.where( for_me[var].sel(year=slice(2015,2100)) != for_ir[var].sel(year=slice(2015,2100)) )[0] ) > 0:
                #                 raise Exception( var+" different" )
                #         else:
                #             for halo in for_ir.spc_halo.values:
                #                 if np.all(np.isnan( for_me[var].sel(year=slice(2015,2100),spc_halo=halo) ))  and  np.all(np.isnan( for_ir[var].sel(year=slice(2015,2100),spc_halo=halo) )):
                #                     pass
                #                 elif halo in ['CH3Br']:
                #                     pass
                #                 else:
                #                     if len( np.where( for_me[var].sel(year=slice(2015,2100),spc_halo=halo) != for_ir[var].sel(year=slice(2015,2100),spc_halo=halo) )[0] ) > 0:
                #                         raise Exception( var+" different on "+halo )
        Par_select = Par_select.sel( all_config=weights_select.all_config )
        ## adding info on selection of configurations
        Par_select.attrs['warning'] = 'Method for the choice of these configurations: 10000 random configurations of OSCAR have been drawn, run on the 8 ssp scenarios of CMIP6, concentrations- and emissions-driven. Any configuration that lead to a divergence on any of these runs before '+str(date_removal)+' were removed. For more information on the exclusion of configurations, check Quilcaille et al, 2020.'
        weights_select.attrs['warning'] = 'Method for the choice of these configurations: 10000 random configurations of OSCAR have been drawn, run on the 8 ssp scenarios of CMIP6, concentrations- and emissions-driven. Any configuration that lead to a divergence on any of these runs before '+str(date_removal)+' were removed. For more information on the exclusion of configurations, check Quilcaille et al, 2020.'
        weights_select.attrs['info'] = 'Weights have been calculated using observational constraints. For more information, check Quilcaille et al, 2020.'

        weights_select.coords['all_config'] = np.arange(weights_select.all_config.size)
        Par_select.coords['all_config'] = np.arange(Par_select.all_config.size)
        weights_select.to_netcdf('P:/imbalancep/Earth_system_modeling/OSCAR-SSP/weights_OSCAR-SSP.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in weights_select})
        Par_select.to_netcdf('P:/imbalancep/Earth_system_modeling/OSCAR-SSP/parameters_OSCAR-SSP.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Par_select})
        # weights_select.to_netcdf('P:/imbalancep/Earth_system_modeling/OSCAR-SSP/weights_OSCAR-SSP_'+str(date_removal)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in weights_select})
        # Par_select.to_netcdf('P:/imbalancep/Earth_system_modeling/OSCAR-SSP/parameters_OSCAR-SSP_'+str(date_removal)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Par_select})

#########################
#########################
##################################################
##################################################











##################################################
## 4. PRODUCING FILES FOR DIFFERENT MIPs
##################################################
#########################
## 4.1. PREPARATION
#########################
## if not set to None, will select configurations for which there is a weight greater than the median/threshold (distribution highly assimetric, mean/median~1.e4)
threshold_select_ProbableRuns = None ## introduces spurious variations

## Dictionary for information on variables
with open('dico_variables_MIPs.csv','r',newline='') as ff:
    dico_variables_MIPs = np.array([line for line in csv.reader(ff)])
head_MIPs = list(dico_variables_MIPs[0,:])

## long names, units and warnings that will be saved in netCDF as description
dico_variables_longnames,dico_variables_units,dico_variables_warnings = {},{},{}
for line in dico_variables_MIPs[1:]:
    if line[head_MIPs.index('CMIP6 name')]!='':
        dico_variables_longnames[ line[head_MIPs.index('CMIP6 name')] ] = line[head_MIPs.index('longname')]
        dico_variables_units[ line[head_MIPs.index('CMIP6 name')] ] = line[head_MIPs.index('unit')]
        dico_variables_warnings[ line[head_MIPs.index('CMIP6 name')] ] = line[head_MIPs.index('warnings')]
    elif line[head_MIPs.index('RCMIP name')]!='':
        dico_variables_longnames[ line[head_MIPs.index('RCMIP name')] ] = line[head_MIPs.index('longname')]
        dico_variables_units[ line[head_MIPs.index('RCMIP name')] ] = line[head_MIPs.index('unit')]
        dico_variables_warnings[ line[head_MIPs.index('RCMIP name')] ] = line[head_MIPs.index('warnings')]

## variables to use for computation
dico_varOSCAR_to_varCMIP6 =  { line[head_MIPs.index( 'CMIP6 name' )]:eval(line[head_MIPs.index( 'variables of OSCAR used for calculation' )]) for line in dico_variables_MIPs[1:] if line[head_MIPs.index( 'CMIP6 name' )]!=''}
dico_varOSCAR_to_varRCMIP =  { line[head_MIPs.index( 'RCMIP name' )]:eval(line[head_MIPs.index( 'variables of OSCAR used for calculation' )]) for line in dico_variables_MIPs[1:] if line[head_MIPs.index( 'RCMIP name' )]!=''}

## Dictionary used for names of spc_halo of OSCAR to names of halogenated compounds used in RCMIP
dico_spc_halo = {'C2F6':'C2F6', 'C3F8':'C3F8', 'C4F10':'C4F10', 'C5F12':'C5F12', 'C6F14':'C6F14', 'C7F16':'C7F16', 'CCl4':'CCl4', 'CF4':'CF4',\
                'CFC11':'CFC-11', 'CFC113':'CFC-113', 'CFC114':'CFC-114', 'CFC115':'CFC-115', 'CFC12':'CFC-12', 'CH3Br':'CH3Br','CH3CCl3':'CH3CCl3',\
                'CH3Cl':'CH3Cl', 'HCFC141b':'HCFC-141b', 'HCFC142b':'HCFC-142b', 'HCFC22':'HCFC-22', 'HFC125':'HFC-125', 'HFC134a':'HFC-134a',\
                'HFC143a':'HFC-143a', 'HFC152a':'HFC-152a', 'HFC227ea':'HFC-227ea', 'HFC23':'HFC-23', 'HFC236fa':'HFC-236fa', 'HFC245fa':'HFC-245fa',\
                'HFC32':'HFC-32', 'HFC365mfc':'HFC-365mfc', 'HFC4310mee':'HFC-43-10mee', 'Halon1202':'Halon-1202', 'Halon1211':'Halon-1211',\
                'Halon1301':'Halon-1301', 'Halon2402':'Halon-2402', 'NF3':'NF3', 'SF6':'SF6', 'cC4F8':'c-C4F8'}
list_spc_halo_NotInOscar = ['C8F18','SO2F2','CH2Cl2','CHCl3']

## Compatible emissions to calculate depending on the experiments. (indexed using 1st xp, extensions have same requirements for compatible emissions)
dico_compatible_emissions = { line[0]:eval(line[2])  for line in dico_experiments_MIPs[1:] }
dico_compatible_emissions.update({ line[1]:eval(line[2])  for line in dico_experiments_MIPs[1:] if line[1]!=''})

## Some experiments have been extended by 100 years so that we can exclude runs that begin to diverge
list_Xp_cut100 = ['1pctCO2-bgc' , '1pctCO2-rad' , '1pctCO2', 'esm-1pctCO2']

## Some variables are defined as differences: only (experiment-control)_t is evaluated, we dont add (control)_mean
list_VAR_NoAddingControl =  ['Effective Climate Sensitivity','Effective Climate Feedback','Instantaneous TCRE','Airborne Fraction|CO2','wetlandFrac','rf_tot','erf_tot'] +\
                            [vv for vv in dico_variables_MIPs[1:,head_MIPs.index('RCMIP name')] if str.split(vv,'|')[0] in ['Radiative Forcing' , 'Effective Radiative Forcing' , 'Heat Uptake' , 'Heat Content' , 'Cumulative Net Land to Atmosphere Flux' , 'Cumulative Net Ocean to Atmosphere Flux' , 'Net Land to Atmosphere Flux|CO2' , 'Net Ocean to Atmosphere Flux']]

## Some variables can be calculated during 'gather_xp' to accelerate the treatment
list_VAR_accelerate = ['cveg_0','csoil1_0','csoil2_0', 'D_Aland','D_csoil1','D_csoil2','D_cveg','D_npp','D_efire','D_fmort1','D_fmort2','D_rh1','D_rh2','D_fmet','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp','D_Ebb_nat','D_Ebb_ant','D_Xhalo_lag','D_CH4_lag','D_N2O_lag','D_Cosurf','D_Ewet',]

def varOSCAR_to_varMIP( VAR, out_tmp, Par, for_tmp , dico_var , option_SumReg , option_DiffControl , type_OSCAR):
    ##-----
    ## translates OSCAR variables to variables fitted to CMIP6
    ## In this function, it is assumed that the seeked results are differences to the control, thus not including PI variables.
    ## //!!\\ For 2 variables (airborne fraction and wetland fraction), IT WOULD BE MORE ACCURATE TO WRITE fraction(xp) - fraction(pi) rather than fraction(xp-pi). Yet, pi is constant enough to neglect this effect.
    ## Besides, several variables (eg the 2 ECS) are not the difference of the variable in xp to the one in pi
    ##-----
    ## calculating variable for project

    ## CLIMATE SYSTEM
    if VAR in ['year','rf_tot','erf_tot','year_breach']:## directly the variable, nothing else to do.
        val = out_tmp[ dico_var[VAR][0] ].copy(deep=True)

    elif VAR == 'tas':## converting into K    AND   ADDING A PREINDUSTRIAL GLOBAL TEMPERATURE
        TMP = xr.open_dataset('input_data/observations/climatology_GSWP3.nc')
        vals = aggreg_region(ds_in=TMP , mod_region='RCP_5reg', weight_var={'Tl':'area','Pl':'area'})
        if option_SumReg:
            if type_OSCAR=='OSCAR':
                val = out_tmp['D_Tg'].copy(deep=True)
            elif type_OSCAR=='OSCAR_landC':
                val = (out_tmp['D_Tl'] * vals.area).sum('reg_land') / vals.area.sum('reg_land')
            if option_DiffControl==False:
                val += 273.15+13.9
        else:
            ## regional temperature
            val = out_tmp['D_Tl'].copy(deep=True)
            if option_DiffControl==False:
                val.loc[{'reg_land':np.arange(1,val.reg_land.size)}] += vals['Tl'].sel(year=np.arange(1901,1920+1),reg_land=np.arange(1,val.reg_land.size)).mean('year')
            ## global temperature
            if type_OSCAR=='OSCAR':
                val.loc[{'reg_land':0}] = out_tmp['D_Tg'].copy(deep=True)
                if option_DiffControl==False:
                    val.loc[{'reg_land':0}] += 273.15+13.9
            elif type_OSCAR=='OSCAR_landC':
                val.loc[{'reg_land':0}] = np.nan
                # val.loc[{'reg_land':0}] = (out_tmp['D_Tl'] * vals.area).sum('reg_land') / vals.area.sum('reg_land')
                # if option_DiffControl==False:
                #     val.loc[{'reg_land':0}] += (vals['Tl'].sel(year=np.arange(1901,1920+1),reg_land=np.arange(1,val.reg_land.size)).mean('year') * vals.area).sum('reg_land') / vals.area.sum('reg_land')
        TMP.close()

    elif VAR == 'ohc':## NOT converting from ZJ to J
        val = out_tmp['D_OHC'].copy(deep=True) # * 1.e21

    elif VAR == 'tos':## calculating and converting to K
        val = out_tmp['D_To'].copy(deep=True)
        if option_DiffControl==False:
            val += Par.To_0 + 273.15


    ## CONCENTRATIONS
    elif VAR in ['xco2','co2'] :## adding CO2_0 and NOT converting from ppm to mol mol-1
        val = (out_tmp['D_CO2']).copy(deep=True)
        if option_DiffControl==False:
            val += Par.CO2_0

    elif VAR == 'co2mass':## adding CO2_0 and converting to atmospheric stock of carbon
        if option_DiffControl==False:
            val = (out_tmp['D_CO2']+Par.CO2_0) * Par.a_CO2
        else:
            val = (out_tmp['D_CO2']) * Par.a_CO2
        ## removing Par.CO2_0, because difference to control

    elif VAR == 'ch4global':## calculating and NOT converting from ppb to mol mol-1
        val = (out_tmp['D_CH4']).copy(deep=True) # * 1.e-9
        if option_DiffControl==False:
            val += Par.CH4_0
        ## removing Par.CH4_0, because difference to control


    ## CARBON CYCLE: LAND STOCKS
    elif VAR in ['cLitter','cLitterLut']:
        val = ( (out_tmp['csoil1_0']+out_tmp['D_csoil1']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
        val += out_tmp['D_Csoil1_bk'].sum(('bio_from','bio_to'))
        if option_DiffControl:
            val -= (out_tmp['csoil1_0']*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR in ['cSoil','cSoilLut']:
        val = ( (out_tmp['csoil2_0']+out_tmp['D_csoil2']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
        val += out_tmp['D_Csoil2_bk'].sum(('bio_from','bio_to'))
        if option_DiffControl:
            val -= (out_tmp['csoil2_0']*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR in ['cVeg','cVegLut']:
        val = ( (out_tmp['cveg_0']+out_tmp['D_cveg']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
        val += out_tmp['D_Cveg_bk'].sum(('bio_from','bio_to'))
        if option_DiffControl:
            val -= (out_tmp['cveg_0']*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR in ['cProduct','cProductLut']:
        val = out_tmp['D_Chwp'].sum( ('bio_from','bio_to','box_hwp') )

    elif VAR == 'cLand':
        val =  ( (out_tmp['csoil1_0']+out_tmp['D_csoil1']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
        val += ( (out_tmp['csoil2_0']+out_tmp['D_csoil2']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
        val += ( (out_tmp['cveg_0']+out_tmp['D_cveg']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
        val += ( out_tmp['D_Csoil1_bk']+out_tmp['D_Csoil2_bk']+out_tmp['D_Cveg_bk'] ).sum(('bio_from','bio_to'))
        val += out_tmp['D_Chwp'].sum( ('bio_from','bio_to','box_hwp') )
        if option_DiffControl:
            val -= ((out_tmp['csoil1_0']+out_tmp['csoil2_0']+out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control


    ## CARBON CYCLE: LAND FLUXES
    # npp = photosynthesis - plant_resp    ?  (equivalent OSCAR)
    # nep = (photosynthesis - plant_resp) - rh - fire  (close to nbp of OSCAR)
    # necb= (photosynthesis - plant_resp) - rh - fire - harv - grazing
    # nbp = (photosynthesis - plant_resp) - rh - fire - harv - luc - grazing   ---> all contributions sumed, equivalent D_Fland-D_Fluc (nbp_OSCAR*Area + D_NBP_bk)
    # fluc = luc - forest_regrowth      /!\ forest regrowth accounted here
    elif VAR in ['npp','nppLut']:
        val = ( (Par.npp_0+out_tmp['D_npp']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        val += out_tmp['D_NPP_bk'] ## actually 0.
        if option_DiffControl:
            val -= (Par.npp_0*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR == 'nbp':
        val = ( (Par.npp_0+out_tmp['D_npp']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land')) \
              - ( (Par.rho1_0*out_tmp['csoil1_0'] + Par.rho2_0*out_tmp['csoil2_0']) * (Par.Aland_0 + out_tmp['D_Aland']) + (out_tmp['D_rh1'] + out_tmp['D_rh2']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land')) \
              - ( (Par.igni_0*out_tmp['cveg_0'] + out_tmp['D_efire']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))# - out_tmp['D_Efire_bk'].sum(('bio_from','bio_to'))
        val -= ( (Par.harv_0+Par.graz_0) * (out_tmp['cveg_0']+out_tmp['D_cveg']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        val += out_tmp['D_NPP_bk'] + ( - out_tmp['D_Efire_bk'] - out_tmp['D_Rh1_bk'] - out_tmp['D_Rh2_bk'] - out_tmp['D_Ehwp'].sum('box_hwp')).sum(('bio_from','bio_to'))
        val += ( - out_tmp['D_Eharv_bk'] - out_tmp['D_Egraz_bk'] ).sum(('bio_from','bio_to'))
        ## EQUIVALENT TO::
        # nbp0 = Par.npp_0 - (Par.igni_0 + Par.harv_0 + Par.graz_0) * out_tmp['cveg_0'] - Par.rho1_0 * out_tmp['csoil1_0'] - Par.rho2_0 * out_tmp['csoil2_0']
        # val = ( (nbp0 + out_tmp['D_nbp']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        if option_DiffControl:
            val -= ((Par.npp_0-Par.rho1_0*out_tmp['csoil1_0']-Par.rho2_0*out_tmp['csoil2_0']-Par.igni_0*out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
            val += ( (Par.harv_0+Par.graz_0) * (out_tmp['cveg_0']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
            ## removing val in PI, because difference to control
        ## all contributions

    elif VAR in ['necb','necbLut']:## similar to nbp, but without wood products
        val = ( (Par.npp_0+out_tmp['D_npp']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land')) \
              - ( (Par.rho1_0*out_tmp['csoil1_0'] + Par.rho2_0*out_tmp['csoil2_0']) * (Par.Aland_0 + out_tmp['D_Aland']) + (out_tmp['D_rh1'] + out_tmp['D_rh2']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land')) \
              - ( (Par.igni_0*out_tmp['cveg_0'] + out_tmp['D_efire']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))# - out_tmp['D_Efire_bk'].sum(('bio_from','bio_to'))
        val -= ( (Par.harv_0+Par.graz_0) * (out_tmp['cveg_0']+out_tmp['D_cveg']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        val += out_tmp['D_NPP_bk'] + ( - out_tmp['D_Efire_bk'] - out_tmp['D_Rh1_bk'] - out_tmp['D_Rh2_bk'] ).sum(('bio_from','bio_to'))
        val += ( - out_tmp['D_Eharv_bk'] - out_tmp['D_Egraz_bk'] ).sum(('bio_from','bio_to'))
        if option_DiffControl:
            val -= ((Par.npp_0-Par.rho1_0*out_tmp['csoil1_0']-Par.rho2_0*out_tmp['csoil2_0']-Par.igni_0*out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
            val -= ( (Par.harv_0+Par.graz_0) * (out_tmp['cveg_0']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR == 'nep':
        val = ( (Par.npp_0+out_tmp['D_npp']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land')) \
              - ( (Par.rho1_0*out_tmp['csoil1_0'] + Par.rho2_0*out_tmp['csoil2_0']) * (Par.Aland_0 + out_tmp['D_Aland']) + (out_tmp['D_rh1'] + out_tmp['D_rh2']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land')) \
              - ( (Par.igni_0*out_tmp['cveg_0'] + out_tmp['D_efire']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))# - out_tmp['D_Efire_bk'].sum(('bio_from','bio_to'))
        val += out_tmp['D_NPP_bk'] + ( - out_tmp['D_Efire_bk'] - out_tmp['D_Rh1_bk'] - out_tmp['D_Rh2_bk'] ).sum(('bio_from','bio_to'))
        if option_DiffControl:
            val -= ((Par.npp_0-Par.rho1_0*out_tmp['csoil1_0']-Par.rho2_0*out_tmp['csoil2_0']-Par.igni_0*out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control
        ## difference to necb: remove contribution of harv

    elif VAR == 'fLuc':
        val = -out_tmp['D_NBP_bk'].sum( ('bio_from','bio_to') )
        # val = - out_tmp['D_NPP_bk'] + ( out_tmp['D_Efire_bk'] + out_tmp['D_Rh1_bk'] + out_tmp['D_Rh2_bk'] + out_tmp['D_Ehwp'].sum('box_hwp')).sum(('reg_land','bio_from','bio_to'))
        ## /!\ accounts here for forest regrowth.

    elif VAR in ['rh','rhLut']:
        val = ( (Par.rho1_0*out_tmp['csoil1_0'] + Par.rho2_0*out_tmp['csoil2_0'] + out_tmp['D_rh1'] + out_tmp['D_rh2']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        if option_DiffControl:
            val -= ((Par.rho1_0*out_tmp['csoil1_0']+Par.rho2_0*out_tmp['csoil2_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR == 'rhLitter':
        val = ( (Par.rho1_0*out_tmp['csoil1_0'] + out_tmp['D_rh1'] ) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        if option_DiffControl:
            val -= ((Par.rho1_0*out_tmp['csoil1_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR == 'rhSoil':
        val = ( (Par.rho2_0*out_tmp['csoil2_0'] + out_tmp['D_rh2']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        if option_DiffControl:
            val -= ((Par.rho2_0*out_tmp['csoil2_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR == 'fFire':
        val = ( (Par.igni_0*out_tmp['cveg_0'] + out_tmp['D_efire']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        val += (Par.p_hwp_bb * out_tmp['D_Ehwp'] ).sum( ('box_hwp','bio_to','bio_from') )
        if option_DiffControl:
            val -= ((Par.igni_0*out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control
    elif VAR == 'fFireAll':
        val = ( (Par.igni_0*out_tmp['cveg_0'] + out_tmp['D_efire']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        val += (Par.p_hwp_bb * out_tmp['D_Ehwp'] ).sum( ('box_hwp','bio_to','bio_from') )
        val += out_tmp['D_Efire_bk'].sum(('bio_from','bio_to')) ## contribution included in D_Eluc
        if option_DiffControl:
            val -= ((Par.igni_0*out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control

    elif VAR == 'cTotFireLut':
        val = ( (Par.igni_0*out_tmp['cveg_0'] + out_tmp['D_efire']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
        val += (Par.p_hwp_bb * out_tmp['D_Ehwp'] ).sum( ('box_hwp','bio_to','bio_from') )
        val += out_tmp['D_Efire_bk'].sum(('bio_from','bio_to')) ## contribution included in D_Eluc
        if option_DiffControl:
            val -= ((Par.igni_0*out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
            ## removing val in PI, because difference to control
        val = val.cumsum('year')

    elif VAR == 'fProductDecompLut':
        val = out_tmp['D_Ehwp'].sum(('box_hwp','bio_from','bio_to'))

    elif VAR == 'fLulccProductLut':
        val = out_tmp['D_Fhwp'].sum(('box_hwp','bio_from','bio_to'))

    elif VAR == 'fLulccResidueLut':
        val = out_tmp['D_Fslash'].sum(('bio_from','bio_to'))

    elif VAR == 'fLulccAtmLut':
        val = xr.zeros_like( out_tmp.year )        
        ## 0 in OSCAR
        

    ## CARBON CYCLE: OCEAN STOCKS
    elif VAR == 'cOcean_deep':## calculating
        val = out_tmp['D_Fcirc'].cumsum('year').sum('box_osurf',min_count=1) ## PI carbon in Deep Ocean from Figure 6.1, AR5 WG1 Ch6 p.471, DIC only
        if option_DiffControl==False:
            val += 37100.

    elif VAR == 'cOcean_surf':## calculating
        val = out_tmp['D_Cosurf'].sum('box_osurf',min_count=1)
        if option_DiffControl==False:
            val += out_tmp['dic_0'] * Par.A_ocean * Par.a_CO2 * Par.mld_0 / Par.a_dic

    elif VAR == 'cOcean':## calculating
        val = out_tmp['D_Fcirc'].cumsum('year').sum('box_osurf',min_count=1) ## PI carbon in Deep Ocean from Figure 6.1, AR5 WG1 Ch6 p.471, DIC only
        val += out_tmp['D_Cosurf'].sum('box_osurf',min_count=1)
        if option_DiffControl==False:
            val += 37100.+ out_tmp['dic_0'] * Par.A_ocean * Par.a_CO2 * Par.mld_0 / Par.a_dic

    elif VAR == 'dissicos':## calculating
        val = out_tmp['D_dic'].copy(deep=True) ##
        if option_DiffControl==False:
            val += out_tmp['dic_0']


    ## CARBON CYCLE: OCEAN FLUXES
    elif VAR == 'fgco2':## NOT converting from PgC yr-1 to kgC m-2 s-1
        val = out_tmp['D_Focean'].copy(deep=True) # * 1.e12 / Par.A_ocean / (365.25*24*3600) # positive: into ocean

    elif VAR == 'dpco2':## calculating, and expressing in ppm and not in Pa.
        val = out_tmp['D_pCO2'] - out_tmp['D_CO2']


    ## WETLANDS
    elif VAR == 'wetlandCH4':## calculating and NOT converting from TgC yr-1 to kg m-2 s-1
        val = (out_tmp['D_Ewet']).copy(deep=True)#.sum(('bio_land'))
        if option_DiffControl==False:
            val += (Par.ewet_0*Par.Aland_0).sum('bio_land')

    elif VAR == 'wetlandFrac':## calculating
        if option_SumReg: val = (Par.Awet_0 + out_tmp['D_Awet']).sum(('reg_land')) / ( (Par.Aland_0 + out_tmp['D_Aland']).sum(('reg_land','bio_land')) )
        else: val = (Par.Awet_0 + out_tmp['D_Awet']) / ( (Par.Aland_0 + out_tmp['D_Aland']).sum(('bio_land')) )


    ## PERMAFROST
    elif VAR == 'permafrostCH4':## calculating
        val = out_tmp['D_Epf_CH4'].sum('reg_pf',min_count=1)

    elif VAR == 'permafrostCO2':## calculating
        val = out_tmp['D_Epf_CO2'].sum('reg_pf',min_count=1)

    elif VAR == 'cPermafrostFrozen':## calculating
        val = (out_tmp['D_Cfroz']).sum('reg_pf',min_count=1)
        if option_DiffControl==False:
            val += Par.Cfroz_0.sum('reg_pf',min_count=1)

    elif VAR == 'cPermafrostThawed':## calculating
        val = out_tmp['D_Cthaw'].sum('reg_pf',min_count=1).sum('box_thaw',min_count=1)


    ## ADDITIONAL VARIABLES
    elif VAR == 'fco2nat':## PgC yr-1
        val = -out_tmp['D_Focean'] - out_tmp['D_Fland'] + out_tmp['D_Epf_CO2'].sum('reg_pf') # positive: into atmosphere

    elif VAR == 'lossch4':## NOT converting from TgC yr-1 to mol m-3 s-1
        val = out_tmp['D_Fsink_CH4'].copy(deep=True)

    elif VAR == 'ph':## calculating
        val = out_tmp['D_pH'].copy(deep=True)
        if option_DiffControl==False:
            val += 8.15946734

    elif VAR == 'pr':## calculating and NOT converting to kg m-2 s-1
        TMP = xr.open_dataset('input_data/observations/climatology_GSWP3.nc')
        vals = aggreg_region(ds_in=TMP , mod_region='RCP_5reg', weight_var={'Tl':'area','Pl':'area'})
        if option_SumReg:
            if type_OSCAR=='OSCAR':
                val = out_tmp['D_Pg']
            elif type_OSCAR=='OSCAR_landC':
                val = (out_tmp['D_Pl'] * vals.area).sum('reg_land') / vals.area.sum('reg_land')
            if option_DiffControl==False:
                val += 990.
        else:
            ## regional temperature
            val = out_tmp['D_Pl'].copy(deep=True)
            if option_DiffControl==False:
                val.loc[{'reg_land':np.arange(1,val.reg_land.size)}] += vals['Pl'].sel(year=np.arange(1901,1920+1),reg_land=np.arange(1,val.reg_land.size)).mean('year')
            ## global temperature
            if type_OSCAR=='OSCAR':
                val.loc[{'reg_land':0}] = out_tmp['D_Pg'].copy(deep=True)
                if option_DiffControl==False:
                    val.loc[{'reg_land':0}] += 990.
            elif type_OSCAR=='OSCAR_landC':
                val.loc[{'reg_land':0}] = np.nan
                # val.loc[{'reg_land':0}] = (out_tmp['D_Pl'] * vals.area).sum('reg_land') / vals.area.sum('reg_land')
                # if option_DiffControl==False:
                #     val.loc[{'reg_land':0}] += (vals['Pl'].sel(year=np.arange(1901,1920+1),reg_land=np.arange(1,val.reg_land.size)).mean('year') * vals.area).sum('reg_land') / vals.area.sum('reg_land')
        TMP.close()



    ## VARIABLES RCMIP
    elif VAR in dico_variables_MIPs[1:,head_MIPs.index( 'RCMIP name' )]:
        ## CONCENTRATIONS RCMIP
        if VAR[:len('Atmospheric Concentrations|')] == 'Atmospheric Concentrations|':
            var = str.split( VAR , '|' )[-1]
            if var in ['CH4','CO2','N2O']:
                val = out_tmp['D_'+var].copy(deep=True)
                if option_DiffControl==False:
                    val += Par[var+'_0']
            elif var in dico_spc_halo.keys():
                val = (out_tmp['D_Xhalo'] ).sel( {'spc_halo':dico_spc_halo[var]}, drop=True ).copy(deep=True)
                if option_DiffControl==False:
                    val += Par['Xhalo_0'].sel( {'spc_halo':dico_spc_halo[var]}, drop=True )
            else:
                raise Exception("Variable Concentrations RCMIP not prepared: "+VAR)
            ## removed: 'FGases', 'HFC', 'PFC', 'CFC', 'Montreal Gases'

        ## NET FLUXES RCMIP
        elif (VAR[:len('Net Land to Atmosphere Flux|')] == 'Net Land to Atmosphere Flux|')   or   (VAR[:len('Cumulative Net Land to Atmosphere Flux|')] == 'Cumulative Net Land to Atmosphere Flux|'):
            opt_cumul = (VAR[:len('Cumulative Net Land to Atmosphere Flux|')] == 'Cumulative Net Land to Atmosphere Flux|')
            var = VAR[len(opt_cumul*'Cumulative '+'Net Land to Atmosphere Flux|'):]
            if var =='CH4':
                val = ( out_tmp['D_Ewet'] + out_tmp['D_Ebb_nat'].sel({'spc_bb':'CH4'}, drop=True).sum('bio_land') ) + out_tmp['D_Epf_CH4'].sum('reg_pf')
                val *= 16/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CH4|Earth System Feedbacks':
                val = ( out_tmp['D_Ewet'] + out_tmp['D_Ebb_nat'].sel({'spc_bb':'CH4'}, drop=True).sum('bio_land') ) + out_tmp['D_Epf_CH4'].sum('reg_pf')
                val *= 16/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CH4|Earth System Feedbacks|Other':
                val = ( out_tmp['D_Ewet'] + out_tmp['D_Ebb_nat'].sel({'spc_bb':'CH4'}, drop=True).sum('bio_land') )
                val *= 16/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CH4|Earth System Feedbacks|Other|Wetlands':
                val = out_tmp['D_Ewet'].copy(deep=True)
                val *= 16/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CH4|Earth System Feedbacks|Other|Natural Biomass Burning':
                val = out_tmp['D_Ebb_nat'].sel({'spc_bb':'CH4'}, drop=True).sum( ('bio_land') )
                val *= 16/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CH4|Earth System Feedbacks|Permafrost':
                val = out_tmp['D_Epf_CH4'].sum( 'reg_pf' )
                val *= 16/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CO2':
                val = out_tmp['D_Epf_CO2'].sum('reg_pf') - out_tmp['D_Fland']
                val *= 1.e3 * 44/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CO2|Earth System Feedbacks':
                val = out_tmp['D_Epf_CO2'].sum('reg_pf') - out_tmp['D_Fland']
                val *= 1.e3 * 44/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CO2|Earth System Feedbacks|Other':
                val = - out_tmp['D_Fland'].copy(deep=True)
                val *= 1.e3 * 44/12.
                if opt_cumul:val = val.cumsum('year')
            elif var =='CO2|Earth System Feedbacks|Permafrost':
                val = out_tmp['D_Epf_CO2'].sum('reg_pf')
                val *= 1.e3 * 44/12.
                if opt_cumul:val = val.cumsum('year')
            else:
                raise Exception("Variable Fluxes RCMIP not prepared: "+VAR)
        elif VAR in ['Net Ocean to Atmosphere Flux|CH4','Cumulative Net Ocean to Atmosphere Flux|CH4']:
            val = xr.zeros_like( out_tmp.year )
        elif VAR in ['Net Ocean to Atmosphere Flux|CO2','Cumulative Net Ocean to Atmosphere Flux|CO2']:
            val = -out_tmp['D_Focean'].copy(deep=True)
            val *= 1.e3 * 44/12.
            if VAR=='Cumulative Net Ocean to Atmosphere Flux|CO2':val = val.cumsum('year')

        ## CARBON POOLS RCMIP
        elif VAR == 'Carbon Pool|Atmosphere':
            if option_DiffControl:
                val = (out_tmp['D_CO2']) * Par.a_CO2
            else:
                val = (out_tmp['D_CO2']+ Par.CO2_0) * Par.a_CO2
            val *= 1.e3 * 44/12.
            ## removing val in PI, because difference to control
        elif VAR == 'Carbon Pool|Soil':
            val = ( (out_tmp['csoil2_0']+out_tmp['D_csoil2']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
            val += out_tmp['D_Csoil2_bk'].sum(('bio_from','bio_to'))
            if option_DiffControl:
                val -= (out_tmp['csoil2_0']*Par.Aland_0).sum(('bio_land'))
            val *= 1.e3 * 44/12.
        elif VAR == 'Carbon Pool|Detritus':
            val = ( (out_tmp['csoil1_0']+out_tmp['D_csoil1']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
            val += out_tmp['D_Csoil1_bk'].sum(('bio_from','bio_to'))
            if option_DiffControl:
                val -= (out_tmp['csoil1_0']*Par.Aland_0).sum(('bio_land'))
            val *= 1.e3 * 44/12.
        elif VAR == 'Carbon Pool|Plant':
            val = ( (out_tmp['cveg_0']+out_tmp['D_cveg']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
            val += out_tmp['D_Cveg_bk'].sum(('bio_from','bio_to'))
            if option_DiffControl:
                val -= (out_tmp['cveg_0']*Par.Aland_0).sum(('bio_land'))
            val *= 1.e3 * 44/12.
        elif VAR == 'Carbon Pool|Other':
            val = out_tmp['D_Chwp'].sum( ('bio_from','bio_to','box_hwp') )
            val *= 1.e3 * 44/12.

        ## (EFFECTIVE) RADIATIVE FORCING RCMIP
        elif (VAR[:len('Effective Radiative Forcing')] == 'Effective Radiative Forcing')  or  (VAR[:len('Radiative Forcing')] == 'Radiative Forcing'):
            if VAR == 'Effective Radiative Forcing':
                val = out_tmp['RF_warm'].copy(deep=True)
            elif VAR == 'Radiative Forcing':
                val = out_tmp['RF'].copy(deep=True)
            elif VAR == 'Effective Radiative Forcing|Anthropogenic':
                val = out_tmp['RF_wmghg'] + out_tmp['RF_slcf'] + Par.w_warm_bcsnow * out_tmp['RF_BCsnow'] + Par.w_warm_lcc * out_tmp['RF_lcc'] + out_tmp['RF_contr']
            elif VAR == 'Radiative Forcing|Anthropogenic':
                val = out_tmp['RF_wmghg'] + out_tmp['RF_slcf'] + out_tmp['RF_BCsnow'] + out_tmp['RF_lcc'] + out_tmp['RF_contr']
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols','Radiative Forcing|Anthropogenic|Aerosols']:
                val = out_tmp['RF_AERtot'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-cloud Interactions','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-cloud Interactions']:
                val = out_tmp['RF_cloud'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions']:
                val = out_tmp['RF_scatter'] + out_tmp['RF_absorb']
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC']:
                val = out_tmp['RF_BC'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC|Biomass Burning','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC|Biomass Burning']:
                D_BC_BCbb = Par.t_BCbb * out_tmp['D_Ebb'].sel({'spc_bb':'BC'}, drop=True).sum('bio_land', min_count=1)
                val = Par.rf_BC * D_BC_BCbb
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC|Fossil and Industrial','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC|Fossil and Industrial']:
                D_BC_BCff = Par.t_BCff * (Par.w_reg_BC * (Par.p_reg_slcf * out_tmp['E_BC'])).sum('reg_slcf', min_count=1)
                val = Par.rf_BC * D_BC_BCff
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC|Other','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|BC|Other']:
                val = Par.rf_BC * Par.G_BC * out_tmp['D_Tg']
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC']:
                val = out_tmp['RF_POA'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC|Biomass Burning','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC|Biomass Burning']:
                D_POA_OMbb = Par.a_POM * Par.t_OMbb * out_tmp['D_Ebb'].sel({'spc_bb':'OC'}, drop=True).sum('bio_land', min_count=1)
                val = Par.rf_POA * D_POA_OMbb
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC|Fossil and Industrial','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC|Fossil and Industrial']:
                D_POA_OMff = Par.a_POM * Par.t_OMff * (Par.w_reg_OC * (Par.p_reg_slcf * out_tmp['E_OC'])).sum('reg_slcf', min_count=1)
                val = Par.rf_POA * D_POA_OMff
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC|Other','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|BC and OC|OC|Other']:
                val = Par.rf_POA * Par.G_POA * out_tmp['D_Tg']
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Mineral Dust','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Mineral Dust']:
                val = out_tmp['RF_dust'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate']:
                val = out_tmp['RF_NO3'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate|Biomass Burning','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate|Biomass Burning']:
                D_NO3_NOX = Par.a_NO3 * Par.t_NOX * (out_tmp['D_Ebb'].sel({'spc_bb':'NOX'}, drop=True).sum('bio_land', min_count=1))
                D_NO3_NH3 = Par.a_NO3 * Par.t_NH3 * (out_tmp['D_Ebb'].sel({'spc_bb':'NH3'}, drop=True).sum('bio_land', min_count=1))
                val = Par.rf_NO3 * (D_NO3_NOX+D_NO3_NH3)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate|Fossil and Industrial','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate|Fossil and Industrial']:
                D_NO3_NOX = Par.a_NO3 * Par.t_NOX * out_tmp['E_NOX']
                D_NO3_NH3 = Par.a_NO3 * Par.t_NH3 * out_tmp['E_NH3']
                val = Par.rf_NO3 * (D_NO3_NOX+D_NO3_NH3)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate|Other','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Nitrate|Other']:
                val = Par.rf_NO3 * Par.G_NO3 * out_tmp['D_Tg']
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate']:
                val = out_tmp['RF_SO4'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate|Biomass Burning','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate|Biomass Burning']:
                D_SO4_SO2 = Par.a_SO4 * Par.t_SO2 * (Par.w_reg_SO2 * (Par.p_reg_slcf * out_tmp['D_Ebb'].sel({'spc_bb':'SO2'}, drop=True).sum('bio_land', min_count=1))).sum('reg_slcf', min_count=1)             
                val = Par.rf_SO4 * D_SO4_SO2
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate|Fossil and Industrial','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate|Fossil and Industrial']:
                D_SO4_SO2 = Par.a_SO4 * Par.t_SO2 * (Par.w_reg_SO2 * (Par.p_reg_slcf * out_tmp['E_SO2'])).sum('reg_slcf', min_count=1)
                val = Par.rf_SO4 * D_SO4_SO2
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate|Other','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Sulfate|Other']:
                D_SO4_DMS = Par.a_SO4 * Par.t_DMS * out_tmp['D_Edms']
                D_SO4_Tg = Par.G_SO4 * out_tmp['D_Tg']
                val = Par.rf_SO4 * ( D_SO4_DMS + D_SO4_Tg )
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Other','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Other']:
                val = out_tmp['RF_SOA'] + out_tmp['RF_salt']
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Other|Secondary Organic Aerosols','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Other|Secondary Organic Aerosols']:
                val = out_tmp['RF_SOA'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Other|Sea salts','Radiative Forcing|Anthropogenic|Aerosols|Aerosols-radiation Interactions|Other|Sea salts']:
                val = out_tmp['RF_salt'].copy(deep=True)
            elif VAR == 'Effective Radiative Forcing|Anthropogenic|Albedo Change':
                val = Par.w_warm_bcsnow * out_tmp['RF_BCsnow'] + Par.w_warm_lcc * out_tmp['RF_lcc']
            elif VAR == 'Effective Radiative Forcing|Anthropogenic|Albedo Change|Deposition of Black Carbon on Snow':
                val = Par.w_warm_bcsnow * out_tmp['RF_BCsnow']
            elif VAR == 'Effective Radiative Forcing|Anthropogenic|Albedo Change|Land Cover Change':
                val = Par.w_warm_lcc * out_tmp['RF_lcc']
            elif VAR == 'Radiative Forcing|Anthropogenic|Albedo Change':
                val = out_tmp['RF_BCsnow'] + out_tmp['RF_lcc']
            elif VAR == 'Radiative Forcing|Anthropogenic|Albedo Change|Deposition of Black Carbon on Snow':
                val = out_tmp['RF_BCsnow'].copy(deep=True)
            elif VAR == 'Radiative Forcing|Anthropogenic|Albedo Change|Land Cover Change':
                val = out_tmp['RF_lcc'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|CH4','Radiative Forcing|Anthropogenic|CH4']:
                val = out_tmp['RF_CH4'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|CO2','Radiative Forcing|Anthropogenic|CO2']:
                val = out_tmp['RF_CO2'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|N2O','Radiative Forcing|Anthropogenic|N2O']:
                val = out_tmp['RF_N2O'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|F-Gases|HFC','Radiative Forcing|Anthropogenic|F-Gases|HFC']:
                val = out_tmp['RF_Xhalo'].sel({'spc_halo':[dico_spc_halo[var] for var in ['HFC125','HFC134a','HFC143a','HFC152a','HFC227ea','HFC23','HFC236fa','HFC245fa','HFC32','HFC365mfc','HFC4310mee'] if var not in list_spc_halo_NotInOscar]},drop=True).sum('spc_halo')
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|F-Gases|PFC','Radiative Forcing|Anthropogenic|F-Gases|PFC']:
                val = out_tmp['RF_Xhalo'].sel({'spc_halo':[dico_spc_halo[var] for var in ['C2F6','C3F8','C4F10','C5F12','C6F14','C7F16','C8F18','cC4F8','CF4'] if var not in list_spc_halo_NotInOscar]},drop=True).sum('spc_halo')
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|F-Gases','Radiative Forcing|Anthropogenic|F-Gases']:
                val = out_tmp['RF_Xhalo'].sel({'spc_halo':[dico_spc_halo[var] for var in ['HFC125','HFC134a','HFC143a','HFC152a','HFC227ea','HFC23','HFC236fa','HFC245fa','HFC32','HFC365mfc','HFC4310mee']+['C2F6','C3F8','C4F10','C5F12','C6F14','C7F16','C8F18','cC4F8','CF4']+['NF3','SF6','SO2F2'] if var not in list_spc_halo_NotInOscar]},drop=True).sum('spc_halo')
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Montreal Gases','Radiative Forcing|Anthropogenic|Montreal Gases']:
                val = out_tmp['RF_Xhalo'].sel({'spc_halo':[dico_spc_halo[var] for var in ['CFC11','CFC113','CFC114','CFC115','CFC12']+['CCl4','CH2Cl2','CH3Br','CH3CCl3','CH3Cl','CHCl3','Halon1202','Halon1211','Halon1301','Halon2402','HCFC141b','HCFC142b','HCFC22'] if var not in list_spc_halo_NotInOscar]},drop=True).sum('spc_halo')
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Montreal Gases|CFC','Radiative Forcing|Anthropogenic|Montreal Gases|CFC']:
                val = out_tmp['RF_Xhalo'].sel({'spc_halo':[dico_spc_halo[var] for var in ['CFC11','CFC113','CFC114','CFC115','CFC12'] if var not in list_spc_halo_NotInOscar]},drop=True).sum('spc_halo')
            elif (VAR[:len('Effective Radiative Forcing|Anthropogenic|F-Gases|')]=='Effective Radiative Forcing|Anthropogenic|F-Gases|')  or  (VAR[:len('Effective Radiative Forcing|Anthropogenic|Montreal Gases|')]=='Effective Radiative Forcing|Anthropogenic|Montreal Gases|') \
                or (VAR[:len('Radiative Forcing|Anthropogenic|F-Gases|')]=='Radiative Forcing|Anthropogenic|F-Gases|')  or  (VAR[:len('Radiative Forcing|Anthropogenic|Montreal Gases|')]=='Radiative Forcing|Anthropogenic|Montreal Gases|'):
                val = out_tmp['RF_Xhalo'].sel({'spc_halo':dico_spc_halo[str.split(VAR,'|')[-1]]},drop=True)
            elif VAR in['Effective Radiative Forcing|Anthropogenic|Stratospheric Ozone','Radiative Forcing|Anthropogenic|Stratospheric Ozone']:
                val = out_tmp['RF_O3s'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone','Radiative Forcing|Anthropogenic|Tropospheric Ozone']:
                val = out_tmp['RF_O3t'].copy(deep=True)
            elif VAR in ['Effective Radiative Forcing|Anthropogenic|Other','Radiative Forcing|Anthropogenic|Other']:
                val = out_tmp['RF_H2Os'].copy(deep=True)
            elif VAR == 'Effective Radiative Forcing|Natural':
                val = out_tmp['RF_solar'] + Par.w_warm_volc * out_tmp['RF_volc']
            elif VAR == 'Radiative Forcing|Natural':
                val = out_tmp['RF_solar'] + out_tmp['RF_volc']
            elif VAR in ['Effective Radiative Forcing|Natural|Solar','Radiative Forcing|Natural|Solar']:
                val = out_tmp['RF_solar'].copy(deep=True)
            elif VAR == 'Effective Radiative Forcing|Natural|Volcanic':
                val = Par.w_warm_volc * out_tmp['RF_volc']
            elif VAR == 'Radiative Forcing|Natural|Volcanic':
                val = out_tmp['RF_volc'].copy(deep=True)
            else:
                raise Exception("Variable not prepared: "+VAR)



        ## EMISSIONS RCMIP
        elif VAR[:len('Emissions|')] == 'Emissions|':
            if VAR == 'Emissions|CO2':
                val = (out_tmp['Eff'] - out_tmp['D_NBP_bk'].sum(('bio_from','bio_to'))) * 1.e3 * 44/12.
            elif VAR == 'Emissions|CO2|MAGICC AFOLU':
                val = - out_tmp['D_NBP_bk'].sum(('bio_from','bio_to')) * 1.e3 * 44/12.
            elif VAR == 'Emissions|CO2|MAGICC Fossil and Industrial':
                val = out_tmp['Eff'] * 1.e3 * 44/12.
            elif VAR == 'Emissions|CH4':
                val = (out_tmp['E_CH4'] + out_tmp['D_Ebb_ant'].sel({'spc_bb':'CH4'}, drop=True).sum('bio_land')) * 16/12.
            elif VAR == 'Emissions|CH4|Other':
                val = out_tmp['D_Ebb_ant'].sel({'spc_bb':'CH4'}, drop=True).sum(('bio_land')) * 16/12.
            elif VAR in ['Emissions|'+cp for cp in ['BC','CO','N2O','NH3','NOx','OC','Sulfur','VOC']]:
                var = str.split(VAR,'|')[1]
                val = (out_tmp['E_'+{'BC':'BC','CO':'CO','N2O':'N2O','NH3':'NH3','NOx':'NOX','OC':'OC','Sulfur':'SO2','VOC':'VOC'}[var]] + out_tmp['D_Ebb'].sel({'spc_bb':{'BC':'BC','CO':'CO','N2O':'N2O','NH3':'NH3','NOx':'NOX','OC':'OC','Sulfur':'SO2','VOC':'VOC'}[var]}, drop=True).sum('bio_land'))
                val *= {'BC':1,'CO':28/12.,'N2O':44./28.,'NH3':17./14,'NOx':46./14,'OC':1.,'Sulfur':64/32.,'VOC':1.}[var]
            elif VAR in ['Emissions|'+cp+'|Other' for cp in ['BC','CO','N2O','NH3','NOx','OC','Sulfur','VOC']]:
                var = str.split(VAR,'|')[1]
                val = out_tmp['D_Ebb'].sel({'spc_bb':{'BC':'BC','CO':'CO','N2O':'N2O','NH3':'NH3','NOx':'NOX','OC':'OC','Sulfur':'SO2','VOC':'VOC'}[var]}, drop=True).sum(('bio_land'))
                val *= {'BC':1,'CO':28/12.,'N2O':44./28.,'NH3':17./14,'NOx':46./14,'OC':1.,'Sulfur':64/32.,'VOC':1.}[var]
            elif ( (VAR[:len('Emissions|F-Gases|')]=='Emissions|F-Gases|')  or  (VAR[:len('Emissions|Montreal Gases|')]=='Emissions|Montreal Gases|') ) \
                and (VAR not in ['Emissions|F-Gases','Emissions|F-Gases|HFC','Emissions|F-Gases|PFC','Emissions|Montreal Gases','Emissions|Montreal Gases|CFC']):
                var = str.split(VAR,'|')[-1]
                val = out_tmp['E_Xhalo'].sel({'spc_halo':dico_spc_halo[var]}, drop=True)
            else:
                raise Exception("Variable Emissions RCMIP not prepared: "+VAR)

        ## OTHER VARIABLES RCMIP
        elif VAR == 'Net Primary Productivity':
            val = ( (Par.npp_0+out_tmp['D_npp']) * (Par.Aland_0 + out_tmp['D_Aland']) ).sum(('bio_land'))
            if option_DiffControl:
                val -= (Par.npp_0*Par.Aland_0).sum(('bio_land'))
            val += out_tmp['D_NPP_bk'] ## actually 0.
            val *= 1.e3 * 44/12.
        elif VAR == 'Airborne Fraction|CO2': 
            ## fraction of the CO2 emitted that remains in the atmosphere, taking into account permafrost CO2 emissions, but not oxidation of CH4 into CO2 (not CO2 emissions)
            val = Par.a_CO2 * (out_tmp['D_CO2']-out_tmp['D_CO2'].isel(year=0))  /  (out_tmp['Eff'].sum('reg_land') + out_tmp['D_Eluc'] ).cumsum('year')
            ## using atmospheric CO2 instead of the following definition, that relies on compatible emissions
            # val = (out_tmp['Eff'].sum('reg_land') + out_tmp['D_Eluc'] + out_tmp['D_Epf_CO2'].sum('reg_pf') - out_tmp['D_Fland'] - out_tmp['D_Focean']).cumsum('year')
            # val /= (out_tmp['Eff'].sum('reg_land') + out_tmp['D_Eluc'] ).cumsum('year')
        elif VAR == 'Effective Climate Sensitivity':
            # a_conv = 3600*24*365.25 / 1E21 # from {W yr} to {ZJ}
            # A_Earth = 510072E9 # m2
            # val = out_tmp['D_Tg'] * OSCAR['RF_CO2'](Var=xr.Dataset({'D_CO2':2.*(Par.CO2_0+out_tmp.D_CO2.isel(year=0))-Par.CO2_0}),Par=Par)
            # val /= out_tmp['RF'] - (a_conv * A_Earth * (out_tmp['RF'] - out_tmp['D_Tg'] / Par.lambda_0))
            val = xr.full_like( other=out_tmp['D_Tg'] , fill_value=Par.lambda_0 * OSCAR['RF_CO2'](Var=xr.Dataset({'D_CO2':2.*(Par.CO2_0+out_tmp.D_CO2.isel(year=0))-Par.CO2_0}),Par=Par) )
        elif VAR == 'Effective Climate Feedback':
            # a_conv = 3600*24*365.25 / 1E21 # from {W yr} to {ZJ}
            # A_Earth = 510072E9 # m2
            # val = (out_tmp['RF'] - (a_conv * A_Earth * (out_tmp['RF'] - out_tmp['D_Tg'] / Par.lambda_0))) / out_tmp['D_Tg']
            val = xr.full_like( other=out_tmp['D_Tg'] , fill_value=1/Par.lambda_0 )
        elif VAR == 'Heat Uptake':
            a_conv = 3600*24*365.25 / 1E21 # from {W yr} to {ZJ}
            A_Earth = 510072E9 # m2
            val = a_conv * A_Earth * (out_tmp['RF'] - out_tmp['D_Tg'] / Par.lambda_0)
        elif VAR == 'Heat Uptake|Ocean':
            a_conv = 3600*24*365.25 / 1E21 # from {W yr} to {ZJ}
            A_Earth = 510072E9 # m2
            val = a_conv * A_Earth * Par.p_ohc * (out_tmp['RF'] - out_tmp['D_Tg'] / Par.lambda_0)
        elif VAR == 'Heat Content|Ocean':
            val = out_tmp['D_OHC']
        elif VAR == 'Instantaneous TCRE':
            val = out_tmp['D_Tg'] / ( (out_tmp['Eff'].sum('reg_land') + out_tmp['D_Eluc'] ).cumsum('year') * 1.e3 * 44/12. )
        elif VAR == 'Surface Air Temperature Change':
            if option_SumReg:
                val = out_tmp['D_Tg']
            else:
                val = out_tmp['D_Tl']
                val.loc[{'reg_land':0}] = out_tmp['D_Tg']
        elif VAR == 'Surface Ocean Temperature Change':
            val = out_tmp['D_To']
        elif VAR == 'Cumulative Emissions|CO2':
            val = ( (out_tmp['Eff'] - out_tmp['D_NBP_bk'].sum(('bio_from','bio_to'))) ).cumsum('year') * 1.e3 * 44/12.
        elif VAR == 'Cumulative Emissions|CO2|MAGICC Fossil and Industrial':
            val = out_tmp['Eff'].cumsum('year') * 1.e3 * 44/12.
        elif VAR == 'Cumulative Emissions|CO2|MAGICC AFOLU':
            val = - out_tmp['D_NBP_bk'].sum(('bio_from','bio_to')).cumsum('year') * 1.e3 * 44/12.
        elif VAR == 'Cumulative Emissions|Other':
            val = xr.zeros_like( out_tmp.year )
        elif VAR in ['Atmospheric Lifetime|CH4','Atmospheric Lifetime|N2O']:
            val = out_tmp['tau_'+str.split(VAR,'|')[1]]
        else:
            raise Exception("Variable RCMIP not prepared: "+VAR)

    ## additional variables from OSCAR
    elif VAR in ['RF_CO2','RF_CH4','RF_N2O','RF_halo','RF_nonCO2','RF_H2Os','RF_O3s','RF_strat','RF_SO4','RF_POA','RF_NO3','RF_SOA','RF_scatter','RF_absorb','RF_cloud','RF_AERtot','RF_O3t','RF_slcf','RF_BCsnow','RF_lcc','RF_alb','D_Fland' , 'RF_solar' , 'RF_volc']:
        val = out_tmp[ dico_var[VAR][0] ].copy(deep=True)


    ## ERROR
    else:
        raise Exception("Variable not prepared: "+VAR)

    ## summing on regions if required
    if option_SumReg  and  ('reg_land' in val.dims)  and  (VAR not in ['tas','pr']): val = val.sum('reg_land')

    return val


def func_add_var( OUT,PAR,FOR, list_var_required , type_OSCAR='OSCAR' ):
    ##-----
    ## This function simply adds variables to the outputs of OSCAR using the model
    ## Using this function instead of the line:
    ##          'out_tmp[var] = OSCAR[var](out_tmp, Par, for_tmp.update(out_tmp),recursive=True)'
    ## inside 'gather_XP' because it is ~10 times faster for variables depending on many such as 'RF', and keep track of the computed ones for the next variables.
    ## However, this function has higher RAM requirements. ----> check if ok when using _BK variables.
    ##-----
    option_LessRAM = True  ## if True, test-->149.2s, for 10.1-x Go  ; if False, test--> 93.5s, for 8.1-x Go  (..?)
    if option_LessRAM:
        for var in list_var_required:
            if var in FOR:
                OUT[var] = FOR[var]
            else:
                if type_OSCAR=='OSCAR_landC':
                    OUT[var] = OSCAR_landC[var](OUT, PAR, FOR.update(OUT),recursive=True)
                elif type_OSCAR=='OSCAR':
                    # print(var)
                    OUT[var] = OSCAR[var](OUT, PAR, FOR.update(OUT),recursive=True)
    else:
        print("WARNING!!!! NOTICED THAT THIS MODE LEADS TO SPURIOUS INITIAL VALUES!!!")
        var,test_to_search,list_backvar = list_var_required[-1],True,list_var_required
        while test_to_search:
            try:
                if var not in OUT:
                    if type_OSCAR=='OSCAR_landC':
                        if var in FOR:
                            OUT[var] = FOR[var]
                        elif var not in OSCAR_landC:
                            raise Exception("This variable is not in "+type_OSCAR+": "+var)
                        else:
                            OUT[var] = OSCAR_landC[var](OUT, PAR, FOR)
                    elif type_OSCAR=='OSCAR':
                        if var in FOR:
                            OUT[var] = FOR[var]
                        elif var not in OSCAR:
                            raise Exception("This variable is not in "+type_OSCAR+": "+var)
                        else:
                            OUT[var] = OSCAR[var](OUT, PAR, FOR)
                    else:
                        raise Exception('Fill in the correct OSCAR.')
                if len(list_backvar)>0:
                    var = list_backvar[-1]
                    list_backvar.remove(var)
                else:
                    test_to_search = False
            except KeyError as err:
                list_backvar.append( var )
                var = err.args[0]
    return OUT


def eval_compat_emi( var_comp, OUT,PAR,FOR,type_OSCAR='OSCAR' ):
    ##-----
    ## This function calculates the compatible emissions of GhG for concentrations-driven experiments.
    ## Mostly used to fill in RCMIP variables such as Airborne Fraction|CO2 or TCRE for concentrations-driven experiments.
    ## A specific warning will be issued for these variables.
    ##-----
    if 'CO2' in var_comp:
        if True:
            ## major edit 29 July 2020. Compatible emissions of CO2 need to be calculated after calculation of compatible emissions of CH4 for the oxidation of CH4
            ## compatible emissions CH4
            OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Ewet','D_Ebb','D_Epf_CH4','D_Fsink_CH4'] , type_OSCAR=type_OSCAR)
            val = - OUT.D_Ewet.sum('reg_land',min_count=1) - OUT.D_Ebb.sel({'spc_bb':'CH4'}).sum('bio_land',min_count=1).sum('reg_land',min_count=1)  -  OUT.D_Epf_CH4.sum('reg_pf',min_count=1)  +  OUT.D_Fsink_CH4
            E_CH4 = PAR.a_CH4 * OUT.D_CH4.diff(dim='year') + 0.5*( val + val.shift(year=1) ).sel(year=np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1))
            Foxi = 1E-3 * (PAR.p_CH4geo * E_CH4 + OUT['D_Epf_CH4'].sum('reg_pf')  - OUT['D_CH4'].dropna('year').differentiate('year'))
            ## compatible emissions CO2
            OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Eluc','D_Epf_CO2','D_Fland','D_Focean'] , type_OSCAR=type_OSCAR)
            val = - OUT.D_Eluc  -  OUT.D_Epf_CO2.sum('reg_pf',min_count=1)  +  OUT.D_Fland  +  OUT.D_Focean  -  Foxi
            FOR['Eff'] = xr.DataArray( np.full(fill_value=0. , shape=(FOR.year.size,FOR.reg_land.size,OUT.config.size)), dims=('year','reg_land','config') )
            FOR['Eff'].loc[{'reg_land':0,'year':np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1)}] = PAR.a_CO2 * OUT.D_CO2.diff(dim='year') + 0.5*(val + val.shift(year=1)).sel(year=np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1))
        else:
            OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
            val = - OUT.D_Eluc  -  OUT.D_Epf_CO2.sum('reg_pf',min_count=1)  +  OUT.D_Fland  +  OUT.D_Focean  -  OUT.D_Foxi_CH4
            FOR['Eff'] = xr.DataArray( np.full(fill_value=0. , shape=(FOR.year.size,FOR.reg_land.size,OUT.config.size)), dims=('year','reg_land','config') )
            FOR['Eff'].loc[{'reg_land':0,'year':np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1)}] = PAR.a_CO2 * OUT.D_CO2.diff(dim='year') + 0.5*(val + val.shift(year=1)).sel(year=np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1))
    if 'N2O' in var_comp:
        OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Ebb','D_Fsink_N2O'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
        val = - OUT.D_Ebb.sel({'spc_bb':'N2O'}).sum('bio_land',min_count=1).sum('reg_land',min_count=1)  +  OUT.D_Fsink_N2O
        FOR['E_N2O'] = xr.DataArray( np.full(fill_value=0. , shape=(FOR.year.size,FOR.reg_land.size,OUT.config.size)), dims=('year','reg_land','config') )
        FOR['E_N2O'].loc[{'reg_land':0,'year':np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1)}] = PAR.a_N2O * OUT.D_N2O.diff(dim='year') + 0.5*( val + val.shift(year=1) ).sel(year=np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1))
    if 'CH4' in var_comp:
        OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Ewet','D_Ebb','D_Epf_CH4','D_Fsink_CH4'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
        val = - OUT.D_Ewet.sum('reg_land',min_count=1) - OUT.D_Ebb.sel({'spc_bb':'CH4'}).sum('bio_land',min_count=1).sum('reg_land',min_count=1)  -  OUT.D_Epf_CH4.sum('reg_pf',min_count=1)  +  OUT.D_Fsink_CH4
        FOR['E_CH4'] = xr.DataArray( np.full(fill_value=0. , shape=(FOR.year.size,FOR.reg_land.size,OUT.config.size)), dims=('year','reg_land','config') )
        FOR['E_CH4'].loc[{'reg_land':0,'year':np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1)}] = PAR.a_CH4 * OUT.D_CH4.diff(dim='year') + 0.5*( val + val.shift(year=1) ).sel(year=np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1))
    if 'Xhalo' in var_comp:
        OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Fsink_Xhalo'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
        val = OUT.D_Fsink_Xhalo
        FOR['E_Xhalo'] = xr.DataArray( np.full(fill_value=0. , shape=(FOR.spc_halo.size,FOR.year.size,FOR.reg_land.size,OUT.config.size)), dims=('spc_halo','year','reg_land','config') )
        FOR['E_Xhalo'].loc[{'reg_land':0,'year':np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1)}] = PAR.a_Xhalo * OUT.D_Xhalo.diff(dim='year')  + 0.5*( val + val.shift(year=1) ).sel(year=np.arange(FOR.year.isel(year=1),FOR.year.isel(year=-1)+1))
    return FOR

if False: ## checking for compatible emissions CO2 (29 July 2020)
    with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
    out_tmp = xr.open_dataset(path_runs+'/'+'ssp585'+'_Out-'+str(setMC)+'.nc')
    for_tmp = xr.open_dataset(path_runs+'/'+'ssp585'+'_For-'+str(setMC)+'.nc')
    aa = eval_compat_emi( dico_compatible_emissions['ssp585'], out_tmp,Par,for_tmp,type_OSCAR='OSCAR' ) ## correction for compatible emissions

    plt.plot( aa.year , aa['Eff'].sel(reg_land=0) * list_noDV['ssp585'][setMC] , color='k',ls='-' )
    plt.plot( aa.year , aa['Eff'].sel(config=0,reg_land=0) * list_noDV['ssp585'][setMC][:,0] , color='k',ls='-' , label="with recursive call for Foxi_CH4" )

    plt.title('Compatible Eff under ssp585, removing diverging runs')
    plt.legend(loc=0)


def gather_XP( xp, weights, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , dico_var , type_OSCAR , option_breach , option_SumReg , option_load_hist , option_JustAddVar):
    ##-----
    ## For an experiment (eg ssp585,ssp585ext), for each set of MC members, this function loads the Xp and its extension if any, AND associated control
    ## Required variables for this MIP are calculated ('func_add_var') using these forcings, outputs and parameters.
    ## Produces the differences to control only for the outputs.
    ## Matching of OSCAR variables to MIP variables ('varOSCAR_to_varMIP') is then executed.
    ## Every set is allocated over a single axis.
    ## option_load_hist:: includes adapted historical run before: for RCMIP, and because of cumulative emission
    ## option_breach:: returns outputs from the date of breach:: specific experiments
    ## option_JustAddVar: if True, will not compute compatible emissions
    ##-----
    ## initializing temporary variable for storing outputs
    TMP_xp = xr.Dataset()
    if threshold_select_ProbableRuns==None:
        TMP_xp.coords['all_config'] = [str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
    else:
        ind = np.where( weights.weights.sel(all_config=[str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]) > weights.weights.median() / threshold_select_ProbableRuns )[0]
        TMP_xp.coords['all_config'] = np.array([str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])])[ind]

    ## LOADING CONTROL
    if xp[0] in dico_Xp_Control.keys():
        ## the experiment is a control. The transformation of variables has to be run in 'experiment only' mode.
        option_DiffControl = False
        xp_c = None
    else:
        ## the experiment is not a control. The transformation of variables has to be run in 'experiment minus control' mode
        option_DiffControl = True
        xp_c = [xp_c for xp_c in dico_Xp_Control if xp[0] in dico_Xp_Control[xp_c]][0]
        ## checking that the control has already been run. Need to be done to accelerate the treatment.
        if os.path.isfile( path_save+'/treated/temporary/'+xp_c+'_TMP_xp_treatment.nc' )==False:
            raise Exception("The control "+xp_c+" need to be run before to accelerate the treatment")
        else:
            TMP_ctrl = xr.open_dataset( path_save+'/treated/temporary/'+xp_c+'_TMP_xp_treatment.nc' )

        if option_SumReg and ('reg_land' in TMP_ctrl):TMP_ctrl = TMP_ctrl.sel(reg_land=0)## just taking global region

    if option_breach: max_yr_brch=-np.inf
    ## looping on sets
    for setMC in list_setMC:
        ## index for selction of likely runs
        if threshold_select_ProbableRuns!=None:ind_threshold_weights = [cfg for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config]

        print("Preparing variables for " + " and ".join(xp) + " over set "+str(setMC))
    

        ## LOADING EXPERIMENT
        Par,out_tmp,for_tmp,mask_xp = load_XP_set( setMC, xp, weights, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , dico_var , type_OSCAR , option_breach , option_SumReg , option_load_hist , option_JustAddVar)
        ## preparing the max_yr_brch
        if option_breach: max_yr_brch = int(np.max([max_yr_brch,for_tmp.year_breach.max()]))


        ##-----
        ## ADDING REQUIRED VARIABLES, MATCHING TO MIP VARIABLES, ALLOCATING
        ##-----
        ## Calculating some basis variables, not to recalculate some of them several times
        if (type_OSCAR!='OSCAR_landC') and (option_JustAddVar==False):
            out_tmp = func_add_var(out_tmp, Par, for_tmp , list_var_required=list_VAR_accelerate , type_OSCAR=type_OSCAR)
        ## starting with year to define this coordinate
        if 'year' not in TMP_xp:
            if option_breach:
                TMP_xp.coords['year'] = np.arange( out_tmp.year.size )
                TMP_ctrl.coords['year'] = np.arange( TMP_ctrl.year.size )
            else:
                TMP_xp.coords['year'] = out_tmp.year
        if option_SumReg==False  and  'reg_land' not in TMP_xp: TMP_xp.coords['reg_land'] = np.arange( out_tmp.reg_land.size )## /!\ WARNING: THE REGION 'UNKNOWN' OF OSCAR WILL BE CHANGED TO A REGION 'WORLD'
        ## check for year_breach
        if ('year_breach' in list_variables_required) and (xp[0] not in ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC']):
            list_variables_required.remove('year_breach')

        ## LOOPING ON CMIP6 VARIABLES
        for VAR in list_variables_required:
            ## correction of the dictionnary of required variables ONLY for landC experiments
            if (xp[0][:len('land-')] == 'land-')  and  (VAR == 'pr'):
                tmp = {VAR:['D_Pl']}
            elif (xp[0][:len('land-')] == 'land-')  and  (VAR == 'tas'):
                tmp = {VAR:['D_Tl']}
            else:
                tmp = {VAR:dico_var[VAR]}
            #tmp = dico_var
            ## calculating variables required
            do_the_transform = True
            for var in tmp[VAR]:
                if var not in out_tmp:
                    if (type_OSCAR=='OSCAR_landC') and (var in OSCAR_landC._processes)   or   (type_OSCAR=='OSCAR'):
                        out_tmp = func_add_var(out_tmp, Par, for_tmp , list_var_required=[var] , type_OSCAR=type_OSCAR)
                    else:
                        do_the_transform = False ## dont do it because variable cant be calculated (eg ocean variables with OSCAR_landC)
            if do_the_transform:
                ## calculating CMIP6 variable
                # val = varOSCAR_to_varMIP( VAR, out_tmp, Par , for_tmp , dico_var=tmp , option_SumReg=option_SumReg , option_DiffControl=option_DiffControl , type_OSCAR=type_OSCAR)
                ## strong difference!!
                val = varOSCAR_to_varMIP( VAR, out_tmp, Par , for_tmp , dico_var=tmp , option_SumReg=option_SumReg , option_DiffControl=False , type_OSCAR=type_OSCAR)
                if option_breach and ('config' in val.dims):
                    val2 = xr.DataArray( np.full(fill_value=np.nan,shape=(val.year.size,val.config.size)) , dims=('year','config') , coords=[np.arange(out_tmp.year.size),val.config] )
                    for cfg in val2.config:
                        if ~np.isnan(for_tmp.year_breach.isel(year=0,config=cfg)):
                            val2.loc[{'config':cfg,'year':np.arange(for_tmp.year.isel(year=-1)-for_tmp.year_breach.isel(year=0,config=cfg)+1)}] =  val.sel(year=np.arange(for_tmp.year_breach.isel(year=0,config=cfg),for_tmp.year.isel(year=-1)+1),config=cfg).values
                            val2.loc[{'config':cfg,'year':np.arange(for_tmp.year.isel(year=-1)-for_tmp.year_breach.isel(year=0,config=cfg)+1)}] -= val.sel(year=for_tmp.year_breach.isel(year=0,config=cfg),config=cfg).values
                    val = val2
                    del val2
                ## avoiding problem of transposition
                # val = val.transpose( tuple(['year'] + ('reg_land' in val.dims)*['reg_land'] + ('config' in val.dims)*['config']) ) ## not working because need hashable
                if ('reg_land' in val.dims):
                    if ('config' in val.dims):
                        val = val.transpose( 'year','reg_land','config' )
                    else:
                        val = val.transpose( 'year','reg_land' )
                else:
                    if ('config' in val.dims):
                        val = val.transpose( 'year','config' )
                ## correction of the region 0: UNKNOWN  TO  WORLD
                if ('reg_land' in val.dims):
                    if  (VAR not in ['tas','pr','Surface Air Temperature Change']):val.loc[{'reg_land':0}] = val.sum('reg_land')

                ## defining in TMP_xp
                if VAR not in TMP_xp:
                    if (option_SumReg==False and 'reg_land' in val.dims):
                        TMP_xp[VAR] = xr.DataArray(    np.full(fill_value=np.nan,shape=[TMP_xp.year.size] + [TMP_xp.reg_land.size] + ('config' in val.dims)*[TMP_xp.all_config.size] ) ,\
                                                    dims=['year'] + ['reg_land'] + ('config' in val.dims)*['all_config'] )
                    else:
                        TMP_xp[VAR] = xr.DataArray(    np.full(fill_value=np.nan,shape=[TMP_xp.year.size] + ('config' in val.dims)*[TMP_xp.all_config.size] ) ,\
                                                    dims=['year'] + ('config' in val.dims)*['all_config'] )
                if ('config' not in val.dims)  and (VAR != 'year'): ## case for forcings: does not depend on config or setMC, allocated directly
                    TMP_xp[VAR].loc[{ 'year':TMP_xp.year }] = val.values ## need to force years, given the situation of breached outputs
                ## allocating
                if ('config' in val.dims):
                    if option_SumReg==False and 'reg_land' in val.dims:
                        TMP_xp[VAR].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config] }] = val.values * np.repeat(mask_xp[:,np.newaxis,:],6,axis=1)                        
                    else:
                        TMP_xp[VAR].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config] }] = val.values * mask_xp
        ## end loop on variables
        ## cleaning
        out_tmp.close()
        for_tmp.close()
        del out_tmp,for_tmp,Par
        ##-----
        ##-----


    ##-----
    ## CORRECTIONS CONTROL
    ##-----
    ## differences to control
    if option_DiffControl:
        if len(TMP_xp.year)  >  len(TMP_ctrl.year):## the Control is too short (1000 years, while some experiments last more than that)
            yr_xp0_start, yr_pi_cut, yr_xp0_end = TMP_xp.year.values[0], TMP_ctrl.year.values[-1], TMP_xp.year.values[-1]

            for VAR in TMP_xp:
                # TMP_xp[VAR].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}]   =  TMP_xp[VAR].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}] - TMP_ctrl[VAR].sel(year=np.arange(yr_xp0_start,yr_pi_cut+1))
                TMP_xp[VAR].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}]   -=  TMP_ctrl[VAR].sel(year=np.arange(yr_xp0_start,yr_pi_cut+1))
                ## using average of the last 10 years of control
                # TMP_xp[VAR].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}]   =  TMP_xp[VAR].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}] - TMP_ctrl[VAR].sel(year=np.arange(yr_pi_cut-10+1,yr_pi_cut+1)).mean('year')
                TMP_xp[VAR].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}]   -=  TMP_ctrl[VAR].sel(year=np.arange(yr_pi_cut-10+1,yr_pi_cut+1)).mean('year')
        else:
            for VAR in TMP_xp:TMP_xp[VAR] = TMP_xp[VAR] - TMP_ctrl[VAR].sel(year=TMP_xp.year)
    ##-----
    ##-----

    ## end loop on sets
    ## cutting breached dataset, to avoid the period with only NaNs
    if option_breach:TMP_xp = TMP_xp.sel(year=np.arange( TMP_xp.year.size-(max_yr_brch-1850)-1 ))## all breach experiment start in 1850. has a check along.

    ## In case this function is run with a control as xp, an intermediary of TMP_xp need to be saved
    if xp[0] in dico_Xp_Control.keys(): TMP_xp.to_netcdf(path_save+'/treated/temporary/'+xp[0]+'_TMP_xp_treatment.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP_xp})
    
    return TMP_xp,xp_c


def load_XP_set( setMC, xp, weights, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , dico_var , type_OSCAR , option_breach , option_SumReg , option_load_hist , option_JustAddVar ):
    ## index for selction of likely runs
    if threshold_select_ProbableRuns!=None:ind_threshold_weights = [cfg for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config]

    ## loading everything related to the experiment
    out_tmp,for_tmp,mask_xp = [],[],[]

    ## PARAMETERS
    with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
    ## immediately correcting Par for -bgc and -rad experiments
    if xp[0] in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc']:
        with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
        Par['D_CO2_rad'] = for_runs_hist.D_CO2.sel(year=1850)
    elif xp[0] in ['1pctCO2-rad']:
        with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
        Par['D_CO2_bgc'] = for_runs_hist.D_CO2.sel(year=1850)
    ## eventually selection of relevant runs
    if threshold_select_ProbableRuns!=None:Par = Par.sel(config=ind_threshold_weights)
        
    ## LOADING ASSOCIATED HISTORICAL, if required  ===> required for cumulative emissions for scenarios!!!!
    if option_load_hist and (dico_experiments_before[xp[0]]!=None):
        if os.path.isfile( path_extra  + '/' + dico_experiments_before[xp[0]] + '_Out2-' + str(setMC) + '.nc' ):
            out_tmp.append( xr.merge( [xr.open_dataset( path_runs + '/' + dico_experiments_before[xp[0]]+ '_Out-' + str(setMC) + '.nc' ) , xr.open_dataset( path_extra  + '/' + dico_experiments_before[xp[0]] + '_Out2-' + str(setMC) + '.nc' )] ) )
        else:
            out_tmp.append( xr.open_dataset( path_runs + '/' + dico_experiments_before[xp[0]]+ '_Out-' + str(setMC) + '.nc' ) )
        for_tmp.append( xr.open_dataset(path_runs+'/'+dico_experiments_before[xp[0]]+'_For-'+str(setMC)+'.nc') )
        ## eventually selection of relevant runs
        if threshold_select_ProbableRuns!=None:out_tmp[-1] , for_tmp[-1] = out_tmp[-1].sel(config=ind_threshold_weights) , for_tmp[-1].sel(config=ind_threshold_weights)
        ## correction of parameter 'Aland_0':: some experiments (blocks 'CMIP5' and 'LAND') have different Aland_0. It has been saved in the forcings, not in the parameters file.
        if 'Aland_0' in for_tmp[-1]:
            Par['Aland_0'] = for_tmp[-1]['Aland_0']
        if option_JustAddVar==False:for_tmp[-1] = eval_compat_emi( dico_compatible_emissions[dico_experiments_before[xp[0]]], out_tmp[-1],Par,for_tmp[-1],type_OSCAR=type_OSCAR ) ## correction for compatible emissions
        mask_xp.append( list_noDV[dico_experiments_before[xp[0]]][setMC] )
        ## eventually selection of relevant runs
        if threshold_select_ProbableRuns!=None:mask_xp[-1] = mask_xp[-1][:,ind_threshold_weights]
        ## removing first value
        # for_tmp[-1] = for_tmp[-1].isel(year=np.arange(1,for_tmp[-1].year.size))
        # out_tmp[-1] = out_tmp[-1].isel(year=np.arange(1,out_tmp[-1].year.size))
        # mask_xp[-1] = mask_xp[-1][1:]
        ## eventually cutting the exceeding years of the experiment
        if dico_experiments_before[xp[0]] in list_Xp_cut100:
            for_tmp[-1] = for_tmp[-1].isel(year=np.arange(0,for_tmp[-1].year.size-100))
            out_tmp[-1] = out_tmp[-1].isel(year=np.arange(0,out_tmp[-1].year.size-100))
            mask_xp[-1] = mask_xp[-1][:-100,:]

    ## LOADING REQUIRED EXPERIMENT
    if os.path.isfile( path_extra  + '/' + xp[0] + '_Out2-' + str(setMC) + '.nc' ):
        out_tmp.append( xr.merge( [xr.open_dataset( path_runs + '/' + xp[0]+ '_Out-' + str(setMC) + '.nc' ) , xr.open_dataset( path_extra  + '/' + xp[0] + '_Out2-' + str(setMC) + '.nc' )] ) )
    else:
        out_tmp.append( xr.open_dataset( path_runs + '/' + xp[0]+ '_Out-' + str(setMC) + '.nc' ) )
    for_tmp.append( xr.open_dataset(path_runs+'/'+xp[0]+'_For-'+str(setMC)+'.nc') )
    ## eventually selection of relevant runs
    if threshold_select_ProbableRuns!=None:out_tmp[-1] , for_tmp[-1] = out_tmp[-1].sel(config=ind_threshold_weights) , for_tmp[-1].sel(config=ind_threshold_weights)
    ## correction of parameter 'Aland_0':: some experiments (blocks 'CMIP5' and 'LAND') have different Aland_0. It has been saved in the forcings, not in the parameters file.
    if 'Aland_0' in for_tmp[-1]:Par['Aland_0'] = for_tmp[-1]['Aland_0']
    ## correction for compatible emissions
    if option_JustAddVar==False:for_tmp[-1] = eval_compat_emi( dico_compatible_emissions[xp[0]], out_tmp[-1],Par,for_tmp[-1],type_OSCAR=type_OSCAR )
    if option_breach:print("breaches in "+str(setMC)+": "+str(int(for_tmp[-1].year_breach.min()))+'-'+str(int(for_tmp[-1].year_breach.max())))
    mask_xp.append( list_noDV[xp[0]][setMC] )
    ## eventually selection of relevant runs
    if threshold_select_ProbableRuns!=None:mask_xp[-1] = mask_xp[-1][:,ind_threshold_weights]
    ## immediately removing first year of outputs: used in OSCAR for initilization, but not to provide (eg. 2014 in ssp585)
    if dico_experiments_before[xp[0]] != None:
        for_tmp[-1] = for_tmp[-1].isel(year=np.arange(1,for_tmp[-1].year.size))
        out_tmp[-1] = out_tmp[-1].isel(year=np.arange(1,out_tmp[-1].year.size))
        mask_xp[-1] = mask_xp[-1][1:]
    ## eventually cutting the exceeding years of the experiment
    if xp[0] in list_Xp_cut100:
        for_tmp[-1] = for_tmp[-1].isel(year=np.arange(0,for_tmp[-1].year.size-100))
        out_tmp[-1] = out_tmp[-1].isel(year=np.arange(0,out_tmp[-1].year.size-100))
        mask_xp[-1] = mask_xp[-1][:-100,:]

    ## LOADING EXTENSION IF REQUIRED
    if len(xp)==2:
        for_tmp.append( xr.open_dataset(path_runs+'/'+xp[1]+'_For-'+str(setMC)+'.nc') )
        if os.path.isfile( path_extra  + '/' + xp[1] + '_Out2-' + str(setMC) + '.nc' ):
            out_tmp.append( xr.merge( [xr.open_dataset( path_runs + '/' + xp[1]+ '_Out-' + str(setMC) + '.nc' ) , xr.open_dataset( path_extra  + '/' + xp[1] + '_Out2-' + str(setMC) + '.nc' )] ) )
        else:
            out_tmp.append( xr.open_dataset( path_runs + '/' + xp[1]+ '_Out-' + str(setMC) + '.nc' ) )
        mask_xp.append( list_noDV[xp[1]][setMC] )
        ## eventually selection of relevant runs
        if threshold_select_ProbableRuns!=None:out_tmp[-1] , for_tmp[-1] , mask_xp[-1] = out_tmp[-1].sel(config=ind_threshold_weights) , for_tmp[-1].sel(config=ind_threshold_weights) , mask_xp[-1][:,ind_threshold_weights]
        if option_JustAddVar==False:for_tmp[-1] = eval_compat_emi( dico_compatible_emissions[xp[1]], out_tmp[-1],Par,for_tmp[-1],type_OSCAR=type_OSCAR ) ## correction for compatible emissions
        ## immediately removing first year of outputs: used in OSCAR for initilization, but not to provide (eg. 2100 in ssp585ext)
        if dico_experiments_before[xp[1]] != None:
            for_tmp[-1] = for_tmp[-1].isel(year=np.arange(1,for_tmp[-1].year.size))
            out_tmp[-1] = out_tmp[-1].isel(year=np.arange(1,out_tmp[-1].year.size))
            mask_xp[-1] = mask_xp[-1][1:]
        else:
            raise Exception("Error in preceding experiments")
        ## eventually cutting the exceeding years of the experiment
        if xp[1] in list_Xp_cut100:
            for_tmp[-1] = for_tmp[-1].isel(year=np.arange(0,for_tmp[-1].year.size-100))
            out_tmp[-1] = out_tmp[-1].isel(year=np.arange(0,out_tmp[-1].year.size-100))
            mask_xp[-1] = mask_xp[-1][:-100,:]

    ## removing some useless coordinates            
    for ii in np.arange(len(for_tmp)):
        for cc in ['data_RF_solar','data_RF_volc','scen']:
            if cc in for_tmp[ii]:
                for_tmp[ii] = for_tmp[ii].drop(cc)
            if cc in out_tmp[ii]:
                out_tmp[ii] = out_tmp[ii].drop(cc)

    ## Concatening forcings: must be careful with calculation of CO2 (compatible) emissions
    if option_load_hist and (dico_experiments_before[xp[0]]!=None): ## special case: presence of a historical, with different requirement of compatible emissions, thus coordinates and variables in the list
        ## removing irrelevant variables
        for ii in np.arange(len(out_tmp)):
            for var in ['D_Tg','D_Td','D_cveg','D_csoil1','D_Cfroz','D_pthaw','D_Chwp','D_Csoil2_bk','D_csoil2','D_CH4_lag','D_Cveg_bk','D_Aland','D_OHC','D_Xhalo_lag','D_Cthaw','D_Pg','D_Csoil1_bk','D_N2O_lag','D_Cosurf','D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4','D_Ebb','D_Fsink_N2O','D_Ewet','D_Epf_CH4','D_Fsink_CH4'] + ['box_osurf','reg_pf','spc_bb','box_thaw','box_hwp']:
                if var in for_tmp[ii]:# variables added because of compatible emissions
                    for_tmp[ii] = for_tmp[ii].drop( var )
                if ('D_CO2' in for_tmp[ii])  and  (xp[0][:len('esm-')]=='esm-'):## add D_CO2 to for_tmp[ii] when calculate compatible emissions. Problem with esm-ssp370-lowNTCF-gidden.
                    for_tmp[ii] = for_tmp[ii].drop( 'D_CO2' )
    for_tmp0 = xr.concat( for_tmp , dim='year' )# no problem anymore
    for ii in np.arange(len(for_tmp)):
        for_tmp[ii].close()
    del for_tmp

    ## correction: some emissions are not perfectly defined (esm-rcp26 after 2100), with NaN out of reg_land=0
    # for_tmp0 = for_tmp0.fillna( 0. )

    ## Concatening outputs: some variables have been added to outputs while computing compatible emissions, but not the sames. (+could not use the 'data_vars' option of 'xr.concat' to directly concatenate datasets)
    out_tmp0 = xr.Dataset()
    for var in (list(OSCAR_landC.var_prog))*(type_OSCAR=='OSCAR_landC') + (list(OSCAR.var_prog))*(type_OSCAR=='OSCAR'):
        out_tmp0[var] = xr.concat( [out_tmp[ii][var] for ii in np.arange(len(out_tmp))] , dim='year' )
    for ii in np.arange(len(out_tmp)):
        out_tmp[ii].close()
    del out_tmp

    ## Concatening masks for divergence
    mask_xp = np.concatenate( mask_xp )

    if option_breach and (for_tmp0.year.isel(year=0)!=1850): raise Exception("not all breach experiment starts in 1850..?")
    ## correction of the mask: need to be shifted if breached experiment
    if option_breach:
        for i_cfg in np.arange(out_tmp0.config.size):
            if ~np.isnan(for_tmp0.year_breach.isel(year=0,config=i_cfg)):
                yy1 = int(for_tmp0.year.isel(year=-1))
                yy2 = int(for_tmp0.year_breach.isel(year=0,config=i_cfg))
                mask_xp[:yy1-yy2+1,i_cfg] = mask_xp[ yy2-1850: ,i_cfg]

    return Par,out_tmp0,for_tmp0,mask_xp



def func_prod_outputs( xp, weights, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings, saved_stat_values , dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=True , option_load_hist=False , option_JustAddVar=False):
    ##-----
    ## Evaluates the required variables (gather_XP) to produce a Dataset with required properties.
    ##-----
    ## Loading outputs
    TMP_xp,xp_c = gather_XP( xp, weights, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , dico_var=dico_var, type_OSCAR=type_OSCAR , option_breach=option_breach , option_SumReg=option_SumReg , option_load_hist=option_load_hist , option_JustAddVar=option_JustAddVar)

    ## producing final file for this experience:
    OUTPUT = xr.Dataset()
    ## STATISTICAL VALUES THAT WILL BE PROVIDED TO THE DIFFERENT MIPs
    OUTPUT.coords['stat_value'] = saved_stat_values
    OUTPUT.coords['year'] = TMP_xp['year']
    if option_SumReg==False  and  'reg_land' not in OUTPUT:
        ## names of regions
        OUTPUT.coords['reg_land'] = np.arange(TMP_xp.reg_land.size)
        if type_OSCAR=='OSCAR':
            OUTPUT.coords['reg_land_long_name'] = ['World','World|R5.2ASIA','World|R5.2LAM','World|R5.2MAF','World|R5.2OECD','World|R5.2REF']
        elif type_OSCAR=='OSCAR_landC':
            OUTPUT.coords['reg_land_long_name'] = ['World|global land','World|R5.2ASIA','World|R5.2LAM','World|R5.2MAF','World|R5.2OECD','World|R5.2REF']
    for VAR in list_variables_required:
        if VAR in TMP_xp.variables:## not necessarily all, for instance because of land- experiments
            if ('all_config' in TMP_xp[VAR].dims):
                # print("Transforming "+VAR+" (config to stat values)")
                ## defining variable in OUTPUT
                if option_SumReg==False  and  'reg_land' in TMP_xp[VAR].dims:
                    OUTPUT[VAR] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUTPUT.year.size,OUTPUT.reg_land.size,OUTPUT.stat_value.size)), dims=('year','reg_land','stat_value') )
                else:
                    OUTPUT[VAR] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUTPUT.year.size,OUTPUT.stat_value.size)), dims=('year','stat_value') )
                ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
                val = np.ma.array(TMP_xp[VAR],mask=np.isnan(TMP_xp[VAR]))
                ww = np.ma.repeat(np.ma.array(weights.weights.sel(all_config=TMP_xp.all_config).values,mask=np.isnan(weights.weights.sel(all_config=TMP_xp.all_config).values))[np.newaxis,:],TMP_xp.year.size,axis=0)
                if option_SumReg==False  and  'reg_land' in TMP_xp[VAR].dims:
                    ww = np.ma.repeat(ww[:,np.newaxis,:],TMP_xp.reg_land.size,axis=1)
                OUTPUT[VAR].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
                OUTPUT[VAR].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUTPUT[VAR].sel(stat_value='mean').values[...,np.newaxis],TMP_xp.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
                for pp in ['median','5pct','95pct']:
                    if pp in saved_stat_values:
                        try:
                            OUTPUT[VAR].loc[{'stat_value':pp}] = quantile(data=val , weights=weights.weights.sel(all_config=TMP_xp.all_config).values , qq={'median':50.,'5pct':5.,'95pct':95.}[pp]/100. )
                        except ValueError:
                            ## NaN slices, several cases: reg_land=0 full of NaN  and/or  variable not defined in t=0 of experiment
                            ## reduced to the last case, by transformation of region 'Unknown' to 'World'
                            # if ('reg_land' in TMP_xp[VAR].dims)  and  np.ma.all(val[:,0,:].mask)  and  np.ma.all(val[0,:,:].mask):
                            #     OUTPUT[VAR].loc[{'stat_value':pp,'year':np.arange(OUTPUT.year[1],OUTPUT.year[-1]+1),'reg_land':np.arange(1,TMP_xp.reg_land.size)}] = quantile(data=val[1:,1:,:] , weights=weights.weights.values , qq={'median':50.,'5pct':5.,'95pct':95.}[pp]/100. )
                            # elif ('reg_land' in TMP_xp[VAR].dims)  and  np.ma.all(val[:,0,:].mask):
                            #     OUTPUT[VAR].loc[{'stat_value':pp,'reg_land':np.arange(1,TMP_xp.reg_land.size)}] = quantile(data=val[:,1:,:] , weights=weights.weights.values , qq={'median':50.,'5pct':5.,'95pct':95.}[pp]/100. )
                            if np.ma.all(val[0,...].mask):# variable not defined in t=0 of experiment
                                OUTPUT[VAR].loc[{'stat_value':pp,'year':np.arange(OUTPUT.year[1],OUTPUT.year[-1]+1)}] = quantile(data=val[1:,...] , weights=weights.weights.sel(all_config=TMP_xp.all_config).values , qq={'median':50.,'5pct':5.,'95pct':95.}[pp]/100. )
                            else:
                                raise Exception("NaN on a full slice: why?")
                if False:
                    plt.plot( TMP_xp.year, TMP_xp[VAR], lw=0.5, color='gray' )
                    for vv in OUTPUT.stat_value.values:
                        if vv != 'std_dev':
                            plt.plot( OUTPUT.year, OUTPUT[VAR].sel(stat_value=vv), lw=2, color='red' if vv=='mean' else 'blue',label=vv )
                    plt.plot( OUTPUT.year, OUTPUT[VAR].sel(stat_value='mean')-OUTPUT[VAR].sel(stat_value='std_dev'), lw=2, color='orange',label='mean-std_dev' )
                    plt.plot( OUTPUT.year, OUTPUT[VAR].sel(stat_value='mean')+OUTPUT[VAR].sel(stat_value='std_dev'), lw=2, color='orange',label='mean+std_dev' )
                    plt.legend(loc=0)
            elif VAR!='year':
                ## Variable as forcing
                OUTPUT[VAR] = TMP_xp[VAR]
            ## adding attributes to the variable
            OUTPUT[VAR].attrs['long_name'] = dico_variables_longnames[VAR]
            OUTPUT[VAR].attrs['unit'] = dico_variables_units[VAR]
            if VAR in dico_variables_warnings:
                OUTPUT[VAR].attrs['warning'] = dico_variables_warnings[VAR]
    ## adding global attributes
    str_xp = ' and '.join(xp)
    if xp_c==None:
        OUTPUT.attrs['info'] = 'Global outputs of OSCARv3 of the experiment '+str_xp+'. Only statistical insights are provided in this file, for a sake of simplicity. The full set of members of each individual experiment can be provided on request.'
    else:
        OUTPUT.attrs['info'] = 'Global outputs of OSCARv3 of the experiment '+str_xp+', minus its control '+xp_c+'. Only statistical insights are provided in this file, for a sake of simplicity. The full set of members of each individual experiment can be provided on request.'
        OUTPUT.attrs['warning'] = 'This outputs are the *differences* of '+str_xp+' to its control '+xp_c+'!'
    wawa = 'Compatible emissions calculated:: over '+xp[0]+': '+', '.join(dico_compatible_emissions[xp[0]])
    if (len(xp)==2):
        wawa += ' ; '+'over '+xp[1]+': '+', '.join(dico_compatible_emissions[xp[1]])
    OUTPUT.attrs['warning_emissions'] = wawa
    OUTPUT.attrs['authors'] = 'Yann Quilcaille, Thomas Gasser'
    OUTPUT.attrs['date'] = datetime.datetime.today().strftime("%d %B %Y")
    if 'region' in OUTPUT:OUTPUT = OUTPUT.drop('region')
    return OUTPUT


def quantile_1D(data, weights, qq): ## function adapted from wquantile, to account for masks ie NaN (pypi.org/project/wquantiles). More efficient than usual weighted percentiles functions.
    """
    Compute the weighted quantile of a 1D numpy array.

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    qq : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.ma.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.ma.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if ((qq > 1.) or (qq < 0.)):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.ma.argsort(data[~data.mask])
    sorted_data = data[~data.mask][ind_sorted]
    sorted_weights = weights[~data.mask][ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.ma.cumsum(sorted_weights)
    Pn = (Sn-0.5*sorted_weights)/np.ma.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(qq, Pn, sorted_data)


def quantile(data, weights, qq): ## function directly from wquantile (pypi.org/project/wquantiles). More efficient than usual weighted percentiles functions.
    """
    Weighted quantile of an array with respect to the last axis.

    Parameters
    ----------
    data : ndarray
        Input array.
    weights : ndarray
        Array with the weights. It must have the same size of the last 
        axis of `data`.
    qq : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile : float
        The output value.
    """
    # TODO: Allow to specify the axis
    nd = data.ndim
    if nd == 0:
        TypeError("data must have at least one dimension")
    elif nd == 1:
        return quantile_1D(data, weights, qq)
    elif nd > 1:
        n = data.shape
        imr = data.reshape((np.prod(n[:-1]), n[-1]))
        result = np.apply_along_axis(quantile_1D, -1, imr, weights, qq)
        return result.reshape(n[:-1])


def func_plot( var , OUT , xp):
    fig = plt.figure(figsize=(15,10))
    if 'stat_value' in OUT[var].dims:
        if 'reg_land' in OUT[var].dims:
            plt.plot( OUT.year , OUT[var].sel(stat_value='mean',reg_land=0) , lw=3 , ls='-', color='k',label='mean' )
            plt.fill_between( OUT.year , OUT[var].sel(stat_value='mean',reg_land=0)-OUT[var].sel(stat_value='std_dev',reg_land=0) , OUT[var].sel(stat_value='mean',reg_land=0)+OUT[var].sel(stat_value='std_dev',reg_land=0) , facecolor='b' , alpha=0.5 ,label='+/- 1 std_dev range' )
            # plt.plot( OUT.year , OUT[var].sel(stat_value='median',reg_land=0) , lw=2 , ls='--', color='k',label='median' )
            # plt.fill_between( OUT.year , OUT[var].sel(stat_value='5pct',reg_land=0) , OUT[var].sel(stat_value='95pct',reg_land=0) , facecolor='r' , alpha=0.25 ,label='90pct range' )
        else:
            plt.plot( OUT.year , OUT[var].sel(stat_value='mean') , lw=3 , ls='-', color='k',label='mean' )
            plt.fill_between( OUT.year , OUT[var].sel(stat_value='mean')-OUT[var].sel(stat_value='std_dev') , OUT[var].sel(stat_value='mean')+OUT[var].sel(stat_value='std_dev') , facecolor='b' , alpha=0.5 ,label='+/- 1 std_dev range' )
            # plt.plot( OUT.year , OUT[var].sel(stat_value='median') , lw=2 , ls='--', color='k',label='median' )
            # plt.fill_between( OUT.year , OUT[var].sel(stat_value='5pct') , OUT[var].sel(stat_value='95pct') , facecolor='r' , alpha=0.25 ,label='90pct range' )
    else:
        if 'reg_land' in OUT[var].dims:
            plt.plot( OUT.year , OUT[var].sel(reg_land=0) , lw=3 , ls='-', color='k',label='single pathway' )
        else:
            plt.plot( OUT.year , OUT[var] , lw=3 , ls='-', color='k',label='single pathway' )
    plt.grid()
    plt.legend(loc=0)
    plt.xlabel( OUT.year.long_name )
    plt.title( xp+', '+var+'\n'+OUT[var].long_name )
    plt.ylabel( '('+OUT[var].unit+')' )
    return fig

#########################
#########################







#########################
## 4.2. CDRMIP
#########################
if option_select_MIP in ['ALL','CDRMIP']:
    list_xp = [ list(dico_experiments_MIPs[ii,0:1+1]) if dico_experiments_MIPs[ii,1]!='' else [dico_experiments_MIPs[ii,0]]   for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-CDRMIP')]=='1')[0] ]
    ## not the 2 about alkalinity
    for xp in list_xp:
        if xp[0] in dico_Xp_Control.keys():
            list_xp.remove(xp)
            list_xp.insert(0,xp)

    saved_stat_values = ['mean','std_dev']

    ## variables that will be provided to CDR-MIP by OSCAR
    list_variables_required = [ dico_variables_MIPs[ii,head_MIPs.index( 'CMIP6 name' )]  for ii in  np.where(dico_variables_MIPs[:,head_MIPs.index('OSCAR-CDRMIP')]=='1')[0] ]

    ## Preparing directories
    for ff in ['treated','treated/CDRMIP','treated/CDRMIP/plots','treated/CDRMIP/intermediary']: os.mkdir(path_save+'/'+ff) if (os.path.isdir( path_save+'/'+ff )==False) else None

    ## Producing and saving single experiments:
    for xp in list_xp:
        print('-'.join(xp)+' ('+str(list_xp.index(xp)+1)+'/'+str(len(list_xp))+')')
        ## producing
        if option_OVERWRITE  or  np.any(xp[0] == np.array([str.split(ff,'_')[0] for ff in os.listdir( path_save+'/treated/CDRMIP/intermediary/' )]))==False:
            OUT = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=True)
            ## saving as netCDF
            OUT.to_netcdf(path_save+'/treated/CDRMIP/intermediary/'+xp[0]+'_'+model+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
        else:
            OUT = xr.open_dataset( path_save+'/treated/CDRMIP/intermediary/'+os.listdir(path_save+'/treated/CDRMIP/intermediary/')[np.where(xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/CDRMIP/intermediary/')]))[0][0]] )

        ## Producing and saving Control
        if xp[0] in dico_Xp_Control.keys():
            pass # this experiment is a control, and has been run without differences to itself
        else:
            xp_c = [ [xp_c for xp_c in dico_Xp_Control if xp[0] in dico_Xp_Control[xp_c]][0] ]
            print(xp_c[0])
            if option_OVERWRITE  or  np.any(xp_c[0] == np.array([str.split(ff,'_')[0] for ff in os.listdir( path_save+'/treated/CDRMIP/intermediary/' )]))==False:
                ## producing using weights used for CMIP6
                OUT_pi = func_prod_outputs( xp_c, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=True)
                ## saving as netCDF (just to keep it somewhere)
                OUT_pi.to_netcdf(path_save+'/treated/CDRMIP/intermediary/'+xp_c[0]+'_OSCARv3.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT_pi})
            else:
                OUT_pi = xr.open_dataset( path_save+'/treated/CDRMIP/intermediary/'+os.listdir(path_save+'/treated/CDRMIP/intermediary/')[np.where(xp_c[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/CDRMIP/intermediary/')]))[0][0]] )

        ## Correction: if xp is a control experiment, for consistency with the addition of the control to final variables (adding control averaged over 50st years), do the same for the control itself.
        if xp[0] in dico_Xp_Control.keys():
            test = OUT.copy(deep=True)
            for VAR in OUT.variables:
                if VAR not in OUT.coords:
                    test[VAR] = xr.full_like( other=OUT[VAR], fill_value=OUT[VAR].isel(year=np.arange(50)).mean('year') )
            OUT = test.copy()
            del test


        ## Producing the files for CDRMIP
        OUT = OUT.sel(stat_value = saved_stat_values)
        for VAR in OUT:
            if xp[0] in dico_Xp_Control.keys():
                pi = 0
            else:
                if 'stat_value' in OUT_pi[VAR].dims:
                    pi = OUT_pi[VAR].sel(stat_value='mean').drop('stat_value').isel(year=np.arange(50)).mean('year')
                else:
                    pi = OUT_pi[VAR].isel(year=np.arange(50)).mean('year')
            # ## Correction of period of control if too short
            # if pi.year[-1] < OUT.year[-1]:
            #     pi2 = xr.Dataset( coords=pi.coords )
            #     pi2.coords['year'] = np.arange(pi.year[-1]+1,OUT.year[-1]+1)
            #     pi2['aa'] = xr.DataArray( data=np.full(fill_value=pi.isel(year=np.arange(-10,-1+1)).mean('year').values,shape=[int(OUT.year[-1]-pi.year[-1])]+list(pi.shape[1:])) , dims=pi.dims )
            #     pi = xr.concat( [pi,pi2['aa']] , dim='year')
            ## Correction for some variables: DO NOT SUM
            if VAR in list_VAR_NoAddingControl:
                pi = 0
            ## adding control
            if 'stat_value' in OUT[VAR].dims:
                OUT[VAR].loc[{'stat_value':'mean'}] = OUT[VAR].loc[{'stat_value':'mean'}] + pi
            else:
                OUT[VAR] = OUT[VAR] + pi
        ## Correcting warning
        str_xp = ' and '.join(xp)
        OUT.attrs['info'] = 'Global outputs of OSCARv3 of the experiment '+str_xp+'. Only statistical insights are provided in this file, for a sake of simplicity. The full set of members of each individual experiment can be provided on request.'
        OUT.attrs['warning'] = ''

        ## Saving
        OUT.to_netcdf(path_save+'/treated/CDRMIP/'+xp[0]+'_'+model+'_'+str(OUT.year.isel(year=0).values)+'-'+str(OUT.year.isel(year=-1).values)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})

        ## plots
        if False:
            for vv in OUT.variables:
                if (vv not in OUT.coords):
                    fig = func_plot( vv , OUT ,xp[0])
                    fig.savefig( path_save+'/treated/CDRMIP/plots/'+xp[0]+'_'+vv)#+'.pdf', format='pdf' )
                    plt.close(fig)
#########################
#########################








#########################
## 4.3. ZECMIP
#########################
if option_select_MIP in ['ALL','ZECMIP']:
    list_xp = [ list(dico_experiments_MIPs[ii,0:1+1]) if dico_experiments_MIPs[ii,1]!='' else [dico_experiments_MIPs[ii,0]]   for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-ZECMIP')]=='1')[0] ]
    for xp in list_xp:
        if xp[0] in dico_Xp_Control.keys():
            list_xp.remove(xp)
            list_xp.insert(0,xp)
    # list_xp += ['piControl']
    # list_xp += ['1pctCO2']

    # list_xp = [ list_xp[k_subsetXP] ] # 0..7+2+1
    # list_xp = [ list_xp[0] , ['piControl'] , ['1pctCO2'] ]

    ## variables that will be provided to ZECMIP by OSCAR
    list_variables_required = [ dico_variables_MIPs[ii,head_MIPs.index( 'CMIP6 name' )]  for ii in  np.where(dico_variables_MIPs[:,head_MIPs.index('OSCAR-ZECMIP')]=='1')[0] ]
    saved_stat_values = ['mean','std_dev']

    ## Preparing directories
    for ff in ['treated','treated/ZECMIP','treated/ZECMIP/plots','treated/ZECMIP/intermediary']+['treated/ZECMIP/'+xp[0] for xp in list_xp]: os.mkdir(path_save+'/'+ff) if (os.path.isdir( path_save+'/'+ff )==False) else None

    # for xp in list_xp:
    # for xp in list_xp[2:4+1]:
    for xp in list_xp[5:7+1]:
        print('-'.join(xp)+' ('+str(list_xp.index(xp)+1)+'/'+str(len(list_xp))+')')
        ## BREACH FILES
        if xp[0] in ['esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC','esm-1pct-brch-750PgC']:
            print('breached')
            if option_OVERWRITE  or  np.any( (xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'== np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) )==False:
                ## producing since breach
                OUT_br = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values , dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=True , option_SumReg=True)
                ## saving as netCDF
                # OUT.to_netcdf(path_save+'/treated/ZECMIP/intermediary/'+xp[0]+'/ALL-VARIABLES_'+xp[0]+'_'+model+'_'+str(OUT.year.isel(year=0).values)+'-'+str(OUT.year.isel(year=-1).values)+'_from-breach.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
                OUT_br.to_netcdf(path_save+'/treated/ZECMIP/intermediary/'+xp[0]+'_'+model+'_from-breach.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT_br})
            else:
                OUT_br = xr.open_dataset( path_save+'/treated/ZECMIP/intermediary/'+os.listdir(path_save+'/treated/ZECMIP/intermediary/')[np.where( (xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'== np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) )[0][0]] )

        ## REGULAR FILES
        if option_OVERWRITE  or  np.any( (xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'!= np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) )==False:
            ## producing
            OUT = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=True)
            ## saving as netCDF
            #OUT.to_netcdf(path_save+'/treated/ZECMIP/intermediary/'+xp[0]+'/ALL-VARIABLES_'+xp[0]+'_'+model+'_'+str(OUT.year.isel(year=0).values)+'-'+str(OUT.year.isel(year=-1).values)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
            OUT.to_netcdf(path_save+'/treated/ZECMIP/intermediary/'+xp[0]+'_'+model+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
        else:
            OUT = xr.open_dataset( path_save+'/treated/ZECMIP/intermediary/'+os.listdir(path_save+'/treated/ZECMIP/intermediary/')[np.where( (xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'!= np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) )[0][0]] )

        ## CONTROL FOR REGULAR FILES          NO NEED OF CONTROL FOR BREACHED FILES (((/!\  TOO LONG TO DO A BREACHED-CONTROL, REQUIRE TO PASS the year_breach of xp[0] to the control, then substantial modification in 'gather_xp'. Assuming control stabilized enough)))
        if xp[0] in dico_Xp_Control.keys():
            pass # this experiment is a control, and has been run without differences to itself
        else:
            xp_c = [ [xp_c for xp_c in dico_Xp_Control if xp[0] in dico_Xp_Control[xp_c]][0] ]
            print(xp_c[0])
            if option_OVERWRITE  or  np.any( (xp_c[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'!= np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) )==False:
                ## producing using weights used for CMIP6
                OUT_pi = func_prod_outputs( xp_c, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=True)
                ## saving as netCDF (just to keep it somewhere)
                OUT_pi.to_netcdf(path_save+'/treated/ZECMIP/intermediary/'+xp_c[0]+'_'+model+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT_pi})
            else:
                OUT_pi = xr.open_dataset( path_save+'/treated/ZECMIP/intermediary/'+os.listdir(path_save+'/treated/ZECMIP/intermediary/')[np.where( (xp_c[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'!= np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_save+'/treated/ZECMIP/intermediary/')])) )[0][0]] )

        ## Correction: if xp is a control experiment, for consistency with the addition of the control to final variables (adding control averaged over 50st years), do the same for the control itself.
        if xp[0] in dico_Xp_Control.keys():OUT = OUT.isel(year=np.arange(50)).mean('year').expand_dims(dim={'year':OUT.year})


        ## CREATING CSV FILES TO SEND
        ## formating as csv
        for vv in OUT.variables:
            if (vv not in OUT.coords):
                ## take control
                if xp[0] in dico_Xp_Control.keys():
                    pi = 0
                else:
                    if 'stat_value' in OUT_pi[vv].dims:
                        pi = OUT_pi[vv].sel(stat_value='mean').drop('stat_value').isel(year=np.arange(50)).mean('year')
                    else:
                        pi = OUT_pi[vv].isel(year=np.arange(50)).mean('year')
                # ## Correction of period of control if too shot
                # if pi.year[-1] < OUT.year[-1]:
                #     pi2 = xr.Dataset( coords=pi.coords )
                #     pi2.coords['year'] = np.arange(pi.year[-1]+1,OUT.year[-1]+1)
                #     pi2['aa'] = xr.DataArray( data=np.full(fill_value=pi.isel(year=np.arange(-50,-1+1)).mean('year').values,shape=[int(OUT.year[-1]-pi.year[-1])]+list(pi.shape[1:])) , dims=pi.dims )
                #     pi = xr.concat( [pi,pi2['aa']] , dim='year')
                ## Correction for some variables: DO NOT SUM
                if vv in list_VAR_NoAddingControl:
                    pi = 0

                ## REGULAR
                with open(path_save+'/treated/ZECMIP/'+xp[0]+'/'+vv+'_'+xp[0]+'_'+model+'_'+str(OUT.year.isel(year=0).values)+'-'+str(OUT.year.isel(year=-1).values)+'.csv','w',newline='') as ff:
                    csv.writer(ff).writerows( np.array([OUT.year.values,pi+OUT[vv].sel(stat_value='mean')]).transpose() )
                ss = 'std_dev'
                with open(path_save+'/treated/ZECMIP/'+xp[0]+'/'+vv+'-'+ss+'_'+xp[0]+'_'+model+'_'+str(OUT.year.isel(year=0).values)+'-'+str(OUT.year.isel(year=-1).values)+'.csv','w',newline='') as ff:
                    csv.writer(ff).writerows( np.array([OUT.year.values,OUT[vv].sel(stat_value=ss)]).transpose() )

                ## BREACHED
                if xp[0] in ['esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC','esm-1pct-brch-750PgC']:
                    with open(path_save+'/treated/ZECMIP/'+xp[0]+'/'+vv+'_'+xp[0]+'_'+model+'_'+str(OUT_br.year.isel(year=0).values)+'-'+str(OUT_br.year.isel(year=-1).values)+'_from-breach.csv','w',newline='') as ff:
                        csv.writer(ff).writerows( np.array([OUT_br.year.values,OUT_br[vv].sel(stat_value='mean').values]).transpose() )
                    ss = 'std_dev'
                    with open(path_save+'/treated/ZECMIP/'+xp[0]+'/'+vv+'-'+ss+'_'+xp[0]+'_'+model+'_'+str(OUT_br.year.isel(year=0).values)+'-'+str(OUT_br.year.isel(year=-1).values)+'_from-breach.csv','w',newline='') as ff:
                        csv.writer(ff).writerows( np.array([OUT_br.year.values,OUT_br[vv].sel(stat_value=ss)]).transpose() )
    ## end loop on experiments


    ## ADDITIONAL VARIABLES FOR ZECMIP
    additional_variables = [['Variable','unit','mean','std_dev']]#,'median','5pct','95pct']
    VALS = xr.Dataset()
    VALS.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
    # list_var_const = ['D_Tg', 'CumEff','Cum D_Focean', 'Eff'] + ['ECS_tot']
    # weights_CMIP6 = func_weights(VAR_CONST=list_var_const , strength_CONST=np.ones(len(list_var_const)),provided_val=prov_val,option_minus_pi=option_minus_pi )
    ## 'ECS' as 'ECS_charney' --> ecs_0
    VALS['vals'] = xr.DataArray(  np.hstack([prov_val[setMC]['ECS_charney'] for setMC in list_setMC]) , dims=('all_config')  )
    ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(VALS.vals))[0]
    mm = np.average( VALS.vals.isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
    ss = np.sqrt(np.average( (VALS.vals.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) ))
    additional_variables.append(['ECS','K',mm,ss])
    # [ quantile_1D(data=np.ma.array(VALS.vals.values,mask=np.isnan(VALS.vals.values)) , weights=weights_CMIP6.weights.values , qq={'median':50.,'5pct':5.,'95pct':95.}[pp]/100. ) for pp in ['median','5pct','95pct'] ]
    ## 'ECS_Gregory as ECS_tot --> sortie 2xCO2
    VALS['vals'] = xr.DataArray(  np.hstack([prov_val[setMC]['ECS_tot'] for setMC in list_setMC]) , dims=('all_config')  )
    ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(VALS.vals))[0]
    mm = np.average( VALS.vals.isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
    ss = np.sqrt(np.average( (VALS.vals.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) ))
    additional_variables.append(['ECS_Gregory','K',mm,ss])
    ## 'TCR'
    TMP = gather_XP( ['1pctCO2'], weights_CMIP6, ['Surface Air Temperature Change','Cumulative Emissions|CO2'], dico_variables_longnames, dico_variables_units, dico_variables_warnings , dico_var=dico_varOSCAR_to_varRCMIP, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=True , option_load_hist=False , option_JustAddVar=False)[0]
    VALS['vals'] = xr.DataArray(  TMP['Surface Air Temperature Change'].sel(year=np.arange(1850+60,1850+80+1)).mean('year') , dims=('all_config')  )
    ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(VALS.vals))[0]
    mm = np.average( VALS.vals.isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
    ss = np.sqrt(np.average( (VALS.vals.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) ))
    additional_variables.append(['TCR','K',mm,ss])
    ## 'TCRE' (K / 1000 PgC)
    VALS['vals'] = xr.DataArray(  1000. * (TMP['Surface Air Temperature Change'] / (TMP['Cumulative Emissions|CO2'] / (1.e3 * 44/12.))).sel(year=1850+70) , dims=('all_config')  )
    ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(VALS.vals))[0]
    mm = np.average( VALS.vals.isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
    ss = np.sqrt(np.average( (VALS.vals.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) ))
    additional_variables.append(['TCRE','K / 1000PgC',mm,ss])
    ## PI temperature
    additional_variables.append( ['PI temperature (assumption)','K',273.15+13.9,0] )
    ## Year breach
    for xp in ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC']:
        VALS['vals'] = xr.DataArray(  np.nan*np.ones(VALS.all_config.size) , dims=('all_config')  )
        for setMC in list_setMC:
            tmp = xr.open_dataset(path_runs+'/'+dico_experiments_before[xp[0]]+'_For-'+str(setMC)+'.nc')
            VALS['vals'].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config] }] = tmp.year_breach
            tmp.close()


        #TMP = gather_XP( [xp], weights_CMIP6, ['year_breach'], dico_variables_longnames, dico_variables_units, dico_variables_warnings , dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=True , option_load_hist=False , option_JustAddVar=False)[0]
        #VALS['vals'] = xr.DataArray(  TMP['year_breach'] , dims=('all_config')  )
        ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(VALS.vals))[0]
        mm = np.average( VALS.vals.isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
        ss = np.sqrt(np.average( (VALS.vals.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) ))
        additional_variables.append(['Year breach on '+xp,'yr',mm,ss])


    with open(path_save+'/treated/ZECMIP/additional_variables.csv','w',newline='') as ff:
        csv.writer(ff).writerows( additional_variables )
#########################
#########################








#########################
## 4.4. LUMIP
#########################
if option_select_MIP in ['ALL','LUMIP']:
    list_xp = [ list(dico_experiments_MIPs[ii,0:1+1]) if dico_experiments_MIPs[ii,1]!='' else [dico_experiments_MIPs[ii,0]]   for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-LUMIP')]=='1')[0] ]
    for xp in list_xp:
        if xp[0] in dico_Xp_Control.keys():
            list_xp.remove(xp)
            list_xp.insert(0,xp)

    saved_stat_values = ['mean','std_dev']
    option_SumReg=False

    ## variables that will be provided to LUMIP by OSCAR
    list_variables_required = [ dico_variables_MIPs[ii,head_MIPs.index( 'CMIP6 name' )]  for ii in  np.where(dico_variables_MIPs[:,head_MIPs.index('OSCAR-LUMIP')]=='1')[0] ]
    ## all variables from Lut to global variable (eg nppLut to npp)

    ## Preparing directories
    for ff in ['treated','treated/LUMIP','treated/LUMIP/plots','treated/LUMIP/intermediary']: os.mkdir(path_save+'/'+ff) if (os.path.isdir( path_save+'/'+ff )==False) else None

    for xp in list_xp:
        print('-'.join(xp)+' ('+str(list_xp.index(xp)+1)+'/'+str(len(list_xp))+')')
        if option_OVERWRITE  or  np.any(xp[0] == np.array([str.split(ff,'_')[0] for ff in os.listdir( path_save+'/treated/LUMIP/intermediary/' )]))==False:
            ## producing
            if xp[0][:len('land-')]=='land-':
                OUT = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings  , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR_landC' , option_breach=False , option_SumReg=option_SumReg)
            else:
                OUT = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings  , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=option_SumReg)
            ## saving as netCDF
            OUT.to_netcdf(path_save+'/treated/LUMIP/intermediary/'+xp[0]+'_'+model+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
        else:
            OUT = xr.open_dataset( path_save+'/treated/LUMIP/intermediary/'+os.listdir(path_save+'/treated/LUMIP/intermediary/')[np.where(xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/LUMIP/intermediary/')]))[0][0]] )


        ## Producing and saving Control
        if xp[0] in dico_Xp_Control.keys():
            pass # this experiment is a control, and has been run without differences to itself
        else:
            xp_c = [ [xp_c for xp_c in dico_Xp_Control if xp[0] in dico_Xp_Control[xp_c]][0] ]
            print(xp_c[0])
            if option_OVERWRITE  or  np.any(xp_c[0] == np.array([str.split(ff,'_')[0] for ff in os.listdir( path_save+'/treated/LUMIP/intermediary/' )]))==False:
                ## producing
                if xp_c[0][:len('land-')]=='land-':
                    OUT_pi = func_prod_outputs( xp_c, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings  , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR_landC' , option_breach=False , option_SumReg=option_SumReg)
                else:
                    OUT_pi = func_prod_outputs( xp_c, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings  , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=option_SumReg)
                ## saving as netCDF (just to keep it somewhere)
                OUT_pi.to_netcdf(path_save+'/treated/LUMIP/intermediary/'+xp_c[0]+'_OSCARv3.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT_pi})
            else:
                OUT_pi = xr.open_dataset( path_save+'/treated/LUMIP/intermediary/'+os.listdir(path_save+'/treated/LUMIP/intermediary/')[np.where(xp_c[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/LUMIP/intermediary/')]))[0][0]] )

        ## Correction: if xp is a control experiment, for consistency with the addition of the control to final variables (adding control averaged over 50st years), do the same for the control itself.
        if xp[0] in dico_Xp_Control.keys():
            test = OUT.copy(deep=True)
            for VAR in OUT.variables:
                if VAR not in OUT.coords:
                    test[VAR] = xr.full_like( other=OUT[VAR], fill_value=OUT[VAR].isel(year=np.arange(50)).mean('year') )
            OUT = test.copy()
            del test

        ## Producing the files for LUMIP
        OUT = OUT.sel(stat_value = saved_stat_values)
        for VAR in OUT:
            if xp[0] in dico_Xp_Control.keys():
                pi = 0
            else:
                if 'stat_value' in OUT_pi[VAR].dims:
                    pi = OUT_pi[VAR].sel(stat_value='mean').drop('stat_value').isel(year=np.arange(50)).mean('year')
                else:
                    pi = OUT_pi[VAR].isel(year=np.arange(50)).mean('year')
            #     if 'stat_value' in OUT_pi[VAR].dims:
            #         pi = OUT_pi[VAR].sel(stat_value='mean').drop('stat_value')
            #     else:
            #         pi = OUT_pi[VAR].isel(year=np.arange(50))
            # ## Correction of period of control if too shot
            # if pi.year[-1] < OUT.year[-1]:
            #     pi2 = xr.Dataset( coords=pi.coords )
            #     pi2.coords['year'] = np.arange(pi.year[-1]+1,OUT.year[-1]+1)
            #     pi2['aa'] = xr.DataArray( data=np.full(fill_value=pi.isel(year=np.arange(-50,-1+1)).mean('year').values,shape=[int(OUT.year[-1]-pi.year[-1])]+list(pi.shape[1:])) , dims=pi.dims )
            #     pi = xr.concat( [pi,pi2['aa']] , dim='year')
            ## Correction for some variables: DO NOT SUM
            if VAR in list_VAR_NoAddingControl:
                pi = 0
            ## adding control
            if 'stat_value' in OUT[VAR].dims:
                OUT[VAR].loc[{'stat_value':'mean'}] = OUT[VAR].loc[{'stat_value':'mean'}] + pi
            else:
                OUT[VAR] = OUT[VAR] + pi
        ## Saving
        OUT.to_netcdf(path_save+'/treated/LUMIP/'+xp[0]+'_'+model+'_'+str(OUT.year.isel(year=0).values)+'-'+str(OUT.year.isel(year=-1).values)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
#########################
#########################









#########################
## 4.5. RCMIP
#########################
if option_select_MIP in ['ALL','RCMIP']:
    list_xp = [ list(dico_experiments_MIPs[ii,0:1+1]) if dico_experiments_MIPs[ii,1]!='' else [dico_experiments_MIPs[ii,0]]   for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-RCMIP')]=='1')[0] ]
    ## Missing (only in phase 2 only, not planned in 2019): 'hist-all-aer2' , 'hist-all-nat2' , 'CDR-yr2010-pulse'
    for xp in list_xp:
        if xp[0] in dico_Xp_Control.keys():
            list_xp.remove(xp)
            list_xp.insert(0,xp)

    # list_xp = [ ['piControl'] , ['esm-piControl'] , ['piControl-CMIP5'] , ['esm-piControl-CMIP5'] ]
    # if k_subsetXP==0:
    #     list_xp = list_xp[ 2:10 ]
    # else:
    #     list_xp = list_xp[ k_subsetXP*10:(k_subsetXP+1)*10 ] ## 0-->6+1*10
    # print(list_xp)

    saved_stat_values = ['mean','std_dev']
    option_SumReg = True
    option_load_hist = True

    ## variables that will be provided to RCMIP by OSCAR
    list_variables_required = [ dico_variables_MIPs[ii,head_MIPs.index( 'RCMIP name' )]  for ii in  np.where(dico_variables_MIPs[:,head_MIPs.index('OSCAR-RCMIP')]=='1')[0] ]+['year']

    ## Preparing directories
    for ff in ['treated','treated/RCMIP','treated/RCMIP/plots','treated/RCMIP/intermediary']: os.mkdir(path_save+'/'+ff) if (os.path.isdir( path_save+'/'+ff )==False) else None

    ## looping on experiments
    dico_RCMIP_model = {'rcp60':'AIM','ssp370':'AIM','ssp370-lowNTCF':'AIM','ssp434':'GCAM4','ssp460':'GCAM4','rcp26':'IMAGE','ssp119':'IMAGE','ssp126':'IMAGE','rcp85':'MESSAGE','ssp245':'MESSAGE-GLOBIOM','rcp45':'MiniCAM','ssp534-over':'REMIND-MAGPIE','ssp585':'REMIND-MAGPIE'}
    dico_RCMIP_quantiles = {'mean':'', 'std_dev':'|Standard deviation', 'median':'|50th quantile', '5pct':'|05th quantile', '95pct':'|95th quantile'}
    TMP_template,TMP_template_3K,TMP_comments = [],[],[]
    for xp in list_xp:
        ## Producing and saving single experiments
        print('-'.join(xp)+' ('+str(list_xp.index(xp)+1)+'/'+str(len(list_xp))+')')
        if option_OVERWRITE  or  np.any(xp[0] == np.array([str.split(ff,'_')[0] for ff in os.listdir( path_save+'/treated/RCMIP/intermediary/' )]))==False:
            ## producing using weights used for CMIP6
            OUT = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varRCMIP, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=option_SumReg , option_load_hist=option_load_hist)
            ## saving as netCDF (just to keep it somewhere)
            OUT.to_netcdf(path_save+'/treated/RCMIP/intermediary/'+xp[0]+'_'+model+'_'+str(OUT.year.isel(year=0).values)+'-'+str(OUT.year.isel(year=-1).values)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
        else:
            OUT = xr.open_dataset( path_save+'/treated/RCMIP/intermediary/'+os.listdir(path_save+'/treated/RCMIP/intermediary/')[np.where(xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/RCMIP/intermediary/')]))[0][0]] )

        ## Producing and saving Control
        if xp[0] in dico_Xp_Control.keys():
            pass # this experiment is a control, and has been run without differences to itself
        else:
            xp_c = [ [xp_c for xp_c in dico_Xp_Control if xp[0] in dico_Xp_Control[xp_c]][0] ]
            print(xp_c[0])
            if np.any(xp_c[0] == np.array([str.split(ff,'_')[0] for ff in os.listdir( path_save+'/treated/RCMIP/intermediary/' )]))==False: # or option_OVERWRITE:
                ## producing using weights used for CMIP6
                OUT_pi = func_prod_outputs( xp_c, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varRCMIP, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=option_SumReg)
                ## saving as netCDF (just to keep it somewhere)
                OUT_pi.to_netcdf(path_save+'/treated/RCMIP/intermediary/'+xp_c[0]+'_'+model+'_'+str(OUT_pi.year.isel(year=0).values)+'-'+str(OUT_pi.year.isel(year=-1).values)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT_pi})
            else:
                OUT_pi = xr.open_dataset( path_save+'/treated/RCMIP/intermediary/'+os.listdir(path_save+'/treated/RCMIP/intermediary/')[np.where(xp_c[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/RCMIP/intermediary/')]))[0][0]] )

        ## Correction: if xp is a control experiment, for consistency with the addition of the control to final variables (adding control averaged over 50st years), do the same for the control itself.
        if xp[0] in dico_Xp_Control.keys():
            test = OUT.copy(deep=True)
            for VAR in OUT.variables:
                if VAR not in OUT.coords:
                    test[VAR] = xr.full_like( other=OUT[VAR], fill_value=OUT[VAR].isel(year=np.arange(50)).mean('year') )
            OUT = test.copy()
            del test

        ## preparation for the datasheet for RCMIP
        ## correction name of xp
        if xp[0] == 'esm-histcmip5':
            name_experiment = 'esm-hist-cmip5'
        elif xp[0] == 'historical-CMIP5':
            name_experiment = 'historical-cmip5'
        elif 'ssp370-lowNTCF' in xp[0]:
            name_experiment = {'esm-ssp370-lowNTCF':'esm-ssp370-lowNTCF-aerchemmip','esm-ssp370-lowNTCF-gidden':'esm-ssp370-lowNTCF-gidden' ,'ssp370-lowNTCF':'ssp370-lowNTCF-aerchemmip','ssp370-lowNTCF-gidden':'ssp370-lowNTCF-gidden'}[xp[0]]
        else:
            name_experiment = xp[0]            
        ## creating the datasheet for RCMIP
        for VAR in OUT:
            ## correction unit:
            if VAR[:len('Cumulative Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Other')] == 'Cumulative Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Other':
                unit = 'Mt CH4'
            else:
                unit = OUT[VAR].unit
            ## correction name variable
            if VAR[-len('Albedo Change|Land Cover Change'):] == 'Albedo Change|Land Cover Change':
                name_VAR = VAR[:-len('Albedo Change|Land Cover Change')] + 'Albedo Change|Other|Land Cover Change'
            elif VAR[-len('Albedo Change|Deposition of Black Carbon on Snow'):] == 'Albedo Change|Deposition of Black Carbon on Snow':
                name_VAR = VAR[:-len('Albedo Change|Deposition of Black Carbon on Snow')] + 'Albedo Change|Other|Deposition of Black Carbon on Snow'
            else:
                name_VAR = VAR
            ## preparing control
            if xp[0] in dico_Xp_Control.keys():
                pi = 0
            else:
                if 'stat_value' in OUT_pi[VAR].dims:
                    pi = OUT_pi[VAR].sel(stat_value='mean').isel(year=np.arange(50)).mean('year')
                else:
                    pi = OUT_pi[VAR].isel(year=np.arange(50)).mean('year')
            ## Correction for some variables: DO NOT SUM
            if VAR in list_VAR_NoAddingControl:
                pi = 0

            ## Allocating over lines
            if 'stat_value' in OUT[VAR].dims:
                if (option_SumReg==False) and ('reg_land' in OUT[VAR].dims):
                    for rr in OUT.reg_land.values:
                        for sv in saved_stat_values:
                            if sv=='mean':
                                TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , OUT.reg_land_long_name.values[rr] , name_VAR+dico_RCMIP_quantiles[sv] , unit]  +  list((pi+OUT[VAR]).sel(stat_value=sv,reg_land=rr).values)  )
                            elif sv=='std_dev':
                                TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , OUT.reg_land_long_name.values[rr] , name_VAR+'|84th quantile' , unit]  +  list((pi+OUT[VAR].sel(stat_value='mean',reg_land=rr)+1.0*OUT[VAR].sel(stat_value=sv,reg_land=rr)).values)  )
                                TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , OUT.reg_land_long_name.values[rr] , name_VAR+'|16th quantile' , unit]  +  list((pi+OUT[VAR].sel(stat_value='mean',reg_land=rr)-1.0*OUT[VAR].sel(stat_value=sv,reg_land=rr)).values)  )
                else:
                    for sv in saved_stat_values:
                        if sv=='mean':
                            TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , 'World' , name_VAR+dico_RCMIP_quantiles[sv] , unit]  +  list((pi+OUT[VAR].sel(stat_value='mean')).values)  )
                        elif sv=='std_dev':
                            TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , 'World' , name_VAR+'|84th quantile' , unit]  +  list((pi+OUT[VAR].sel(stat_value='mean')+1.0*OUT[VAR].sel(stat_value=sv)).values)  )
                            TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , 'World' , name_VAR+'|16th quantile' , unit]  +  list((pi+OUT[VAR].sel(stat_value='mean')-1.0*OUT[VAR].sel(stat_value=sv)).values)  )

            else:
                if (option_SumReg==False) and ('reg_land' in OUT[VAR].dims):
                    for rr in OUT.reg_land.values:
                        TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , OUT.reg_land_long_name.values[rr] , name_VAR , unit]  +  list((pi+OUT[VAR]).sel(reg_land=rr).values)  )
                else:
                    TMP_template.append(  [model , (dico_RCMIP_model[xp[0]] if xp[0] in dico_RCMIP_model else 'unspecified') , name_experiment , 'World' , name_VAR , unit]  +  list((pi+OUT[VAR]).values)  )

        ## plots
        if False:
            for vv in list(OUT.variables):
                if (vv not in OUT.coords):
                    fig = func_plot( vv , OUT , xp[0])
                    fig.savefig( path_save+'/treated/RCMIP/plots/'+xp[0]+'_'+str(list(OUT.variables).index(vv)+1))#+'.pdf', format='pdf' )
                    plt.close(fig)

        ## warnings
        if len(TMP_comments)==0: ## only once
            TMP_comments.append(  ['Yann Quilcaille','Calculated as mean minus 1 standard deviation','yann.quilcaille@iiasa.ac.at',model,'all','all','16th quantile','all']  )
            TMP_comments.append(  ['Yann Quilcaille','Calculated as mean plus 1 standard deviation','yann.quilcaille@iiasa.ac.at',model,'all','all','84th quantile','all']  )
        str_xp = ' and '.join(xp)
        warn = 'Global outputs of OSCARv3 of the experiment '+str_xp+'. Only statistical insights are provided in this file, for a sake of simplicity. The full set of members of each individual experiment can be provided on request.'
        if 'warning' in OUT.attrs:
            TMP_comments.append(  ['Yann Quilcaille',warn,'yann.quilcaille@iiasa.ac.at',model,xp[0],'all','all','all']  )
        if 'warning_emissions' in OUT.attrs:
            TMP_comments.append(  ['Yann Quilcaille',OUT.warning_emissions,'yann.quilcaille@iiasa.ac.at',model,xp[0],'all','all','all']  )
    for VAR in OUT:
        if OUT[VAR].warning != '':
            TMP_comments.append(  ['Yann Quilcaille',OUT[VAR].warning,'yann.quilcaille@iiasa.ac.at',model,'all','all',VAR,'all']  )

    ## saving for RCMIP
    with open(path_save+'/treated/RCMIP/for_template.csv','w',newline='') as ff:
        csv.writer(ff).writerows( TMP_template )
    # with open(path_save+'/treated/RCMIP/for_template_3K.csv','w',newline='') as ff:
    #     csv.writer(ff).writerows( TMP_template_3K )
    with open(path_save+'/treated/RCMIP/for_comments.csv','w',newline='') as ff:
        csv.writer(ff).writerows( TMP_comments )
#########################
#########################





#########################
## 4.6. OSCAR-CMIP6
#########################
if option_select_MIP in ['ALL','CMIP6']  and  option_AddingVarToCMIP6==False:
    list_xp = [ list(dico_experiments_MIPs[ii,0:1+1]) if dico_experiments_MIPs[ii,1]!='' else [dico_experiments_MIPs[ii,0]]   for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-CMIP6')]=='1')[0] ]
    ## not the 2 about alkalinity
    for xp in list_xp:
        if xp[0] in dico_Xp_Control.keys():
            list_xp.remove(xp)
            list_xp.insert(0,xp)

    saved_stat_values = ['mean','std_dev']

    ## controls
    # list_xp = [ list_xp[0:7+1][k_subsetXP] ]

    if k_subsetXP==0:
        list_xp = list_xp[ 8:10 ]
    else:
        list_xp = list_xp[ k_subsetXP*10:(k_subsetXP+1)*10 ] ## 10*10

    ## variables that will be provided to CDR-MIP by OSCAR
    list_variables_required = [ dico_variables_MIPs[ii,head_MIPs.index( 'CMIP6 name' )]  for ii in  np.where(dico_variables_MIPs[:,head_MIPs.index('OSCAR-CMIP6')]=='1')[0] ]

    ## Preparing directories
    for ff in ['treated','treated/OSCAR-CMIP6','treated/OSCAR-CMIP6/plots','treated/OSCAR-CMIP6/intermediary']: os.mkdir(path_save+'/'+ff) if (os.path.isdir( path_save+'/'+ff )==False) else None

    ## Producing and saving single experiments:
    for xp in list_xp:
        print('-'.join(xp)+' ('+str(list_xp.index(xp)+1)+'/'+str(len(list_xp))+')')
        ## producing
        if option_OVERWRITE  or  np.any(xp[0] == np.array([str.split(ff,'_')[0] for ff in os.listdir( path_save+'/treated/OSCAR-CMIP6/intermediary/' )]))==False:
            ## producing
            if xp[0][:len('land-')]=='land-':
                OUT = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR_landC' , option_breach=False , option_SumReg=False)
            else:
                OUT = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=False)
            ## saving as netCDF
            OUT.to_netcdf(path_save+'/treated/OSCAR-CMIP6/intermediary/'+xp[0]+'_'+model+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
        else:
            pass
            # OUT = xr.open_dataset( path_save+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_save+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
#########################
#########################

##################################################
##################################################






















##################################################
## X. Stash
##################################################


##-------------------------------------------------
## number of years
##-------------------------------------------------
dico_year_xp = {}
for xp in list_experiments:
    out_TMP = xr.open_dataset(path_runs+'/'+xp+'_Out-'+str(1)+'.nc' )
    dico_year_xp[xp] = out_TMP.year.size
    out_TMP.close()

N_year = np.sum(list(dico_sizesMC.values())) * np.sum(np.array(list(dico_year_xp.values()))-1)
##-------------------------------------------------
##-------------------------------------------------





# ##-------------------------------------------------
# ## appending a simple variable to all experiments
# ##-------------------------------------------------
if option_AddingVarToCMIP6:
    list_xp = [ list(dico_experiments_MIPs[ii,0:1+1]) if dico_experiments_MIPs[ii,1]!='' else [dico_experiments_MIPs[ii,0]]   for ii in  np.where(dico_experiments_MIPs[:,list(dico_experiments_MIPs[0,:]).index('OSCAR-CMIP6')]=='1')[0] ]
    ## controls not provided, because already providing differences to controls
    saved_stat_values = ['mean','std_dev']

    list_variables_required = [ 'RF_solar','RF_volc' ]################ CHANGED

    ## Producing and saving single experiments:
    for xp in list_xp:
        print(xp[0])
        if xp[0][:len('land-')]!='land-':
            print('-'.join(xp)+' ('+str(list_xp.index(xp)+1)+'/'+str(len(list_xp))+')')
            ## load the old one
            OUT1 = xr.open_dataset( path_save+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_save+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp[0]==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_save+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
            OUT = OUT1.copy(deep=True)
            OUT1.close()
            del OUT1
            ## check if RF_CO2
            test_go = False
            for var in list_variables_required:
                if var not in OUT:test_go = True
            if test_go:
                print('Missing on '+xp[0])
                ## producing
                OUT2 = func_prod_outputs( xp, weights_CMIP6, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , saved_stat_values=saved_stat_values, dico_var=dico_varOSCAR_to_varCMIP6, type_OSCAR='OSCAR' , option_breach=False , option_SumReg=False , option_JustAddVar=True)
                ## adding
                for var in list_variables_required:
                    OUT[var] = OUT2[var]
                ## saving as netCDF
                OUT.to_netcdf(path_save+'/treated/OSCAR-CMIP6/intermediary/'+xp[0]+'_'+model+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in OUT})
                OUT2.close()
                del OUT2
            ## cleaning
            OUT.close()
            del OUT
#-------------------------------------------------
#-------------------------------------------------




#-------------------------------------------------
## Interannual variability
#-------------------------------------------------
# test = xr.open_dataset('extra_data/constraints/Gsat_yr_historical_IPSL-CM6A-LR_1850-2029.nc')

# plt.plot( test.year, test.Gsat.transpose()-273.15 )
# plt.plot( test.year, (test.Gsat-test.Gsat.sel(year=np.arange(1850,1900)).mean('year')).transpose() )

# val = test.Gsat-test.Gsat.sel(year=np.arange(1850,1900)).mean('year')
# val.sel(year=np.arange(2006,2014+1)).mean( ('year','member') )
# val.sel(year=np.arange(2006,2014+1)).std( ('year','member') )


# (  test.Gsat.sel(year=np.arange(2006,2015+1)) - test.Gsat.sel(year=np.arange(1850,1900+1)).mean('year')  ).mean()

# (  test.Gsat.sel(year=np.arange(2006,2015+1)) - test.Gsat.sel(year=np.arange(1850,1900+1)).mean('year') - (test.Gsat.sel(year=np.arange(2006,2015+1)) - test.Gsat.sel(year=np.arange(1850,1900+1)).mean('year')).mean('member')  ).std()
#-------------------------------------------------
#-------------------------------------------------



#-------------------------------------------------
## Fit ECS
#-------------------------------------------------
# def func_err(val):
#     ## function to fit: ECS = 3 K [1.5 to 4.5K] (likely range from IPCC AR5, likely=66-100%. Considered as one std_dev of the normal: 68.27%)
#     ## using 3K [1.6K to 4.5 K] --> improves greatly the fit.
#     tmp = main_values_ECS( val , dT=1.e-4 )
#     return ((tmp[0]-3.)/3.)**2. + ((tmp[2]-1.6)/2.)**2. + ((tmp[3]-4.5)/4.5)**2.

## Tring typical values for fits from Roe et al, 2007: Checking which first guess provides the best fit.
# for fg in [ [0.58,0.17],[0.63,0.21],\
#             [0.67,0.10],[0.60,0.14],\
#             [0.64,0.20],[0.56,0.16],\
#             [0.82,0.10],[0.65,0.14],[0.15,0.28],\
#             [0.86,0.35],\
#             [0.72,0.17],[0.75,0.19],[0.77,0.21]]:
#     print( np.round(main_values_ECS(fg),3) )
#     out = fmin( func_err , fg )
#     print( np.round(out,3) )
#     print( np.round(main_values_ECS(out),5) )
#     print(" ")
#     print(" ")
#     print(" ")
#-------------------------------------------------
#-------------------------------------------------


#-------------------------------------------------
## old version of gather_xp, included load_xp
#-------------------------------------------------
# def gather_XP( xp, weights, list_variables_required, dico_variables_longnames, dico_variables_units, dico_variables_warnings , dico_var , type_OSCAR , option_breach , option_SumReg , option_load_hist , option_JustAddVar):
#     ##-----
#     ## For an experiment (eg ssp585,ssp585ext), for each set of MC members, this function loads the Xp and its extension if any, their control, produces the differences only for the outputs.
#     ## Required variables for this MIP are calculated ('func_add_var') using these forcings, outputs and parameters.
#     ## Matching of OSCAR variables to MIP variables ('varOSCAR_to_varMIP') is then executed.
#     ## Every set is allocated over a single axis.
#     ## option_load_hist:: includes adapted historical run before: for RCMIP, and because of cumulative emission
#     ## option_breach:: returns outputs from the date of breach:: specific experiments
#     ## option_JustAddVar: if True, will not compute compatible emissions
#     ##-----
#     ## initializing temporary variable for storing outputs
#     TMP_xp = xr.Dataset()
#     if threshold_select_ProbableRuns==None:
#         TMP_xp.coords['all_config'] = [str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
#     else:
#         ind = np.where( weights.weights.sel(all_config=[str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]) > weights.weights.median() / threshold_select_ProbableRuns )[0]
#         TMP_xp.coords['all_config'] = np.array([str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])])[ind]

#     if option_breach: max_yr_brch=-np.inf
#     ## looping on sets
#     for setMC in list_setMC:
#         ## index for selction of likely runs
#         if threshold_select_ProbableRuns!=None:
#             ind_threshold_weights = [cfg for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config]

#         ##-----
#         ## LOADING
#         ##-----
#         print("Preparing variables for " + " and ".join(xp) + " over set "+str(setMC))
#         ## loading everything related to the experiment
#         out_tmp,for_tmp,mask_xp = [],[],[]
#         ## parameters applied throughout experiments
#         with xr.open_dataset(path_runs+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
#         ## immediately correcting Par for -bgc and -rad experiments
#         if xp[0] in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc']:
#             with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
#             Par['D_CO2_rad'] = for_runs_hist.D_CO2.sel(year=1850)
#         elif xp[0] in ['1pctCO2-rad']:
#             with xr.open_dataset(path_runs+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
#             Par['D_CO2_bgc'] = for_runs_hist.D_CO2.sel(year=1850)
#         ## eventually selection of relevant runs
#         if threshold_select_ProbableRuns!=None:
#             Par = Par.sel(config=ind_threshold_weights)
#         ## loading if required corresponding historical  ===> required for cumulative emissions for scenarios!!!!
#         if option_load_hist and (dico_experiments_before[xp[0]]!=None):
#             out_tmp.append( xr.open_dataset(path_runs+'/'+dico_experiments_before[xp[0]]+'_Out-'+str(setMC)+'.nc') )
#             for_tmp.append( xr.open_dataset(path_runs+'/'+dico_experiments_before[xp[0]]+'_For-'+str(setMC)+'.nc') )
#             ## eventually selection of relevant runs
#             if threshold_select_ProbableRuns!=None:
#                 out_tmp[-1] = out_tmp[-1].sel(config=ind_threshold_weights)
#                 for_tmp[-1] = for_tmp[-1].sel(config=ind_threshold_weights)
#             ## correction of parameter 'Aland_0':: some experiments (blocks 'CMIP5' and 'LAND') have different Aland_0. It has been saved in the forcings, not in the parameters file.
#             if 'Aland_0' in for_tmp[-1]:
#                 Par['Aland_0'] = for_tmp[-1]['Aland_0']
#             if option_JustAddVar==False:for_tmp[-1] = eval_compat_emi( dico_compatible_emissions[dico_experiments_before[xp[0]]], out_tmp[-1],Par,for_tmp[-1],type_OSCAR=type_OSCAR ) ## correction for compatible emissions
#             mask_xp.append( list_noDV[dico_experiments_before[xp[0]]][setMC] )
#             ## eventually selection of relevant runs
#             if threshold_select_ProbableRuns!=None:
#                 mask_xp[-1] = mask_xp[-1][:,ind_threshold_weights]
#             ## removing first value
#             for_tmp[-1] = for_tmp[-1].isel(year=np.arange(1,for_tmp[-1].year.size))
#             out_tmp[-1] = out_tmp[-1].isel(year=np.arange(1,out_tmp[-1].year.size))
#             mask_xp[-1] = mask_xp[-1][1:]
#             ## eventually cutting the exceeding years of the experiment
#             if dico_experiments_before[xp[0]] in list_Xp_cut100:
#                 for_tmp[-1] = for_tmp[-1].isel(year=np.arange(0,for_tmp[-1].year.size-100))
#                 out_tmp[-1] = out_tmp[-1].isel(year=np.arange(0,out_tmp[-1].year.size-100))
#                 mask_xp[-1] = mask_xp[-1][:-100,:]
#         ## loading in all cases the required xp
#         out_tmp.append( xr.open_dataset(path_runs+'/'+xp[0]+'_Out-'+str(setMC)+'.nc') )
#         for_tmp.append( xr.open_dataset(path_runs+'/'+xp[0]+'_For-'+str(setMC)+'.nc') )
#         ## eventually selection of relevant runs
#         if threshold_select_ProbableRuns!=None:
#             out_tmp[-1] = out_tmp[-1].sel(config=ind_threshold_weights)
#             for_tmp[-1] = for_tmp[-1].sel(config=ind_threshold_weights)
#         ## correction of parameter 'Aland_0':: some experiments (blocks 'CMIP5' and 'LAND') have different Aland_0. It has been saved in the forcings, not in the parameters file.
#         if 'Aland_0' in for_tmp[-1]:
#             Par['Aland_0'] = for_tmp[-1]['Aland_0']
#         if option_JustAddVar==False:for_tmp[-1] = eval_compat_emi( dico_compatible_emissions[xp[0]], out_tmp[-1],Par,for_tmp[-1],type_OSCAR=type_OSCAR ) ## correction for compatible emissions
#         if option_breach:print("breaches in "+str(setMC)+": "+str(int(for_tmp[-1].year_breach.min()))+'-'+str(int(for_tmp[-1].year_breach.max())))
#         mask_xp.append( list_noDV[xp[0]][setMC] )
#         ## eventually selection of relevant runs
#         if threshold_select_ProbableRuns!=None:
#             mask_xp[-1] = mask_xp[-1][:,ind_threshold_weights]
#         ## immediately removing first year of outputs: used in OSCAR for initilization, but not to provide (eg. 2014 in ssp585)
#         if dico_experiments_before[xp[0]] != None:
#             for_tmp[-1] = for_tmp[-1].isel(year=np.arange(1,for_tmp[-1].year.size))
#             out_tmp[-1] = out_tmp[-1].isel(year=np.arange(1,out_tmp[-1].year.size))
#             mask_xp[-1] = mask_xp[-1][1:]
#         ## eventually cutting the exceeding years of the experiment
#         if xp[0] in list_Xp_cut100:
#             for_tmp[-1] = for_tmp[-1].isel(year=np.arange(0,for_tmp[-1].year.size-100))
#             out_tmp[-1] = out_tmp[-1].isel(year=np.arange(0,out_tmp[-1].year.size-100))
#             mask_xp[-1] = mask_xp[-1][:-100,:]
#         ## loading extension if required
#         if len(xp)==2:
#             for_tmp.append( xr.open_dataset(path_runs+'/'+xp[1]+'_For-'+str(setMC)+'.nc') )
#             out_tmp.append( xr.open_dataset(path_runs+'/'+xp[1]+'_Out-'+str(setMC)+'.nc') )
#             mask_xp.append( list_noDV[xp[1]][setMC] )
#             ## eventually selection of relevant runs
#             if threshold_select_ProbableRuns!=None:
#                 out_tmp[-1] = out_tmp[-1].sel(config=ind_threshold_weights)
#                 for_tmp[-1] = for_tmp[-1].sel(config=ind_threshold_weights)
#                 mask_xp[-1] = mask_xp[-1][:,ind_threshold_weights]
#             if option_JustAddVar==False:for_tmp[-1] = eval_compat_emi( dico_compatible_emissions[xp[1]], out_tmp[-1],Par,for_tmp[-1],type_OSCAR=type_OSCAR ) ## correction for compatible emissions
#             ## immediately removing first year of outputs: used in OSCAR for initilization, but not to provide (eg. 2100 in ssp585ext)
#             if dico_experiments_before[xp[1]] != None:
#                 for_tmp[-1] = for_tmp[-1].isel(year=np.arange(1,for_tmp[-1].year.size))
#                 out_tmp[-1] = out_tmp[-1].isel(year=np.arange(1,out_tmp[-1].year.size))
#                 mask_xp[-1] = mask_xp[-1][1:]
#             else:
#                 raise Exception("Error in preceding experiments")
#             ## eventually cutting the exceeding years of the experiment
#             if xp[1] in list_Xp_cut100:
#                 for_tmp[-1] = for_tmp[-1].isel(year=np.arange(0,for_tmp[-1].year.size-100))
#                 out_tmp[-1] = out_tmp[-1].isel(year=np.arange(0,out_tmp[-1].year.size-100))
#                 mask_xp[-1] = mask_xp[-1][:-100,:]
#         ## removing some useless coordinates            
#         for ii in np.arange(len(for_tmp)):
#             for cc in ['data_RF_solar','data_RF_volc','scen']:
#                 if cc in for_tmp[ii]:
#                     for_tmp[ii] = for_tmp[ii].drop(cc)
#                 if cc in out_tmp[ii]:
#                     out_tmp[ii] = out_tmp[ii].drop(cc)
#         ## Concatening forcings: situation of CO2 compatible emissions over historical (year x config) and CO2 prescribed emissions but not used (config) correctly handled.
#         if option_load_hist and (dico_experiments_before[xp[0]]!=None): ## special case: presence of a historical, with different requirement of compatible emissions, thus coordinates and variables in the list
#             ## removing irrelevant variables
#             for ii in np.arange(len(out_tmp)):
#                 for var in ['D_Tg','D_Td','D_cveg','D_csoil1','D_Cfroz','D_pthaw','D_Chwp','D_Csoil2_bk','D_csoil2','D_CH4_lag','D_Cveg_bk','D_Aland','D_OHC','D_Xhalo_lag','D_Cthaw','D_Pg','D_Csoil1_bk','D_N2O_lag','D_Cosurf','D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4','D_Ebb','D_Fsink_N2O','D_Ewet','D_Epf_CH4','D_Fsink_CH4'] + ['box_osurf','reg_pf','spc_bb','box_thaw','box_hwp']:
#                     if var in for_tmp[ii]:# variables added because of compatible emissions
#                         for_tmp[ii] = for_tmp[ii].drop( var )
#                     if ('D_CO2' in for_tmp[ii])  and  (xp[0][:len('esm-')]=='esm-'):## add D_CO2 to for_tmp[ii] when calculate compatible emissions. Problem with esm-ssp370-lowNTCF-gidden.
#                         for_tmp[ii] = for_tmp[ii].drop( 'D_CO2' )
#         for_tmp0 = xr.concat( for_tmp , dim='year' )# no problem anymore
#         for ii in np.arange(len(for_tmp)):
#             for_tmp[ii].close()
#         del for_tmp
#         ## correction: some emissions are not perfectly defined (esm-rcp26 after 2100), with NaN out of reg_land=0
#         # for_tmp0 = for_tmp0.fillna( 0. )
#         ## Concatening outputs: some variables have been added to outputs while computing compatible emissions, but not the sames. (+could not use the 'data_vars' option of 'xr.concat' to directly concatenate datasets)
#         out_tmp0 = xr.Dataset()
#         for var in (list(OSCAR_landC.var_prog))*(type_OSCAR=='OSCAR_landC') + (list(OSCAR.var_prog))*(type_OSCAR=='OSCAR'):
#             out_tmp0[var] = xr.concat( [out_tmp[ii][var] for ii in np.arange(len(out_tmp))] , dim='year' )
#         for ii in np.arange(len(out_tmp)):
#             out_tmp[ii].close()
#         del out_tmp
#         ## Concatening masks for divergence
#         mask_xp = np.concatenate( mask_xp )
#         ## loading control
#         if xp[0] in dico_Xp_Control.keys():
#             ## the experiment is a control. The transformation of variables has to be run in 'experiment only' mode.
#             option_DiffControl = False
#             xp_c = None
#         else:
#             ## the experiment is not a control. The transformation of variables has to be run in 'experiment-control' monde
#             option_DiffControl = True
#             xp_c = [xp_c for xp_c in dico_Xp_Control if xp[0] in dico_Xp_Control[xp_c]][0]
#             out_ctrl = xr.open_dataset(path_runs+'/'+xp_c+'_Out-'+str(setMC)+'.nc' )
#             for_ctrl = xr.open_dataset(path_runs+'/'+xp_c+'_For-'+str(setMC)+'.nc' )
#             ## corrected forcings: if the experiment is concentrations-driven, some compatible experiments may be required.
#             if option_JustAddVar==False:for_ctrl = eval_compat_emi( dico_compatible_emissions[xp_c], out_ctrl,Par,for_ctrl,type_OSCAR=type_OSCAR )
#             if threshold_select_ProbableRuns!=None:
#                 out_ctrl = out_ctrl.sel(config=ind_threshold_weights)
#                 for_ctrl = for_ctrl.sel(config=ind_threshold_weights)
#         ##-----
#         ##-----


#         ##-----
#         ## CORRECTIONS
#         ##-----
#         ## differences to control
#         if option_DiffControl:
#             for_tmp = for_tmp0 ## must substract compatible emissions ONLY for those that have been calculated
#             if out_tmp0.year.isel(year=-1)  >  out_ctrl.year.isel(year=-1):## the Control is too short (1000 years, while some experiments last more than that)
#                 yr_xp0_start, yr_pi_cut, yr_xp0_end = out_tmp0.year.isel(year=0), out_ctrl.year.isel(year=-1), out_tmp0.year.isel(year=-1)
#                 out_tmp = xr.Dataset( coords=out_tmp0.coords )
#                 for VAR in out_tmp0.variables:
#                     if (VAR not in out_tmp0.coords):
#                         out_tmp[VAR] = xr.DataArray( np.full(fill_value=np.nan,shape=out_tmp0[VAR].shape) , dims=out_tmp0[VAR].dims  )
#                         out_tmp[VAR].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}]   =  out_tmp0[VAR].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}] - out_ctrl[VAR].sel(year=np.arange(yr_xp0_start,yr_pi_cut+1))
#                         ## using average of the last 10 years of control
#                         out_tmp[VAR].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}]   =  out_tmp0[VAR].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}] - out_ctrl[VAR].sel(year=np.arange(yr_pi_cut-10+1,yr_pi_cut+1)).mean('year')
#                 ## substracting compatible emissions from control
#                 for var in dico_compatible_emissions[xp[0]]:
#                     for_tmp[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}]  =  for_tmp0[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}] - for_ctrl[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]].sel(year=np.arange(yr_xp0_start,yr_pi_cut+1))
#                     for_tmp[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}]  =  for_tmp0[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}] - for_ctrl[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]].sel(year=np.arange(yr_pi_cut-50+1,yr_pi_cut+1)).mean('year')
#             else:## the Control last longer than the experiment (most experiments)
#                 out_tmp  =  out_tmp0 - out_ctrl.sel(year=out_tmp0.year)
#                 ## substracting compatible emissions from control
#                 for var in dico_compatible_emissions[xp[0]]:
#                     for_tmp[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]]  =  for_tmp0[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]] - for_ctrl[{'CO2':'Eff','CH4':'E_CH4','N2O':'E_N2O','Xhalo':'E_Xhalo'}[var]].sel(year=out_tmp0.year)
#         else:
#             for_tmp = for_tmp0
#             out_tmp = out_tmp0
#         if option_DiffControl:
#             out_ctrl.close()
#             for_ctrl.close()
#             del out_ctrl,for_ctrl
#         ## cleaning
#         out_tmp0.close()
#         for_tmp0.close()
#         del out_tmp0,for_tmp0
#         ## preparing the max_yr_brch
#         if option_breach: max_yr_brch = int(np.max([max_yr_brch,for_tmp.year_breach.max()]))
#         if option_breach and (for_tmp.year.isel(year=0)!=1850): raise Exception("not all breach experiment starts in 1850..?")
#         ## correction of the mask: need to be shifted if breached experiment
#         if option_breach:
#             for i_cfg in np.arange(out_tmp.config.size):
#                 if ~np.isnan(for_tmp.year_breach.isel(year=0,config=i_cfg)):
#                     yy1 = int(for_tmp.year.isel(year=-1))
#                     yy2 = int(for_tmp.year_breach.isel(year=0,config=i_cfg))
#                     mask_xp[:yy1-yy2+1,i_cfg] = mask_xp[ yy2-1850: ,i_cfg]
#         ##-----
#         ##-----


#         ##-----
#         ## ADDING REQUIRED VARIABLES, MATCHING TO MIP VARIABLES, ALLOCATING
#         ##-----
#         ## Calculating some basis variables, not to recalculate some of them several times
#         if (type_OSCAR!='OSCAR_landC') and (option_JustAddVar==False):
#             out_tmp = func_add_var(out_tmp, Par, for_tmp , list_var_required=list_VAR_accelerate , type_OSCAR=type_OSCAR)
#         ## starting with year to define this coordinate
#         if 'year' not in TMP_xp:
#             if option_breach:
#                 TMP_xp.coords['year'] = np.arange( out_tmp.year.size )
#             else:
#                 TMP_xp.coords['year'] = out_tmp.year
#         if option_SumReg==False  and  'reg_land' not in TMP_xp: TMP_xp.coords['reg_land'] = np.arange( out_tmp.reg_land.size )## /!\ WARNING: THE REGION 'UNKNOWN' OF OSCAR WILL BE CHANGED TO A REGION 'WORLD'
#         ## check for year_breach
#         if ('year_breach' in list_variables_required) and (xp[0] not in ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC']):
#             list_variables_required.remove('year_breach')
#         ## looping on CMIP6 variables
#         for VAR in list_variables_required:
#             ## correction of the dictionnary of required variables ONLY for landC experiments
#             if (xp[0][:len('land-')] == 'land-')  and  (VAR == 'pr'):
#                 tmp = {VAR:['D_Pl']}
#             elif (xp[0][:len('land-')] == 'land-')  and  (VAR == 'tas'):
#                 tmp = {VAR:['D_Tl']}
#             else:
#                 tmp = {VAR:dico_var[VAR]}
#             #tmp = dico_var
#             ## calculating variables required
#             do_the_transform = True
#             for var in tmp[VAR]:
#                 if var not in out_tmp:
#                     if (type_OSCAR=='OSCAR_landC') and (var in OSCAR_landC._processes)   or   (type_OSCAR=='OSCAR'):
#                         out_tmp = func_add_var(out_tmp, Par, for_tmp , list_var_required=[var] , type_OSCAR=type_OSCAR)
#                     else:
#                         do_the_transform = False ## dont do it because variable cant be calculated (eg ocean variables with OSCAR_landC)
#             if do_the_transform:
#                 ## calculating CMIP6 variable
#                 val = varOSCAR_to_varMIP( VAR, out_tmp, Par , for_tmp , dico_var=tmp , option_SumReg=option_SumReg , option_DiffControl=option_DiffControl , type_OSCAR=type_OSCAR)
#                 if option_breach and ('config' in val.dims):
#                     val2 = xr.DataArray( np.full(fill_value=np.nan,shape=(val.year.size,val.config.size)) , dims=('year','config') , coords=[np.arange(out_tmp.year.size),val.config] )
#                     for cfg in val2.config:
#                         if ~np.isnan(for_tmp.year_breach.isel(year=0,config=cfg)):
#                             val2.loc[{'config':cfg,'year':np.arange(for_tmp.year.isel(year=-1)-for_tmp.year_breach.isel(year=0,config=cfg)+1)}] =  val.sel(year=np.arange(for_tmp.year_breach.isel(year=0,config=cfg),for_tmp.year.isel(year=-1)+1),config=cfg).values
#                             val2.loc[{'config':cfg,'year':np.arange(for_tmp.year.isel(year=-1)-for_tmp.year_breach.isel(year=0,config=cfg)+1)}] -= val.sel(year=for_tmp.year_breach.isel(year=0,config=cfg),config=cfg).values
#                     val = val2
#                     del val2
#                 ## avoiding problem of transposition
#                 # val = val.transpose( tuple(['year'] + ('reg_land' in val.dims)*['reg_land'] + ('config' in val.dims)*['config']) ) ## not working because need hashable
#                 if ('reg_land' in val.dims):
#                     if ('config' in val.dims):
#                         val = val.transpose( 'year','reg_land','config' )
#                     else:
#                         val = val.transpose( 'year','reg_land' )
#                 else:
#                     if ('config' in val.dims):
#                         val = val.transpose( 'year','config' )
#                 ## correction of the region 0: UNKNOWN  TO  WORLD
#                 if ('reg_land' in val.dims):
#                     if  (VAR not in ['tas','pr','Surface Air Temperature Change']):val.loc[{'reg_land':0}] = val.sum('reg_land')

#                 ## defining in TMP_xp
#                 if VAR not in TMP_xp:
#                     if (option_SumReg==False and 'reg_land' in val.dims):
#                         TMP_xp[VAR] = xr.DataArray(    np.full(fill_value=np.nan,shape=[TMP_xp.year.size] + [TMP_xp.reg_land.size] + ('config' in val.dims)*[TMP_xp.all_config.size] ) ,\
#                                                     dims=['year'] + ['reg_land'] + ('config' in val.dims)*['all_config'] )
#                     else:
#                         TMP_xp[VAR] = xr.DataArray(    np.full(fill_value=np.nan,shape=[TMP_xp.year.size] + ('config' in val.dims)*[TMP_xp.all_config.size] ) ,\
#                                                     dims=['year'] + ('config' in val.dims)*['all_config'] )
#                 if ('config' not in val.dims)  and (VAR != 'year'): ## case for forcings: does not depend on config or setMC, allocated directly
#                     TMP_xp[VAR].loc[{ 'year':TMP_xp.year }] = val.values ## need to force years, given the situation of breached outputs
#                 ## allocating
#                 if ('config' in val.dims):
#                     if option_SumReg==False and 'reg_land' in val.dims:
#                         TMP_xp[VAR].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config] }] = val.values * np.repeat(mask_xp[:,np.newaxis,:],6,axis=1)                        
#                     else:
#                         TMP_xp[VAR].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC]) if str(setMC)+'-'+str(cfg) in TMP_xp.all_config] }] = val.values * mask_xp
#         ## end loop on variables
#         ## cleaning
#         out_tmp.close()
#         for_tmp.close()
#         del out_tmp,for_tmp,Par
#         ##-----
#         ##-----
#     ## end loop on sets
#     ## cutting breached dataset, to avoid the period with only NaNs
#     if option_breach:TMP_xp = TMP_xp.sel(year=np.arange( TMP_xp.year.size-(max_yr_brch-1850)-1 ))## all breach experiment start in 1850. has a check along.
#     return TMP_xp,xp_c
#-------------------------------------------------
#-------------------------------------------------




##################################################
##################################################





















 
