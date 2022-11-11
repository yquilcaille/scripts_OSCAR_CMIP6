import os
import csv
import math
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
import matplotlib.pyplot as plt
import time
import pandas as pd

from scipy.optimize import fmin

import sys
sys.path.append("H:/MyDocuments/Repositories/OSCARv31_CMIP6") ## line required for run on server ebro
from core_fct.fct_process import OSCAR
from run_scripts.RCMIP_phase2.weighted_quantile import weighted_quantile


# 'C:\Users\quilcail\AppData\Roaming\Python\Python37\Scripts' #rcmip.exe in this folder. Had to do a pip --user.
import pyrcmip as pyr
import scmdata as scm

##################################################
##################################################

## info
folder_raw = 'results/CMIP6_v3.1/'
folder_extra = 'results/CMIP6_v3.1_extra/'
folder_rcmip = 'results/RCMIP_phase2/'
folder_interm = 'intermediary' ## not intermediary...

## 
option_mask = 'mask_unique' ## mask_all | mask_select | mask_indiv | mask_unique






##################################################
##   LOADING FUNCTIONS
##################################################
def load_Par(Nset=20):
    Par = xr.open_mfdataset([folder_raw + 'Par-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    Par = Par.assign_coords(config=np.arange(len(Par.config)))
    return Par


def load_For(exp, Nset=20):
    For = xr.open_mfdataset([folder_raw + exp + '_For-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    For = For.assign_coords(config=np.arange(len(For.config)))
    For = For.transpose(*(['year'] + [dim for dim in list(For.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
    return For


def load_Out(exp, Nset=20):
    Out = xr.open_mfdataset([folder_raw + exp + '_Out-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    Out = Out.assign_coords(config=np.arange(len(Out.config)))
    Out = Out.transpose(*(['year'] + [dim for dim in list(Out.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
    Out2 = xr.open_mfdataset([folder_extra + exp + '_Out2-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    Out2 = Out2.assign_coords(config=np.arange(len(Out2.config)))
    Out2 = Out2.transpose(*(['year'] + [dim for dim in list(Out2.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
    return xr.merge([Out, Out2])


def load_mask(exp, Nset=20):
    for n in range(Nset):
        with open(folder_raw + 'treated/masks/masknoDV_' + exp + '_' + str(n) + '.csv', 'r') as f: 
            TMP = np.array([line for line in csv.reader(f)], dtype=float)
        if n==0: mask = TMP.copy()
        else: mask = np.append(mask, TMP, axis=1)
    mask = xr.DataArray(mask, coords={'year': 1850 + np.arange(len(mask)), 'config': np.arange(len(mask.T))}, dims=['year', 'config'])
    return mask#.notnull()


def get_var(var, exp):
    return OSCAR[var](load_Out(exp), load_Par(), load_For(exp), recursive=True)


def load_ALL(scen, Nset=20):
    ## getting results from OSCAR
    if ('ssp' in scen):
        Var0 = load_Out(  dico_experiments_before[scen], Nset=Nset  ).sel(config=configs)
        Var1 = load_Out(scen, Nset=Nset).sel(config=configs)
        if scen in ['ssp534-over','esm-ssp534-over']:
            Var2 = load_Out(scen+'-ext', Nset=Nset).sel(config=configs)## extension required!!
            mask_tmp = mask_all_exp[scen+'-ext']
        else:
            Var2 = load_Out(scen+'ext', Nset=Nset).sel(config=configs)## extension required!!
            mask_tmp = mask_all_exp[scen+'ext']
        Var = xr.concat([Var0, Var1.sel(year=slice(Var0.year[-1].values + 1, None, None)), Var2.sel(year=slice(Var1.year[-1].values + 1, None, None))], dim='year')
        del Var0, Var1, Var2
        For0 = load_For(  dico_experiments_before[scen], Nset=Nset  ).sel(config=configs)
        For1 = load_For(scen, Nset=Nset).sel(config=configs)
        if scen in ['ssp534-over','esm-ssp534-over']:
            For2 = load_For(scen+'-ext', Nset=Nset).sel(config=configs)
        else:
            For2 = load_For(scen+'ext', Nset=Nset).sel(config=configs)
        For = xr.concat([For0, For1.sel(year=slice(For0.year[-1].values + 1, None, None)), For2.sel(year=slice(For1.year[-1].values + 1, None, None))], dim='year')
        del For0, For1, For2

    elif ('rcp' in scen):
        Var0 = load_Out(  dico_experiments_before[scen], Nset=Nset  ).sel(config=configs)
        Var1 = load_Out(scen, Nset=Nset).sel(config=configs)
        Var = xr.concat([Var0, Var1.sel(year=slice(Var0.year[-1].values + 1, None, None))], dim='year')
        mask_tmp = mask_all_exp[scen]
        del Var0, Var1
        For0 = load_For(  dico_experiments_before[scen], Nset=Nset  ).sel(config=configs)
        For1 = load_For(scen, Nset=Nset).sel(config=configs)
        For = xr.concat([For0, For1.sel(year=slice(For0.year[-1].values + 1, None, None))], dim='year')
        del For0, For1
    else:
        Var = load_Out(scen, Nset=Nset).sel(config=configs)
        For = load_For(scen, Nset=Nset).sel(config=configs)
        mask_tmp = mask_all_exp[scen]
    ## Parameters
    Par = load_Par(Nset=Nset).sel(config=configs)

    ## ADAPTING parameters: Aland_0
    if 'Aland_0' in For:Par['Aland_0'] = For['Aland_0']
    ## ADAPTING parameters: Aland_0
    if scen in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc']:
        with xr.open_dataset(folder_raw + 'historical' + '_For-' + str(nset) + '.nc') as TMP: for_tmp = TMP.load()
        Par['D_CO2_rad'] = for_tmp.D_CO2.sel(year=1850)
    elif scen in ['1pctCO2-rad']:
        with xr.open_dataset(folder_raw + 'historical' + '_For-' + str(nset) + '.nc') as TMP: for_tmp = TMP.load()
        Par['D_CO2_bgc'] = for_tmp.D_CO2.sel(year=1850)
    ## ADAPTING lengths of experiments
    if scen in ['1pctCO2','1pctCO2-bgc','1pctCO2-cdr','1pctCO2-rad','G2', 'esm-1pctCO2']:
        Var = Var.isel(year=slice(0,150+1))
        For = For.isel(year=slice(0,150+1))

    ## correct drift using control
    Var0 = load_Out(  dico_controls[scen]  ).sel(config=configs)
    Var = Var - Var0 + Var0.mean('year')
    del Var0

    ## masking
    For *= mask_tmp
    Var *= mask_tmp

    ## computing
    print('Computing dasks')
    Var_calc = xr.merge( [Var[vv].compute() for vv in vars_oscar_required if vv in Var] )
    For = xr.merge( [For[vv].compute() for vv in vars_oscar_required if vv in For] )
    Par = Par.compute()

    return Var_calc, For, Par


def var_OSCAR_to_RCMIP(Out, Par , For, scen, vars_required):
    OUT = xr.Dataset()
    for var in vars_required:
        if var == 'Surface Air Temperature Change':
            OUT[var] = Out['D_Tg']

        elif var == 'Surface Air Ocean Blended Temperature Change':
            OUT[var] = Out['D_Tg']  *  0.75/0.89 # cf IPCC SR1.5C, Ch1, table 1.1

        elif var == 'Heat Uptake':
            a_conv = 3600*24*365.25 / 1E21 # from {W yr} to {ZJ}
            A_Earth = 510072E9 # m^2
            OUT[var] = a_conv * A_Earth * (Out['RF'] - Out['D_Tg'] / Par.lambda_0)

        elif var == 'Heat Uptake|Ocean':
            a_conv = 3600*24*365.25 / 1E21 # from {W yr} to {ZJ}
            A_Earth = 510072E9 # m^2
            OUT[var] = a_conv * A_Earth * Par.p_ohc * (Out['RF'] - Out['D_Tg'] / Par.lambda_0)

        elif var == 'Heat Content|Ocean':
            OUT[var] = Out['D_OHC'] * 1.

        elif var == 'Effective Climate Feedback':
            OUT[var] = 1/Par.lambda_0 * xr.ones_like( other=Out['D_Tg'] )

        elif 'Effective Radiative Forcing' in var:
            if var == 'Effective Radiative Forcing':
                OUT[var] = Out['RF_warm']
            else:
                rf = str.split(var,'|')[2]## /!\ |Anthropogenic?
                dico_rf = { 'CO2':'RF_CO2', 'CH4':'RF_CH4', 'N2O':'RF_N2O', 'Aerosols':'RF_AERtot', 'Tropospheric Ozone':'RF_O3t', 'Stratospheric Ozone':'RF_O3s' }
                if rf in dico_rf.keys():
                    OUT[var] = Out[ dico_rf[rf] ]
                else:
                    if rf == 'F-Gases':
                        set_spc = [spc for spc in Par.spc_halo.values if spc not in Par.p_fracrel.dropna('spc_halo').spc_halo]
                    elif rf == 'Montreal Gases':
                        set_spc = [spc for spc in Par.spc_halo.values if spc in Par.p_fracrel.dropna('spc_halo').spc_halo]
                    OUT[var] = Out['RF_Xhalo'].sel( spc_halo=[cp for cp in Out.spc_halo.values if cp in set_spc] ).sum('spc_halo')

        elif var in ['Atmospheric Concentrations|CO2', 'Carbon Pool|Atmosphere']:
            OUT[var] = (Out['D_CO2'] + Par.CO2_0) * ((var=='Carbon Pool|Atmosphere') * Par.a_CO2 + (var=='Atmospheric Concentrations|CO2'))

        elif var == 'Emissions|CO2':
            if scen[len('esm-')] == 'esm-':
                OUT[var] = For['Eff'].sum('reg_land')
            else:
                OUT[var] = Out['Eff_comp']

        elif var == 'Net Land to Atmosphere Flux|CO2':
            OUT[var] = -1. * (Out['D_Fland']  - Out['D_Eluc']  - Out['D_Epf_CO2'].sum('reg_pf')  - 1.e-3 * Out['D_Epf_CH4'].sum('reg_pf')) ## assuming directly oxidation of CH4

        elif var == 'Net Ocean to Atmosphere Flux|CO2':
            OUT[var] = -1. * Out['D_Focean']

    return OUT


def configs_to_quantiles( INPUT, quantiles, axis_along='year', option_with_mean=False  ):
    OUTPUT = xr.Dataset()
    OUTPUT.coords[axis_along] = INPUT[axis_along]
    OUTPUT.coords['ensemble_member'] = option_with_mean*['mean'] + list(quantiles)
    ## correction for all variables, used for the NaN in abrupt-0p5CO2
    if 'Emissions|CO2' in INPUT:
        tmp = list( set(  np.where(np.isnan(INPUT['Emissions|CO2']))[list(INPUT['Emissions|CO2'].dims).index('config')]  ) )
        cfgs = [INPUT.config.values[i_cfg] for i_cfg in np.arange(INPUT.config.size) if i_cfg not in tmp ]
        INPUT = INPUT.sel(config=cfgs)
        weights = WEIGHTS.weights.sel(config=cfgs)
    else:
        weights = WEIGHTS.weights
    ## looping on variables
    for var in INPUT.variables:
        if var not in INPUT.coords:
            print('Distribution: '+var)
            ## init
            OUTPUT[var] = np.nan * xr.DataArray( np.zeros((OUTPUT[axis_along].size,OUTPUT.ensemble_member.size)), dims=(axis_along,'ensemble_member') )
            ## filling
            for tt in INPUT[axis_along]:
                if np.all(np.isnan(INPUT[var].loc[{axis_along:tt}].values)): ## case for Eff
                    pass
                elif np.all(  INPUT[var].loc[{axis_along:tt}].values == INPUT[var].loc[{axis_along:tt}].values[0]  ): ## case for atm CO2 in concentration driven runs
                    OUTPUT[var].loc[{'ensemble_member':[str(qq) for qq in quantiles], axis_along:tt}] = INPUT[var].loc[{axis_along:tt}].values[0]
                else:
                    cfgs = np.where(np.isnan(INPUT[var].loc[{axis_along:tt}].values)==False)[0]
                    # cfgs = range( INPUT.config.size )
                    hist,edges = np.histogram( a=INPUT[var].loc[{axis_along:tt}].isel(config=cfgs).values, bins=len(cfgs), weights=weights.isel(config=cfgs).values, density=True )
                    hist /= np.sum( hist * np.diff(edges) )
                    cumhist = np.cumsum( hist * np.diff(edges) )
                    val = np.interp(x=quantiles, xp=cumhist, fp=0.5*(edges[1:]+edges[:-1]) )
                    ## plot of cumulative distribution
                    # plt.plot( 0.5*(edges[1:]+edges[:-1]) , cumhist )
                    # plt.plot( val , quantiles , ls='--')
                    ## plot of distributions (effect of nbins, then density)
                    # plt.plot( 0.5*(edges[1:]+edges[:-1]) , hist / np.max(hist) )
                    # plt.plot( 0.5*(val[1:]+val[:-1]) , np.diff(quantiles) / np.diff(val)  /  np.max(np.diff(quantiles) / np.diff(val)) , ls='--')
                    OUTPUT[var].loc[{'ensemble_member':[str(qq) for qq in quantiles], axis_along:tt}] = val
            ## average
            if option_with_mean:
                mm = np.average(a=INPUT[var].values, weights=weights.values, axis=list(INPUT[var].dims).index(axis_along))
                OUTPUT[var].loc[{'ensemble_member':'mean'}] = mm
                # plt.plot( INPUT[var].year , INPUT[var], color='grey' )
                # plt.plot( INPUT[var].year , mm, color='k' )
    return OUTPUT


## function for scm timeseries
timeserie_scm = lambda vars_oscar_rcmip_distrib,var,scen,ind:scm.run.ScmRun(    data=vars_oscar_rcmip_distrib[var].sel(ensemble_member=ind),
                                                                                columns={   "model":dico_IAM_scen[scen], 
                                                                                            "scenario":dico_name_scen[scen], 
                                                                                            "climate_model":"OSCARv3.1",
                                                                                            "variable":var, 
                                                                                            "region":'World', 
                                                                                            "unit":dico_units[var], 
                                                                                            "ensemble_member":ind }, 
                                                                                            #"run_id":list_scen.index(scen)*len(vars_rcmip_required)*len(quantiles) + vars_rcmip_required.index(var)*len(quantiles) + list(quantiles).index(ind) }, 
                                                                                index=vars_oscar_rcmip_distrib.year.values 
                                                                            )
##################################################
##################################################





##################################################
##   ADDING INFORMATIONS ON EXPERIMENTS
##################################################
list_experiments = set([zou.split('_')[1] for zou in os.listdir(folder_raw + 'treated/masks/') if '.csv' in zou])

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
dico_Xp_Control = { 'piControl':['1pctCO2-4xext', '1pctCO2-bgc', '1pctCO2-cdr', '1pctCO2-rad', '1pctCO2', 'abrupt-0p5xCO2', 'abrupt-2xCO2', 'abrupt-4xCO2', 'G1', 'G2', 'G6solar', 'hist-1950HC', 'hist-aer', 'hist-bgc', 'hist-CO2', 'hist-GHG', 'hist-nat', 'hist-piAer', 'hist-piNTCF', 'hist-sol', 'hist-stratO3', 'hist-volc', 'historical', 'hist-noLu', 'ssp119', 'ssp119ext', 'ssp126-ssp370Lu', 'ssp126', 'ssp126ext', 'ssp245-aer', 'ssp245-CO2', 'ssp245-GHG', 'ssp245-nat', 'ssp245-sol', 'ssp245-stratO3', 'ssp245-volc', 'ssp245', 'ssp245ext', 'ssp370-lowNTCF', 'ssp370-lowNTCFext', 'ssp370-lowNTCF-gidden', 'ssp370-lowNTCFext-gidden', 'ssp370-ssp126Lu', 'ssp370', 'ssp370ext', 'ssp434', 'ssp434ext', 'ssp460', 'ssp460ext', 'ssp534-over-bgc', 'ssp534-over-bgcExt', 'ssp534-over-ext', 'ssp534-over', 'ssp585-bgc', 'ssp585-bgcExt', 'ssp585-ssp126Lu', 'ssp585', 'ssp585ext', 'yr2010CO2'] ,
                    'esm-piControl':['esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-1pct-brch-750PgC', 'esm-1pctCO2', 'esm-abrupt-4xCO2', 'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC', 'esm-hist', 'esm-pi-cdr-pulse', 'esm-pi-CO2pulse', 'esm-ssp119', 'esm-ssp119ext', 'esm-ssp126', 'esm-ssp126ext', 'esm-ssp245', 'esm-ssp245ext', 'esm-ssp370-lowNTCF', 'esm-ssp370-lowNTCFext', 'esm-ssp370-lowNTCF-gidden', 'esm-ssp370-lowNTCFext-gidden', 'esm-ssp370', 'esm-ssp370ext', 'esm-ssp460', 'esm-ssp460ext', 'esm-ssp434', 'esm-ssp434ext', 'esm-ssp534-over-ext', 'esm-ssp534-over', 'esm-ssp585-ssp126Lu-ext', 'esm-ssp585-ssp126Lu', 'esm-ssp585', 'esm-ssp585ext', 'esm-yr2010CO2-cdr-pulse', 'esm-yr2010CO2-CO2pulse', 'esm-yr2010CO2-control', 'esm-yr2010CO2-noemit'],
                    'esm-piControl-CMIP5':['esm-histcmip5','esm-rcp26', 'esm-rcp45', 'esm-rcp60', 'esm-rcp85'],
                    'piControl-CMIP5':['historical-CMIP5', 'rcp26', 'rcp45', 'rcp60', 'rcp85'],
                    'land-piControl':['land-cClim', 'land-cCO2', 'land-crop-grass', 'land-hist', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv'],
                    'land-piControl-altLu1':['land-hist-altLu1'],
                    'land-piControl-altLu2':['land-hist-altLu2'],
                    'land-piControl-altStartYear':['land-hist-altStartYear'],
                    'spinup':[],'esm-spinup':[],'esm-spinup-CMIP5':[],'spinup-CMIP5':[],'land-spinup':[],'land-spinup-altLu1':[],'land-spinup-altLu2':[],'land-spinup-altStartYear':[] }
dico_controls = dict(  [(vv,kk) for kk in dico_Xp_Control.keys() for vv in dico_Xp_Control[kk]] )
##################################################
##################################################







##################################################
##   GET MINIMAL MASK
##################################################
if option_mask=='mask_all':
    mask_all = xr.open_dataarray(folder_rcmip + 'mask_all_exp.nc')
    mask_all_exp = {}
    for exp in list_experiments:mask_all_exp[exp] = mask_all

elif option_mask=='mask_select':
    mask_all = xr.open_dataarray(folder_rcmip + 'mask_sel_exp.nc')
    mask_all_exp = {}
    for exp in list_experiments:mask_all_exp[exp] = mask_all

elif option_mask=='mask_indiv':
    ## loading masks for every experiment. Making an array "Year x Config x Scen" takes too much memory, instead we make within a dataset arrays for each experiment
    mask_all_exp = {}
    ## looping on experiments
    for exp in list_experiments:
        print("loading masks: " + str(np.round(100.*(list(list_experiments).index(exp)+1)/len(list_experiments)))+'%',end='\r' )
        masks = []
        masks.append( load_mask(exp) )
        ## adding previous ones for continuity
        ss = dico_experiments_before[exp]
        while ss != None:
            masks.append(  load_mask(ss)  )
            ss = dico_experiments_before[ss]
        ## correcting mistake on years in function load_mask
        tmp = np.concatenate( [masks[-1].values[0,np.newaxis,:]] + [np.array(mm.values[1:,:],dtype=np.float32) for mm in masks[::-1]] )
        ## setting as NaN rather than False
        tmp[np.where(tmp==0.)] = np.nan
        ## creating the array
        mask_all_exp[exp] = xr.DataArray( tmp, coords={'year':1850+np.arange(len(tmp)), 'config':np.arange(len(tmp.T)) }, dims=['year', 'config'])

elif option_mask=='mask_unique':
    # nan much better than True / False to multiply directly variables.
    # mask_all = load_mask('piControl').all('year')
    mask_all = load_mask('piControl').prod('year')
    for setMC in range(20):
        ## loading unique mask for this set
        with open(folder_raw+'/treated/masks/mask_all_exp_'+str(setMC)+'.csv','r',newline='') as ff:
            mask_all.loc[{ 'config':setMC*500 + np.where( np.isnan(np.array([line for line in csv.reader(ff)] ,dtype=np.float32)[:,0]) )[0] }] = np.nan # nan much better than True / False to multiply directly variables.
    mask_all_exp = {}
    for exp in list_experiments:mask_all_exp[exp] = mask_all

else:
    raise Exception('Option for masks not known.')
##################################################
##################################################







##################################################
##   PREPARING RCMIP FORMAT
##################################################
## scenarios used
list_scen = ['1pctCO2', 'abrupt-2xCO2', 'abrupt-4xCO2', 'abrupt-0p5xCO2'] + \
                ['historical', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585' ] + \
                ['esm-hist'] + \
                ['historical-CMIP5', 'rcp26', 'rcp45', 'rcp60', 'rcp85'] + \
                ['esm-ssp119', 'esm-ssp126', 'esm-ssp245', 'esm-ssp370', 'esm-ssp434', 'esm-ssp460', 'esm-ssp534-over', 'esm-ssp585'] +\
                ['1pctCO2-cdr', 'esm-1pctCO2', 'esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-pi-CO2pulse', 'esm-pi-cdr-pulse']

# list_scen = ['1pctCO2-cdr', 'esm-1pctCO2', 'esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-pi-CO2pulse', 'esm-pi-cdr-pulse']
# list_scen = ['esm-ssp119', 'esm-ssp126', 'esm-ssp245', 'esm-ssp370', 'esm-ssp434', 'esm-ssp460', 'esm-ssp534-over', 'esm-ssp585']


dico_IAM_scen = {   '1pctCO2':'idealized',
                    'abrupt-2xCO2':'idealized', 'abrupt-4xCO2':'idealized', 'abrupt-0p5xCO2':'idealized',
                    '1pctCO2-cdr':'idealized', 'esm-1pctCO2':'idealized', 'esm-1pct-brch-750PgC':'idealized', 'esm-1pct-brch-1000PgC':'idealized', 'esm-1pct-brch-2000PgC':'idealized', 'esm-pi-CO2pulse':'idealized', 'esm-pi-cdr-pulse':'idealized',
                    'historical':'historical_CMIP6',
                    'ssp119':'IMAGE','esm-ssp119':'IMAGE',
                    'ssp126':'IMAGE','esm-ssp126':'IMAGE',
                    'ssp245':'MESSAGE_GLOBIOM','esm-ssp245':'MESSAGE_GLOBIOM',
                    'ssp370':'AIM_CGE','esm-ssp370':'AIM_CGE',
                    'ssp434':'GCAM4','esm-ssp434':'GCAM4',
                    'ssp460':'GCAM4','esm-ssp460':'GCAM4',
                    'ssp534-over':'REMIND_MAGPIE','esm-ssp534-over':'REMIND_MAGPIE',
                    'ssp585':'REMIND_MAGPIE','esm-ssp585':'REMIND_MAGPIE',
                    'esm-hist':'esm_historical_CMIP5',
                    'historical-CMIP5':'historical_CMIP5',
                    'rcp26':'IMAGE',
                    'rcp45':'MiniCAM',
                    'rcp60':'AIM',
                    'rcp85':'MESSAGE'}

dico_name_scen = {scen:scen for scen in list_scen}
dico_name_scen['historical-CMIP5'] = 'historical-cmip5'


## variables that will be saved in RCMIP files
vars_rcmip_required = ['Surface Air Temperature Change',\
                        'Surface Air Ocean Blended Temperature Change',\
                        'Heat Uptake|Ocean', 'Heat Content|Ocean', 'Effective Climate Feedback',\
                        'Effective Radiative Forcing',\
                        'Effective Radiative Forcing|Anthropogenic|CO2',\
                        'Effective Radiative Forcing|Anthropogenic|CH4',\
                        'Effective Radiative Forcing|Anthropogenic|N2O',\
                        'Effective Radiative Forcing|Anthropogenic|Aerosols',\
                        'Effective Radiative Forcing|Anthropogenic|Montreal Gases',\
                        'Effective Radiative Forcing|Anthropogenic|F-Gases',\
                        'Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone',\
                        'Effective Radiative Forcing|Anthropogenic|Stratospheric Ozone',\
                        'Atmospheric Concentrations|CO2', 'Emissions|CO2', 'Carbon Pool|Atmosphere', 'Net Land to Atmosphere Flux|CO2', 'Net Ocean to Atmosphere Flux|CO2']

## units
dico_units = {  'Surface Air Temperature Change':'K',\
                'Surface Air Ocean Blended Temperature Change':'K',\
                'Heat Uptake':'ZJ/yr', 'Heat Uptake|Ocean':'ZJ/yr', 'Heat Content|Ocean':'ZJ', 'Effective Climate Feedback':'W/m^2/K',\
                'Effective Radiative Forcing':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|CO2':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|CH4':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|N2O':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|Aerosols':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|Montreal Gases':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|F-Gases':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone':'W/m^2',\
                'Effective Radiative Forcing|Anthropogenic|Stratospheric Ozone':'W/m^2',\
                'Atmospheric Concentrations|CO2':'ppm', 'Emissions|CO2':'PgC/yr', 'Carbon Pool|Atmosphere':'PgC', 'Net Land to Atmosphere Flux|CO2':'PgC/yr', 'Net Ocean to Atmosphere Flux|CO2':'PgC/yr' }

## variables that OSCAR will use for their computation
vars_oscar_required = ['D_Tg','D_OHC', 'RF', 'RF_warm', 'RF_CO2', 'RF_CH4', 'RF_N2O', 'RF_AERtot', 'RF_O3t', 'RF_O3s', 'RF_Xhalo', 'D_CO2', 'Eff_comp', 'Eff', 'D_Fland', 'D_Eluc', 'D_Epf_CO2', 'D_Epf_CH4', 'D_Focean']

## configurations kept
# configs = mask_all.where(mask_all).dropna('config').config
configs = mask_all_exp['1pctCO2'].config

## Loading indicators
file_indic = folder_rcmip + 'oscar_indicators_full-configs_'+option_mask+'.nc'
print("Loading "+file_indic)
indic = xr.load_dataset( file_indic )
## correction, some coordinates become variables....
for var in ['RCMIP variable', 'RCMIP region', 'RCMIP scenario']:
    indic.coords[var] = indic[var]
indic = indic.drop('distrib')

# ## indicators to use for weighting OSCAR by Yann Quilcaille for RCMIP-phase 2
# ind_list = [#'Equilibrium Climate Sensitivity',
#             'Cumulative Net Land to Atmosphere Flux|CO2 World esm-hist-2011',
#             'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1980',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1990',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2000',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2002',
#             ]

## indicators to use for weighting OSCAR by Yann Quilcaille for CMIP6
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
WEIGHTS.coords['config'] = mask_all_exp['1pctCO2'].config.sel(config=configs)
val = indic['w'].sel(index=[i for i in indic.index if str(indic.indicator.sel(index=i).values) in ind_list]).prod('index')
WEIGHTS['weights'] = xr.DataArray( data=val , dims=('config') )

## quantiles used
quantiles = np.arange(0.0,1.0+1.e-10,0.001)# np.arange(0.001,0.999+1.e-10,0.001)
##################################################
##################################################



# aa = vars_for_rcmip['Surface Air Temperature Change'].isel(year=-1).values
# bb = vars_for_rcmip['Emissions|CO2'].isel(year=-1).values
# 1396, 7608, 8219
# diff = np.where( ~np.isnan(aa) != ~np.isnan(bb) )


# plt.plot( vars_for_rcmip['year'] , vars_for_rcmip['Net Land to Atmosphere Flux|CO2'].sel(config=4),ls='--', label='ok' )
# plt.plot( vars_for_rcmip['year'] , vars_for_rcmip['Net Land to Atmosphere Flux|CO2'].sel(config=1396),ls='--', label='1396' )
# plt.grid()
# plt.legend()
# plt.show()



##################################################
##   TREATING RCMIP FORMAT FOR TIMESERIES
##################################################
## Variable that will gather all scm timeseries
ensembles_oscar = []
## loop on scenarios
for scen in list_scen:
    print('Working on '+scen)
    if os.path.isfile( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+dico_name_scen[scen]+'.nc' ):
        # OUT = scm.run.ScmRun.from_nc( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+scen+'.nc' ) ## a bit faster, correct format.
        OUT = scm.dataframe.ScmDataFrame.from_nc( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+dico_name_scen[scen]+'.nc' ) ## a bit faster, correct format.
        ensembles_oscar.append( OUT )

    else:
        time0 = time.process_time()
        ## Loading everything required, with required corrections
        Var_calc, For, Par = load_ALL(scen, Nset=20)

        ## transforming to RCMIP variables
        print("Finished loading results. Transforming now.")
        vars_for_rcmip = var_OSCAR_to_RCMIP(Var_calc, Par , For, scen, vars_rcmip_required)

        # ## temporary cut date
        # if scen in ['esm-ssp119','esm-ssp126','esm-ssp434']:
        #     vars_for_rcmip = vars_for_rcmip.sel(year=range(1850,2300))

        ## distribution
        vars_oscar_rcmip_distrib = configs_to_quantiles( INPUT=vars_for_rcmip, quantiles=quantiles, option_with_mean=False  )

        ## Writing in scm format
        if True:## saving using scm.dataframe.ScmDataFrame
            list_df = []
            for qq in quantiles:
                ## prepary data easy to understand for scm.dataframe.ScmDataFrame
                val = vars_oscar_rcmip_distrib.sel(ensemble_member=qq).drop('ensemble_member')
                vars_val = [var for var in val if var not in val.coords]
                ## transformation into panda dataframe
                oscar_as_a_panda = val.to_dataframe()
                # run_id = list( list_scen.index(scen) * len(quantiles) * len(vars_val)  +  list(quantiles).index(qq) * len(vars_val) + np.arange(len(vars_val)) )
                # run_id = [scen+'-'+var+'-'+str(qq) for var in vars_val]
                # list_df.append( scm.dataframe.ScmDataFrame(data=oscar_as_a_panda, index=vars_oscar_rcmip_distrib.year.values, columns={"model":'OSCARv3.1', "scenario":scen, "variable":vars_val, "region":'World', "unit":[dico_units[var] for var in vars_val], "ensemble_member":qq, "run_id":run_id }) )
                list_df.append( scm.dataframe.ScmDataFrame( data=oscar_as_a_panda,
                                                            index=vars_oscar_rcmip_distrib.year.values,
                                                            columns={"model":dico_IAM_scen[scen], "climate_model":"OSCARv3.1", "scenario":dico_name_scen[scen], "variable":vars_val, "region":'World', "unit":[dico_units[var] for var in vars_val], "ensemble_member":qq },
                                                          ) )
            ## concatenating all of them
            OUT = scm.dataframe.df_append( dfs=list_df )
            del list_df

            ## saving intermediary file
            OUT.to_nc( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+dico_name_scen[scen]+'.nc', dimensions=["ensemble_member"])

            ## adding metadata. Not found with dataframe -_-
            TMP = scm.run.ScmRun.from_nc( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+dico_name_scen[scen]+'.nc' ) ## a bit faster, correct format.
            TMP.metadata['info'] = 'Global outputs of OSCARv3.1 for '+dico_name_scen[scen]+', provided for RCMIP phase 2.'
            TMP.metadata['climate_model'] = 'OSCARv3.1'
            TMP.metadata['contact'] = 'Yann Quilcaille (quilcail@iiasa.ac.at) and Thomas Gasser (gasser@iiasa.ac.at)'
            TMP.to_nc( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+dico_name_scen[scen]+'.nc', dimensions=["ensemble_member"])

            ## appending to ensembles_oscar
            ensembles_oscar.append( OUT )

        else:## saving using scm.run
            # Avoid following lines, slower than using pandas and scm.dataframe
            ## creating scm file
            tmp = [timeserie_scm(vars_oscar_rcmip_distrib,var,scen,ind) for var in vars_rcmip_required for ind in quantiles]
            OUT = scm.run.run_append( tmp ) 

            ## saving intermediary file
            OUT.metadata['info'] = 'Global outputs of OSCARv3.1 for '+dico_name_scen[scen]+', provided for RCMIP phase 2.'
            OUT.metadata['climate_model'] = 'OSCARv3.1'
            OUT.metadata['contact'] = 'Yann Quilcaille (quilcail@iiasa.ac.at) and Thomas Gasser (gasser@iiasa.ac.at)'
            OUT.to_nc( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+dico_name_scen[scen]+'.nc', dimensions=["ensemble_member"])

            ## appending to ensembles_oscar
            ensembles_oscar.extend( tmp )

        ## cleaning
        del Var_calc, For, Par, vars_oscar_rcmip_distrib, OUT
        # ## checking in format ScmRun rather than dataframe
        # TMP = scm.run.ScmRun.from_nc( folder_rcmip + folder_interm+'/RCMIP-phase2-OSCARv3-1_'+scen+'.nc' ) ## a bit faster, correct format.

        print( 'Time for '+scen+': '+str(time.process_time() - time0)+'s' )
        print(" ")

## concatenating scm files
# OSCAR_to_SAVE = scm.dataframe.df_append( dfs=ensembles_oscar )
# OSCAR_to_SAVE.to_nc( folder_rcmip + '/RCMIP-phase2-OSCARv3-1.nc', dimensions=["ensemble_member"])

print("***********")
print("Still problems to save all scenarios in one file. 'ensemble_member' not unique. Adding a 'run_id' not varying from scenario to scenario would not change it, for the tried saving dimension was 'ensemble_member'. A 'run_id' increasing over scenarios did not prove successful. To take later.")
print("For now, using intermediary files.")
print("***********")

## saving scm file
# OSCAR_to_SAVE.metadata['info'] = 'Global outputs of OSCARv3.1 for experiments of CMIP6, provided for RCMIP phase 2.'
# OSCAR_to_SAVE.metadata['contact'] = 'Yann Quilcaille (quilcail@iiasa.ac.at) and Thomas Gasser (gasser@iiasa.ac.at)'
# OSCAR_to_SAVE.to_nc(folder_rcmip + 'RCMIP-phase2-OSCARv3-1.nc', dimensions=["ensemble_member"])
##################################################
##################################################








##################################################
##   INDICATORS
##################################################
if True:
    ## distribution
    indic_distrib = configs_to_quantiles( INPUT=indic, quantiles=quantiles, axis_along='index', option_with_mean=False  )

    ## creating normal csv. WITH ONLY ECS!!
    INDIC_CSV = [  ['RCMIP name', 'climate_model', 'unit', 'ensemble_member', 'value']  ]
    for ind in indic_distrib.index.values:
        if indic_distrib['indicator'].sel(index=ind).values in ['Equilibrium Climate Sensitivity']:#,'Transient Climate Response to Emissions']:
            tmp = [ indic_distrib['indicator'].sel(index=ind).values, \
                    'OSCARv3.1',\
                    indic_distrib['unit'].sel(index=ind).values]
            for qq in quantiles:
                INDIC_CSV.append(  tmp + [ list(quantiles).index(qq) , indic_distrib['x'].sel(index=ind,ensemble_member=qq).values ]  ) ## QUANTILE NOT WRITTEN
    ## writing csv
    with open(folder_rcmip + 'RCMIP-phase2-OSCARv3-1-indicators.csv', 'w',newline='') as ff:
        csv.writer(ff).writerows( INDIC_CSV )
##################################################
##################################################











## windows cmd, to run in admin mode
# C:/ProgramData/Anaconda3/Scripts/activate.bat
# pip install --upgrade pip ## no need for that
# pip install pyrcmip==0.3.0

## does not read on P drive, transfering everything on a temporary repository.
# chdir C:/Users/quilcail/Documents/Repositories/temporary/

## checking files:
# rcmip validate RCMIP-phase2-OSCARv3-1_1pctCO2.nc RCMIP-phase2-OSCARv3-1-indicators.csv rcmip_model_metadata_OSCARv3-1.csv

## submission using*.nc, not working cf pb in rcmip upload
## rcmip upload --token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3Nfa2V5X2lkIjoiQUtJQVpQNFFZSkhHQVdOQ1BIT1ciLCJzZWNyZXRfYWNjZXNzX2tleSI6IndNZUF3Q3lndW04U2M2NUZGVS9yWDdYY1A0MktLeHlNU0ZpL0JybHAiLCJvcmciOiJpaWFzYSJ9.HyLnb1Qli80j659dMLSSFRtrItEYyZD-kN64M5D_-KU --model OSCARv3.1 --version 2.0.0 C:/Users/quilcail/Documents/Repositories/temporary/*.nc C:/Users/quilcail/Documents/Repositories/temporary/RCMIP-phase2-OSCARv3-1-indicators.csv C:/Users/quilcail/Documents/Repositories/temporary/rcmip_model_metadata_OSCARv3-1.csv

## submission of previous version, not all experiments
## rcmip upload --token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3Nfa2V5X2lkIjoiQUtJQVpQNFFZSkhHQVdOQ1BIT1ciLCJzZWNyZXRfYWNjZXNzX2tleSI6IndNZUF3Q3lndW04U2M2NUZGVS9yWDdYY1A0MktLeHlNU0ZpL0JybHAiLCJvcmciOiJpaWFzYSJ9.HyLnb1Qli80j659dMLSSFRtrItEYyZD-kN64M5D_-KU --model OSCARv3.1 --version 2.0.1 RCMIP-phase2-OSCARv3-1_1pctCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-0p5xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-2xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-4xCO2.nc RCMIP-phase2-OSCARv3-1_esm-hist.nc RCMIP-phase2-OSCARv3-1_historical.nc RCMIP-phase2-OSCARv3-1_historical-cmip5.nc RCMIP-phase2-OSCARv3-1_rcp26.nc RCMIP-phase2-OSCARv3-1_rcp45.nc RCMIP-phase2-OSCARv3-1_rcp60.nc RCMIP-phase2-OSCARv3-1_rcp85.nc RCMIP-phase2-OSCARv3-1_ssp119.nc RCMIP-phase2-OSCARv3-1_ssp126.nc RCMIP-phase2-OSCARv3-1_ssp245.nc RCMIP-phase2-OSCARv3-1_ssp370.nc RCMIP-phase2-OSCARv3-1_ssp434.nc RCMIP-phase2-OSCARv3-1_ssp460.nc RCMIP-phase2-OSCARv3-1_ssp534-over.nc RCMIP-phase2-OSCARv3-1_ssp585.nc RCMIP-phase2-OSCARv3-1-indicators.csv rcmip_model_metadata_OSCARv3-1.csv

## submission of previous version, all but fullGHG experiments
# rcmip upload --token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3Nfa2V5X2lkIjoiQUtJQVpQNFFZSkhHQVdOQ1BIT1ciLCJzZWNyZXRfYWNjZXNzX2tleSI6IndNZUF3Q3lndW04U2M2NUZGVS9yWDdYY1A0MktLeHlNU0ZpL0JybHAiLCJvcmciOiJpaWFzYSJ9.HyLnb1Qli80j659dMLSSFRtrItEYyZD-kN64M5D_-KU --model OSCARv3.1 --version 2.0.2 RCMIP-phase2-OSCARv3-1_1pctCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-0p5xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-2xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-4xCO2.nc RCMIP-phase2-OSCARv3-1_esm-hist.nc RCMIP-phase2-OSCARv3-1_historical.nc RCMIP-phase2-OSCARv3-1_historical-cmip5.nc RCMIP-phase2-OSCARv3-1_rcp26.nc RCMIP-phase2-OSCARv3-1_rcp45.nc RCMIP-phase2-OSCARv3-1_rcp60.nc RCMIP-phase2-OSCARv3-1_rcp85.nc RCMIP-phase2-OSCARv3-1_ssp119.nc RCMIP-phase2-OSCARv3-1_ssp126.nc RCMIP-phase2-OSCARv3-1_ssp245.nc RCMIP-phase2-OSCARv3-1_ssp370.nc RCMIP-phase2-OSCARv3-1_ssp434.nc RCMIP-phase2-OSCARv3-1_ssp460.nc RCMIP-phase2-OSCARv3-1_ssp534-over.nc RCMIP-phase2-OSCARv3-1_ssp585.nc RCMIP-phase2-OSCARv3-1_esm-ssp119.nc RCMIP-phase2-OSCARv3-1_esm-ssp126.nc RCMIP-phase2-OSCARv3-1_esm-ssp245.nc RCMIP-phase2-OSCARv3-1_esm-ssp370.nc RCMIP-phase2-OSCARv3-1_esm-ssp434.nc RCMIP-phase2-OSCARv3-1_esm-ssp460.nc RCMIP-phase2-OSCARv3-1_esm-ssp534-over.nc RCMIP-phase2-OSCARv3-1_esm-ssp585.nc RCMIP-phase2-OSCARv3-1_1pctCO2-cdr.nc RCMIP-phase2-OSCARv3-1_esm-1pctCO2.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-750PgC.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-1000PgC.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-2000PgC.nc RCMIP-phase2-OSCARv3-1_esm-pi-CO2pulse.nc RCMIP-phase2-OSCARv3-1_esm-pi-cdr-pulse.nc RCMIP-phase2-OSCARv3-1-indicators.csv rcmip_model_metadata_OSCARv3-1.csv

## submission of previous version, all but fullGHG experiments
# rcmip upload --token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3Nfa2V5X2lkIjoiQUtJQVpQNFFZSkhHQVdOQ1BIT1ciLCJzZWNyZXRfYWNjZXNzX2tleSI6IndNZUF3Q3lndW04U2M2NUZGVS9yWDdYY1A0MktLeHlNU0ZpL0JybHAiLCJvcmciOiJpaWFzYSJ9.HyLnb1Qli80j659dMLSSFRtrItEYyZD-kN64M5D_-KU --model OSCARv3.1 --version 2.0.3 RCMIP-phase2-OSCARv3-1_1pctCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-0p5xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-2xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-4xCO2.nc RCMIP-phase2-OSCARv3-1_esm-hist.nc RCMIP-phase2-OSCARv3-1_historical.nc RCMIP-phase2-OSCARv3-1_historical-cmip5.nc RCMIP-phase2-OSCARv3-1_rcp26.nc RCMIP-phase2-OSCARv3-1_rcp45.nc RCMIP-phase2-OSCARv3-1_rcp60.nc RCMIP-phase2-OSCARv3-1_rcp85.nc RCMIP-phase2-OSCARv3-1_ssp119.nc RCMIP-phase2-OSCARv3-1_ssp126.nc RCMIP-phase2-OSCARv3-1_ssp245.nc RCMIP-phase2-OSCARv3-1_ssp370.nc RCMIP-phase2-OSCARv3-1_ssp434.nc RCMIP-phase2-OSCARv3-1_ssp460.nc RCMIP-phase2-OSCARv3-1_ssp534-over.nc RCMIP-phase2-OSCARv3-1_ssp585.nc RCMIP-phase2-OSCARv3-1_esm-ssp119.nc RCMIP-phase2-OSCARv3-1_esm-ssp126.nc RCMIP-phase2-OSCARv3-1_esm-ssp245.nc RCMIP-phase2-OSCARv3-1_esm-ssp370.nc RCMIP-phase2-OSCARv3-1_esm-ssp434.nc RCMIP-phase2-OSCARv3-1_esm-ssp460.nc RCMIP-phase2-OSCARv3-1_esm-ssp534-over.nc RCMIP-phase2-OSCARv3-1_esm-ssp585.nc RCMIP-phase2-OSCARv3-1_1pctCO2-cdr.nc RCMIP-phase2-OSCARv3-1_esm-1pctCO2.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-750PgC.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-1000PgC.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-2000PgC.nc RCMIP-phase2-OSCARv3-1_esm-pi-CO2pulse.nc RCMIP-phase2-OSCARv3-1_esm-pi-cdr-pulse.nc RCMIP-phase2-OSCARv3-1-indicators.csv rcmip_model_metadata_OSCARv3-1.csv

## submission of last version, all but fullGHG experiments, with updated indicators (chg in temperature)
# rcmip upload --token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3Nfa2V5X2lkIjoiQUtJQVpQNFFZSkhHQVdOQ1BIT1ciLCJzZWNyZXRfYWNjZXNzX2tleSI6IndNZUF3Q3lndW04U2M2NUZGVS9yWDdYY1A0MktLeHlNU0ZpL0JybHAiLCJvcmciOiJpaWFzYSJ9.HyLnb1Qli80j659dMLSSFRtrItEYyZD-kN64M5D_-KU --model OSCARv3.1 --version 2.2.0 RCMIP-phase2-OSCARv3-1_1pctCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-0p5xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-2xCO2.nc RCMIP-phase2-OSCARv3-1_abrupt-4xCO2.nc RCMIP-phase2-OSCARv3-1_esm-hist.nc RCMIP-phase2-OSCARv3-1_historical.nc RCMIP-phase2-OSCARv3-1_historical-cmip5.nc RCMIP-phase2-OSCARv3-1_rcp26.nc RCMIP-phase2-OSCARv3-1_rcp45.nc RCMIP-phase2-OSCARv3-1_rcp60.nc RCMIP-phase2-OSCARv3-1_rcp85.nc RCMIP-phase2-OSCARv3-1_ssp119.nc RCMIP-phase2-OSCARv3-1_ssp126.nc RCMIP-phase2-OSCARv3-1_ssp245.nc RCMIP-phase2-OSCARv3-1_ssp370.nc RCMIP-phase2-OSCARv3-1_ssp434.nc RCMIP-phase2-OSCARv3-1_ssp460.nc RCMIP-phase2-OSCARv3-1_ssp534-over.nc RCMIP-phase2-OSCARv3-1_ssp585.nc RCMIP-phase2-OSCARv3-1_esm-ssp119.nc RCMIP-phase2-OSCARv3-1_esm-ssp126.nc RCMIP-phase2-OSCARv3-1_esm-ssp245.nc RCMIP-phase2-OSCARv3-1_esm-ssp370.nc RCMIP-phase2-OSCARv3-1_esm-ssp434.nc RCMIP-phase2-OSCARv3-1_esm-ssp460.nc RCMIP-phase2-OSCARv3-1_esm-ssp534-over.nc RCMIP-phase2-OSCARv3-1_esm-ssp585.nc RCMIP-phase2-OSCARv3-1_1pctCO2-cdr.nc RCMIP-phase2-OSCARv3-1_esm-1pctCO2.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-750PgC.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-1000PgC.nc RCMIP-phase2-OSCARv3-1_esm-1pct-brch-2000PgC.nc RCMIP-phase2-OSCARv3-1_esm-pi-CO2pulse.nc RCMIP-phase2-OSCARv3-1_esm-pi-cdr-pulse.nc RCMIP-phase2-OSCARv3-1-indicators.csv rcmip_model_metadata_OSCARv3-1.csv


# tmp = []
# for scen in ['esm-ssp119', 'esm-ssp126', 'esm-ssp245', 'esm-ssp370', 'esm-ssp434', 'esm-ssp460', 'esm-ssp534-over', 'esm-ssp585'] + ['1pctCO2-cdr', 'esm-1pctCO2', 'esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-pi-CO2pulse', 'esm-pi-cdr-pulse']:
#     tmp.append( 'RCMIP-phase2-OSCARv3-1_'+dico_name_scen[scen]+'.nc' )
# ' '.join(tmp)


# C:/Users/quilcail/Documents/Repositories/temporary/
# RCMIP-phase2-OSCARv3-1_esm-ssp585.nc









