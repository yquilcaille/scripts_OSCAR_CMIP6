import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# update for ETH-exo, quick fix of a plot
os.chdir( '/home/yquilcaille/OSCAR-CMIP6' )

import csv
import sys
#sys.path.append("H:\MyDocuments\Repositories\OSCARv31_CMIP6") ## line required for run on server ebro
#from core_fct.fct_process import OSCAR,OSCAR_landC
import scipy.stats as scp
import seaborn as sns ## for colors
import pandas as pd

#from run_scripts.RCMIP_phase2.weighted_quantile import weighted_quantile

## 1. OPTIONS
## 2. PREPARATION
## 3. CDRMIP
## 4. ZECMIP
## 5. LUMIP
## 6. RCMIP
## 7. OSCAR-CMIP6


## This whole script uses the results of the treatment of each MIP to produce figures
## remark: standard deviation are calculated as 1/N sum_i(x_i-x_mean), instead of 1/(N-1.5) sum_i(x_i-x_mean) --> doesnt make a difference, N=10000


##################################################
## 1. OPTIONS
##################################################
# update ETH-exo for quick update of figures:
dico_paths = {'masks':'/home/yquilcaille/OSCAR-CMIP6/masks/', \
              'weights':'/home/yquilcaille/OSCAR-CMIP6/weights/', \
              'assessed_ranges':'/home/yquilcaille/OSCAR-CMIP6/run_scripts/RCMIP_phase2/',\
              'RCMIP_phase2':'/home/yquilcaille/OSCAR-CMIP6/RCMIP_phase2/'
             }
path_all = '/net/exo/landclim/yquilcaille/OSCARv31_CMIP6/results/CMIP6_v3.1/'
#'H:/MyDocuments/Repositories/OSCARv31_CMIP6/results/CMIP6_v3.1' # folder where treated outputs will be saved::  'C:/Users/quilcail/Documents/Repositories/OSCARv3_CMIP6/results/CMIP6_v3.0'  |  'E:/OSCARv3_CMIP6/results/CMIP6_v3.0'

list_setMC = np.arange(0,19+1)
type_weights = 'assessed_ranges'     # constraints_4 |  assessed_ranges
option_maskDV = 'LOAD_unique'
option_overwrite  = False # used only to recalculate files before plot
option_which_plots = [] # 'ZEC','7.1','7.2-figure','7.2-table','7.3','7.4-cdr','7.4-a','7.6','7.7','7.8','7.9','7.10','7.11','7.12','7.13','7.14','7.15','7.16'
fac_size = 2.0 ## factors used for sizes of lines, texts, markers,... Required to compensate for changes in computer/server -- size of output (only lw has 0.75* this factor)
##################################################
##################################################
print(option_which_plots)




##################################################
## 2. PREPARATION
##################################################
####################
## 2.1. Experiments
####################
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
    elif (xp[-3:] in ['ext','Ext']) and (xp not in ['1pctCO2-4xext']): xp2 = xp[:-3]
    elif xp in ['ssp534-over-bgc','ssp585-bgc']:xp2 = 'hist-bgc'
    elif xp[:3]=='ssp': xp2 = 'historical'
    elif xp[:7]=='esm-ssp': xp2 = 'esm-hist'
    elif xp[:3]=='rcp': xp2 = 'historical-CMIP5'
    elif xp[:7]=='esm-rcp': xp2 = 'esm-histcmip5'
    else: xp2 = None
    if (xp2 in list_experiments) or xp2==None:dico_experiments_before[xp] = xp2
    else: raise Exception("Correct the name of this experiment")

dico_Xp_Control = { 'piControl':['1pctCO2-4xext', '1pctCO2-bgc', '1pctCO2-cdr', '1pctCO2-rad', '1pctCO2', 'abrupt-0p5xCO2', 'abrupt-2xCO2', 'abrupt-4xCO2', 'G1', 'G2', 'G6solar', 'hist-1950HC', 'hist-aer', 'hist-bgc', 'hist-CO2', 'hist-GHG', 'hist-nat', 'hist-piAer', 'hist-piNTCF', 'hist-sol', 'hist-stratO3', 'hist-volc', 'historical', 'hist-noLu', 'ssp119', 'ssp119ext', 'ssp126-ssp370Lu', 'ssp126', 'ssp126ext', 'ssp245-aer', 'ssp245-CO2', 'ssp245-GHG', 'ssp245-nat', 'ssp245-sol', 'ssp245-stratO3', 'ssp245-volc', 'ssp245', 'ssp245ext', 'ssp370-lowNTCF', 'ssp370-lowNTCFext', 'ssp370-ssp126Lu', 'ssp370', 'ssp370ext', 'ssp434', 'ssp434ext', 'ssp460', 'ssp460ext', 'ssp534-over-bgc', 'ssp534-over-bgcExt', 'ssp534-over-ext', 'ssp534-over', 'ssp585-bgc', 'ssp585-bgcExt', 'ssp585-ssp126Lu', 'ssp585', 'ssp585ext', 'yr2010CO2'] ,
                    'esm-piControl':['esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-1pct-brch-750PgC', 'esm-1pctCO2', 'esm-abrupt-4xCO2', 'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC', 'esm-hist', 'esm-pi-cdr-pulse', 'esm-pi-CO2pulse', 'esm-ssp119', 'esm-ssp119ext', 'esm-ssp126', 'esm-ssp126ext', 'esm-ssp245', 'esm-ssp245ext', 'esm-ssp370-lowNTCF', 'esm-ssp370-lowNTCFext', 'esm-ssp370', 'esm-ssp370ext', 'esm-ssp460', 'esm-ssp460ext', 'esm-ssp434', 'esm-ssp434ext', 'esm-ssp534-over-ext', 'esm-ssp534-over', 'esm-ssp585-ssp126Lu-ext', 'esm-ssp585-ssp126Lu', 'esm-ssp585', 'esm-ssp585ext', 'esm-yr2010CO2-cdr-pulse', 'esm-yr2010CO2-CO2pulse', 'esm-yr2010CO2-control', 'esm-yr2010CO2-noemit'],
                    'esm-piControl-CMIP5':['esm-histcmip5','esm-rcp26', 'esm-rcp45', 'esm-rcp60', 'esm-rcp85'],
                    'piControl-CMIP5':['historical-CMIP5', 'rcp26', 'rcp45', 'rcp60', 'rcp85'],
                    'land-piControl':['land-cClim', 'land-cCO2', 'land-crop-grass', 'land-hist', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv'],
                    'land-piControl-altLu1':['land-hist-altLu1'],
                    'land-piControl-altLu2':['land-hist-altLu2'],
                    'land-piControl-altStartYear':['land-hist-altStartYear'] }
####################
####################










####################
## 2.2. Masks
####################
if option_maskDV=='LOAD_indiv':
    for name_experiment in list_experiments:
        print('Loading masks for '+name_experiment)
        list_noDV[name_experiment] = {}
        for setMC in list_setMC:
            # with open(path_all+'/treated/masks/masknoDV_'+name_experiment+'_'+str(setMC)+'.csv','r',newline='') as ff:
            with open(dico_paths['masks']+'masknoDV_'+name_experiment+'_'+str(setMC)+'.csv','r',newline='') as ff:
                list_noDV[name_experiment][setMC] = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)
elif option_maskDV=='LOAD_unique':
    mask_all,list_noDV = {},{}
    for setMC in list_setMC:
        print('Loading unique mask for set '+str(setMC))
        ## loading unique mask for this set
        # with open(path_all+'/treated/masks/mask_all_exp_'+str(setMC)+'.csv','r',newline='') as ff:
        with open(dico_paths['masks']+'mask_all_exp_'+str(setMC)+'.csv','r',newline='') as ff:
            mask_all[setMC] = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)
        for name_experiment in list_experiments:
            if name_experiment not in list_noDV.keys():list_noDV[name_experiment] = {}
            ## checking the length of the run to conform to the previous code
            #out_TMP = xr.open_dataset(path_all+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' )
            # nn = out_TMP.year.size
            nn = 151 + 100 + 1000
            list_noDV[name_experiment][setMC] = np.repeat( mask_all[setMC][np.newaxis,:,0], nn, axis=0 )
            #out_TMP.close()

## information on set monte-carlo
dico_sizesMC = {}
for setMC in list_setMC:dico_sizesMC[setMC] = int(list_noDV[list_experiments[0]][setMC].shape[1])
# dico_sizesMC = {setMC:500 for setMC in list_setMC}
####################
####################







####################
## 2.3. Weights
####################
if type_weights == 'constraints_4':
    #weights_CMIP6 = xr.open_dataset(path_all+'/treated/weights/weights_CMIP6.nc')
    #weights_RCMIP = xr.open_dataset(path_all+'/treated/weights/weights_RCMIP.nc')
    weights_CMIP6 = xr.open_dataset(dico_paths['weights'] + 'weights_CMIP6.nc')
    weights_RCMIP = xr.open_dataset(dico_paths['weights'] + 'weights_RCMIP.nc')
elif type_weights == 'assessed_ranges':
    ## Loading indicators
    folder_rcmip = 'results/RCMIP_phase2/'
    if option_maskDV == 'LOAD_indiv':
        #indic = xr.load_dataset( folder_rcmip + 'oscar_indicators_full-configs_mask_indiv.nc' )
        indic = xr.load_dataset( dico_paths['RCMIP_phase2'] + 'oscar_indicators_full-configs_mask_indiv.nc' )
    elif option_maskDV == 'LOAD_unique':
        #indic = xr.load_dataset( folder_rcmip + 'oscar_indicators_full-configs_mask_unique.nc' )
        indic = xr.load_dataset( dico_paths['RCMIP_phase2'] + 'oscar_indicators_full-configs_mask_unique.nc' )
    ## correction, some coordinates become variables....
    for var in ['RCMIP variable', 'RCMIP region', 'RCMIP scenario']:
        indic.coords[var] = indic[var]
    # indic = indic.drop('distrib')
    ## indicators to use for weighting OSCAR by Yann Quilcaille for RCMIP-phase 2
    ind_list = ['Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011','Surface Air Ocean Blended Temperature Change World ssp245 2000-2019','Cumulative compatible emissions CMIP5 historical-CMIP5','Cumulative compatible emissions CMIP5 RCP2.6','Cumulative compatible emissions CMIP5 RCP4.5','Cumulative compatible emissions CMIP5 RCP6.0','Cumulative compatible emissions CMIP5 RCP8.5']
    ## checking that all ind_list are within
    if len( [ind for ind in ind_list if ind not in indic.indicator] )>0: raise Exception('Warning, missing indicators.')
    ## calculating products of weights
    WEIGHTS = xr.Dataset()
    WEIGHTS.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]#mask_all.config.sel(config=configs)
    val = indic['w'].sel(index=[i for i in indic.index if str(indic.indicator.sel(index=i).values) in ind_list]).prod('index')
    val *= np.hstack( [mask_all[kk][:,0] for kk in list_setMC] )
    WEIGHTS['weights'] = xr.DataArray( data=val.values , dims=('all_config') )
    weights_CMIP6 = WEIGHTS.copy()
    weights_RCMIP = WEIGHTS.copy()
    del WEIGHTS

weights_ones = xr.Dataset()
weights_ones['weights'] = xr.ones_like( weights_CMIP6.weights )
####################
####################






####################
## 2.4. information on indicators
####################
#file_ranges = 'run_scripts/RCMIP_phase2/RCMIP-phase2_assessed-ranges-v2-2-0.csv'
file_ranges = dico_paths['assessed_ranges']+'RCMIP-phase2_assessed-ranges-v2-2-0.csv'
rcmip = xr.Dataset.from_dataframe(pd.read_csv(file_ranges))
## complying to IPCC script
rcmip = rcmip.rename({'RCMIP name':'indicator'})
## load as well IPCC indicators for consistent weighting
file_ranges_ipcc = 'run_scripts/RCMIP_phase2/ipcc-wg1-emulator-assessed-ranges_v2+.csv'
ipcc = xr.Dataset.from_dataframe(pd.read_csv(file_ranges_ipcc))
## those in common will be rewritten: [ind for ind in rcmip['indicator'].values if ind in ipcc['indicator'].values]
## load as well extra indicators for testing
file_ranges_extra = 'extra_data/extra_assessed-ranges.csv'
extra = xr.Dataset.from_dataframe(pd.read_csv(file_ranges_extra)).rename({'RCMIP name':'indicator'})
## those in common will be rewritten: [ind for ind in extra['indicator'].values if ind in ipcc['indicator'].values]
## manual merging of rcmip using ipcc to increment index while not adding indicators already in rcmip
ipcc = ipcc.sel( index=[ind for ind in ipcc.index.values if ipcc['indicator'].sel(index=ind).values not in rcmip['indicator']] )
ipcc.coords['index'] = np.arange( rcmip.index[-1]+1, rcmip.index[-1]+1+ipcc.index.size )
extra = extra.sel( index=[ind for ind in extra.index.values if extra['indicator'].sel(index=ind).values not in rcmip['indicator']] )
extra.coords['index'] = np.arange( ipcc.index[-1]+1, ipcc.index[-1]+1+extra.index.size )
rcmip = xr.merge( [rcmip,ipcc,extra] )
####################
####################



####################
## 2.5. Functions
####################
## colors
CB_color_cycle = sns.color_palette( 'colorblind' )

if False:sns.palplot( CB_color_cycle )

list_letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)','(s)','(t)','(u)','(v)','(w)','(x)','(y)','(z)']

def func_plot( varx,vary , OUT, label, col='k',ls='-',lw=0.75*fac_size*3,marker=None,markevery=None,mec=None,ms=fac_size*0 ,edgecolor=None,alpha=0.25,zorder=10):
    if type(varx)!=str:
        for_x = varx
    else:
        for_x = OUT[varx]
    if 'stat_value' in OUT[vary].dims:
        if 'reg_land' in OUT[vary].dims:
            plt.plot( for_x , OUT[vary].sel(stat_value='mean',reg_land=0) , lw=0.75*fac_size*lw , ls=ls, color=col,marker=marker,markevery=markevery,mec=mec,ms=fac_size*ms , label=label ,zorder=zorder)
            plt.fill_between( for_x , OUT[vary].sel(stat_value='mean',reg_land=0)-1.0*OUT[vary].sel(stat_value='std_dev',reg_land=0) , OUT[vary].sel(stat_value='mean',reg_land=0)+1.0*OUT[vary].sel(stat_value='std_dev',reg_land=0) , edgecolor=edgecolor,ls=ls,lw=0.75*fac_size*lw*2/3., facecolor=col , alpha=alpha ,zorder=zorder )
        else:
            plt.plot( for_x , OUT[vary].sel(stat_value='mean') , lw=0.75*fac_size*lw , ls=ls, color=col,marker=marker,markevery=markevery,mec=mec,ms=fac_size*ms , label=label ,zorder=zorder )
            plt.fill_between( for_x , OUT[vary].sel(stat_value='mean')-1.0*OUT[vary].sel(stat_value='std_dev') , OUT[vary].sel(stat_value='mean')+1.0*OUT[vary].sel(stat_value='std_dev') , edgecolor=edgecolor,ls=ls,lw=0.75*fac_size*lw*2/3., facecolor=col , alpha=alpha ,zorder=zorder )
    else:
        if 'reg_land' in OUT[vary].dims:
            plt.plot( for_x , OUT[vary].sel(reg_land=0) , lw=0.75*fac_size*lw , ls=ls, color=col,marker=marker,markevery=markevery,mec=mec,ms=fac_size*ms , label=label ,zorder=zorder)
        else:
            plt.plot( for_x , OUT[vary] , lw=0.75*fac_size*lw , ls=ls, color=col,marker=marker,markevery=markevery,mec=mec,ms=fac_size*ms , label=label ,zorder=zorder)



def func_get( setMC , xp , list_var_OSCAR ,mode_ext=False , option_NeedDiffControl=True):
    with xr.open_dataset(path_all+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
    ## experiment
    if mode_ext:
        if xp in ['esm-ssp370-lowNTCF-gidden','ssp370-lowNTCF-gidden']:
            xp = {'esm-ssp370-lowNTCF-gidden':'esm-ssp370-lowNTCFext-gidden' , 'ssp370-lowNTCF-gidden':'ssp370-lowNTCFext-gidden'}
        else:
            xp = xp+'ext'
    out_tmp = xr.open_dataset(path_all+'/'+xp+'_Out-'+str(setMC)+'.nc')
    for_tmp = xr.open_dataset(path_all+'/'+xp+'_For-'+str(setMC)+'.nc')
    # control
    test_NotControl = xp not in dico_Xp_Control.keys()
    if test_NotControl and option_NeedDiffControl:
        xp_c = [xp_c for xp_c in dico_Xp_Control if xp in dico_Xp_Control[xp_c]][0]
        out_ctrl = xr.open_dataset(path_all+'/'+xp_c+'_Out-'+str(setMC)+'.nc')
        for_ctrl = xr.open_dataset(path_all+'/'+xp_c+'_For-'+str(setMC)+'.nc')
        ## cutting period to the one of out_tmp
        if out_tmp.year.size < out_ctrl.year.size:
            out_ctrl = out_ctrl.sel(year=out_tmp.year)
            for_ctrl = for_ctrl.sel(year=out_tmp.year)
        for var in list_var_OSCAR:
            out_ctrl[var] = OSCAR[var](out_ctrl, Par, for_ctrl.update(out_ctrl),recursive=True)


    ## correction required?
    if xp in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc' , 'ssp585-bgcExt','ssp534-over-bgcExt']:
        with xr.open_dataset(path_all+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
        Par['D_CO2_rad'] = for_runs_hist.D_CO2.sel(year=1850)
    elif xp in ['1pctCO2-rad']:
        with xr.open_dataset(path_all+'/historical_For-'+str(setMC)+'.nc') as TMP: for_runs_hist = TMP.load()
        Par['D_CO2_bgc'] = for_runs_hist.D_CO2.sel(year=1850)
    for var in list_var_OSCAR:
        out_tmp[var] = OSCAR[var](out_tmp, Par, for_tmp.update(out_tmp),recursive=True)
    ## difference control
    if option_NeedDiffControl:
        if out_tmp.year.size > out_ctrl.year.size:
            yr_xp0_start, yr_pi_cut, yr_xp0_end = out_tmp.year.isel(year=0), out_ctrl.year.isel(year=-1), out_tmp.year.isel(year=-1)
            out_tmp2 = xr.Dataset( coords=out_tmp.coords )
            for VAR in out_tmp.variables:
                if (VAR not in out_tmp.coords):
                    out_tmp2[VAR] = xr.DataArray( np.full(fill_value=np.nan,shape=out_tmp[VAR].shape) , dims=out_tmp[VAR].dims  )
                    out_tmp2[VAR].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}]   =  out_tmp[VAR].loc[{'year':np.arange(yr_xp0_start,yr_pi_cut+1)}] - out_ctrl[VAR].sel(year=np.arange(yr_xp0_start,yr_pi_cut+1))
                    ## using average of the last 10 years of control
                    out_tmp2[VAR].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}]   =  out_tmp[VAR].loc[{'year':np.arange(yr_pi_cut+1,yr_xp0_end+1)}] - out_ctrl[VAR].sel(year=np.arange(yr_pi_cut-10+1,yr_pi_cut+1)).mean('year')
            out_tmp = out_tmp2
            del out_tmp2
        else:
            if test_NotControl: out_tmp = out_tmp - out_ctrl
    ## mask
    # with open(path_all+'/treated/masks/masknoDV_'+xp+'_'+str(setMC)+'.csv','r',newline='') as ff:
    #     mask = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)
    mask = list_noDV[xp][setMC]
    MASK = xr.Dataset()
    for cc in ['year','config']:
        MASK.coords[cc] = out_tmp.coords[cc]
    MASK['mask'] = xr.DataArray( mask, dims=('year','config') )
    ff = xr.Dataset()
    for vv in for_tmp:
        if 'config' in for_tmp[vv].dims:
            ff[vv] = for_tmp[vv]*MASK.mask
        else:
            ff[vv] = for_tmp[vv].copy()
    return out_tmp*MASK.mask , ff , Par,mask


def func_add_var( OUT,PAR,FOR, list_var_required , type_OSCAR='OSCAR' ):
    for var in list_var_required:
        if var in FOR:
            OUT[var] = FOR[var]
        else:
            if type_OSCAR=='OSCAR_landC':
                OUT[var] = OSCAR_landC[var](OUT, PAR, FOR.update(OUT),recursive=True)
            elif type_OSCAR=='OSCAR':
                OUT[var] = OSCAR[var](OUT, PAR, FOR.update(OUT),recursive=True)
    return OUT


def eval_compat_emi( var_comp, OUT,PAR,FOR,type_OSCAR='OSCAR' ):
    ##-----
    ## This function calculates the compatible emissions of GhG for concentrations-driven experiments.
    ## Mostly used to fill in RCMIP variables such as Airborne Fraction|CO2 or TCRE for concentrations-driven experiments.
    ## A specific warning will be issued for these variables.
    ##-----
    ## Xhalo
    OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Fsink_Xhalo'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
    OUT['E_Xhalo_comp'] = (PAR.a_Xhalo * OUT.D_Xhalo.differentiate('year')
        + OUT.D_Fsink_Xhalo)
    ## N2O
    OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Ebb','D_Fsink_N2O'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
    OUT['E_N2O_comp'] = (PAR.a_N2O * OUT.D_N2O.differentiate('year') 
        - OUT.D_Ebb.sel({'spc_bb':'N2O'}, drop=True).sum('bio_land', min_count=1).sum('reg_land', min_count=1)
        + OUT.D_Fsink_N2O)
    ## CH4
    OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Ewet','D_Ebb','D_Epf_CH4','D_Fsink_CH4'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
    OUT['E_CH4_comp'] = (PAR.a_CH4 * OUT.D_CH4.differentiate('year') 
        - OUT.D_Ewet.sum('reg_land', min_count=1)
        - OUT.D_Ebb.sel({'spc_bb':'CH4'}, drop=True).sum('bio_land', min_count=1).sum('reg_land', min_count=1)
        - OUT.D_Epf_CH4.sum('reg_pf', min_count=1) 
        + OUT.D_Fsink_CH4)
    ## Foxi!
    OUT['D_Foxi_CH4'] = 1E-3 * (PAR.p_CH4geo * OUT.E_CH4_comp
        + OUT.D_Epf_CH4.sum('reg_pf', min_count=1) 
        - PAR.a_CH4 * OUT.D_CH4.differentiate('year'))
    ## FF CO2
    OUT = func_add_var(OUT, PAR, FOR , list_var_required=['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4'] , type_OSCAR=type_OSCAR) ## no OSCAR_landC experiments need compatible emissions
    OUT['Eff_comp'] = (PAR.a_CO2 * OUT.D_CO2.differentiate('year') 
        - OUT.D_Eluc
        - OUT.D_Epf_CO2.sum('reg_pf', min_count=1) 
        + OUT.D_Fland
        + OUT.D_Focean
        - OUT.D_Foxi_CH4)
    return xr.merge([FOR, OUT['E_Xhalo_comp'], OUT['E_N2O_comp'], OUT['E_CH4_comp'], OUT['D_Foxi_CH4'], OUT['Eff_comp']])


## general pdf function
def pdf_indic(DIST,xx):
    dist, param = DIST[:DIST.find('[')], eval( DIST[DIST.find('['):] )
    categ_dist = {'N': scp.norm, 'GN1': scp.gennorm, 'PN': scp.powernorm, 'EN': scp.exponnorm, 'GL': scp.genlogistic, 'NIG': scp.norminvgauss}[dist]

    ## normal distrib
    if dist == 'N':
        trans = [lambda x: x, abs]

    ## generalized normal distrib
    elif dist == 'GN1':
        trans = [lambda x: 1+abs(x), lambda x: x, abs]

    ## power-normal distrib
    elif dist == 'PN':
        trans = [abs, lambda x: x, abs]

    ## exponentially modified normal distrib
    elif dist == 'EN':
        trans = [abs, lambda x: x, abs]

    ## generalized logistic distrib
    elif dist == 'GL':
        trans = [abs, lambda x: x, abs]

    ## normal inverse Gaussian distrib
    elif dist == 'NIG':
        trans = [abs, lambda x: x, lambda x: x, abs] # warning! extra condition not implemented: abs(b) <= a

    ## calculating distibution and returning
    return categ_dist.pdf(xx, *param)

def indic_mean_stddev( VAR, xx_in ):
    yy = pdf_indic( str(indic['distrib'].isel(index=indic.indicator.values.tolist().index(VAR)).values), xx_in )
    ind = np.where( ~np.isinf(yy) & ~np.isnan(yy) )[0]
    mm = np.sum( (yy*(xx_in[1]-xx_in[0]) * xx_in)[ind] )
    ss = np.sqrt( np.nansum( (yy*(xx_in[1]-xx_in[0]) * (xx_in - mm)**2.)[ind] ) * len(xx_in[ind]) / (len(xx_in[ind])-1) )
    return mm,ss

####################
####################
##################################################
##################################################






##################################################
## 3. CDRMIP
##################################################
if False:
    ## C1: reversibility: '1pctCO2-cdr' tas vs co2, with std_dev
    ## C2: pulses: 'esm-pi-CO2pulse', 'esm-pi-cdr-pulse', 'esm-yr2010CO2-CO2pulse', 'esm-yr2010CO2-cdr-pulse', tas vs time
    ## C3: af-reforestation: 'esm-ssp585-ssp126Lu', 'esm-ssp585-ssp126Lu-ext', 'esm-ssp585', 'esm-ssp585ext': tas vs time

    ## C1:
    ax = plt.subplot(3,1,1)
    plt.title('Reversibility')
    plt.grid()
    VAR = 'tas'
    xp = '1pctCO2-cdr'
    ## loading XP minus control
    OUT = xr.open_dataset( path_all+'/treated/CDRMIP/'+os.listdir(path_all+'/treated/CDRMIP/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/CDRMIP/')]))[0][0]] )
    ## ploting
    func_plot('xco2',VAR,OUT.sel(year=np.arange(1850,1850+140+1)),col='k',label=None,marker='>',markevery=35,mec='r',ms=fac_size*8)
    func_plot('xco2',VAR,OUT.sel(year=np.arange(1850+140,1850+2*140+1)),col='k',label=xp,marker='<',markevery=35,mec='r',ms=fac_size*8)
    func_plot('xco2',VAR,OUT.sel(year=np.arange(1850+2*140,1850+2*140+1+1000)),col='k',label=None,marker='v',markevery=35,mec='r',ms=fac_size*8)
    plt.xlabel( 'Atmospheric CO2 (ppm)' )
    plt.ylabel( OUT[VAR].long_name+' ('+OUT[VAR].unit+')' )
    ## cleaning
    OUT.close()
    plt.legend(loc='upper left')
    box = ax.get_position()
    ax.set_position([box.x0-0.03, box.y0+0.09-0.00, box.width*1.1, box.height*1.0])


    ## C2:
    dico_col = {'esm-pi-CO2pulse':'r', 'esm-pi-cdr-pulse':'g', 'esm-yr2010CO2-CO2pulse':'r', 'esm-yr2010CO2-cdr-pulse':'g'}
    dico_ls = {'esm-pi-CO2pulse':'--', 'esm-pi-cdr-pulse':'--', 'esm-yr2010CO2-CO2pulse':'-', 'esm-yr2010CO2-cdr-pulse':'-'}
    VAR = 'tas'
    for ii in np.arange(2):
        ax = plt.subplot(3,2,3+ii)
        plt.title( ['Pulses control','Pulses 2010'][ii] )
        for xp in [['esm-pi-CO2pulse', 'esm-pi-cdr-pulse'], ['esm-yr2010CO2-CO2pulse', 'esm-yr2010CO2-cdr-pulse']][ii]:
            ## loading XP minus control
            OUT = xr.open_dataset( path_all+'/treated/CDRMIP/'+os.listdir(path_all+'/treated/CDRMIP/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/CDRMIP/')]))[0][0]] )
            ## ploting
            func_plot('year',VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
        plt.xlabel( 'Year' )
        plt.ylabel( OUT[VAR].long_name+' ('+OUT[VAR].unit+')' )
        ## cleaning
        OUT.close()
        plt.grid()
        plt.legend(loc=0)
        box = ax.get_position()
        ax.set_position([box.x0-0.03+ii*0.045, box.y0+0.09-0.06, box.width*1.1, box.height*1.0])


    ## C3:
    ax = plt.subplot(3,1,3)
    dico_col = {'esm-ssp585-ssp126Lu':'g', 'esm-ssp585-ssp126Lu-ext':'g', 'esm-ssp585':'k', 'esm-ssp585ext':'k'}
    dico_ls = {'esm-ssp585-ssp126Lu':'-', 'esm-ssp585-ssp126Lu-ext':'-', 'esm-ssp585':'-', 'esm-ssp585ext':'-'}
    plt.title('Reforestation')
    plt.grid()
    VAR = 'cVeg'
    for xp in ['esm-ssp585-ssp126Lu', 'esm-ssp585']:
        ## loading XP minus control
        OUT = xr.open_dataset( path_all+'/treated/CDRMIP/'+os.listdir(path_all+'/treated/CDRMIP/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/CDRMIP/')]))[0][0]] )
        ## ploting
        func_plot('year',VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.xlabel( 'Year' )
    plt.ylabel( OUT[VAR].long_name+' ('+OUT[VAR].unit+')' )
    ## cleaning
    OUT.close()
    plt.legend(loc='upper left')
    box = ax.get_position()
    ax.set_position([box.x0-0.03, box.y0+0.09-0.12, box.width*1.1, box.height*1.0])
##################################################
##################################################






##################################################
## 4. LUMIP
##################################################
if False:
    ## effects CO2 and climt: 'land-cCO2', 'land-cClim', cLand vs time
    ## effects practices: 'land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv', cLand vs time
    ## effects data: 'land-hist', 'land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear', cLand vs time

    ## effects CO2 and climt: 'land-cCO2', 'land-cClim', cLand vs time
    ax = plt.subplot(3,1,1)
    dico_col = {'land-cCO2':'r', 'land-cClim':'k'}
    dico_ls = {'land-cCO2':'-', 'land-cClim':'-'}
    plt.title('CO2 and climate')
    plt.grid()
    VAR = 'cLand'
    for xp in ['land-cCO2', 'land-cClim']:
        ## loading XP minus control
        OUT = xr.open_dataset( path_all+'/treated/LUMIP/'+os.listdir(path_all+'/treated/LUMIP/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/LUMIP/')]))[0][0]] )
        ## ploting
        func_plot('year',VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.xlabel( 'Year' )
    plt.ylabel( OUT[VAR].long_name+' ('+OUT[VAR].unit+')' )
    ## cleaning
    OUT.close()
    plt.legend(loc='upper left')
    box = ax.get_position()
    ax.set_position([box.x0-0.03, box.y0+0.09-0.00, box.width*1.1, box.height*1.0])


    ## effects practices: 'land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv', cLand vs time
    ax = plt.subplot(3,1,2)
    dico_col = {'land-crop-grass':'g', 'land-noLu':'purple', 'land-noShiftcultivate':'b', 'land-noWoodHarv':'k'}
    dico_ls = {'land-crop-grass':'-', 'land-noLu':'-', 'land-noShiftcultivate':'-', 'land-noWoodHarv':'-'}
    plt.title('Practices')
    plt.grid()
    VAR = 'cLand'
    for xp in ['land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv']:
        ## loading XP minus control
        OUT = xr.open_dataset( path_all+'/treated/LUMIP/'+os.listdir(path_all+'/treated/LUMIP/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/LUMIP/')]))[0][0]] )
        ## ploting
        func_plot('year',VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.xlabel( 'Year' )
    plt.ylabel( OUT[VAR].long_name+' ('+OUT[VAR].unit+')' )
    ## cleaning
    OUT.close()
    plt.legend(loc='upper left')
    box = ax.get_position()
    ax.set_position([box.x0-0.03, box.y0+0.09-0.06, box.width*1.1, box.height*1.0])


    ## effects data: 'land-hist', 'land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear', cLand vs time
    ax = plt.subplot(3,1,3)
    dico_col = {'land-hist':'k', 'land-hist-altLu1':'r', 'land-hist-altLu2':'g', 'land-hist-altStartYear':'b'}
    dico_ls = {'land-hist':'-', 'land-hist-altLu1':'-', 'land-hist-altLu2':'-', 'land-hist-altStartYear':'-'}
    plt.title('Datasets LU')
    plt.grid()
    VAR = 'cLand'
    for xp in ['land-hist', 'land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear']:
        ## loading XP minus control
        OUT = xr.open_dataset( path_all+'/treated/LUMIP/'+os.listdir(path_all+'/treated/LUMIP/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/LUMIP/')]))[0][0]] )
        ## ploting
        func_plot('year',VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.xlabel( 'Year' )
    plt.ylabel( OUT[VAR].long_name+' ('+OUT[VAR].unit+')' )
    ## cleaning
    OUT.close()
    plt.legend(loc='upper left')
    box = ax.get_position()
    ax.set_position([box.x0-0.03, box.y0+0.09-0.12, box.width*1.1, box.height*1.0])
##################################################
##################################################





##################################################
## 5. ZECMIP
##################################################
if 'ZEC' in option_which_plots:
    ## just some small treatment for ZEC after bells
    list_xp = ['esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC']
    list_VAR = ['D_Tg']

    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(1850,1850+100+1000+1)
        for xp in list_xp:
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , list_VAR )
                for var in list_VAR:
                    if var+'_'+xp not in TMP:
                        TMP[var+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                    val = out_tmp[var].sel(year=out_tmp.year)
                    TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = val
                out_tmp.close()
                for_tmp.close()

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_ZEC.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_ZEC.nc' )

    # statistical values
    OUTPUT = xr.Dataset()
    OUTPUT.coords['stat_value'] = ['mean','std_dev']
    OUTPUT.coords['year'] = np.arange(1000+1)
    for xp in list_xp:
        for VAR in list_VAR:
            OUTPUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUTPUT.year.size,OUTPUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp].sel(year=np.arange(1850+100,1850+100+1000+1)) - TMP[VAR+'_'+xp].sel(year=1850+100),mask=np.isnan(TMP[VAR+'_'+xp].sel(year=np.arange(1850+100,1850+100+1000+1))))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],1000+1,axis=0)
            OUTPUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUTPUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUTPUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

    ## Figure
    ## Bells: 'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC'
    ## Branched: 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-1pct-brch-750PgC', tas vs time AND tas vs time from branch
    counter = 0
    fig = plt.figure( figsize=(30,20) )
    ax = plt.subplot(2,2,1)
    VAR = 'tas'
    dico_col = {'esm-1pct-brch-1000PgC':CB_color_cycle[0], 'esm-1pct-brch-2000PgC':CB_color_cycle[1], 'esm-1pct-brch-750PgC':CB_color_cycle[2]}
    dico_ls = {'esm-1pct-brch-1000PgC':'-', 'esm-1pct-brch-2000PgC':'-', 'esm-1pct-brch-750PgC':'-'}
    plt.title('Branched experiments, full period',size=fac_size*14 )#,fontweight='bold')
    plt.grid()
    for xp in ['esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC'][::-1]:
        OUT = xr.open_dataset( path_all+'/treated/ZECMIP/intermediary/'+os.listdir(path_all+'/treated/ZECMIP/intermediary/')[np.where( (xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'!= np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) )[0][0]] )
        func_plot(np.arange(OUT.year.size),VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.ylabel( 'Change in global mean\nsurface temperature (K)'  ,size=fac_size*14 )#,fontweight='bold')
    ## cleaning
    OUT.close()
    plt.legend(prop={'size':fac_size*12}, loc="lower left")#,loc=0,bbox_to_anchor=(-0.125,0.62))
    box = ax.get_position()
    # ax.set_position([box.x0+0.05, box.y0+0.01, box.width*1.05, box.height*1.0])
    ax.tick_params(labelsize=fac_size*13)
    plt.xlim(0,3000-1850)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.95*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ax = plt.subplot(2,2,2)
    VAR = 'tas'
    dico_col = {'esm-1pct-brch-1000PgC':CB_color_cycle[0], 'esm-1pct-brch-2000PgC':CB_color_cycle[1], 'esm-1pct-brch-750PgC':CB_color_cycle[2]}
    dico_ls = {'esm-1pct-brch-1000PgC':'-', 'esm-1pct-brch-2000PgC':'-', 'esm-1pct-brch-750PgC':'-'}
    plt.title('Branched experiments, zero emissions phase',size=fac_size*14 )#,fontweight='bold')
    plt.grid()
    for xp in ['esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC'][::-1]:
        OUT = xr.open_dataset( path_all+'/treated/ZECMIP/intermediary/'+os.listdir(path_all+'/treated/ZECMIP/intermediary/')[np.where( (xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'== np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) )[0][0]] )
        func_plot(np.arange(OUT.year.size),VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    ## cleaning
    OUT.close()
    box = ax.get_position()
    # ax.set_position([box.x0+0.05, box.y0+0.01, box.width*1.05, box.height*1.0])
    ax.tick_params(labelsize=fac_size*13)
    plt.xlim(0,1000)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.95*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ax = plt.subplot(2,2,3)
    dico_col = {'esm-bell-1000PgC':CB_color_cycle[0], 'esm-bell-2000PgC':CB_color_cycle[1], 'esm-bell-750PgC':CB_color_cycle[2]}
    dico_ls = {'esm-bell-1000PgC':'-', 'esm-bell-2000PgC':'-', 'esm-bell-750PgC':'-'}
    plt.title('Bells, full period',size=fac_size*14 )#,fontweight='bold')
    plt.grid()
    VAR = 'tas'
    for xp in ['esm-bell-750PgC', 'esm-bell-1000PgC', 'esm-bell-2000PgC'][::-1]:
        OUT = xr.open_dataset( path_all+'/treated/ZECMIP/intermediary/'+os.listdir(path_all+'/treated/ZECMIP/intermediary/')[np.where( (xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'!= np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) )[0][0]] )
        func_plot(np.arange(OUT.year.size),VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.xlabel( 'Year' ,size=fac_size*15)
    plt.ylabel( 'Change in global mean\nsurface temperature (K)'  ,size=fac_size*14 )#,fontweight='bold')
    ## cleaning
    OUT.close()
    plt.legend(prop={'size':fac_size*12}, loc="lower center")#,loc=0,bbox_to_anchor=(-0.125,0.62))
    box = ax.get_position()
    # ax.set_position([box.x0+0.05, box.y0+0.01, box.width*1.05, box.height*1.0])
    plt.xlim(0,2950-1850)
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.95*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ax = plt.subplot(2,2,4)
    dico_col = {'esm-bell-1000PgC':CB_color_cycle[0], 'esm-bell-2000PgC':CB_color_cycle[1], 'esm-bell-750PgC':CB_color_cycle[2]}
    dico_ls = {'esm-bell-1000PgC':'-', 'esm-bell-2000PgC':'-', 'esm-bell-750PgC':'-'}
    plt.title('Bells, zero emissions phase',size=fac_size*14 )#,fontweight='bold')
    plt.grid()
    VAR = 'D_Tg'
    for xp in ['esm-bell-750PgC', 'esm-bell-1000PgC', 'esm-bell-2000PgC'][::-1]:func_plot(np.arange(OUTPUT.year.size),VAR+'_'+xp,OUTPUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.xlabel( 'Year after branch'  ,size=fac_size*15)
    ## cleaning
    OUT.close()
    box = ax.get_position()
    # ax.set_position([box.x0+0.05, box.y0+0.01, box.width*1.05, box.height*1.0])
    plt.xlim(0,1000)
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.95*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1
    
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/ZEC.pdf',dpi=300 )
    plt.close(fig)


    if False:
        ## max temperature in whole trajectory
        for xp in ['esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC'][::-1]:
            print(xp)
            OUT = xr.open_dataset( path_all+'/treated/ZECMIP/intermediary/'+os.listdir(path_all+'/treated/ZECMIP/intermediary/')[np.where( (xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'!= np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) )[0][0]] )
            dd = OUT['tas'].sel(stat_value='mean').argmax('year')
            print( OUT['tas'].isel(year=dd.values) )
            print(" ")

        ## max temperature in breached trajectory
        for xp in ['esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC'][::-1]:
            print(xp)
            OUT = xr.open_dataset( path_all+'/treated/ZECMIP/intermediary/'+os.listdir(path_all+'/treated/ZECMIP/intermediary/')[np.where( (xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'== np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) )[0][0]] )
            dd = OUT['tas'].sel(stat_value='mean').argmax('year')
            print( OUT['tas'].isel(year=dd.values) )
            print(" ")

        ## temperature in breached trajectory at a date
        for xp in ['esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC'][::-1]:
            print(xp)
            OUT = xr.open_dataset( path_all+'/treated/ZECMIP/intermediary/'+os.listdir(path_all+'/treated/ZECMIP/intermediary/')[np.where( (xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) * ('from-breach.nc'== np.array([str.split(ff,'_')[-1] for ff in os.listdir(path_all+'/treated/ZECMIP/intermediary/')])) )[0][0]] )
            print(  OUT['tas'].sel(year=500).values  )
            OUT.close()
            print(" ")
        ## temperature in breached bells at a date
        for xp in ['esm-bell-750PgC', 'esm-bell-1000PgC', 'esm-bell-2000PgC'][::-1]:
            print(xp)
            print(  OUTPUT['D_Tg'+'_'+xp].sel(year=100).values  )
            print(" ")
##################################################
##################################################





##################################################
## 6. RCMIP
##################################################
##################################################
##################################################







##################################################
## 7. CMIP6 paper
##################################################
#########################
## 7.0. Figure distribution
#########################
##  in _TREAT-ALL_OSCARv3-CMIP6
#########################
#########################



#########################
## 7.1. Figure Climate Response
#########################
## abrupt-4xCO2, abrupt-2xCO2, abrupt-0p5xCO2:: tas vs time AND distrib ECS tot for each at the end
if '7.1' in option_which_plots:
    ## prepare values
    if option_overwrite:
        VALS = xr.Dataset()
        VALS.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        for xp in ['abrupt-0p5xCO2' , 'abrupt-2xCO2' , 'abrupt-4xCO2']:
            VALS['vals_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(VALS.all_config.size)) , dims=('all_config')  )
            VALS['vals_RescaleCO2_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(VALS.all_config.size)) , dims=('all_config')  )
        for setMC in list_setMC:
            out_2x,for_2x,Par,mask = func_get( setMC , 'abrupt-2xCO2' , ['D_Tg','RF_CO2','RF'] )
            for xp in ['abrupt-0p5xCO2' , 'abrupt-2xCO2' , 'abrupt-4xCO2']:
                print(xp+' on '+str(setMC))
                if xp == 'abrupt-2xCO2':
                    out_abrupt,for_abrupt,Par,mask = out_2x,for_2x,Par,mask
                else:
                    out_abrupt,for_abrupt,Par,mask = func_get( setMC , xp , ['D_Tg','RF_CO2','RF'] )
                ## calculation (already as difference to their control)
                VALS['vals_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] }] = out_abrupt['D_Tg'].isel(year=-1) * out_2x['RF'].isel(year=-1) / out_abrupt['RF'].isel(year=-1)
                VALS['vals_RescaleCO2_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] }] = out_abrupt['D_Tg'].isel(year=-1) * out_2x['RF_CO2'].isel(year=-1) / out_abrupt['RF_CO2'].isel(year=-1)

        VALS.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_climresp.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in VALS})
    else:
        VALS = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_climresp.nc' )

    if False:
        for i in indic.index.values:
            x_vl, x_l, x_c, x_u, x_vu = [float(rcmip[var][i]) for var in ['very_likely__lower', 'likely__lower', 'central', 'likely__upper', 'very_likely__upper']]
            name = str(rcmip.indicator[i].values)
            if name in ['Equilibrium Climate Sensitivity','ECS_FROM_ABRUPT4XCO2_EXPT', 'Transient Climate Response', 'Transient Climate Response to Emissions']:
                cfg = np.where( (~np.isnan(indic.x).sel(index=i).values)  &  ~np.isnan(indic.m.sel(index=i).values) )[0]
                values = indic['x'].sel(index=i).values * indic['m'].sel(index=i).values
                ww = weights_CMIP6.weights.values
                ind = np.where( ~np.isnan(ww) & ~np.isnan(values))[0]
                for type_const in ['Unconstrained', 'Constrained']:
                    if type_const == 'Constrained':
                        abs_val = [weighted_quantile(values[ind], pct, ww[ind]) for pct in [0.05, 0.17, 0.50, 0.83, 0.95]]
                        mm = np.average( values[ind], axis=0, weights=ww[ind] )
                        ss = np.sqrt( np.average( (values[ind]-mm)**2.,axis=0, weights=ww[ind] )*len(ind)/(len(ind)-1.) )
                    elif  type_const == 'Unconstrained':
                        abs_val = [weighted_quantile(values[ind], pct, np.ones(len(ww[ind]))) for pct in [0.05, 0.17, 0.50, 0.83, 0.95]]
                        mm = np.average( values[ind], axis=0, weights=np.ones(len(ww[ind])) )
                        ss = np.sqrt( np.average( (values[ind]-mm)**2.,axis=0, weights=np.ones(len(ww[ind])) )*len(ind)/(len(ind)-1.) )
                    print(type_const+' '+name+': ' + ' '.join(map(str,abs_val)) )
                    print(str(mm)+'+/-'+str(ss))
                    print(' ')

    ## PLOT
    counter = 0
    fig = plt.figure( figsize=(30,20) )
    dico_col = {'abrupt-4xCO2':CB_color_cycle[4], 'abrupt-2xCO2':CB_color_cycle[1], 'abrupt-0p5xCO2':CB_color_cycle[0]}
    dico_ls = {'abrupt-4xCO2':'-', 'abrupt-2xCO2':'-', 'abrupt-0p5xCO2':'-'}
    n_bins_distrib = 100

    ax = plt.subplot(131)
    # VAR = 'Net Ocean to Atmosphere Flux|CO2'
    VAR = 'tas'
    for xp in ['abrupt-4xCO2' , 'abrupt-2xCO2', 'abrupt-0p5xCO2']:
        ## loading XP minus control
        OUT = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
        ## ploting
        # func_plot('year',VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
        func_plot(np.arange(0,OUT['year'].size),VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)
    plt.xlabel( 'Year' , size=fac_size*15 )
    # plt.title( '(a) '+VAR+' ('+OUT[VAR].unit+')'  , size=fac_size*14)#,fontweight='bold')
    plt.xlim(0,1000)
    # plt.xticks( np.arange(0,OUT['year'].size,100) )
    ax.tick_params(labelsize=fac_size*13)
    ## cleaning
    OUT.close()
    plt.grid()
    plt.ylim( ax.get_ylim()[0],1.1*ax.get_ylim()[1] )
    plt.legend(loc='upper left',prop={'size':fac_size*13})
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0+0.02, box.width*1.0, box.height*1.0])
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ## second column
    for xp in ['abrupt-4xCO2' , 'abrupt-2xCO2' , 'abrupt-0p5xCO2']:
        ax = plt.subplot(3,3,2+['abrupt-4xCO2', 'abrupt-2xCO2' , 'abrupt-0p5xCO2'].index(xp)*3)
        dico_lw_rescale= {'vals_':3.,'vals_RescaleCO2_':1.5}
        dico_ls_rescale= {'vals_':'-','vals_RescaleCO2_':'--'}
        plt.ylim(0,1.7*1.5)
        ax.set_axisbelow(True)
        for type_rescale in ['vals_']:#'vals_','vals_RescaleCO2_']:
            ## ploting our values
            ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(VALS[type_rescale+xp]))[0]
            mm = np.average( VALS[type_rescale+xp].isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
            ss = np.sqrt(  np.average( (VALS[type_rescale+xp].isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )   *    (np.sum(list(dico_sizesMC.values()))-1.5)/np.sum(list(dico_sizesMC.values()))  )
            ## Pearson's moment coefficient of skewness: (average(X**3.) - 3*average(X)*std_dev(X)**2. - average(X)**3.)  /  std_dev(X)**3.
            tmp = np.average( VALS[type_rescale+xp].isel(all_config=ind)**3. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind)  )
            skew = (tmp-3*mm*ss**2.-mm**3.)  /  ss**3.
            out = plt.hist( x=VALS[type_rescale+xp].isel(all_config=ind).values ,bins=n_bins_distrib,density=True,weights=weights_CMIP6.weights.isel(all_config=ind).values , alpha=0.5 , color=dico_col[xp] , histtype='step',lw=0.75*fac_size*dico_lw_rescale[type_rescale],ls=dico_ls_rescale[type_rescale] )
            plt.xlim(1.5,4.5)
            xl,yl,pos = ax.get_xlim(),[np.nanmin(out[0]),np.nanmax(out[0])],0.4+0.1*['abrupt-0p5xCO2' , 'abrupt-2xCO2' , 'abrupt-4xCO2'].index(xp)
            plt.axhline( y=1.05*(yl[1]-yl[0]) , xmin=(mm-ss-xl[0])/(xl[1]-xl[0]),xmax=(mm+ss-xl[0])/(xl[1]-xl[0]),color=dico_col[xp],lw=0.75*fac_size*dico_lw_rescale[type_rescale]*1.0,ls=dico_ls_rescale[type_rescale] , label=str(np.round(mm,2))+'$\pm$'+str(np.round(ss,2))+'K' )# xp+': '+str(np.round(mm,2))+'$\pm$'+str(np.round(ss,2))+'K ['+str(np.round(skew,2))+']'
            plt.scatter( y=1.05*(yl[1]-yl[0]) , x=mm , facecolor=dico_col[xp],edgecolor='k',marker='o',s=fac_size* 1.5*dico_lw_rescale[type_rescale]*20 )
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False,# labels along the bottom edge are off
            labelsize=fac_size*15)
        # plt.ylim( 0 , ax.get_ylim()[1]*1.25  )
        if xp == 'abrupt-4xCO2':
            # plt.title( '(c) ECS, with all feedbacks (K)' , size=fac_size*14 )#,fontweight='bold' )
            ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        elif xp == 'abrupt-2xCO2':
            ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        elif xp == 'abrupt-0p5xCO2':
            pass
        plt.grid()
        ax.tick_params(labelsize=fac_size*13)
        plt.legend(loc=0,prop={'size':fac_size*13})
        box = ax.get_position()
        ax.set_position([box.x0+0.0, box.y0+0.02, box.width*1.0, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
        counter += 1

    ## third column
    tmp = ['rf_tot','RF_CO2', 'RF_O3t', 'RF_absorb','RF_scatter','RF_cloud' ]# all about GhG and 'RF_H2Os','RF_O3s','RF_BCsnow'
    dico_tmp = { 'rf_tot':'RF','RF_CO2':'CO$_2$', 'RF_H2Os':'H$_2$O$^s$','RF_O3s':'O$_3^s$', 'RF_O3t':'O$_3$ (trop.)', 'RF_absorb':'Absorbing\naerosols', 'RF_scatter':'Scattering\naerosols', 'RF_cloud':'Clouds', 'RF_BCsnow':'Deposition of\nBC on snow' }
    for xp in ['abrupt-4xCO2' , 'abrupt-2xCO2' , 'abrupt-0p5xCO2']:
        ax = plt.subplot(3,3,3+['abrupt-4xCO2', 'abrupt-2xCO2' , 'abrupt-0p5xCO2'].index(xp)*3)
        OUT = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
        plt.ylim(  { 'abrupt-4xCO2':[-1.5,8] , 'abrupt-2xCO2':[-1,4.] , 'abrupt-0p5xCO2':[0.5,-4] }[xp] )
        plt.xlim(0,len(tmp)+1)
        xl,yl = plt.xlim(),plt.ylim()
        for var in tmp:
            if 'stat_value' in OUT[var].dims:
                mm,ss = OUT[var].isel(year=-1).values
                # plt.axvline( x=1+tmp.index(var) , ymin=(mm-ss-yl[0])/(yl[1]-yl[0]),ymax=(mm+ss-yl[0])/(yl[1]-yl[0]),color='k',lw=0.75*fac_size*3*1.0,ls='-' , label=xp+': '+str(np.round(mm,2))+'$\pm$'+str(np.round(ss,2))+'K ['+str(np.round(skew,2))+']' ,zorder=100)
                plt.bar( x=1+tmp.index(var) , bottom=mm-ss, height=2*ss , width=0.05, color='k',edgecolor='k' ,zorder=100 )
                print(var+' in '+xp+': '+str(mm)+'+/-'+str(ss))
            else:
                mm = OUT[var].isel(year=-1).values
                print(var+' in '+xp+': '+str(mm))
            plt.bar( x=1+tmp.index(var) , height=mm , width=0.15, color=dico_col[xp],edgecolor=dico_col[xp] )
        plt.xticks( np.arange(0,len(tmp)+1), ['']+[dico_tmp[var] for var in tmp] , rotation = 45)
        if xp == 'abrupt-4xCO2':
            # plt.title( '(b) Contributions to RF (W.m$^{-2}$)' , size=fac_size*14 )#,fontweight='bold' )
            ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        elif xp == 'abrupt-2xCO2':
            ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        elif xp == 'abrupt-0p5xCO2':
            pass
        plt.grid()
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=fac_size*11)
        box = ax.get_position()
        ax.set_position([box.x0+0.0, box.y0+0.02, box.width*1.0, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
        counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/abrupt.pdf',dpi=300 )
    plt.close(fig)
#########################
#########################



#########################
## 7.2. Carbon Cycle Response
#########################
if '7.2-figure' in option_which_plots:
    counter = 0
    fig = plt.figure( figsize=(30,20) )
    dico_col = {'1pctCO2':CB_color_cycle[7], '1pctCO2-rad':CB_color_cycle[3], '1pctCO2-bgc':CB_color_cycle[0]}
    dico_ls = {'1pctCO2':'-', '1pctCO2-rad':'-', '1pctCO2-bgc':'-'}
    n_bins_distrib = 200
    for xp in ['1pctCO2', '1pctCO2-rad', '1pctCO2-bgc']:
        ## loading XP minus control
        if True:
            # list_VAR = ['permafrostCO2' , 'tas'  ,  'nbp'  ,  'fgco2']
            list_VAR = ['co2', 'tas', 'nbp', 'fgco2']
            OUT = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
            dico_label_var = {'co2':'Atmospheric CO2 (ppm)', 'tas':'Increase in global mean surface temperature (K)'  ,  'nbp':'Net Biome Productivity (PgC.yr$^{-1}$)'  ,  'fgco2':'Ocean carbon sink (PgC.yr$^{-1}$)'  ,  'permafrostCO2':'CO$_2$ emissions from permafrost (PgC.yr$^{-1}$)'}
            OUT['co2'] += 284.31702
        else:
            list_VAR = ['Atmospheric Concentrations|CO2','Surface Air Temperature Change','Net Land to Atmosphere Flux|CO2','Net Ocean to Atmosphere Flux|CO2','Cumulative Net Land to Atmosphere Flux|CO2','Cumulative Net Ocean to Atmosphere Flux|CO2']
            raise Exception("Dont use RCMIP intermediary")
            # OUT = xr.open_dataset( path_all+'/treated/RCMIP/intermediary/'+os.listdir(path_all+'/treated/RCMIP/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/RCMIP/intermediary/')]))[0][0]] )
            OUT = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
            dico_label_var = {'Atmospheric Concentrations|CO2':'Atmospheric CO2 (ppm)' , 'Surface Air Temperature Change':'Increase in global mean surface temperature (K)' , 'Net Land to Atmosphere Flux|CO2':'Atmosphere-land flux of CO$_2$ (PgC.yr$^{-1}$)' , 'Net Ocean to Atmosphere Flux|CO2':'Atmosphere-ocean flux of CO$_2$ (PgC.yr$^{-1}$)' , 'Cumulative Net Land to Atmosphere Flux|CO2':'Cumulative atmosphere-land flux of CO$_2$ (PgC.yr$^{-1}$)' , 'Cumulative Net Ocean to Atmosphere Flux|CO2':'Cumulative atmosphere-ocean flux of CO$_2$ (PgC.yr$^{-1}$)'}
            OUT['Atmospheric Concentrations|CO2'] += 284.31702
        for VAR in list_VAR:
            ax = plt.subplot(int(len(list_VAR)/2),2,list_VAR.index(VAR)+1)
            if VAR in ['Net Land to Atmosphere Flux|CO2','Net Ocean to Atmosphere Flux|CO2','Cumulative Net Land to Atmosphere Flux|CO2','Cumulative Net Ocean to Atmosphere Flux|CO2']:
                OUT[VAR] /= -(1.e3 * 44/12.)
            ## ploting
            func_plot(np.arange(0,150+1),VAR,OUT,col=dico_col[xp],ls=dico_ls[xp],label=xp)#'co2'
            if list_VAR.index(VAR)+1>2*(len(list_VAR)/2-1):
                plt.xlabel( 'Year' , size=fac_size*14)#,fontweight='bold' )
            else:
                ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
            plt.title( dico_label_var[VAR]  , size=fac_size*14 )#,fontweight='bold')
            OUT.close()
            plt.grid()
            plt.xlim(0,150)
            ax.tick_params(labelsize=fac_size*13)
            box = ax.get_position()
            ax.set_position([box.x0+0.0, box.y0+0.02-0.005*int(list_VAR.index(VAR)/2), box.width*1.0, box.height*1.0])
            if xp == '1pctCO2-bgc':
                plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.95*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
                counter += 1
        plt.legend(loc='center',prop={'size':fac_size*15},ncol=3 ,bbox_to_anchor=(-0.125,-0.20))
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/carbon-cycle_response.pdf',dpi=300 )
    plt.close(fig)
    


## table, alpha, beta, gamma
if '7.2-table' in option_which_plots:
    ## preparing data
    list_dom = ['atmosphere_plus','land_plus','ocean','atmosphere','land']
    # list_var = ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp'] + ['D_Fland','D_Focean','D_Epf_CO2','D_Epf_CH4'] + ['D_Eluc','D_Foxi_CH4']
    # VAR_accelerate = ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp']
    list_xp = ['1pctCO2-rad','1pctCO2-bgc','1pctCO2']
    list_tt = [70.,140.]
    dico_names_tt = {70:'2xCO2',140:'4xCO2'}

    if option_overwrite:
        TMP_xp = xr.Dataset()
        TMP_xp.coords['all_config'] = np.array([str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])])
        TMP_xp.coords['year'] = np.arange( 1850,1850+140+1 )
        for xp in list_xp:
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp'] + ['D_Fland','D_Focean','D_Epf_CO2','D_Epf_CH4'] + ['D_Eluc','D_Foxi_CH4'] )

                ## doing temperatures
                Txp = ((out_tmp['D_Tg']-out_tmp['D_Tg'].isel(year=0)) * mask[:,:]).sel(year=np.arange(1850,1850+list_tt[-1]+1))
                ## saving
                if 'D_Tg_'+xp not in TMP_xp:
                    TMP_xp['D_Tg_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=[TMP_xp.year.size,TMP_xp.all_config.size]) , dims=['year','all_config'] )
                TMP_xp['D_Tg_'+xp].loc[{'year':Txp.year,'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]}] = Txp.transpose('year','config')#out_tmp[vv]
                del Txp

                ## doing co2
                co2 = ((out_tmp['D_CO2']-out_tmp['D_CO2'].isel(year=0)) * mask[:,:]).sel(year=np.arange(1850,1850+list_tt[-1]+1))
                ## saving
                if 'D_CO2_'+xp not in TMP_xp:
                    TMP_xp['D_CO2_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=[TMP_xp.year.size,TMP_xp.all_config.size]) , dims=['year','all_config'] )
                TMP_xp['D_CO2_'+xp].loc[{'year':co2.year,'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]}] = co2.transpose('year','config')#out_tmp[vv]
                del co2

                ## looping on domains
                for dom in list_dom:
                    if dom =='atmosphere_plus':
                        ## integer of flux going into atmosphere
                        Flux = (-out_tmp['D_Fland']-out_tmp['D_Focean']+out_tmp['D_Eluc']+out_tmp['D_Epf_CO2'].sum('reg_pf',min_count=1)+out_tmp['D_Foxi_CH4']).sel(year=np.arange(1850,1850+list_tt[-1]+1))#.sum('year')
                    elif dom =='atmosphere':
                        ## integer of flux going into atmosphere
                        Flux = (-out_tmp['D_Fland']-out_tmp['D_Focean']+out_tmp['D_Eluc']).sel(year=np.arange(1850,1850+list_tt[-1]+1))#.sum('year')
                    elif dom =='land_plus':
                        ## integer of flux going into land
                        Flux = (out_tmp['D_Fland']-out_tmp['D_Eluc']-out_tmp['D_Epf_CO2'].sum('reg_pf',min_count=1)-out_tmp['D_Epf_CH4'].sum('reg_pf',min_count=1)*1.e-3).sel(year=np.arange(1850,1850+list_tt[-1]+1))
                        # Flux =  ( (Par.csoil1_0+out_tmp['D_csoil1']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum('bio_land' , min_count=1)
                        # Flux += ( (Par.csoil2_0+out_tmp['D_csoil2']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum('bio_land' , min_count=1)
                        # Flux += ( (Par.cveg_0+out_tmp['D_cveg']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum('bio_land' , min_count=1))
                        # Flux += ( out_tmp['D_Csoil1_bk']+out_tmp['D_Csoil2_bk']+out_tmp['D_Cveg_bk'] ).sum('bio_from' , min_count=1).sum('bio_to',min_count=1)
                        # Flux += out_tmp['D_Chwp'].sum('bio_from',min_count=1).sum('bio_to',min_count=1).sum('box_hwp',min_count=1) )
                        # Flux -= ((Par.csoil1_0+Par.csoil2_0+Par.cveg_0)*Par.Aland_0).sum('bio_land',min_count=1)
                        # Flux += -out_tmp['D_Epf_CO2'].sum('reg_pf',min_count=1)-out_tmp['D_Epf_CH4'].sum('reg_pf',min_count=1)*1.e-3
                        # Flux = Flux.sum( 'reg_land' , min_count=1 ).sel(year=np.arange(1850,1850+140+1))##.sel(year=1850+140)
                    elif dom =='land':
                        ## integer of flux going into land
                        Flux = (out_tmp['D_Fland']-out_tmp['D_Eluc']).sel(year=np.arange(1850,1850+list_tt[-1]+1))
                        # Flux =  ( (Par.csoil1_0+out_tmp['D_csoil1']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum('bio_land' , min_count=1)
                        # Flux += ( (Par.csoil2_0+out_tmp['D_csoil2']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum('bio_land' , min_count=1)
                        # Flux += ( (Par.cveg_0+out_tmp['D_cveg']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum('bio_land' , min_count=1))
                        # Flux += ( out_tmp['D_Csoil1_bk']+out_tmp['D_Csoil2_bk']+out_tmp['D_Cveg_bk'] ).sum('bio_from' , min_count=1).sum('bio_to',min_count=1)
                        # Flux += out_tmp['D_Chwp'].sum('bio_from',min_count=1).sum('bio_to',min_count=1).sum('box_hwp',min_count=1) )
                        # Flux -= ((Par.csoil1_0+Par.csoil2_0+Par.cveg_0)*Par.Aland_0).sum('bio_land',min_count=1)
                        # Flux = Flux.sum( ('reg_land') ).sel(year=np.arange(1850,1850+140+1))##.sel(year=1850+140)
                    elif dom =='ocean':
                        ## integer of flux going into ocean
                        Flux = out_tmp['D_Focean'].sel(year=np.arange(1850,1850+list_tt[-1]+1))#.sum('year')
                    ## saving
                    if dom+'_'+xp not in TMP_xp:
                        TMP_xp[dom+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=[TMP_xp.year.size,TMP_xp.all_config.size]) , dims=['year','all_config'] )
                    ## allocating
                    TMP_xp[dom+'_'+xp].loc[{'year':Flux.year,'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]}] = Flux.cumsum('year').transpose('year','config')#out_tmp[vv]
                    del Flux
                ## cleaning
                out_tmp.close()

        TMP_xp.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_table_betgam.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP_xp})
    else:
        TMP_xp = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_table_betgam.nc' )

    ## preparing
    gamma,beta = xr.Dataset(),xr.Dataset()
    gamma.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
    beta.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
    gamma.coords['stat_value'] = ['mean','std_dev']
    beta.coords['stat_value'] = ['mean','std_dev']
    ## adding climate sensitivity
    alpha = xr.Dataset()
    alpha.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
    alpha.coords['stat_value'] = ['mean','std_dev']

    for tt in list_tt:
        alpha[dico_names_tt[tt]] = ( TMP_xp['D_Tg_1pctCO2'] / TMP_xp['D_CO2_1pctCO2'] ).sel(year=1850+tt)
        ## Method RB:
        for dom in list_dom:
            gamma[dom+'_RB'+'_'+dico_names_tt[tt]] = ( TMP_xp[dom+'_1pctCO2-rad'] / TMP_xp['D_Tg_1pctCO2-rad'] ).sel(year=1850+tt)
            beta[dom+'_RB'+'_'+dico_names_tt[tt]]  = ( (TMP_xp[dom+'_1pctCO2-bgc']-gamma[dom+'_RB'+'_'+dico_names_tt[tt]]*TMP_xp['D_Tg_1pctCO2-bgc']) / TMP_xp['D_CO2_1pctCO2'] ).sel(year=1850+tt)
        ## Method RF:
        for dom in list_dom:
            gamma[dom+'_RF'+'_'+dico_names_tt[tt]] = ( TMP_xp[dom+'_1pctCO2-rad'] / TMP_xp['D_Tg_1pctCO2-rad'] ).sel(year=1850+tt)
            beta[dom+'_RF'+'_'+dico_names_tt[tt]]  = ( (TMP_xp[dom+'_1pctCO2']-gamma[dom+'_RF'+'_'+dico_names_tt[tt]]*TMP_xp['D_Tg_1pctCO2']) / TMP_xp['D_CO2_1pctCO2'] ).sel(year=1850+tt)
        ## Method BF:
        for dom in list_dom:
            gamma[dom+'_BF'+'_'+dico_names_tt[tt]] = ( (TMP_xp[dom+'_1pctCO2']-TMP_xp[dom+'_1pctCO2-bgc']) / (TMP_xp['D_Tg_1pctCO2']-TMP_xp['D_Tg_1pctCO2-bgc']) ).sel(year=1850+tt)
            beta[dom+'_BF'+'_'+dico_names_tt[tt]]  = ( (TMP_xp[dom+'_1pctCO2-bgc']*TMP_xp['D_Tg_1pctCO2']-TMP_xp[dom+'_1pctCO2']*TMP_xp['D_Tg_1pctCO2-bgc']) / (TMP_xp['D_Tg_1pctCO2']-TMP_xp['D_Tg_1pctCO2-bgc']) / TMP_xp['D_CO2_1pctCO2'] ).sel(year=1850+tt)

    ## statistical values
    for tt in list_tt:
        for dom in list_dom:
            for method in ['RB','RF','BF']:
                for IsConst in ['Const','NonConst']:
                    gamma[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst] = xr.DataArray(  np.ones((gamma.stat_value.size)) , dims=('stat_value')  )
                    beta[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst] = xr.DataArray(  np.ones((beta.stat_value.size)) , dims=('stat_value')  )
                    ww = weights_CMIP6.weights.values * indic['m'].isel(index=0).values
                    ## gamma
                    ind = np.where( ~np.isnan(ww) & ~np.isnan(gamma[dom+'_'+method+'_'+dico_names_tt[tt]]))[0]
                    if IsConst == 'Const':
                        mm = np.average( gamma[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) ,axis=0, weights=ww[ind] )
                        gamma[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'mean'}] = mm
                        gamma[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'std_dev'}] = np.sqrt(np.average( (gamma[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) - mm)**2. ,axis=0, weights=ww[ind] ))
                    else:
                        mm = np.average( gamma[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) ,axis=0, weights=np.ones(len(ww[ind])) )
                        gamma[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'mean'}] = mm
                        gamma[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'std_dev'}] = np.sqrt(np.average( (gamma[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) - mm)**2. ,axis=0, weights=np.ones(len(ww[ind])) ))
                    ## beta
                    ind = np.where( ~np.isnan(ww) & ~np.isnan(beta[dom+'_'+method+'_'+dico_names_tt[tt]]))[0]
                    if IsConst == 'Const':
                        mm = np.average( beta[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) ,axis=0, weights=ww[ind] )
                        beta[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'mean'}] = mm
                        beta[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'std_dev'}] = np.sqrt(np.average( (beta[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) - mm)**2. ,axis=0, weights=ww[ind] ))
                    else:
                        mm = np.average( beta[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) ,axis=0, weights=np.ones(len(ww[ind])) )
                        beta[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'mean'}] = mm
                        beta[dom+'_'+method+'_'+dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'std_dev'}] = np.sqrt(np.average( (beta[dom+'_'+method+'_'+dico_names_tt[tt]].isel(all_config=ind) - mm)**2. ,axis=0, weights=np.ones(len(ww[ind])) ))
    ## alpha
    for tt in list_tt:
        for IsConst in ['Const','NonConst']:
            alpha[dico_names_tt[tt]+'_'+IsConst] = xr.DataArray(  np.ones((alpha.stat_value.size)) , dims=('stat_value')  )
            ww = weights_CMIP6.weights.values * indic['m'].isel(index=0).values
            ind = np.where( ~np.isnan(ww) & ~np.isnan(alpha[dico_names_tt[tt]]))[0]
            if IsConst == 'Const':
                mm = np.average( alpha[dico_names_tt[tt]].isel(all_config=ind) ,axis=0, weights=ww[ind] )
                alpha[dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'mean'}] = mm
                alpha[dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'std_dev'}] = np.sqrt(np.average( (alpha[dico_names_tt[tt]].isel(all_config=ind) - mm)**2. ,axis=0, weights=ww[ind] ))
            else:
                mm = np.average( alpha[dico_names_tt[tt]].isel(all_config=ind) ,axis=0, weights=np.ones(len(ww[ind])) )
                alpha[dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'mean'}] = mm
                alpha[dico_names_tt[tt]+'_'+IsConst].loc[{'stat_value':'std_dev'}] = np.sqrt(np.average( (alpha[dico_names_tt[tt]].isel(all_config=ind) - mm)**2. ,axis=0, weights=np.ones(len(ww[ind])) ))

    alpha.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/alpha.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in alpha})
    beta.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/beta.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in beta})
    gamma.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/gamma.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in gamma})

    alpha = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/alpha.nc' )
    beta = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/beta.nc' )
    gamma = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/gamma.nc' )

    ## printing values for beta and gamma, mean and std only
    for dom in ['land_plus', 'ocean']:
        for xCO2 in ['2xCO2','4xCO2']:
            for const in ['Const','NonConst']:
                for method in ['RB','RF','BF']:
                    print('BETA :: '+dom+' at '+xCO2+' for '+method+' '+const+': '+str(beta[dom+'_'+method+'_'+xCO2+'_'+const].values[0])+'+/-'+str(beta[dom+'_'+method+'_'+xCO2+'_'+const].values[1]))
                    print('GAMMA :: '+dom+' at '+xCO2+' for '+method+' '+const+': '+str(gamma[dom+'_'+method+'_'+xCO2+'_'+const].values[0])+'+/-'+str(gamma[dom+'_'+method+'_'+xCO2+'_'+const].values[1]))
    ## printing values for alpha, mean, std and and quantiles
    if False:
        for xCO2 in ['2xCO2','4xCO2']:
            for const in ['Const','NonConst']:
                print('ALPHA :: at '+xCO2+' '+const+': '+str(alpha[xCO2+'_'+const].values[0])+'+/-'+str(alpha[xCO2+'_'+const].values[1]))
                ww = weights_CMIP6.weights.values * indic['m'].isel(index=0).values
                ind = np.where( ~np.isnan(ww) & ~np.isnan(alpha[dico_names_tt[tt]]))[0]
                values = alpha[xCO2].isel(all_config=ind)
                if const == 'Const':
                    abs_val = [weighted_quantile(values, pct, ww[ind]) for pct in [0.05, 0.17, 0.50, 0.83, 0.95]]
                elif const == 'NonConst':
                    abs_val = [weighted_quantile(values, pct, np.ones(len(ww[ind])) ) for pct in [0.05, 0.17, 0.50, 0.83, 0.95]]
                print(const+' quantiles: ' + ' '.join(map(str,abs_val)) )
                



## values CMIP5 for beta gamma
if False:
    list_MDL = ['bcc-csm1-1','CanESM2','CESM1-BGC','HadGEM2-ES','IPSL-CM5A-LR','MPI-ESM-LR']
    dico_gamma = {}
    for mdl in list_MDL:
        ## getting experiment rad
        xp = 'esmFdbk1'# '1pctCO2'  |  'esmFdbk1'
        with open('C:/Users/quilcail/Documents/Repositories/OSCARv31_CMIP6/CMIP5/CMIP5_'+mdl+'_'+xp+'.csv','r',newline='') as ff:
            TMP = np.array([line for line in csv.reader(ff)])
        head,TMP = list(TMP[0]),np.array(TMP[1:],dtype=np.float32)
        ## getting 2nd experiment
        xp = 'piControl' # 'esmFixClim1' |  'piControl'
        with open('C:/Users/quilcail/Documents/Repositories/OSCARv31_CMIP6/CMIP5/CMIP5_'+mdl+'_'+xp+'.csv','r',newline='') as ff:
            TMP2 = np.array([line for line in csv.reader(ff)])
        head2,TMP2 = list(TMP2[0]),np.array(TMP2[1:],dtype=np.float32)
        ## calculating gamma
        if mdl == 'HadGEM2-ES':
            stock = (TMP[:140,head.index('cVeg')]+TMP[:140,head.index('cSoil')])  -  (TMP2[:140,head2.index('cVeg')]+TMP2[:140,head2.index('cSoil')])
        else:
            stock = (TMP[:140,head.index('cVeg')]+TMP[:140,head.index('cLitter')]+TMP[:140,head.index('cSoil')])  -  (TMP2[:140,head2.index('cVeg')]+TMP2[:140,head2.index('cLitter')]+TMP2[:140,head2.index('cSoil')])
        temp = TMP[:140,head.index('tas')] - TMP2[:140,head.index('tas')]
        # dico_gamma[mdl]  =  stock[-1] / temp[-1]
        dico_gamma[mdl]  =  np.mean(stock[140-10:140]) / np.mean(temp[140-10:140])
        plt.plot( stock/temp , label=mdl )
    plt.legend(loc=0)
    plt.grid()

#########################
#########################




#########################
## 7.3. Solar geo-engineering
#########################
## G1: For['RF_solar'] -= Par.rf_CO2 * (np.log1p( For.D_CO2.isel(year=0) / Par.CO2_0) - np.log1p( (TMP['CO2'].loc[{'year':year_PI,'region':'Globe'}] - Par['CO2_0']) / Par.CO2_0))
## G2: For['RF_solar'] -= Par.rf_CO2 * (np.log1p( For.D_CO2 / Par.CO2_0) - np.log1p( (TMP['CO2'].loc[{'year':year_PI,'region':'Globe'}] - Par['CO2_0']) / Par.CO2_0))
if '7.3' in option_which_plots:
    ## figure
    counter = 0
    fig = plt.figure( figsize=(30,20) )
    # VAR_plot = ['fFireAll']
    VAR_plot = ['tas','cLand','cOcean','RF_AERtot','pr']# + ['RF_scatter','RF_absorb','RF_cloud']
    dico_col = {'abrupt-4xCO2':CB_color_cycle[0], '1pctCO2':CB_color_cycle[1],'G1':CB_color_cycle[2], 'G2':CB_color_cycle[4]}
    dico_ls = {'abrupt-4xCO2':'-', '1pctCO2':'-','G1':'--', 'G2':'--'}
    dico_title_VAR = {'cLand':'Change in land\ncarbon stock (PgC)','cOcean':'Change in oceanic\ncarbon stock (PgC)','RF_AERtot':'Radiative forcing\nof aerosols (W.m$^{-2}$)','rf_tot':'Radiative forcing (W.m$^{-2}$)', 'RF_solar':'Radiative forcing\nof solar activity (W.m$^{-2}$)' , 'RF_CO2':'Radiative forcing\nof CO$_2$ (W.m$^{-2}$)','pr':'Change in global\nprecipitation (mm.yr$^{-1}$)','tas':'Change in global mean\nsurface temperature (K)' , 'RF_scatter':'Radiative forcing\nof scattering aerosols (W.m$^{-2}$)','RF_absorb':'Radiative forcing\nof absorbing aerosols (W.m$^{-2}$)','RF_cloud':'Radiative forcing\nof aerosols-clouds (W.m$^{-2}$)'}
    ## looping on variables / subplots
    for ii in [0,1]:
        list_xp = [ ['abrupt-4xCO2','G1'], ['1pctCO2', 'G2'] ][ii]
        period = [np.arange(1850,2750+1),np.arange(1850,2000+1)][ii]
        for var in VAR_plot:
            ax = plt.subplot(2,len(VAR_plot),ii*len(VAR_plot)+VAR_plot.index(var)+1)
            ## looping on experiments inside a subplot
            for xp in list_xp:
                OUT = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
                func_plot(np.arange(len(period)),var,OUT.sel(year=period),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*3,label=xp)
                if 'reg_land' in OUT[var].dims:
                    print(var+" in "+xp+" in 2000:"+str(OUT[var].sel(year=2000,stat_value='mean',reg_land=0).values)+'+/-'+str(OUT[var].sel(year=2000,stat_value='std_dev',reg_land=0).values))
                else:
                    print(var+" in "+xp+" in 2000:"+str(OUT[var].sel(year=2000,stat_value='mean').values)+'+/-'+str(OUT[var].sel(year=2000,stat_value='std_dev').values))
            ## polishing
            plt.grid()
            plt.xlim(0,len(period)-1)
            if ii == 0:
                plt.title( dico_title_VAR[var] , size=fac_size*14,rotation=0  )#,fontweight='bold'
            if var == VAR_plot[0]:
                # plt.legend(loc=0,prop={'size':fac_size*12} ,bbox_to_anchor=(-0.125,0.62))
                plt.legend(loc='upper left',prop={'size':fac_size*12})
                ax.yaxis.set_label_coords(-0.40,0.5)
            ax.tick_params(labelsize=fac_size*13)
            box = ax.get_position()
            ax.set_position([box.x0-0.01+0.012*(VAR_plot.index(var)), box.y0+0.02, box.width*1.0, box.height*1.0])
            # plt.text(x=period[0]+0.075*(period[-1]-period[0]),y=ax.get_ylim()[1]-0.1*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=fac_size*'('+  ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p'][ii*len(VAR_plot)+VAR_plot.index(var)] +')',fontdict={'size':fac_size*13})
        for var in VAR_plot:
            ax = plt.subplot(2,len(VAR_plot),ii*len(VAR_plot)+VAR_plot.index(var)+1)
            plt.xlabel('Time (year)', size=fac_size*13)
            plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
            counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/geomip_G1-2.pdf',dpi=300 )
    plt.close(fig)

#########################
#########################



#########################
## 7.4. Carbon geoengineering (1pctCO2-cdr) + (AGWP, AGTP)
#########################
if '7.4-cdr' in option_which_plots: # 1pctCO2-cdr
    ## figure
    counter = 0
    fig = plt.figure( figsize=(30,20) )
    # VAR_plot = ['tas','cLand','cOcean','RF_AERtot','pr']# + ['RF_scatter','RF_absorb','RF_cloud']
    # VAR_plot = ['RF_CO2','RF_solar','RF_AERtot','tas','pr']
    # VAR_plot = ['cLand','tas','cOcean','cPermafrostFrozen']
    VAR_plot = [ ['cLand','cOcean'], ['fFireAll','wetlandCH4','cPermafrostFrozen'], ['tas','pr'] ]
    list_col = [ CB_color_cycle[3],CB_color_cycle[0],CB_color_cycle[7] ]
    dico_title_VAR = {'cLand':'Change in land carbon stock (PgC)','cOcean':'Change in oceanic carbon stock (PgC)','RF_AERtot':'Radiative forcing of aerosols (W.m$^{-2}$)','rf_tot':'Radiative forcing (W.m$^{-2}$)', 'RF_solar':'Radiative forcing of solar activity (W.m$^{-2}$)' , 'RF_CO2':'Radiative forcing of CO$_2$ (W.m$^{-2}$)','pr':'Change in global precipitation (mm.yr$^{-1}$)','tas':'Change in global mean\nsurface temperature (K)' , 'RF_scatter':'Radiative forcing of scattering aerosols (W.m$^{-2}$)','RF_absorb':'Radiative forcing of absorbing aerosols (W.m$^{-2}$)','RF_cloud':'Radiative forcing of aerosols-clouds (W.m$^{-2}$)','cPermafrostFrozen':'Change in the permafrost carbon stock (PgC)', 'fFireAll':'Carbon emissions due to fire,\nall sources (PgC.yr$^{-1}$)', 'wetlandCH4':'Methane emissions from wetlands (TgC.yr$^{-1}$)'}
    ## looping on variables / subplots
    xp = '1pctCO2-cdr'
    OUT = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
    for VAR in VAR_plot:
        for var in VAR:
            ax = plt.subplot(len(VAR),len(VAR_plot), VAR_plot.index(VAR)+1+VAR.index(var)*len(VAR_plot))
            ## looping on experiments inside a subplot
            func_plot('co2',var,OUT.sel(year=np.arange(1850,1850+140+1)),col=list_col[0],label='+1$\%$ CO$_2$ / yr',marker='>',markevery=35,mec=list_col[0],ms=fac_size*8,lw=0.75*fac_size*3)
            func_plot('co2',var,OUT.sel(year=np.arange(1850+140,1850+2*140+1)),col=list_col[1],label='-1$\%$ CO$_2$ / yr',marker='<',markevery=35,mec=list_col[1],ms=fac_size*8,lw=0.75*fac_size*3)
            if var in ['cPermafrostFrozen']:
                func_plot('co2',var,OUT.sel(year=np.arange(1850+2*140,1850+2*140+1+1000)),col=list_col[2],label='constant CO2',marker='^',markevery=35,mec=list_col[2],ms=fac_size*8,lw=0.75*fac_size*3)
            else:
                func_plot('co2',var,OUT.sel(year=np.arange(1850+2*140,1850+2*140+1+1000)),col=list_col[2],label='constant CO2',marker='v',markevery=35,mec=list_col[2],ms=fac_size*8,lw=0.75*fac_size*3)
            if 'reg_land' in OUT[var].dims:
                print(var+" in "+xp+" in 1850+140:"+str(OUT[var].sel(year=1850+140,stat_value='mean',reg_land=0).values)+'+/-'+str(OUT[var].sel(year=1850+140,stat_value='std_dev',reg_land=0).values))
                print(var+" in "+xp+" in 1850+280:"+str(OUT[var].sel(year=1850+280,stat_value='mean',reg_land=0).values)+'+/-'+str(OUT[var].sel(year=1850+280,stat_value='std_dev',reg_land=0).values))
            else:
                print(var+" in "+xp+" in 1850+140:"+str(OUT[var].sel(year=1850+140,stat_value='mean').values)+'+/-'+str(OUT[var].sel(year=1850+140,stat_value='std_dev').values))
                print(var+" in "+xp+" in 1850+280:"+str(OUT[var].sel(year=1850+280,stat_value='mean').values)+'+/-'+str(OUT[var].sel(year=1850+280,stat_value='std_dev').values))
            if var == VAR[-1]:plt.xlabel( 'Atmospheric CO2 (ppm)', size=fac_size*13)#,fontweight='bold' )
            # func_plot('year',var,OUT.sel(year=period),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*3,label=xp)
            ## polishing
            plt.grid()
            plt.xlim( -10. , ax.get_xlim()[1] )
            if var in ['cPermafrostFrozen']:plt.ylim( ax.get_ylim()[0] , 0.01*(ax.get_ylim()[1]-ax.get_ylim()[0]) )
            else:plt.ylim( -0.01*(ax.get_ylim()[1]-ax.get_ylim()[0]) , ax.get_ylim()[1] )
            plt.title( dico_title_VAR[var] , size=fac_size*12,rotation=0  )#,fontweight='bold'
            if (VAR == VAR_plot[-2])  and  (var==VAR[-1]):
                plt.legend(loc='center',prop={'size':fac_size*13} ,bbox_to_anchor=(0.5,-0.30),ncol=3)
                ax.yaxis.set_label_coords(-0.40,0.5)
            ax.tick_params(labelsize=fac_size*11)
            if var!= VAR[-1]:ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
            box = ax.get_position()
            ax.set_position([box.x0-0.00, box.y0+0.04, box.width*1.0, box.height*1.0])
            plt.text(x=ax.get_xlim()[0]+0.1*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.90*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
            counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/1pctCO2-cdr.pdf',dpi=300 )
    plt.close(fig)

    ## computing time to return to 10% of perturbation
    for VAR in VAR_plot:
        for var in VAR:
            # threshold = {'tas':0.1 , 'cLand':50. , 'cOcean':50.  , 'cPermafrostFrozen':50. }[var]
            for ii in [0]:#[-1,0,1]:
                val = np.abs( (OUT[var].sel(stat_value='mean')+ii*OUT[var].sel(stat_value='std_dev')).sel(year=np.arange(1850+140,1850+140+1000+1)) )
                val_thres = np.abs( OUT[var].sel(stat_value='mean',year=1850+140) )
                if 'reg_land' in val.dims:val, val_thres = val.sel(reg_land=0), val_thres.sel(reg_land=0)
                aa = np.where( val < 0.1*val_thres )[0]
                if len(aa)!=0:
                    print( var+' returns below 10% of 1850+140 '+str(aa[0])+'yrs after)' )
                else:
                    print( var+'finishs with a ratio (value end)/(value 1850+140) '+str( (val.isel(year=-1)/val_thres) ) )



if '7.4-a' in option_which_plots: # AGWP, AGTP
    ## preparing data
    list_var = ['RF','D_Tg']
    # VAR_accelerate = ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp']
    list_xp = ['esm-pi-CO2pulse','esm-pi-cdr-pulse','esm-yr2010CO2-cdr-pulse','esm-yr2010CO2-CO2pulse','esm-yr2010CO2-control']#['esm-yr2010CO2-cdr-pulse','esm-yr2010CO2-CO2pulse','esm-yr2010CO2-control']

    if option_overwrite:
        TMP_xp = xr.Dataset()
        TMP_xp.coords['all_config'] = np.array([str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])])
        TMP_xp.coords['year'] = np.arange( 1850,2010+1000+1 )
        for xp in list_xp:
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                if xp in ['esm-yr2010CO2-cdr-pulse','esm-yr2010CO2-CO2pulse','esm-yr2010CO2-control']:
                    out_tmp,for_tmp,Par,mask = func_get( setMC , xp , list_var , option_NeedDiffControl=False ) ## using differences to esm-yr2010CO2-control
                else:
                    out_tmp,for_tmp,Par,mask = func_get( setMC , xp , list_var , option_NeedDiffControl=True ) ## using differences to esm-control
                ## preparing variable
                for vv in list_var:
                    if vv+'_'+xp not in TMP_xp:
                        TMP_xp[vv+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=[TMP_xp.year.size,TMP_xp.all_config.size]) , dims=['year','all_config'] )
                    ## allocating
                    TMP_xp[vv+'_'+xp].loc[{'year':out_tmp.year,'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]}] = out_tmp[vv]
                ## cleaning
                out_tmp.close()

        TMP_xp.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_pulses.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP_xp})
    else:
        TMP_xp = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_pulses.nc' )


    ## treating data
    dico_pulse = {'1860: CO2 pulse':['esm-pi-CO2pulse'] , '1860: CDR pulse':['esm-pi-cdr-pulse'] , '2015: CO2 pulse':['esm-yr2010CO2-CO2pulse','esm-yr2010CO2-control'] , '2015: CDR pulse':['esm-yr2010CO2-cdr-pulse','esm-yr2010CO2-control']  }
    VALS = xr.Dataset()
    VALS.coords['all_config'] = TMP_xp.all_config
    VALS.coords['year'] = np.arange(1150+1)
    for type_pulse in dico_pulse.keys():
        for var in ['AGWP','AGTP']:
            VALS[var+': '+type_pulse] = xr.DataArray( np.full(fill_value=np.nan,shape=(VALS.year.size,VALS.all_config.size)), dims=('year','all_config') )
            ## preparing values to analyze
            if var == 'AGWP':
                if len(dico_pulse[type_pulse])==1:
                    tmp = TMP_xp['RF'+'_'+dico_pulse[type_pulse][0]].sel(year=np.arange(int(type_pulse[:4]),TMP_xp.year[-1]+1)).cumsum('year')  /  100.
                else:
                    tmp = ( TMP_xp['RF'+'_'+dico_pulse[type_pulse][0]] - TMP_xp['RF'+'_'+dico_pulse[type_pulse][1]] ).sel(year=np.arange(int(type_pulse[:4]),TMP_xp.year[-1]+1)).cumsum('year')  /  100.
            else:
                if len(dico_pulse[type_pulse])==1:
                    tmp = TMP_xp['D_Tg'+'_'+dico_pulse[type_pulse][0]].sel(year=np.arange(int(type_pulse[:4]),TMP_xp.year[-1]+1))  /  100.
                else:
                    tmp = ( TMP_xp['D_Tg'+'_'+dico_pulse[type_pulse][0]] - TMP_xp['D_Tg'+'_'+dico_pulse[type_pulse][1]] ).sel(year=np.arange(int(type_pulse[:4]),TMP_xp.year[-1]+1))  /  100.
            VALS[var+': '+type_pulse].loc[{'year':np.arange(TMP_xp.year[-1]-int(type_pulse[:4])+1)}] = np.ma.array(tmp,mask=np.isnan(tmp))



    OUT = xr.Dataset()
    OUT.coords['stat_value'] = ['mean','std_dev']
    OUT.coords['year'] = np.arange(1150+1)
    ## AGWP and AGTP
    for type_pulse in dico_pulse.keys():
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP_xp.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP_xp.all_config).values))[np.newaxis,:],TMP_xp.year[-1]-int(type_pulse[:4])+1,axis=0)
        for var in ['AGWP','AGTP']:
            OUT[var+': '+type_pulse] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            val = VALS[var+': '+type_pulse].loc[{'year':np.arange(TMP_xp.year[-1]-int(type_pulse[:4])+1)}]
            OUT[var+': '+type_pulse].loc[{'stat_value':'mean','year':np.arange(TMP_xp.year[-1]-int(type_pulse[:4])+1)}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[var+': '+type_pulse].loc[{'stat_value':'std_dev','year':np.arange(TMP_xp.year[-1]-int(type_pulse[:4])+1)}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[var+': '+type_pulse].sel(stat_value='mean',year=np.arange(TMP_xp.year[-1]-int(type_pulse[:4])+1)).values[...,np.newaxis],TMP_xp.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    for var in ['AGWP','AGTP']:
        ## dependency of AGWP and AGTP to background
        OUT[var+': '+'background'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        val = VALS[var+': '+'1860: CO2 pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-1860+1)}]  -  VALS[var+': '+'2015: CO2 pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-2015+1)}]
        OUT[var+': '+'background'].loc[{'stat_value':'mean','year':np.arange(TMP_xp.year[-1]-2015+1)}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[var+': '+'background'].loc[{'stat_value':'std_dev','year':np.arange(TMP_xp.year[-1]-2015+1)}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[var+': '+'background'].sel(stat_value='mean',year=np.arange(TMP_xp.year[-1]-2015+1)).values[...,np.newaxis],TMP_xp.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
        ## dependency of AGWP and AGTP to sign
        OUT[var+': '+'sign'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        val = VALS[var+': '+'2015: CDR pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-2015+1)}]  +  VALS[var+': '+'2015: CO2 pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-2015+1)}]
        OUT[var+': '+'sign'].loc[{'stat_value':'mean','year':np.arange(TMP_xp.year[-1]-2015+1)}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[var+': '+'sign'].loc[{'stat_value':'std_dev','year':np.arange(TMP_xp.year[-1]-2015+1)}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[var+': '+'sign'].sel(stat_value='mean',year=np.arange(TMP_xp.year[-1]-2015+1)).values[...,np.newaxis],TMP_xp.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
        ## cross-dependency of AGWP and AGTP to sign & background
        OUT[var+': '+'cross'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        val = -1. * (VALS[var+': '+'1860: CDR pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-1860+1)}] - VALS[var+': '+'2015: CDR pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-2015+1)}])  -  (VALS[var+': '+'1860: CO2 pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-1860+1)}]  -  VALS[var+': '+'2015: CO2 pulse'].loc[{'year':np.arange(TMP_xp.year[-1]-2015+1)}])
        OUT[var+': '+'cross'].loc[{'stat_value':'mean','year':np.arange(TMP_xp.year[-1]-2015+1)}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[var+': '+'cross'].loc[{'stat_value':'std_dev','year':np.arange(TMP_xp.year[-1]-2015+1)}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[var+': '+'cross'].sel(stat_value='mean',year=np.arange(TMP_xp.year[-1]-2015+1)).values[...,np.newaxis],TMP_xp.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    # VALS.close()
    # del VALS


    ## Figure
    counter = 0
    fig = plt.figure( figsize=(30,20) )
    dico_col = {'1860: CO2 pulse':CB_color_cycle[7], '1860: CDR pulse':CB_color_cycle[3], '2015: CO2 pulse':CB_color_cycle[7], '2015: CDR pulse':CB_color_cycle[3] , 'background':'k', 'sign':'k', 'cross':'k'}
    dico_ls = {'1860: CO2 pulse':'--', '1860: CDR pulse':'--', '2015: CO2 pulse':'-', '2015: CDR pulse':'-', 'background':'-', 'sign':'-', 'cross':'-'}
    list_VAR = ['AGWP' , 'AGTP']
    dico_unit_var = {'AGWP':'(W.m$^{-2}$.(PgC.yr$^{-1}$)$^{-1}$)'  ,  'AGTP':'(mK.(PgC.yr$^{-1}$)$^{-1}$)'}

        
    for ii in [0,1]:
        period = [ np.arange(100+1) , np.arange(100,990+1) ][ii]

        ax,type_pulse = plt.subplot(3,2,1+ii),'2015: CO2 pulse'
        for VAR in ['AGWP','AGTP']:
            fact = {'AGWP':1.,'AGTP':1.e3}[VAR]
            ax.tick_params(axis='y', labelcolor={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR])
            func_plot(period,VAR+': '+type_pulse, fact * OUT.isel(year=period),col={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR],ls='-',label=VAR+' '+dico_unit_var[VAR])
            ax.tick_params(labelsize=fac_size*12)
            box = ax.get_position()
            ax.set_position([box.x0-0.03+ii*0.02, box.y0+0.04, box.width*1.05, box.height*1.05])
            if ii==0:plt.ylim( {'AGWP':[0.,0.3*1.25],'AGTP':[0.,2.1*1.25]}[VAR] )
            if VAR=='AGWP':
                if ii==0:plt.ylabel( 'Reference:'+'\n'+r'Pulse$^{CO_2}_{2015}$' , size=fac_size*15  )
                # if ii==0:plt.ylabel( 'Reference:'+'\n'+r'$\Uparrow^{CO2}_{2015}$' , size=fac_size*15  )
                plt.grid()
                ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
                plt.xlim(period[0],period[-1])
                if (ii==0):plt.legend( loc='upper left',prop={'size':fac_size*13})
                ax = ax.twinx()
            if (ii==0) and (var=='AGTP'):
                plt.legend( loc='lower right',prop={'size':fac_size*13})
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
        counter += 1

        ax,type_pulse = plt.subplot(3,2,3+ii),'background'
        for VAR in ['AGWP','AGTP']:
            fact = {'AGWP':1.,'AGTP':1.e3}[VAR]
            ax.tick_params(axis='y', labelcolor={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR])
            func_plot(period,VAR+': '+'background', fact * OUT.isel(year=period),col={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR],ls='-',label=type_pulse)
            ax.tick_params(labelsize=fac_size*12)
            box = ax.get_position()
            ax.set_position([box.x0-0.03+ii*0.02, box.y0+0.04, box.width*1.05, box.height*1.05])
            if VAR=='AGWP':
                if ii==0:plt.ylabel( 'Background:\n'+r'Pulse$^{CO_2}_{1860}$ - Pulse$^{CO_2}_{2015}$' , size=fac_size*15  )
                # if ii==0:plt.ylabel( 'Background:\n'+r'$\Uparrow^{CO_2}_{1860} - \Uparrow^{CO_2}_{2015}$' , size=fac_size*15  )
                plt.grid()
                ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
                plt.xlim(period[0],period[-1])
                ax = ax.twinx()
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
        counter += 1


        ax,type_pulse = plt.subplot(3,2,5+ii),'sign'
        for VAR in ['AGWP','AGTP']:
            fact = {'AGWP':1.,'AGTP':1.e3}[VAR]
            ax.tick_params(axis='y', labelcolor={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR])
            func_plot(period,VAR+': '+'sign', fact * OUT.isel(year=period),col={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR],ls='-',label=type_pulse)
            ax.tick_params(labelsize=fac_size*12)
            box = ax.get_position()
            ax.set_position([box.x0-0.03+ii*0.02, box.y0+0.04, box.width*1.05, box.height*1.05])
            if VAR=='AGWP':
                if ii==0:plt.ylabel( 'Sign:\n'+r'-Pulse$^{CDR}_{2015}$ - Pulse$^{CO_2}_{2015}$', size=fac_size*15  )
                # if ii==0:plt.ylabel( 'Sign:\n'+r'$-\Downarrow^{CDR}_{2015} - \Uparrow^{CO_2}_{2015}$', size=fac_size*15  )
                plt.grid()
                # ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
                plt.xlim(period[0],period[-1])
                plt.xlabel( ['Short','Long'][ii]+' term (year)' , size=fac_size*14 )
                ax = ax.twinx()
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
        counter += 1

        # ax,type_pulse = plt.subplot(4,2,7+ii),'cross'
        # for VAR in ['AGWP','AGTP']:
        #     fact = {'AGWP':1.,'AGTP':1.e3}[VAR]
        #     ax.tick_params(axis='y', labelcolor={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR])
        #     func_plot(period,VAR+': '+'cross', fact * OUT.isel(year=period),col={'AGWP':CB_color_cycle[0],'AGTP':CB_color_cycle[1]}[VAR],ls='-',label=type_pulse)
        #     ax.tick_params(labelsize=fac_size*12)
        #     box = ax.get_position()
        #     ax.set_position([box.x0-0.03+ii*0.02, box.y0+0.04, box.width*1.05, box.height*1.05])
        #     if VAR=='AGWP':
        #         if ii==0:plt.ylabel( 'Cross-variation:\n'+r'$-\left(\Downarrow^{CDR}_{1860} - \Downarrow^{CDR}_{2015}\right)$     '+'\n'+r'$- \left(\Uparrow^{CO_2}_{1860} - \Uparrow^{CO_2}_{2015}\right)$' , size=fac_size*15  )
        #         plt.grid()
        #         plt.xlim(period[0],period[-1])
        #         plt.xlabel( ['Short','Long'][ii]+' term (year)' , size=fac_size*14 )
        #         ax = ax.twinx()
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/AGWP-AGTP.pdf',dpi=300 )
    plt.close(fig)

#########################
#########################



#########################
## 7.5. Zero Emissions Commitment
#########################
## cf section 5, figure for ZECMIP
#########################
#########################



#########################
# 7.6. Historical and scenarios attribution
#########################
## add proba that climate change is from natural causes? cf distributions
if '7.6' in option_which_plots:
    # list_xp_left = [['historical'] , ['hist-GHG', 'hist-CO2', 'hist-1950HC'] , ['hist-aer','hist-piNTCF','hist-piAer'] , ['hist-nat', 'hist-sol', 'hist-volc'] , ['hist-noLu'] , ['hist-bgc'] , ['hist-stratO3']]
    # list_xp_right = [['ssp245'] , ['ssp245-GHG', 'ssp245-CO2'] , ['ssp245-aer'] , ['ssp245-nat', 'ssp245-sol', 'ssp245-volc'] , [] , []]
    list_xp = ['historical' , 'hist-GHG', 'hist-CO2', 'hist-1950HC' , 'hist-aer','hist-piNTCF','hist-piAer' , 'hist-nat', 'hist-sol', 'hist-volc' , 'hist-noLu' , 'hist-bgc' , 'hist-stratO3'] \
                + ['ssp245' , 'ssp245-GHG', 'ssp245-CO2' , 'ssp245-aer' , 'ssp245-nat', 'ssp245-sol', 'ssp245-volc' , 'ssp245-stratO3']
    list_VAR = ['D_Tg','RF']

    ## preparing data
    if option_overwrite:
        TMP_xp = xr.Dataset()
        TMP_xp.coords['all_config'] = np.array([str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])])
        TMP_xp.coords['year'] = np.arange( 1850,2100+1 ) ## hist and ssp245 without extension
        for xp in list_xp:#['hist-bgc' , 'hist-stratO3'] + ['ssp245' , 'ssp245-GHG', 'ssp245-CO2' , 'ssp245-aer' , 'ssp245-nat', 'ssp245-sol', 'ssp245-volc' , 'ssp245-stratO3']:
            for var in list_VAR:
                TMP_xp[var+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=[TMP_xp.year.size,TMP_xp.all_config.size]) , dims=['year','all_config'] )
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , ['D_Tg','RF'] )
                ## allocating
                for var in list_VAR:
                    TMP_xp[var+'_'+xp].loc[{'year':out_tmp.year,'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]}] = out_tmp[var] #- out_ctrl[VAR]
                ## cleaning
                out_tmp.close()
                # out_ctrl.close()

        TMP_xp.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_attrib.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP_xp})
    else:
        TMP_xp = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_attrib.nc' )


    # TMP_xp[var].sel(year=slice(2006,2015)).mean('year') - TMP_xp[var].sel(year=slice(1850,1900)).mean('year')



    def func_attrib_plot( val,xp,lbl ):
        ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(val))[0]
        mm = np.average( val.isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
        ss = np.sqrt(np.average( (val.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) ))
        ## histogram
        hst = np.histogram( a=val.isel(all_config=ind) ,bins=n_bins_distrib,density=True,weights=weights_CMIP6.weights.isel(all_config=ind) )
        hst = [hst[0]/np.max(hst[0]),hst[1]]
        plt.plot( hst[1][:-1]+0.5*(hst[1][1]-hst[1][0]) , hst[0]   , alpha=1 , lw=0.75*fac_size*2, color=dico_col[xp],ls=dico_ls[xp] , label=lbl+': '+str(np.round(mm,2))+'K ('+str(np.round(ss,2))+')' )
        ## median and std_dev
        plt.ylim(0,1.1)
        ii = np.argmin(np.abs(100+hst[1][:-1]+0.5*(hst[1][1]-hst[1][0])-(100+mm))) ## 100 as offset, just because both negative and positive values
        plt.axvline( x=mm , ymax=hst[0][ii] / ax.get_ylim()[1] ,color=dico_col[xp],lw=0.75*fac_size*2,ls=(0,(5,5)))
        ## observations
        if xp=='historical':
            mo,so = 0.87 * 0.99/0.86 , np.sqrt((0.12/0.955 * (1.37-0.65) / (1.18-0.54))**2. + 0.1467**2.)
            yy = scp.norm.pdf(np.linspace(plt.xlim()[0],plt.xlim()[1],500)  , loc=mo , scale=so )
            yy /= np.max(yy)
            plt.plot( np.linspace(plt.xlim()[0],plt.xlim()[1],500) ,yy, color=dico_col['Obs'],lw=0.75*fac_size*2,ls=dico_ls['Obs'] , label='Obs: '+str(np.round(mo,2))+'K ('+str(np.round(so,2))+')' )


    #-------------------------
    ## FIGURE HISTORICAL
    #-------------------------
    var = 'RF' ## ['D_Tg','RF']

    fig = plt.figure(figsize=(30,20))
    counter = 0
    ## first line
    n_bins_distrib = 75
    temp = np.arange(2006,2015+1)
    ax = plt.subplot(3,1,1)
    if var =='RF':
        plt.title( '$\mathregular{RF_{2006-2015} - RF_{1850-1900}}$ (W.m$^{-2}$)'  , size=fac_size*13 )#, fontweight='bold')
    else:
        plt.title( '$\mathregular{\Delta T_{2006-2015} - \Delta T_{1850-1900}}$ (K)'  , size=fac_size*13)#, fontweight='bold')
    dico_col = {'historical':CB_color_cycle[0],'Obs':CB_color_cycle[3]}
    dico_ls = {'historical':'-','Obs':'-'}
    for xp in ['historical']:
        val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year')
        # plot
        func_attrib_plot( val,xp,xp )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.ylabel( '(a)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0+0.06 , box.width*0.95, box.height*1.05])
    plt.legend(loc=0,prop={'size':fac_size*12})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ## second line
    ax = plt.subplot(3,1,2)
    dico_col = {'hist-GHG':CB_color_cycle[9],'hist-aer':CB_color_cycle[7],'hist-nat':CB_color_cycle[5],'hist-noLu':CB_color_cycle[4]}
    dico_ls = {'hist-GHG':'-','hist-aer':'-','hist-nat':'-','hist-noLu':'-.'}
    for xp in ['hist-GHG','hist-aer','hist-nat','hist-noLu']:
        if xp=='hist-noLu':
            val = (TMP_xp[var+'_'+'historical'].sel(year=temp).mean('year')-TMP_xp[var+'_'+xp].sel(year=temp).mean('year')) \
                    - (TMP_xp[var+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')-TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year'))
            func_attrib_plot( val,xp,'historical - hist-noLu' )
        else:
            val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year')
            func_attrib_plot( val,xp,xp )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.ylabel( '(b)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0+0.06 , box.width*0.95, box.height*1.16])
    plt.legend(loc=0,prop={'size':fac_size*12})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ## third line
    ax = plt.subplot(3,3,7)
    dico_col = {'hist-piNTCF':CB_color_cycle[1],'hist-piAer':CB_color_cycle[2],'diff_NTCFAer':CB_color_cycle[6], 'diff_NTCF_aer':CB_color_cycle[8]}
    dico_ls = {'hist-piNTCF':'-.','hist-piAer':'-.','diff_NTCFAer':'-.','diff_NTCF_aer':'-.'}
    for xp in ['hist-piNTCF','hist-piAer','diff_NTCFAer','diff_NTCF_aer']:
        if xp in ['hist-piNTCF','hist-piAer']:
            val = (TMP_xp[var+'_'+'historical'].sel(year=temp).mean('year')-TMP_xp[var+'_'+xp].sel(year=temp).mean('year')) \
                    - (TMP_xp[var+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')-TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year'))
            func_attrib_plot( val,xp,'historical - '+xp )
        elif xp == 'diff_NTCFAer':
            val = (TMP_xp[var+'_'+'hist-piNTCF']-TMP_xp[var+'_'+'hist-piAer']).sel(year=temp).mean('year') \
                    - (TMP_xp[var+'_'+'hist-piNTCF']-TMP_xp[var+'_'+'hist-piAer']).sel(year=np.arange(1850,1900+1)).mean('year')
            func_attrib_plot( val,xp,'hist-piNTCF - hist-piAer' )
        elif xp == 'diff_NTCF_aer':
            val = (TMP_xp[var+'_'+'historical']-TMP_xp[var+'_'+'hist-piNTCF']-TMP_xp[var+'_'+'hist-aer']).sel(year=temp).mean('year') \
                    - (TMP_xp[var+'_'+'historical']-TMP_xp[var+'_'+'hist-piNTCF']-TMP_xp[var+'_'+'hist-aer']).sel(year=np.arange(1850,1900+1)).mean('year')
            func_attrib_plot( val,xp,'historical - hist-piNTCF - hist-aer' )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.title( '(c)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    plt.ylim(0,1.3)
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0-0.01 , box.width*1.05, box.height*1.15])
    plt.legend(loc=0,prop={'size':fac_size*12})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ax = plt.subplot(3,3,8)
    dico_col = {'hist-sol':CB_color_cycle[1],'hist-volc':CB_color_cycle[2]}
    dico_ls = {'hist-sol':'-','hist-volc':'-'}
    for xp in ['hist-sol','hist-volc']:
        val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year')
        func_attrib_plot( val,xp,xp )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.title( '(d)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    plt.ylim(0,1.3)
    box = ax.get_position()
    ax.set_position([box.x0-0.025, box.y0-0.01 , box.width*1.05, box.height*1.15])
    plt.legend(loc=0,prop={'size':fac_size*12})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ax = plt.subplot(3,3,9)
    dico_col = {'hist-CO2':CB_color_cycle[1],'hist-1950HC':CB_color_cycle[2],'hist-stratO3':CB_color_cycle[6],'hist-bgc':CB_color_cycle[8] , 'diff_all':CB_color_cycle[0]}
    dico_ls = {'hist-CO2':'-','hist-stratO3':'-','hist-1950HC':'-.','hist-bgc':'-.' , 'diff_all':'-.'}
    for xp in ['hist-CO2','hist-stratO3','hist-1950HC','hist-bgc' , 'diff_all']:
        if xp in ['hist-bgc']:
            val = (TMP_xp[var+'_'+'historical'].sel(year=temp).mean('year')-TMP_xp[var+'_'+xp].sel(year=temp).mean('year')) \
                    - (TMP_xp[var+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')-TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year'))
            func_attrib_plot( val,xp,'historical - '+xp )
        elif xp in ['hist-1950HC']:
            val = (TMP_xp[var+'_'+'historical'].sel(year=temp).mean('year')-TMP_xp[var+'_'+xp].sel(year=temp).mean('year'))
            ## /!\ hist-1950HC starts in 1950, but is similar to historical before
            func_attrib_plot( val,xp,'historical - '+xp )
        elif xp in ['hist-CO2','hist-stratO3']:
            val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year')
            func_attrib_plot( val,xp,xp )
        elif xp == 'diff_all':
            val = (TMP_xp[var+'_'+'hist-GHG']-TMP_xp[var+'_'+'hist-CO2']-TMP_xp[var+'_'+'historical']+TMP_xp[var+'_'+'hist-1950HC']).sel(year=temp).mean('year') \
                    - (TMP_xp[var+'_'+'hist-GHG']-TMP_xp[var+'_'+'hist-CO2']-0.).sel(year=np.arange(1850,1900+1)).mean('year')
            ## /!\ hist-1950HC starts in 1950, but is similar to historical before
            func_attrib_plot( val,xp,'hist-GHG - hist-CO2 - (historical - hist-1950HC)' )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    plt.ylim(0,1.3)
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.title( '(e)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    box = ax.get_position()
    ax.set_position([box.x0-0.05, box.y0-0.01 , box.width*1.05, box.height*1.15])
    plt.legend(loc=0,prop={'size':fac_size*11})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/attrib-'+var+'-hist.pdf',dpi=300 )
    plt.close(fig)
    #-------------------------
    #-------------------------



    #-------------------------
    ## FIGURE SSP245
    #-------------------------
    counter = 0
    fig = plt.figure(figsize=(30,20))
    ## first line
    n_bins_distrib = 75
    temp = np.arange(2091,2100+1)
    ax = plt.subplot(3,1,1)
    plt.title( 'SSP2-4.5: $\mathregular{\Delta T_{2091-2100} - \Delta T_{1850-1900}}$ (K)'  , size=fac_size*13, fontweight='bold')
    dico_col = {'ssp245':CB_color_cycle[0]}
    dico_ls = {'ssp245':'-'}
    for xp in ['ssp245']:
        val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')
        # plot
        func_attrib_plot( val,xp,xp )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.ylabel( '(a)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0+0.06 , box.width*0.95, box.height*1.05])
    plt.legend(loc=0,prop={'size':fac_size*12})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ## second line
    ax = plt.subplot(3,1,2)
    dico_col = {'ssp245-GHG':CB_color_cycle[9],'ssp245-aer':CB_color_cycle[7],'ssp245-nat':CB_color_cycle[5]}
    dico_ls = {'ssp245-GHG':'-','ssp245-aer':'-','ssp245-nat':'-'}
    for xp in ['ssp245-GHG','ssp245-aer','ssp245-nat']:
        val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+'hist-'+str.split(xp,'-')[1]].sel(year=np.arange(1850,1900+1)).mean('year')
        func_attrib_plot( val,xp,xp )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.ylabel( '(b)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0+0.06 , box.width*0.95, box.height*1.16])
    plt.legend(loc=0,prop={'size':fac_size*12})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ## third line
    ax = plt.subplot(3,2,5)
    dico_col = {'ssp245-sol':CB_color_cycle[1],'ssp245-volc':CB_color_cycle[2]}
    dico_ls = {'ssp245-sol':'-','ssp245-volc':'-'}
    for xp in ['ssp245-sol','ssp245-volc']:
        val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+'hist-'+str.split(xp,'-')[1]].sel(year=np.arange(1850,1900+1)).mean('year')
        func_attrib_plot( val,xp,xp )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.title( '(c)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    plt.ylim(0,1.3)
    box = ax.get_position()
    ax.set_position([box.x0+0.0, box.y0-0.0 , box.width*1.05, box.height*1.125])
    plt.legend(loc=0,prop={'size':fac_size*12})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1

    ax = plt.subplot(3,2,6)
    dico_col = {'ssp245-CO2':CB_color_cycle[1],'ssp245-stratO3':CB_color_cycle[2]}
    dico_ls = {'ssp245-CO2':'-','ssp245-stratO3':'-'}
    for xp in ['ssp245-CO2','ssp245-stratO3']:
        val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+'hist-'+str.split(xp,'-')[1]].sel(year=np.arange(1850,1900+1)).mean('year')
        func_attrib_plot( val,xp,xp )
    ## cutting off xticks for all
    ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
    ## polishing
    plt.grid()
    plt.ylim(0,1.3)
    if var=='RF':
        plt.xlabel( 'RF (W.m$^{-2}$)'  , size=fac_size*13 , rotation=0)
    else:
        plt.xlabel( '$\Delta$T (K)'  , size=fac_size*13 , rotation=0)
    plt.title( '(d)'  , size=fac_size*15 , rotation=0)
    ## shape of subplots
    box = ax.get_position()
    ax.set_position([box.x0-0.05, box.y0-0.0 , box.width*1.05, box.height*1.125])
    plt.legend(loc=0,prop={'size':fac_size*11})
    ax.tick_params(labelsize=fac_size*13)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
    counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/attrib-'+var+'-ssp245.pdf',dpi=300 )
    plt.close(fig)
    #-------------------------
    #-------------------------

    #-------------------------
    # ratios
    #-------------------------
    if False:
        plt.figure()
        var = 'D_Tg'
        n_bins_distrib = 75
        temp = np.arange(2006,2015+1)
        dico_col = {'hist-GHG':CB_color_cycle[9],'hist-aer':CB_color_cycle[7],'hist-nat':CB_color_cycle[5],'hist-noLu':CB_color_cycle[4]}
        dico_ls = {'hist-GHG':'-','hist-aer':'-','hist-nat':'-','hist-noLu':'-.'}
        for xp in ['hist-GHG','hist-aer','hist-nat','hist-noLu']:
            if xp=='hist-noLu':
                val = (TMP_xp[var+'_'+'historical'].sel(year=temp).mean('year')-TMP_xp[var+'_'+xp].sel(year=temp).mean('year')) \
                        - (TMP_xp[var+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')-TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year'))
            else:
                val = TMP_xp[var+'_'+xp].sel(year=temp).mean('year') - TMP_xp[var+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year')
            val = val / (TMP_xp[var+'_'+'historical'].sel(year=temp).mean('year') - TMP_xp[var+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year'))
            func_attrib_plot( val,xp,"ratio using "+xp )
        plt.legend(loc=0,prop={'size':fac_size*12})
    #-------------------------
    #-------------------------


if False:## old version of the attribution plot, kept in stash
    ## ploting
    n_bins_distrib = 50
    dico_bins = { 'historical':[0,4.7] , 'ssp245':[0,4.7] , 'GHG':[-0.5,4.5] , 'aer':[-1.1,0.5] , 'nat':[-0.1,0.1] , 'noLu':[-0.5,0.3] , 'bgc':[0.5,1.2] }
    dico_effect_xp = { 'historical':'All'  , 'hist-GHG':'Anthropogenic\nGhG' , 'hist-aer':'Anthropogenic\n NTCF' , 'hist-nat':'Solar &\nvolcanic' , 'hist-noLu':'Land use' , 'hist-bgc':'Carbon\ncycle'}
    dico_lbl = {'historical':'All','hist-GHG':'GhG','hist-CO2':'CO2','hist-1950HC':'CFC & HCFC','hist-aer':'NTCF','hist-piNTCF':'NTCF','hist-piAer':'Aerosols','hist-nat':'Sol. & Volc.','hist-sol':'Sol.','hist-volc':'Volc.','hist-noLu':'Land use','hist-bgc':'Carbon cycle','ssp245':'All','ssp245-GHG':'GhG','ssp245-CO2':'CO2','ssp245-aer':'NTCF','ssp245-nat':'Sol. & volc.','ssp245-sol':'Sol.','ssp245-volc':'Volc.'}
    # 0,1,2,3 :: observations, without non-linearities, with non-linearities, all effects
    dico_ls = {'historical':3,'hist-GHG':1,'hist-CO2':1,'hist-1950HC':2,'hist-aer':1,'hist-piNTCF':2,'hist-piAer':2,'hist-nat':1,'hist-sol':1,'hist-volc':1,'hist-noLu':2,'hist-bgc':2,'ssp245':3,'ssp245-GHG':1,'ssp245-CO2':1,'ssp245-aer':1,'ssp245-nat':1,'ssp245-sol':1,'ssp245-volc':1}
    list_ls = ['-','-',(0,(5,1)),'-'] 
    # 0,1,2,3 :: order of curves. Reserved: 0 for obs, 3 for all
    dico_col = {'historical':3,'hist-GHG':1,'hist-CO2':2,'hist-1950HC':4,'hist-aer':1,'hist-piNTCF':2,'hist-piAer':4,'hist-nat':1,'hist-sol':2,'hist-volc':4,'hist-noLu':1,'hist-bgc':1,'ssp245':3,'ssp245-GHG':1,'ssp245-CO2':2,'ssp245-aer':1,'ssp245-nat':1,'ssp245-sol':2,'ssp245-volc':4}
    list_col = [CB_color_cycle[0]]+CB_color_cycle[2:3+1]+CB_color_cycle[5:]#['gold','darkblue', 'dodgerblue','crimson' , 'olive']
    # NTCF = 'BC, OC, SO$_2$, NH$_3$,\nCO, VOC, NO$_X$'


    eps,eps_inter = 0.06, 0.0
    dico_shift_box = { 'historical':0 , 'ssp245':0 , 'GHG':eps , 'aer':eps , 'nat':eps , 'noLu':eps , 'bgc':eps }
    plt.figure(figsize=(10,20))
    for XP in list_xp_left+list_xp_right:
        if len(XP)>0:
            ## preparing subplot
            if XP in list_xp_left:
                ind_pos = list_xp_left.index(XP)
                ax = plt.subplot( len(list_xp_left),2,2*ind_pos+1 )
                temp = np.arange(2006,2015+1)
            else:
                ind_pos = list_xp_right.index(XP)
                ax = plt.subplot( len(list_xp_left),2,2*ind_pos+2 )
                temp = np.arange(2091,2100+1)
            ## plotting results
            bb = dico_bins[str.split(XP[0],'-')[-1]]
            for xp in XP:
                if (str.split(xp,'-')[0] == 'ssp245'):
                    val = TMP_xp[xp].sel(year=temp).mean('year') - TMP_xp['historical'].sel(year=np.arange(1850,1900+1)).mean('year')
                elif (xp in ['hist-piNTCF','hist-piAer','hist-noLu','hist-bgc']):
                    val = (TMP_xp['historical'].sel(year=temp).mean('year')-TMP_xp[xp].sel(year=temp).mean('year')) \
                         - (TMP_xp['historical'].sel(year=np.arange(1850,1900+1)).mean('year')-TMP_xp[xp].sel(year=np.arange(1850,1900+1)).mean('year'))
                elif (xp in ['hist-1950HC']):
                    val = TMP_xp['historical'].sel(year=temp).mean('year')-TMP_xp[xp].sel(year=temp).mean('year') ## hist-1950HC starts in 1950, shares the same 1850-1900 with 'historical'
                else:
                    val = TMP_xp[xp].sel(year=temp).mean('year') - TMP_xp[xp].sel(year=np.arange(1850,1900+1)).mean('year')
                ind = np.where( ~np.isnan(weights_CMIP6.weights) & ~np.isnan(val))[0]
                mm = np.average( val.isel(all_config=ind) ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) )
                ss = np.sqrt(np.average( (val.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights_CMIP6.weights.isel(all_config=ind) ))
                ## histogram
                hst = np.histogram( a=val.isel(all_config=ind) ,bins=np.linspace(bb[0],bb[1],n_bins_distrib),density=True,weights=weights_CMIP6.weights.isel(all_config=ind) )
                plt.plot( hst[0] , hst[1][:-1]+0.5*(hst[1][1]-hst[1][0]) , alpha=1 , lw=0.75*fac_size*2, color=list_col[dico_col[xp]],ls=list_ls[dico_ls[xp]] , label=dico_lbl[xp]+' '+str(np.round(mm,2))+'K ('+str(np.round(ss,2))+')' )
                if xp==XP[-1]:
                    plt.xlim(0,1.6*ax.get_xlim()[1])
                ## median and std_dev
                ii = np.argmin(np.abs(100+hst[1][:-1]+0.5*(hst[1][1]-hst[1][0])-(100+mm))) ## 100 as offset, just because both negative and positive values
                # plt.axhline( y=mm , xmax=hst[0][ii] / ax.get_xlim()[1] ,color=list_col[dico_col[xp]],lw=0.75*fac_size*2,ls=(0,(5,5)))
            ## observations
            if XP == ['historical']:
                mo,so = 0.87 * 0.99/0.86 , np.sqrt((0.12/0.955 * (1.37-0.65) / (1.18-0.54))**2. + 0.1467**2.)
                yy = scp.norm.pdf(np.linspace(plt.xlim()[0],plt.xlim()[1],500)  , loc=mo , scale=so )
                plt.plot( yy , np.linspace(plt.xlim()[0],plt.xlim()[1],500) , color=list_col[0],lw=0.75*fac_size*2,ls=list_ls[0] , label='Obs: '+str(np.round(mo,2))+'K ('+str(np.round(so,2))+')' )
            ## cutting off xticks for all
            ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
            ## cutting off xticks for ssp245
            if xp in list_xp_right:
                ax.set_yticklabels(['']*len([item.get_text() for item in ax.get_yticklabels()]))
            ## polishing
            plt.grid()
            plt.ylim(bb[0],bb[1])
            if XP in list_xp_left:
                plt.ylabel( dico_effect_xp[XP[0]]  , size=fac_size*13 , rotation=0)
            if XP == ['historical']:
                ax.yaxis.set_label_coords(-0.12,0.4)
            elif XP == ['hist-bgc']:
                ax.yaxis.set_label_coords(-0.12-0.20,0.375)
            else:
                ax.yaxis.set_label_coords(-0.12-0.20,0.3)
            # elif XP in [['hist-1950HC'] , ['hist-aer','hist-piNTCF','hist-piAer'] , ['hist-nat'], ['hist-bgc']]:
            #     ax.yaxis.set_label_coords(-0.12-0.10,0.20)
            if XP==list_xp_left[0]:
                plt.title( '$\mathregular{\Delta T_{2006-2015} - \Delta T_{1850-1900}}$ (K)'  , size=fac_size*13, fontweight='bold')
            elif XP==list_xp_right[0]:
                plt.title( 'SSP2-4.5: $\mathregular{\Delta T_{2091-2100} - \Delta T_{1850-1900}}$ (K)'  , size=fac_size*13, fontweight='bold')
            ## shape of subplots
            if str.split(XP[0],'-')[-1] in ['CO2','sol','volc','1950HC']:
                fact_width = 0.215/(0.215+dico_shift_box[str.split(XP[0],'-')[-1]])
            else:
                fact_width = 0.28/(0.28+dico_shift_box[str.split(XP[0],'-')[-1]])
            box = ax.get_position()
            ax.set_position([box.x0+0.0+dico_shift_box[str.split(XP[0],'-')[-1]]+eps_inter*(XP in list_xp_right), box.y0+0.07+ind_pos*(-0.01) , box.width*0.95*fact_width, box.height*1.16])
            plt.legend(loc=0,prop={'size':fac_size*9})
    ## creating shadow subplot for legend:
    ax = plt.subplot( 3,2,6 )
    box = ax.get_position()
    ax.set_position([box.x0+0.0+eps_inter, box.y0+0.07+(len(list_xp_left)-1)*(-0.01) , box.width*0.95*fact_width, box.height*1.16])
    plt.plot( np.arange(10) , np.nan*np.arange(10) , lw=0.75*fac_size*2, color=list_col[0], ls=list_ls[0] , label='Observations' )
    plt.plot( np.arange(10) , np.nan*np.arange(10) , lw=0.75*fac_size*2, color=list_col[3], ls=list_ls[3] , label='All effects' )
    plt.plot( np.arange(10) , np.nan*np.arange(10) , lw=0.75*fac_size*2, color='white', ls=None , label=' ' )
    plt.plot( np.arange(10) , np.nan*np.arange(10) , lw=0.75*fac_size*2, color='k', ls=list_ls[1] , label='Without non-linearities' )
    plt.plot( np.arange(10) , np.nan*np.arange(10) , lw=0.75*fac_size*2, color='k', ls=list_ls[2] , label='With non-linearities' )
    plt.legend(loc='upper right',prop={'size':fac_size*13})
    ax.axis('off')
#########################
#########################








#########################
## 7.7. Land-Use experiments
#########################
## Section 4. Figure LUMIP
## Add LASC?
if '7.7' in option_which_plots:
    ## effects CO2 and climate: 'land-cCO2', 'land-cClim', Eluc, LSNK, LASC, cLand vs time
    ## effects practices: 'land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv', Eluc, LSNK, LASC, cLand vs time
    ## effects data: 'land-hist', 'land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear', Eluc, LSNK, LASC, cLand vs time
    ## effects CO2 and climt: 'land-cCO2', 'land-cClim', Eluc, LSNK, LASC, cLand vs time

    list_VAR = ['D_Fland','D_Eluc','LASC','cLand']

    if option_overwrite:
        ## preparing data for figures
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(1700,2014+1)
        for xp in ['land-cCO2', 'land-cClim'] + ['land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv'] + ['land-hist', 'land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear']:
            print(xp)
            for setMC in list_setMC:
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , ['csoil1_0','csoil2_0','cveg_0','D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp'] + ['D_Fland'] + ['D_Eluc'] )
                for var in list_VAR:
                    if var not in TMP:
                        TMP[var+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                    if var in ['D_Fland','D_Eluc']:
                        TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = out_tmp[var] * mask
                    elif var in ['cLand']:
                        val = out_tmp['D_Fland'] - out_tmp['D_Eluc']
                        TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = val.cumsum('year') * mask
                        # val =  ( (out_tmp['csoil1_0']+out_tmp['D_csoil1']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
                        # val += ( (out_tmp['csoil2_0']+out_tmp['D_csoil2']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
                        # val += ( (out_tmp['cveg_0']+out_tmp['D_cveg']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
                        # val += ( out_tmp['D_Csoil1_bk']+out_tmp['D_Csoil2_bk']+out_tmp['D_Cveg_bk'] ).sum(('bio_from','bio_to'))
                        # val += out_tmp['D_Chwp'].sum( ('bio_from','bio_to','box_hwp') )
                        # val -= ((out_tmp['csoil1_0']+out_tmp['csoil2_0']+out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
                        # TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = val.sum('reg_land').transpose( 'year','config' ) * mask
                    elif var in ['LASC']:
                        TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] =  (out_tmp['D_nbp'] * out_tmp['D_Aland']).sum('bio_land', min_count=1).sum('reg_land', min_count=1) * mask
                    else:
                        raise Exception("Not prepared")
                out_tmp.close()
                for_tmp.close()
        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_land.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_land.nc' )

    ## statistical values
    OUT = xr.Dataset()
    OUT.coords['stat_value'] = ['mean','std_dev']
    OUT.coords['year'] = np.arange(1700,2014+1)
    for xp in ['land-cCO2', 'land-cClim'] + ['land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv'] + ['land-hist', 'land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear']:
        for VAR in list_VAR:
            OUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    ## statistical values of differences to land-hist
    for xp in ['land-cCO2', 'land-cClim'] + ['land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv']:
        for VAR in list_VAR:
            OUT[VAR+'_land-hist - '+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array( TMP[VAR+'_land-hist'] - TMP[VAR+'_'+xp] ,mask=np.isnan(TMP[VAR+'_'+xp]-TMP[VAR+'_land-hist']))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_land-hist - '+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_land-hist - '+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_land-hist - '+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    for xp in ['land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear']:
        for VAR in list_VAR:
            OUT[VAR+'_'+xp+' - land-hist'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array( TMP[VAR+'_'+xp] - TMP[VAR+'_land-hist'] ,mask=np.isnan(TMP[VAR+'_'+xp]-TMP[VAR+'_land-hist']))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+xp+' - land-hist'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp+' - land-hist'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp+' - land-hist'].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data


    ## figure
    dico_title_VAR = {'D_Fland':'Land carbon sink (PgC.yr$^{-1}$)','D_Eluc':'CO$_2$ emissions from LUC (PgC.yr$^{-1}$)','LASC':'Loss of additional sink capacity (PgC.yr$^{-1}$)','cLand':'Land carbon stock (PgC)'}
    # dico_var_ylim= {'D_Fland':[-2.5,6.5],'D_Eluc':[-0.2,2.5],'LASC':[-0.75,0.3]}
    ## looping on lines for groups of experiments
    VAR_plot = ['D_Fland','D_Eluc','cLand']
    fig = plt.figure( figsize=(30,30) )
    counter = 0 
    for ind_line in [0,1,2,3]:
        if ind_line == 0:
            dico_col = {'land-hist':'k'}
            dico_ls = {'land-hist':'-'}
            list_xp = ['land-hist']
            ylbl = 'Historical'
        elif ind_line == 1:
            dico_col = {'land-hist - land-cCO2':CB_color_cycle[0], 'land-hist - land-cClim':CB_color_cycle[3]}
            dico_ls = {'land-hist - land-cCO2':'-', 'land-hist - land-cClim':'-'}
            list_xp = ['land-hist - land-cCO2', 'land-hist - land-cClim']
            ylbl = 'CO$_2$ and climate'
        elif ind_line == 2:
            dico_col = {'land-hist - land-crop-grass':CB_color_cycle[0], 'land-hist - land-noLu':CB_color_cycle[4], 'land-hist - land-noShiftcultivate':CB_color_cycle[3], 'land-hist - land-noWoodHarv':CB_color_cycle[2]}
            dico_ls = {'land-hist - land-crop-grass':'-', 'land-hist - land-noLu':'-', 'land-hist - land-noShiftcultivate':'-', 'land-hist - land-noWoodHarv':'-'}
            list_xp = ['land-hist - land-crop-grass', 'land-hist - land-noLu', 'land-hist - land-noShiftcultivate', 'land-hist - land-noWoodHarv']
            ylbl = 'Practices'
        elif ind_line == 3:
            dico_col = {'land-hist-altLu1 - land-hist':CB_color_cycle[0], 'land-hist-altLu2 - land-hist':CB_color_cycle[3], 'land-hist-altStartYear - land-hist':CB_color_cycle[2]}#'land-hist':CB_color_cycle[4], 
            dico_ls = {'land-hist-altLu1 - land-hist':'-', 'land-hist-altLu2 - land-hist':'-', 'land-hist-altStartYear - land-hist':'-'}#'land-hist':'-', 
            list_xp = ['land-hist-altLu1 - land-hist', 'land-hist-altLu2 - land-hist', 'land-hist-altStartYear - land-hist']#'land-hist', 
            ylbl = 'Datasets'
        lw=2
        period = [1850,2010]
        ## looping on variables / subplots
        for var in VAR_plot:
            ax = plt.subplot(4,len(VAR_plot),ind_line*len(VAR_plot)+VAR_plot.index(var)+1)
            ## looping on experiments inside a subplot
            for xp in list_xp:
                func_plot('year',var+'_'+xp,OUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*lw,label=xp)
                mm,ss = OUT[var+'_'+xp].sel(year=period[1],stat_value='mean'), OUT[var+'_'+xp].sel(year=period[1],stat_value='std_dev')
                print('OSCAR: '+var+' on '+xp+': '+str(mm.values)+'+/-'+str(ss.values))
            ## polishing
            plt.grid()
            plt.xlim(period[0],period[1])
            # if var in dico_var_ylim.keys():
            #     plt.ylim( dico_var_ylim[var] )
            if VAR_plot.index(var) == 0:
                plt.ylabel( ylbl , size=fac_size*15 )#,fontweight='bold',rotation=0  )
                # if ind_line==0:
                #     ax.yaxis.set_label_coords(-0.4,0.65)
                # else:
                # ax.yaxis.set_label_coords(-0.35,0.65)
                # plt.legend(loc=0,prop={'size':fac_size*10.4},bbox_to_anchor=(-0.125,0.65))
                plt.legend( loc={'Historical':'upper left','CO$_2$ and climate':'upper left','Practices':'lower left','Datasets':'upper right'}[ylbl],prop={'size':fac_size*10} )
            if ind_line == 0:
                plt.title( dico_title_VAR[var]  , size=fac_size*15)#,fontweight='bold')
            if ind_line == 3:
                plt.xlabel( 'Year' , size=fac_size*15 )
            else:
                ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
            ax.tick_params(labelsize=fac_size*13)
            box = ax.get_position()
            ax.set_position([box.x0+0.055-VAR_plot.index(var)*0.02, box.y0+0.04-ind_line*(0.02), box.width*0.95, box.height*1.1])
            plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
            counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/lumip-hist.pdf',dpi=300 )
    plt.close(fig)

    OUT['cLand'+'_'+'land-hist'].sel(year=2010)
    OUT['cLand'+'_'+'land-hist - land-cCO2'].sel(year=2010)
    OUT['cLand'+'_'+'land-hist - land-crop-grass'].sel(year=2010)
    OUT['cLand'+'_'+'land-hist-altLu1 - land-hist'].sel(year=2010)
    

    # val = np.ma.array( TMP['cLand'+'_'+'land-hist']-TMP['cLand'+'_'+'land-hist'].sel(year=1959) ,mask=np.isnan(TMP['cLand'+'_'+'land-hist']-TMP['cLand'+'_'+'land-hist'].sel(year=1959)))
    # ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
    # mm = np.ma.average( a=val , axis=-1 , weights=ww ).data
    # ss = np.ma.sqrt(np.ma.average( (val - np.repeat(mm[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    # mm[2010-1700]


if False:## old figure, not in differences
    ## figure
    dico_title_VAR = {'D_Fland':'Land carbon sink (PgC.yr$^{-1}$)','D_Eluc':'CO$_2$ emissions from LUC (PgC.yr$^{-1}$)','LASC':'Loss of additional sink capacity (PgC.yr$^{-1}$)','cLand':'Land carbon stock (PgC)'}
    dico_var_ylim= {'D_Fland':[-2.5,6.5],'D_Eluc':[-0.2,2.5],'LASC':[-0.75,0.3]}
    ## looping on lines for groups of experiments
    VAR_plot = ['D_Fland','D_Eluc','cLand']
    plt.figure( figsize=(30,20) )
    for ind_line in [0,1,2]:
        if ind_line == 0:
            dico_col = {'land-cCO2':CB_color_cycle[0], 'land-cClim':CB_color_cycle[3]}
            dico_ls = {'land-cCO2':'-', 'land-cClim':'-'}
            list_xp = ['land-cCO2', 'land-cClim']
            period = [1850,2010]
            ylbl = 'CO2 and climate'
            lw=2
        elif ind_line == 1:
            dico_col = {'land-crop-grass':CB_color_cycle[0], 'land-noLu':CB_color_cycle[4], 'land-noShiftcultivate':CB_color_cycle[3], 'land-noWoodHarv':CB_color_cycle[2]}
            dico_ls = {'land-crop-grass':'-', 'land-noLu':'-', 'land-noShiftcultivate':'-', 'land-noWoodHarv':'-'}
            list_xp = ['land-crop-grass', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv']
            period = [1850,2010]
            ylbl = 'Practices'
            lw=2
        elif ind_line == 2:
            dico_col = {'land-hist':CB_color_cycle[4], 'land-hist-altLu1':CB_color_cycle[0], 'land-hist-altLu2':CB_color_cycle[3], 'land-hist-altStartYear':CB_color_cycle[2]}
            dico_ls = {'land-hist':'-', 'land-hist-altLu1':'-', 'land-hist-altLu2':'-', 'land-hist-altStartYear':'-'}
            list_xp = ['land-hist', 'land-hist-altLu1', 'land-hist-altLu2', 'land-hist-altStartYear']
            period = [1850,2010]
            ylbl = 'Datasets'
            lw=2
        ## looping on variables / subplots
        for var in VAR_plot:
            ax = plt.subplot(3,len(VAR_plot),ind_line*len(VAR_plot)+VAR_plot.index(var)+1)
            ## looping on experiments inside a subplot
            for xp in list_xp:
                func_plot('year',var+'_'+xp,OUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*lw,label=xp)
            ## polishing
            plt.grid()
            plt.xlim(period[0],period[1])
            if var in dico_var_ylim.keys():
                plt.ylim( dico_var_ylim[var] )
            if VAR_plot.index(var) == 0:
                plt.ylabel( ylbl , size=fac_size*15,fontweight='bold',rotation=0  )
                if ind_line==0:
                    ax.yaxis.set_label_coords(-0.4,0.65)
                else:
                    ax.yaxis.set_label_coords(-0.35,0.65)
                plt.legend(loc=0,prop={'size':fac_size*13},bbox_to_anchor=(-0.125,0.65))
            if ind_line == 0:
                plt.title( dico_title_VAR[var]  , size=fac_size*15,fontweight='bold')
            if ind_line == 2:
                plt.xlabel( 'Year' , size=fac_size*15 )
            else:
                ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
            ax.tick_params(labelsize=fac_size*13)
            box = ax.get_position()
            ax.set_position([box.x0+0.045, box.y0+0.04-ind_line*(0.02), box.width*0.95, box.height*1.1])
            plt.text(x=period[0]+0.05*(period[1]-period[0]),y=ax.get_ylim()[1]-0.1*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=fac_size*'('+  ['a','b','c','d','e','f','g','h','i'][ind_line*len(VAR_plot)+VAR_plot.index(var)] +')',fontdict={'size':fac_size*13})
#########################
#########################




#########################
## 7.8. Main CMIP6 scenarios
#########################
if '7.8' in option_which_plots:
    list_xp = ['ssp585', 'ssp370',  'ssp460', 'ssp534-over', 'ssp434', 'ssp245', 'ssp126', 'ssp119'] + ['esm-ssp585', 'esm-ssp370',  'esm-ssp460', 'esm-ssp534-over', 'esm-ssp434', 'esm-ssp245', 'esm-ssp126', 'esm-ssp119']
    list_VAR = ['Eff' , 'co2' , 'erf_tot' , 'fgco2' , 'RF_nonCO2' , 'tas', 'nbp', 'RF_O3t', 'pr', 'permafrostCO2', 'RF_AERtot', 'ohc','D_Tg_shift']
    # list_VAR = ['D_Eluc','D_Epf_CO2','D_Epf_CH4','D_Focean','D_Fland','D_Cfroz','D_Ewet'] + ['RF_CO2']+['RF_CH4', 'RF_N2O', 'RF_halo','RF_nonCO2']+['RF_H2Os', 'RF_O3s','RF_strat']+['RF_SO4', 'RF_POA', 'RF_NO3', 'RF_SOA', 'RF_dust', 'RF_salt','RF_scatter']+['RF_BC','RF_absorb']+['RF_cloud','RF_AERtot']+['RF_O3t','RF_slcf']+['RF_BCsnow', 'RF_lcc','RF_alb'] + ['RF','D_Tg','D_Pg']

    OUT = xr.Dataset()
    for xp in list_xp:
        ## variables in CMIP6
        OUT0 = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
        for var in list_VAR:
            if var not in ['Eff','D_Tg_shift']:OUT[var+'_'+xp] = OUT0[var].copy(deep=True)
        OUT0.close()
    ## adding control for all var
    OUT0 = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where( 'piControl' ==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
    OUT0e = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where( 'esm-piControl' ==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
    for xp in list_xp:
        for var in list_VAR:
            if var not in ['Eff','D_Tg_shift','pr','ohc']:
                if xp[:3+1]=='esm-':
                    if 'stat_value' in OUT[var+'_'+xp].dims:
                        OUT[var+'_'+xp].loc[{'stat_value':'mean'}] += OUT0e[var].sel(stat_value='mean',year=np.arange(1850,1900+1)).mean('year')
                    else:
                        OUT[var+'_'+xp] += OUT0e[var].sel(year=np.arange(1850,1900+1)).mean('year')
                else:
                    if 'stat_value' in OUT[var+'_'+xp].dims:
                        OUT[var+'_'+xp].loc[{'stat_value':'mean'}] += OUT0[var].sel(stat_value='mean',year=np.arange(1850,1900+1)).mean('year')
                    else:
                        OUT[var+'_'+xp] += OUT0[var].sel(year=np.arange(1850,1900+1)).mean('year')
    OUT0.close()
    OUT0e.close()

    ## preparing data
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = np.array([str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])])
        TMP.coords['year'] = np.arange( 1850,2500+1 )
        for xp in ['historical','esm-hist']+list_xp:
            if 'D_Tg_shift'+'_'+xp not in TMP:
                TMP['D_Tg_shift'+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=[TMP.year.size,TMP.all_config.size]) , dims=['year','all_config'] )
            for ext in ['','ext']:
                if xp in ['ssp534-over','esm-ssp534-over'] and ext=='ext': ext='-ext'
                if xp in ['historical','esm-hist'] and ext=='ext':
                    pass
                else:
                    for setMC in list_setMC:
                        print(xp+ext+'/'+str(setMC))
                        out_tmp,for_tmp,Par,mask = func_get( setMC , xp+ext , ['D_Tg'] )
                        TMP['D_Tg_shift'+'_'+xp].loc[{'year':out_tmp.year,'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]}] = out_tmp["D_Tg"]
                        out_tmp.close()
        ## shifting: D_Tg to D_Tg_shift
        for xp in list_xp:TMP['D_Tg_shift'+'_'+xp] = TMP['D_Tg_shift'+'_'+xp] - TMP['D_Tg_shift'+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')
        ## Eff
        for xp in list_xp:
            TMP['Eff'+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
            for setMC in list_setMC:
                for ext in ['','ext']:
                    if xp in ['ssp534-over','esm-ssp534-over'] and ext=='ext': ext='-ext'
                    print(xp+ext+'/'+str(setMC))
                    out_tmp,for_tmp,Par,mask = func_get( setMC , xp+ext , ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4'] )
                    for_tmp = eval_compat_emi( ['CO2'], out_tmp,Par,for_tmp )
                    tmp_period = {2014:np.arange(1850,2014+1), 2100:np.arange(2015,2100+1), 2500:np.arange(2101,2500+1)}[int(for_tmp.year[-1])]
                    if 'all_config' in for_tmp['Eff']:
                        TMP['Eff'+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':tmp_period }] = (for_tmp['Eff'].sum('reg_land') * mask).sel( year=tmp_period )
                    else:
                        TMP['Eff'+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':tmp_period }] = (for_tmp['Eff'].sum('reg_land').expand_dims({'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]},axis=1) * mask).sel( year=tmp_period )
        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_scen.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_scen.nc' )

    ## statistical values
    for xp in list_xp:
        for VAR in ['Eff','D_Tg_shift']:
            OUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp].sel(year=OUT.year),mask=np.isnan(TMP[VAR+'_'+xp].sel(year=OUT.year)))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],OUT.year.size,axis=0)
            OUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

    ## figure
    list_xp_plot = ['ssp585', 'ssp370',  'ssp460', 'ssp534-over', 'ssp434', 'ssp245', 'ssp126', 'ssp119']
    period = np.arange(2015,2300+1)
    dico_col = {'ssp119':(0/255.,170/255.,208/255.), 'ssp126':(0/255.,52/255.,102/255.), 'ssp245':(239/255.,85/255.,15/255.), 'ssp370':(224/255.,0/255.,0/255.), 'ssp370-lowNTCF':(224/255.,0/255.,0/255.), 'ssp434':(255/255.,169/255.,0/255.), 'ssp460':(196/255.,121/255.,0/255.), 'ssp534-over':(127/255.,0/255.,110/255.), 'ssp585':(153/255.,0/255.,2/255.)}
    dico_ls = {'ssp119':'-', 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp370-lowNTCF':'--', 'ssp434':'-', 'ssp460':'-', 'ssp534-over':'-', 'ssp585':'-'}
    # VAR_plot = ['Eff'  , 'co2'      , 'erf_tot'  , 'D_Tg_shift',\
    #             'fgco2', 'RF_AERtot', 'RF_nonCO2', 'pr',\
    #             'nbp'  , 'RF_O3t'   , 'ohc'      , 'permafrostCO2']
    VAR_plot = ['Eff'         , 'fgco2'     , 'nbp'      , 'co2',\
                'erf_tot'     , 'RF_nonCO2' , 'RF_AERtot', 'RF_O3t',\
                'D_Tg_shift'  , 'ohc'       , 'pr'       , 'permafrostCO2']

    # dico_title_VAR = {'CumD_Eluc':'Cumulated CO$_2$ emissions from LUC (PgC)','fLuc':'CO$_2$ emissions from LUC (PgC)','RF':'Radiative forcing (W.m$^{-2}$)','D_Tg':'Change in global mean\nsurface temperature (K)','CumD_Fland':'Cumulated land sink,\n permafrost emissions included (PgC)','RF_AERtot':'Radiative forcing\nfrom aerosols (W.m$^{-2}$)','RF_nonCO2':'Radiative forcing from CH$_4$, N$_2$O\nand halogenated compounds (W.m$^{-2}$)','Cfroz':'Carbon stock in permafrost (PgC)','CumD_Focean':'Cumulated ocean sink (PgC)','RF_O3':'Radiative forcing\nfrom ozone (W.m$^{-2}$)','CumEmiCH4':'Cumulated CH4 emissions from\nwetlands, permafrost and biomass burning (TgC)','pr':'Change in global\nmean precipitation (mm.yr$^{-1}$)'}
    dico_title_VAR = {'Eff':'Compatible fossil-fuel\nCO$_2$ emissions (PgC.yr$^{-1}$)'  ,  'co2':'Atmospheric CO$_2$ (ppm)'  ,  'fgco2':'Ocean sink of carbon (PgC.yr$^{-1}$)'  ,  'nbp':'Net land flux of carbon (PgC.yr$^{-1}$)'  ,  'permafrostCO2':'CO$_2$ emissions from permafrost (PgC.yr$^{-1}$)'  ,  'tas':'Change in global surface air temperature\nwith reference to 1850 (K)'  ,  'D_Tg_shift':'Change in global surface air temperature\nwith reference to 1850-1900 (K)'  ,  'RF_AERtot':'Radiative forcing of\naerosols (W.m$^{-2}$)'  ,  'RF_O3t':'Radiative forcing\nof tropospheric ozone (W.m$^{-2}$)'  ,  'erf_tot':'Effective Radiative forcing (W.m$^{-2}$)'  ,  'RF_nonCO2':'Radiative forcing\nof non-CO2 WMGHG (W.m$^{-2}$)'  ,  'ohc':'Change in ocean heat content\nwith reference to 1850 (ZJ)'  ,  'pr':'Change in global mean precipitation\nwith reference to 1850 (mm.yr$^{-1}$)'}
    fig = plt.figure( figsize=(30,20) )
    counter = 0
    ## looping on variables / subplots
    for var in VAR_plot:
        ax = plt.subplot(3,4,VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for xp in list_xp_plot[::-1]:func_plot('year',var+'_'+xp,OUT.sel(year=period),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*3*0.8,label=xp,alpha=0.15)
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[-1])
        plt.title( dico_title_VAR[var] , size=fac_size*14*0.8,rotation=0  )#,fontweight='bold'
        if var == VAR_plot[-2]:#[2]:
            plt.legend(prop={'size':fac_size*16*0.8} ,ncol=len(list_xp_plot),loc='center', bbox_to_anchor=(-0.05,-0.15))
            # plt.legend(prop={'size':fac_size*14*0.8} , bbox_to_anchor=(1.6,-0.18)) # ,frameon=False
        ax.tick_params(labelsize=fac_size*14*0.8)
        if VAR_plot.index(var) < 2*4:
            ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        else:
            pass
            # plt.xlabel( 'Year' , size=fac_size*15*0.8 )
        if var=='permafrostCO2':pass#plt.ylim(0,ax.get_ylim()[1])
        elif var=='RF_AERtot':pass#plt.ylim(-2.,0)
        elif var=='tas':
            plt.yticks( [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0] )
            plt.ylim(0.75,8.5)
        box = ax.get_position()
        # ax.set_position([box.x0-0.06, box.y0+0.05-(VAR_plot.index(var)//3)*0.045, box.width*1.0, box.height*1.1])
        ax.set_position([box.x0-0.03, box.y0+0.06-0.02*(VAR_plot.index(var)//4), box.width*1.0, box.height*1.1])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13*0.8})
        counter += 1

    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/scen.pdf',dpi=600 )
    plt.close(fig)

    for xp in list_xp:
        VAR = 'D_Tg_shift'
        for date in [[2081,2100]]:#[[2041,2050] , [2091,2100] , [2291,2300] , [2491,2500]]:
            val = np.ma.array(  TMP[VAR+'_'+xp].sel(year=np.arange(date[0],date[1]+1)).mean('year') - TMP[VAR+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')  ,  mask=np.isnan(TMP[VAR+'_'+xp].sel(year=np.arange(date[0],date[1]+1)).mean('year')) )
            ww = np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))
            mm = np.ma.average( a=val , axis=0 , weights=ww )
            ss = np.ma.sqrt(np.ma.average( (val - np.repeat(mm[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=0 , weights=ww ))
            print( VAR+', '+xp+' on '+str(date[0])+'-'+str(date[1])+': '+str(np.round(mm,2))+' +/- '+str(np.round(ss,2)) )
        VAR = 'erf_tot'
        mm = OUT[VAR+'_'+xp].loc[{'stat_value':'mean','year':2100}].values
        ss = OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev','year':2100}].values
        print( VAR+', '+xp+' on '+str(2100)+': '+str(np.round(mm,2))+' +/- '+str(np.round(ss,2)) )
        print("")
    for xp in list_xp:
        VAR = 'co2'
        for date in [2100,2300]:
            if 'stat_value' in OUT[VAR+'_'+xp].dims:
                mm = OUT[VAR+'_'+xp].loc[{'stat_value':'mean','year':date}].values
                ss = OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev','year':date}].values
                print( VAR+', '+xp+' on '+str(date)+': '+str(np.round(mm,2))+' +/- '+str(np.round(ss,2)) )
            else:
                mm = OUT[VAR+'_'+xp].loc[{'year':date}].values
                print( VAR+', '+xp+' on '+str(date)+': '+str(np.round(mm,2)) )
        print("")
    # print(OUT[var+'_'+xp].isel(year=209,reg_land=0).values)
    # OUT[var+'_'+xp].sel(reg_land=0,stat_value='mean').argmax('year')



if False:## results from CMIP6
    RES_CMIP6 = xr.Dataset()
    RES_CMIP6.coords['year'] = np.arange(1700,2500+1)
    ## first round to define dimensions
    list_models,list_xp,list_members =[],[],[]
    for mip in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6'):
        for institute in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip):
            for model in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute):
                if model not in list_models:list_models.append(model)
                for xp in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model):
                    if xp not in list_xp:list_xp.append(xp)
                    for member in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model+'/'+xp):
                        if member not in list_members:list_members.append(member)
    if option_overwrite:
        ## definitions
        counter = 0
        RES_CMIP6.coords['model'] = list_models
        RES_CMIP6.coords['xp'] = list_xp
        RES_CMIP6.coords['member'] = list_members
        RES_CMIP6['tas'] = xr.DataArray(  np.full(fill_value=np.nan,shape=(RES_CMIP6.year.size,RES_CMIP6.xp.size,RES_CMIP6.model.size,RES_CMIP6.member.size)) , dims=('year','xp','model','member')  )
        ## second round to get data
        for mip in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6'):
            for institute in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip):
                for model in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute):
                    for xp in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model):
                        for member in os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model+'/'+xp):
                            counter += 1
                            gnr = os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model+'/'+xp+'/'+member+'/Amon/tas')
                            if len(gnr)>1:
                                raise Exception("Multiple gn or gr?")
                            else:
                                ver = os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model+'/'+xp+'/'+member+'/Amon/tas/'+gnr[0])
                            if len(ver)>1:
                                raise Exception("Multiple versions?")
                            else:
                                file_W = os.listdir('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model+'/'+xp+'/'+member+'/Amon/tas/'+gnr[0]+'/'+ver[0]+'/')
                            with open('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/CMIP6/'+mip+'/'+institute+'/'+model+'/'+xp+'/'+member+'/Amon/tas/'+gnr[0]+'/'+ver[0]+'/'+file_W[0],'r',newline='') as ff:
                                TMP = np.array([line for line in csv.reader(ff)])
                            for line in TMP:
                                ind_start = [ind for ind in np.arange(len(TMP)) if (len(TMP[ind])>0) and ('YEARS' in TMP[ind][0])][0]
                            TMP2 = np.array( [eval(line[0].replace('       ',',')[1:]) for line in TMP[ind_start+1:]],dtype=np.float32)
                            head = TMP[ind_start][0].replace('           ',',').replace(' ','').split(',')
                            if xp in ['1pctCO2-4xext', '1pctCO2-bgc', '1pctCO2-cdr', '1pctCO2-rad', '1pctCO2', 'abrupt-0p5xCO2', 'abrupt-2xCO2', 'abrupt-4xCO2', 'G1', 'G2'] + ['esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-1pct-brch-750PgC', 'esm-1pctCO2', 'esm-abrupt-4xCO2', 'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC']:
                                RES_CMIP6['tas'].loc[{'year':np.arange(1850,1850+len(TMP2)),'xp':xp,'model':model,'member':member}] = TMP2[:,head.index('WORLD')]
                            else:
                                RES_CMIP6['tas'].loc[{'year':TMP2[:,head.index('YEARS')],'xp':xp,'model':model,'member':member}] = TMP2[:,head.index('WORLD')]
                            del TMP,TMP2,head

        RES_CMIP6.to_netcdf('H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/temperature_CMIP6.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in RES_CMIP6})
    else:
        RES_CMIP6 = xr.open_dataset( 'H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/CMIP6_data/temperature_CMIP6.nc' )


    if False:## plot
        xp = 'abrupt-2xCO2'

        ## plot and counting how many done
        counter = {str(model.values):[] for model in RES_CMIP6.model}
        for model in RES_CMIP6.model:
            for member in RES_CMIP6.member:
                if np.any(~np.isnan(RES_CMIP6['tas'].sel(xp=xp,model=model,member=member))):
                    counter[str(model.values)].append(str(member.values))
                    if len(counter[str(model.values)])==1:
                        plt.plot( RES_CMIP6.year , RES_CMIP6['tas'].sel(xp=xp,model=model,member=member) , color=sns.color_palette( 'colorblind', n_colors=RES_CMIP6.model.size )[list(RES_CMIP6.model).index(model)] , label=str(model.values) , lw=0.75*fac_size*0.5)
                    else:
                        plt.plot( RES_CMIP6.year , RES_CMIP6['tas'].sel(xp=xp,model=model,member=member) , color=sns.color_palette( 'colorblind', n_colors=RES_CMIP6.model.size )[list(RES_CMIP6.model).index(model)] , lw=0.75*fac_size*0.5)
        ## mean and std dev
        mm = RES_CMIP6['tas'].sel(xp=xp).mean('member').mean('model')
        ss = np.sqrt(  ((RES_CMIP6['tas'].sel(xp=xp).mean('member') - mm)**2.).sum('model') / ((~np.isnan(RES_CMIP6['tas'].sel(xp=xp).mean('member'))).sum('model') - 1)  )
        # ss = np.sqrt( ((RES_CMIP6['tas'].sel(xp=xp)-mm)**2.).sum(('model','member')) / ((~np.isnan(RES_CMIP6['tas'].sel(xp=xp))).sum(('model','member')) - 1) )
        ## plot mean and std dev
        plt.plot( RES_CMIP6.year , mm , lw=0.75*fac_size*5 ,label='average ('+str( len([mod for mod in counter.keys() if len(counter[mod])>0]) )+' models)')
        plt.fill_between( RES_CMIP6.year , mm-ss , mm+ss , alpha=0.33 ,zorder=100 )
        plt.grid()
        plt.legend(loc=0)

    if False:## values
        xp = 'historical'
        # for xp in ['esm-ssp585', 'esm-ssp370',  'esm-ssp460', 'esm-ssp534-over', 'esm-ssp434', 'esm-ssp245', 'esm-ssp126', 'esm-ssp119']:
        #     if xp in RES_CMIP6.xp:
        #         print(xp)
        # for xp in ['ssp585', 'ssp370',  'ssp460', 'ssp534-over', 'ssp434', 'ssp245', 'ssp126', 'ssp119']:
        for date in [[2041,2050] , [2091,2100] , [2291,2300] , [2491,2500]]:
            counter = {str(model.values):[] for model in RES_CMIP6.model}
            for model in RES_CMIP6.model:
                for member in RES_CMIP6.member:
                    val1,val0 = RES_CMIP6['tas'].sel(xp=xp,model=model,member=member,year=np.arange(date[0],date[1]+1)) , RES_CMIP6['tas'].sel(xp=xp,model=model,member=member,year=np.arange(1850,1900+1))
                    if (np.all(~np.isnan(val1))) and (np.all(~np.isnan(val0))):
                        counter[str(model.values)].append(str(member.values))
            vals = RES_CMIP6['tas'].sel(xp=xp,year=np.arange(date[0],date[1]+1)).mean('year') - RES_CMIP6['tas'].sel(xp=xp,year=np.arange(1850,1900+1)).mean('year')
            mm = vals.mean('member').mean('model')
            cnt = len([mod for mod in counter.keys() if len(counter[mod])>0])
            ss = np.sqrt( ((vals.mean('member')-mm)**2.).sum('model') / (cnt-1) )
            # mm = vals.mean( ('model','member') )
            # cnt = len([mod for mod in counter.keys() if len(counter[mod])>0])
            # ss = np.sqrt( ((vals-mm)**2.).sum(('model','member')) / (cnt-1) )
            print( xp+' with '+str(cnt)+' models: '+str(np.round(mm.values,2))+' +/- '+str(np.round(ss.values,2))+' K' )

#########################
#########################





#########################
## 7.9. Variant CMIP6 scenarios targeting the carbon cycle
#########################
if '7.9' in option_which_plots:
    list_xp = ['ssp585', 'ssp585-bgc', 'ssp534-over', 'ssp534-over-bgc']
    list_VAR = ['D_Eluc','D_Epf_CO2','D_Epf_CH4','D_Focean','D_Fland','D_Cfroz','D_Ewet','D_Fcirc','D_Cosurf']# + ['RF_CO2']+['RF_CH4', 'RF_N2O', 'RF_halo','RF_nonCO2']+['RF_H2Os', 'RF_O3s','RF_strat']+['RF_SO4', 'RF_POA', 'RF_NO3', 'RF_SOA', 'RF_dust', 'RF_salt','RF_scatter']+['RF_BC','RF_absorb']+['RF_cloud','RF_AERtot']+['RF_O3t','RF_slcf']+['RF_BCsnow', 'RF_lcc','RF_alb'] + ['RF','D_Tg','D_Pg']
    VAR_accelerate = ['D_Aland','csoil1_0','csoil2_0','cveg_0','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp']

    ## preparing data for figures
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(2014,2500+1)
        # for xp in list_xp:
        dic_tmp = {'ssp585':'ssp585','ssp585ext':'ssp585', 'ssp585-bgc':'ssp585-bgc','ssp585-bgcExt':'ssp585-bgc', 'ssp534-over':'ssp534-over','ssp534-over-ext':'ssp534-over', 'ssp534-over-bgc':'ssp534-over-bgc','ssp534-over-bgcExt':'ssp534-over-bgc'}
        for xp in ['ssp585','ssp585ext', 'ssp585-bgc','ssp585-bgcExt', 'ssp534-over','ssp534-over-ext', 'ssp534-over-bgc','ssp534-over-bgcExt']:
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                mode_ext = False# for mode_ext in [False,True]:
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , VAR_accelerate+list_VAR , mode_ext=mode_ext )
                # out_tmp['D_Ewet'] = OSCAR['D_Ewet'](out_tmp, Par, for_tmp.update(out_tmp),recursive=True)
                for var in list_VAR:
                    if var+'_'+dic_tmp[xp] not in TMP:
                        TMP[var+'_'+dic_tmp[xp]] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                    val = out_tmp[var].sel(year=out_tmp.year)
                    if 'reg_pf' in out_tmp[var].dims: val = val.sum('reg_pf')
                    if 'reg_land' in out_tmp[var].dims: val = val.sum('reg_land')
                    if 'box_osurf' in out_tmp[var].dims: val = val.sum('box_osurf')
                    TMP[var+'_'+dic_tmp[xp]].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = val

                ## additional variable: Cfroz
                if 'Cfroz'+'_'+dic_tmp[xp] not in TMP:
                    TMP['Cfroz'+'_'+dic_tmp[xp]] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                TMP['Cfroz_'+dic_tmp[xp]].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = (Par.Cfroz_0+out_tmp['D_Cfroz']).sum('reg_pf').transpose()
                ## additional variable: D_cLand
                if 'D_cLand'+'_'+dic_tmp[xp] not in TMP:
                    TMP['D_cLand'+'_'+dic_tmp[xp]] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                val = out_tmp['D_Fland'] - out_tmp['D_Eluc']
                if (out_tmp.year[0]-1).values in TMP.year:
                    before = ( TMP['D_cLand_'+dic_tmp[xp]].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year[0]-1 }] ).values
                else:
                    before = 0.
                TMP['D_cLand_'+dic_tmp[xp]].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = before + val.cumsum('year') * mask
                ## additional variable: D_cOcean
                if 'D_cOcean'+'_'+dic_tmp[xp] not in TMP:
                    TMP['D_cOcean'+'_'+dic_tmp[xp]] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                val = TMP['D_Fcirc'+'_'+dic_tmp[xp]].cumsum('year') + TMP['D_Cosurf'+'_'+dic_tmp[xp]]
                TMP['D_cOcean_'+dic_tmp[xp]].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':TMP.year }] = val.loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':TMP.year }]
                out_tmp.close()
                for_tmp.close()
            ## additional variable: D_Fland-D_Epf-D_Eluc
            TMP['D_Fland-D_Epf-D_Eluc'+'_'+dic_tmp[xp]] = TMP['D_Fland'+'_'+dic_tmp[xp]]-TMP['D_Epf_CO2'+'_'+dic_tmp[xp]]-TMP['D_Epf_CH4'+'_'+dic_tmp[xp]]*1.e-3 - TMP['D_Eluc'+'_'+dic_tmp[xp]]
            ## additional variable: CUMSUM D_Fland-D_Epf-D_Eluc
            TMP['CUMSUM D_Fland-D_Epf-D_Eluc'+'_'+dic_tmp[xp]] = (TMP['D_Fland'+'_'+dic_tmp[xp]]-TMP['D_Epf_CO2'+'_'+dic_tmp[xp]]-TMP['D_Epf_CH4'+'_'+dic_tmp[xp]]*1.e-3 - TMP['D_Eluc'+'_'+dic_tmp[xp]]).cumsum('year')
            ## additional variable: cLand + Cfroz
            TMP['D_cLand+D_Cfroz'+'_'+dic_tmp[xp]] = TMP['D_cLand_'+dic_tmp[xp]] + TMP['Cfroz_'+dic_tmp[xp]]

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantCcycle.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantCcycle.nc' )

    # statistical values
    OUT = xr.Dataset()
    OUT.coords['stat_value'] = ['mean','std_dev']
    OUT.coords['year'] = TMP.year
    for xp in list_xp:
        for VAR in list_VAR + ['D_Focean','D_cOcean','D_Fland-D_Epf-D_Eluc','D_cLand','D_cLand+D_Cfroz']:
            OUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    ## difference
    for VAR in list_VAR+['D_Focean','D_cOcean','D_Fland-D_Epf-D_Eluc','D_cLand','CUMSUM D_Fland-D_Epf-D_Eluc','D_cLand+D_Cfroz']:
        OUT[VAR+'_'+'diff-ssp534-over'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
        val = np.ma.array( TMP[VAR+'_'+'ssp534-over'] - TMP[VAR+'_'+'ssp534-over-bgc'] ,mask=np.isnan(TMP[VAR+'_'+'ssp534-over'] - TMP[VAR+'_'+'ssp534-over-bgc']))
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        OUT[VAR+'_'+'diff-ssp534-over'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[VAR+'_'+'diff-ssp534-over'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+'diff-ssp534-over'].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

        OUT[VAR+'_'+'diff-ssp585'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
        val = np.ma.array( TMP[VAR+'_'+'ssp585'] - TMP[VAR+'_'+'ssp585-bgc'] ,mask=np.isnan(TMP[VAR+'_'+'ssp585'] - TMP[VAR+'_'+'ssp585-bgc']))
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        OUT[VAR+'_'+'diff-ssp585'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[VAR+'_'+'diff-ssp585'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+'diff-ssp585'].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data


    ## figure
    period = np.arange(2015,2300+1)
    dico_col = {'ssp534-over':(127/255.,0/255.,110/255.), 'ssp534-over-bgc':(127/255.,0/255.,110/255.), 'ssp585':(153/255.,0/255.,2/255.), 'ssp585-bgc':(153/255.,0/255.,2/255.)}
    dico_ls = { 'ssp534-over':'-', 'ssp534-over-bgc':'--', 'ssp585':'-', 'ssp585-bgc':'--'}
    VAR_plot = ['D_Focean','D_cOcean','D_Fland-D_Epf-D_Eluc','D_cLand+D_Cfroz']#,'D_cLand']
    dico_title_VAR = {'D_Fland-D_Epf-D_Eluc':'Net carbon flux from\natmosphere to land (PgC.yr$^{-1}$)','D_Focean':'Oceanic carbon sink (PgC.yr$^{-1}$)','D_cOcean':'Change in the oceanic\ncarbon stock (PgC)','D_cLand':'Change in the land\ncarbon stock (PgC)','D_cLand+D_Cfroz':'Change in the total\nland carbon stock (PgC)'}
    fig = plt.figure( figsize=(30,20) )
    counter = 0
    ## scenarios
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for xp in list_xp:
            func_plot('year',var+'_'+xp,OUT.sel(year=period),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*3*0.9,label=xp)
            date_print = 2100
            mm,ss = OUT[var+'_'+xp].sel(year=date_print,stat_value='mean'), OUT[var+'_'+xp].sel(year=date_print,stat_value='std_dev')
            print('OSCAR: '+var+' on '+xp+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[-1])
        plt.title( dico_title_VAR[var] , size=fac_size*14*0.9,rotation=0  )#,fontweight='bold'
        ax.tick_params(labelsize=fac_size*13*0.9)
        ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        box = ax.get_position()
        ax.set_position([box.x0+0.05-0.005*(VAR_plot.index(var)%len(VAR_plot)), box.y0+0.02, box.width*0.9, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*0.9*13})
        counter += 1
        if var == VAR_plot[0]:
            plt.legend(loc=0,prop={'size':fac_size*11*0.9} ,bbox_to_anchor=(-0.1025,0.60))
            plt.ylabel( 'Scenarios' , size=fac_size*16*0.9,rotation=0  )#,fontweight='bold'
            ax.yaxis.set_label_coords(-0.40,0.6)

    ## differences
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),len(VAR_plot)+VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for xp_diff in ['ssp534-over','ssp585']:
            func_plot('year',var+'_diff-'+xp_diff,OUT.sel(year=period),col=dico_col[xp_diff],ls=dico_ls[xp_diff],lw=0.75*fac_size*3*0.9,label=xp_diff+' - '+xp_diff+'-bgc')
            date_print = 2100
            mm,ss = OUT[var+'_diff-'+xp_diff].sel(year=date_print,stat_value='mean'), OUT[var+'_diff-'+xp_diff].sel(year=date_print,stat_value='std_dev')
            print('OSCAR: '+var+'_diff-'+xp_diff+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[-1])
        ax.tick_params(labelsize=fac_size*13*0.9)
        plt.xlabel( 'Year' , size=fac_size*15*0.9 )
        box = ax.get_position()
        ax.set_position([box.x0+0.05-0.005*(VAR_plot.index(var)%len(VAR_plot)), box.y0+0.02, box.width*0.9, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13*0.9})
        counter += 1
        if var == VAR_plot[0]:
            plt.legend(loc=0,prop={'size':fac_size*9*0.9} ,bbox_to_anchor=(-0.1000,0.60))
            plt.ylabel( 'Differences' , size=fac_size*16*0.9,rotation=0  )#,fontweight='bold'
            ax.yaxis.set_label_coords(-0.40,0.6)
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/variant-bgc.pdf',dpi=300 )
    plt.close(fig)
#########################
#########################









#########################
## 7.10. Variant CMIP6 scenarios targeting LU
#########################
if '7.10' in option_which_plots:
    list_xp = ['ssp585', 'ssp585-ssp126Lu', 'ssp370' , 'ssp370-ssp126Lu' , 'ssp126', 'ssp126-ssp370Lu']
    list_VAR = ['D_Eluc','D_Fland','RF_lcc' ]# + ['RF_CO2']+['RF_CH4', 'RF_N2O', 'RF_halo','RF_nonCO2']+['RF_H2Os', 'RF_O3s','RF_strat']+['RF_SO4', 'RF_POA', 'RF_NO3', 'RF_SOA', 'RF_dust', 'RF_salt','RF_scatter']+['RF_BC','RF_absorb']+['RF_cloud','RF_AERtot']+['RF_O3t','RF_slcf']+['RF_BCsnow', 'RF_lcc','RF_alb'] + ['RF','D_Tg','D_Pg']
    VAR_accelerate = ['D_Aland','csoil1_0','csoil2_0','cveg_0','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp']

    ## preparing data for figures
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(2014,2100+1)
        for xp in list_xp:
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                mode_ext = False # for mode_ext in [False,True]:
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , VAR_accelerate+list_VAR , mode_ext=mode_ext )
                # out_tmp['D_Ewet'] = OSCAR['D_Ewet'](out_tmp, Par, for_tmp.update(out_tmp),recursive=True)
                for var in list_VAR:
                    if var+'_'+xp not in TMP:
                        TMP[var+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                    val = out_tmp[var].sel(year=out_tmp.year)
                    if 'reg_pf' in out_tmp[var].dims: val = val.sum('reg_pf')
                    if 'reg_land' in out_tmp[var].dims: val = val.sum('reg_land')
                    TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = val
                ## additional variable: D_cLand
                if 'D_cLand'+'_'+xp not in TMP:
                    TMP['D_cLand'+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                val = out_tmp['D_Fland'] - out_tmp['D_Eluc']
                TMP['D_cLand_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = val.cumsum('year') * mask
                # val =  ( (out_tmp['csoil1_0']+out_tmp['D_csoil1']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
                # val += ( (out_tmp['csoil2_0']+out_tmp['D_csoil2']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
                # val += ( (out_tmp['cveg_0']+out_tmp['D_cveg']) * (Par.Aland_0+out_tmp['D_Aland']) ).sum(('bio_land'))
                # val += ( out_tmp['D_Csoil1_bk']+out_tmp['D_Csoil2_bk']+out_tmp['D_Cveg_bk'] ).sum(('bio_from','bio_to'))
                # val += out_tmp['D_Chwp'].sum( ('bio_from','bio_to','box_hwp') )
                # val -= ((out_tmp['csoil1_0']+out_tmp['csoil2_0']+out_tmp['cveg_0'])*Par.Aland_0).sum(('bio_land'))
                # TMP['D_cLand_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = val.sum('reg_land').transpose( 'year','config' )
                out_tmp.close()
                for_tmp.close()

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantLU.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantLU.nc' )

    # statistical values
    OUT = xr.Dataset()
    OUT.coords['stat_value'] = ['mean','std_dev']
    OUT.coords['year'] = TMP.year
    for xp in list_xp:
        for VAR in list_VAR + ['D_cLand']:
            OUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array( TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]) )
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    ## difference
    for VAR in list_VAR+['D_cLand']:
        OUT[VAR+'_'+'diff-ssp585-ssp126Lu - ssp585'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
        val = np.ma.array( TMP[VAR+'_'+'ssp585-ssp126Lu'] - TMP[VAR+'_'+'ssp585'] ,mask=np.isnan(TMP[VAR+'_'+'ssp585-ssp126Lu'] - TMP[VAR+'_'+'ssp585']))
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        OUT[VAR+'_'+'diff-ssp585-ssp126Lu - ssp585'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[VAR+'_'+'diff-ssp585-ssp126Lu - ssp585'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+'diff-ssp585-ssp126Lu - ssp585'].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

        OUT[VAR+'_'+'diff-ssp370-ssp126Lu - ssp370'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
        val = np.ma.array( TMP[VAR+'_'+'ssp370-ssp126Lu'] - TMP[VAR+'_'+'ssp370'] ,mask=np.isnan(TMP[VAR+'_'+'ssp370-ssp126Lu'] - TMP[VAR+'_'+'ssp370']))
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        OUT[VAR+'_'+'diff-ssp370-ssp126Lu - ssp370'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[VAR+'_'+'diff-ssp370-ssp126Lu - ssp370'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+'diff-ssp370-ssp126Lu - ssp370'].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

        OUT[VAR+'_'+'diff-ssp126-ssp370Lu - ssp126'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
        val = np.ma.array( TMP[VAR+'_'+'ssp126-ssp370Lu'] - TMP[VAR+'_'+'ssp126'] ,mask=np.isnan(TMP[VAR+'_'+'ssp126-ssp370Lu'] - TMP[VAR+'_'+'ssp126']))
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        OUT[VAR+'_'+'diff-ssp126-ssp370Lu - ssp126'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[VAR+'_'+'diff-ssp126-ssp370Lu - ssp126'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+'diff-ssp126-ssp370Lu - ssp126'].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data


    ## figure
    period = np.arange(2015,2100+1)
    dico_col = {'ssp126':(0/255.,52/255.,102/255.), 'ssp126-ssp370Lu':(0/255.,52/255.,102/255.), 'ssp370':(224/255.,0/255.,0/255.), 'ssp370-ssp126Lu':(224/255.,0/255.,0/255.), 'ssp585':(153/255.,0/255.,2/255.) , 'ssp585-ssp126Lu':(153/255.,0/255.,2/255.)}

    dico_ls = {'ssp585':'-', 'ssp585-ssp126Lu':'-.', 'ssp370':'-' , 'ssp370-ssp126Lu':'-.' , 'ssp126':'-', 'ssp126-ssp370Lu':'-.'}
    VAR_plot = ['D_Eluc','D_Fland','D_cLand','RF_lcc']
    dico_title_VAR = {'D_Eluc':'CO$_2$ emissions from LUC (PgC.yr$^{-1}$)','D_Fland':'Land carbon sink (PgC.yr$^{-1}$)','D_cLand':'Change in the land\ncarbon stock (PgC)', 'RF_lcc':'Radiative forcing from\nalbedo of land cover change (W.m$^{-2})$'}
    fig = plt.figure( figsize=(30,20) )
    counter = 0
    ## scenarios
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for xp in list_xp:
            func_plot('year',var+'_'+xp,OUT.sel(year=period),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*3*0.8,label=xp,alpha=0.15)
            date_print = 2100
            mm,ss = OUT[var+'_'+xp].sel(year=date_print,stat_value='mean'), OUT[var+'_'+xp].sel(year=date_print,stat_value='std_dev')
            print('OSCAR: '+var+' on '+xp+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[-1])
        plt.title( dico_title_VAR[var] , size=fac_size*14*0.8)#,fontweight='bold',rotation=0  )
        ax.tick_params(labelsize=fac_size*13*0.8)
        ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        box = ax.get_position()
        ax.set_position([box.x0+0.04-0.005*(VAR_plot.index(var)%len(VAR_plot)), box.y0+0.02, box.width*0.925, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
        counter += 1
        if var == VAR_plot[0]:
            plt.legend(loc=0,prop={'size':fac_size*11*0.8} ,bbox_to_anchor=(-0.14,0.60))
            plt.ylabel( 'Scenarios' , size=fac_size*13*0.8,rotation=0)#,fontweight='bold'  )
            ax.yaxis.set_label_coords(-0.40,0.6)

    ## differences
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),len(VAR_plot)+VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for xp_diff in ['ssp585-ssp126Lu - ssp585' , 'ssp370-ssp126Lu - ssp370' , 'ssp126-ssp370Lu - ssp126']:
            func_plot('year',var+'_diff-'+xp_diff,OUT.sel(year=period),col=dico_col[str.split(xp_diff,' - ')[0]],ls=dico_ls[str.split(xp_diff,' - ')[1]],lw=0.75*fac_size*3*0.8,label=xp_diff,alpha=0.15)
            date_print = 2100
            mm,ss = OUT[var+'_diff-'+xp_diff].sel(year=date_print,stat_value='mean'), OUT[var+'_diff-'+xp_diff].sel(year=date_print,stat_value='std_dev')
            print('OSCAR: '+var+' on '+xp_diff+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[-1])
        ax.tick_params(labelsize=fac_size*13*0.8)
        plt.xlabel( 'Year' , size=fac_size*15*0.8 )
        box = ax.get_position()
        ax.set_position([box.x0+0.04-0.005*(VAR_plot.index(var)%len(VAR_plot)), box.y0+0.02, box.width*0.925, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*13})
        counter += 1
        if var == VAR_plot[0]:
            plt.legend(loc=0,prop={'size':fac_size*11*0.8} ,bbox_to_anchor=(-0.14,0.60))
            plt.ylabel( 'Differences' , size=fac_size*13*0.8,rotation=0)#,fontweight='bold'  )
            ax.yaxis.set_label_coords(-0.40,0.6)
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/variant-LU.pdf',dpi=300 )
    plt.close(fig)
#########################
#########################







#########################
## 7.11. Historical constrained: emissions/concentrations driven
#########################
if False:## former version
    list_VAR = ['Eff','D_CO2','D_Tg','D_Eluc','D_Epf_CO2','D_Focean','D_Fland','RF_O3t','RF_nonCO2','RF_scatter', 'RF_absorb', 'RF_cloud','RF_AERtot','RF','RF_warm','D_Pg','D_OHC']

    ## preparing data for figures
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(1850,2014+1)
        for xp in ['historical', 'esm-hist' , 'piControl' , 'esm-piControl']:
            for setMC in list_setMC:
                print(xp+'/'+str(setMC))
                if xp in ['historical','esm-hist']:
                    out_tmp,for_tmp,Par,mask = func_get( setMC , xp , ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp'] + list_VAR[1:] )
                elif xp in ['piControl' , 'esm-piControl']:
                    out_tmp,for_tmp,Par,mask = func_get( setMC , xp , ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp'] + list_VAR[1:] , option_NeedDiffControl=False )
                for_tmp = eval_compat_emi( ['CO2'], out_tmp,Par,for_tmp )
                for var in list_VAR:
                    if var not in TMP:
                        TMP[var+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                    if var in for_tmp and 'reg_land' in for_tmp[var].dims:yy = for_tmp[var].sum('reg_land') * mask
                    elif 'reg_land' in out_tmp[var].dims:yy = for_tmp[var].sum('reg_land') * mask
                    elif 'reg_pf' in out_tmp[var].dims:yy = out_tmp[var].sum('reg_pf') * mask
                    else: yy = out_tmp[var] * mask
                    TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':np.arange(1850,min([2014,out_tmp.year[-1]])+1) }] = yy.sel(year=np.arange(1850,min([2014,out_tmp.year[-1]])+1))
                out_tmp.close()
                for_tmp.close()
            ## adding new variable
            TMP['D_LandNet'+'_'+xp] = TMP['D_Fland'+'_'+xp] - TMP['D_Eluc'+'_'+xp] - TMP['D_Epf_CO2'+'_'+xp].sum('reg_pf') - 1.e-3 * TMP['D_Epf_CH4'+'_'+xp].sum('reg_pf')## not entire oxidation flux of CH4, not everything from Land, eg FF&I
            TMP['D_Pg_shift'+'_'+xp] = TMP['D_Pg'+'_'+xp] - TMP['D_Pg'+'_'+xp].sel(year=np.arange(1979,1989+1)).mean('year')
            TMP['D_OHC_shift'+'_'+xp] = TMP['D_OHC'+'_'+xp] - TMP['D_OHC'+'_'+xp].sel(year=np.arange(1955,2006+1)).mean('year')
            TMP['D_Tg_shift'+'_'+xp] = TMP['D_Tg'+'_'+xp] - TMP['D_Tg'+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year')
            TMP['RF_AER-rad'+'_'+xp] = TMP['RF_scatter'+'_'+xp] + TMP['RF_absorb'+'_'+xp] 

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_hist.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/'+'data_figures_hist.nc' )
        
        
    # statistical values
    OUTPUT = xr.Dataset()
    OUTPUT.coords['stat_value'] = ['mean','std_dev']
    OUTPUT.coords['year'] = np.arange(1850,2014+1)
    for xp in ['piControl' , 'esm-piControl','historical', 'esm-hist']:
        for VAR in list_VAR+['D_LandNet','D_Pg_shift','D_OHC_shift','D_Tg_shift','RF_AER-rad']:
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]))
            ## constrained
            OUTPUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUTPUT.year.size,OUTPUT.stat_value.size)), dims=('year','stat_value') )
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUTPUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUTPUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUTPUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
            # ## unconstrained
            # OUTPUT[VAR+'_'+xp+'_noConst'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUTPUT.year.size,OUTPUT.stat_value.size)), dims=('year','stat_value') )
            # ww = np.ma.repeat(np.ma.array(weights_ones.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_ones.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            # OUTPUT[VAR+'_'+xp+'_noConst'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            # OUTPUT[VAR+'_'+xp+'_noConst'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUTPUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    ## adding control
    for VAR in list_VAR+['D_Fland-D_Eluc','D_Pg_shift','D_OHC_shift','RF_AER-rad']:
        OUTPUT[VAR+'_'+'historical'].loc[{'stat_value':'mean'}] = OUTPUT[VAR+'_'+'historical'].loc[{'stat_value':'mean'}] + OUTPUT[VAR+'_'+'piControl'].loc[{'stat_value':'mean'}].isel(year=np.arange(-50,-1+1)).mean('year')
        OUTPUT[VAR+'_'+'esm-hist'].loc[{'stat_value':'mean'}] = OUTPUT[VAR+'_'+'esm-hist'].loc[{'stat_value':'mean'}] + OUTPUT[VAR+'_'+'esm-piControl'].loc[{'stat_value':'mean'}].isel(year=np.arange(-50,-1+1)).mean('year')


    ## PLOT
    period = [1850,2014]
    # VAR_plot = [ 'Eff','D_CO2','D_Tg','D_Focean','D_Fland-D_Eluc','D_Epf_CO2','RF_AERtot','RF_O3t','RF_warm' ]
    VAR_plot = [ 'Eff','D_CO2','RF_warm','D_Focean','RF_nonCO2','D_Tg_shift','D_Fland-D_Eluc','RF_O3t','D_Pg_shift','D_Epf_CO2','RF_AERtot','D_OHC_shift' ]
    list_xp = ['historical','esm-hist']
    dico_col = {'historical':CB_color_cycle[1],'esm-hist':CB_color_cycle[2],'obs':CB_color_cycle[4:4+5]}
    dico_ls = {'historical':'-','esm-hist':'--'}
    dico_titles = {'Eff':'Fossil-fuel CO$_2$ emissions (PgC.yr$^{-1}$)','D_CO2':'Change in atmospheric CO$_2$\nwith reference to 1750 (ppm)','D_Eluc':'CO$_2$ emissions from LUC (PgC.yr$^{-1}$)','D_Focean':'Ocean sink of carbon (PgC.yr$^{-1}$)','D_Fland':'Land sink of carbon (PgC.yr$^{-1}$)','D_Fland-D_Eluc':'Sum of land sink of carbon\n& CO$_2$ emissions from LUC (PgC.yr$^{-1}$)','D_Epf_CO2':'CO$_2$ emissions from permafrost (PgC.yr$^{-1}$)','D_Tg_shift':'Change in global surface air temperature\nwith reference to 1850-1900 (K)','RF_AER-rad':'Radiative forcing of\naerosols-radiation effects (W.m$^{-2}$)','RF_AERtot':'Radiative forcing of\naerosols (W.m$^{-2}$)','RF_O3t':'Radiative forcing\nof tropospheric ozone (W.m$^{-2}$)','RF_warm':'Effective Radiative forcing (W.m$^{-2}$)','RF_nonCO2':'Radiative forcing\nof non-CO2 WMGHG (W.m$^{-2}$)','D_OHC_shift':'Ocean heat content\nwith refence to 1955-2006(ZJ)','D_Pg_shift':'Change in global mean precipitation\nwith reference to 1979-1989 (mm.yr$^{-1}$)'}
    alpha_plot = 0.3
    dico_obs = {'Eff':['carbon-cycle_GCB'],'D_CO2':['carbon-cycle_GCB'],'D_Eluc':['carbon-cycle_GCB'],'D_Fland-D_Eluc':['carbon-cycle_GCB'],'D_Focean':['carbon-cycle_GCB'],'D_Fland':['carbon-cycle_GCB'],'D_Epf_CO2':[],'RF_AER-rad':['radiative-forcing_AR5'],'RF_O3t':['radiative-forcing_AR5'],'RF_warm':['radiative-forcing_AR5'],'RF_AERtot':['radiative-forcing_AR5'],'D_Tg_shift':['temperature_Cowtan2014'],'RF_nonCO2':['radiative-forcing_AR5'],'D_Pg_shift' :['precipitation_GPCP'],'D_OHC_shift':['ocean-heat_NOAA-NODC']}#'temperature_BerkeleyEarth','temperature_GISTEMP','temperature_HadCRUT4','temperature_NOAA-GlobalTemp'

    fig = plt.figure( figsize=(20,30) )
    counter = 0
    for VAR in VAR_plot:
        ax = plt.subplot(4,3,VAR_plot.index(VAR)+1)
        ax.set_axisbelow(True)
        for xp in list_xp:
            func_plot('year',VAR+'_'+xp,OUTPUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],edgecolor=None,ls='-',label=xp,alpha=alpha_plot,zorder=100)
        for oo in dico_obs[VAR]:
            obs = xr.open_dataset('H:/MyDocuments/Repositories/OSCARv31_CMIP6/input_data/observations/'+oo+'.nc' )
            lw_obs = 3
            if VAR=='D_Tg_shift':
                date_const = [[2006,2015]]
                const = [[0.87 * 0.99/0.86  ,  np.sqrt((0.12/0.955 * (1.37-0.65) / (1.18-0.54))**2. + 0.1467**2.)]]
                lw_obs = 2
                lbl = 'IPCC SR 1.5C (constraint)'
                to_plot = obs['D_Tg'] - obs['D_Tg'].sel(year=np.arange(1850,1900+1)).mean('year')
                lbl2 = 'Cowtan et al, 2014'
            elif VAR=='D_Pg_shift':
                # obs = xr.open_dataset( 'H:/MyDocuments/Repositories/OSCARv31_CMIP6/extra_data/precipitation_GPCP.nc'  )
                to_plot = obs['Pg'] - obs['Pg'].sel(year=np.arange(1979,1989+1)).mean('year')
                # to_plot_std = obs['Pg_err']

                # ref = obs['Pg'].sel(year=np.arange(1979,1989+1)).mean('year')
                # date_const = [[1979,1989], [1990,1999], [2000,2009]]
                # const = [[0,0], [(obs['Pg'].sel(year=np.arange(1990,1999+1)).mean('year')-ref).values,0.], [(obs['Pg'].sel(year=np.arange(2000,2009+1)).mean('year')-ref).values,0.]]

                lbl = 'GPCP 2.3'
                lw_obs = 2
            elif VAR=='D_OHC_shift':
                to_plot = obs['OHC'].sel(depth_lim=2000)
                to_plot_std = obs['std_OHC'].sel(depth_lim=2000)
                # date_const = [[2010,2010]]
                # const = [[obs['OHC'].sel(depth_lim=2000,year=2010).values,obs['std_OHC'].sel(depth_lim=2000,year=2010).values]]
                lbl = 'NOAA-NODC'
            elif VAR=='Eff':
                # to_plot = obs[VAR]
                date_const = [[1960,1969], [1970,1979], [1980,1989], [1990,1999], [2000,2009]]
                const = [[3.0,0.2], [4.7,0.2], [5.5,0.3], [6.4,0.3], [7.8,0.4]]#[[3.1,0.2], [4.7,0.2], [5.4,0.3], [6.3,0.3], [7.8,0.4]]
                lbl = 'GCB 2020'#'GCB 2018'
            elif VAR=='D_Fland-D_Eluc':
                # to_plot = obs['Fland'] - obs['Eluc']
                date_const = [[1960,1969], [1970,1979], [1980,1989], [1990,1999], [2000,2009]]
                const = [[-0.1,0.5], [0.7,0.6], [0.4,0.6], [1.2,0.6], [1.1,0.6]]#[[-0.3,0.6], [0.7,0.5], [0.3,0.6], [1.1,0.5], [1.3,0.5]]
                lbl = 'GCB 2020'#'GCB 2018'
            elif VAR in ['D_Epf_CO2']:
                raise Exception("Not prepared")
            elif VAR in ['D_Focean']:#'D_Eluc',,'D_Fland']:
                # to_plot = obs[VAR[2:]]
                date_const = [[1960,1969], [1970,1979], [1980,1989], [1990,1999], [2000,2009]]
                const = [[1.0,0.6], [1.3,0.6], [1.7,0.6], [2.0,0.6], [2.2,0.6]]#[[1.0,0.5], [1.3,0.5], [1.7,0.5], [2.0,0.5], [2.1,0.5]]
                lbl = 'GCB 2020'#'GCB 2018'
            elif VAR in ['RF_AERtot','RF_O3t','RF_nonCO2']:
                # to_plot = obs[VAR]
                date_const = [[2011,2011]]
                const = [{'RF_AERtot':[-0.9 , np.sqrt((-0.95+0.45)**2.+(-1.2+0.45)**2.)/1.282 , np.sqrt((0.05+0.45)**2.+(0+0.45)**2.)/1.282]  ,  'RF_O3t':[0.4,0.2/1.282]  ,  'RF_nonCO2':[1.01,0.07/1.282]}[VAR]]
                lbl = 'AR5 Ch8'
            elif VAR in ['D_CO2']:
                date_const = [[2011,2011]]
                const = [[390.5-278.6,2./1.282]]
                lbl = 'AR5 Ch8'
            elif VAR in ['RF_warm']:
                # obs = obs.sel(year=np.arange(1850,2011+1))
                # to_plot = 0.
                # for var in obs:to_plot = to_plot + obs[var]
                date_const = [[2011,2011]]
                const = [[2.3,1.1]]
                lbl = 'AR5'
            else:
                raise Exception("unprepared case")
        # if VAR in ['D_Pg_shift' ]:
        #     plt.plot( obs.year, to_plot ,color=dico_col['obs'][dico_obs[VAR].index(oo)],ls='-',lw=0.75*fac_size*lw_obs * 1.,label=lbl)
        if VAR in ['D_Tg_shift' ]:
            plt.plot( obs.year, to_plot ,color=dico_col['obs'][dico_obs[VAR].index(oo)],ls='-',lw=0.75*fac_size*lw_obs * 1.25,label=lbl2 )
        if VAR in ['D_Pg_shift']:
            plt.plot( obs.year, to_plot ,color=dico_col['obs'][dico_obs[VAR].index(oo)],ls='-',lw=0.75*fac_size*lw_obs * 1.25,label=lbl )
        elif VAR in ['D_OHC_shift' ]:
            plt.plot( obs.year, to_plot ,color=dico_col['obs'][dico_obs[VAR].index(oo)],ls='-',lw=0.75*fac_size*lw_obs * 1.25,label=lbl,zorder=100)
            plt.fill_between( obs.year, to_plot-to_plot_std , to_plot+to_plot_std , facecolor=dico_col['obs'][dico_obs[VAR].index(oo)] , edgecolor=dico_col['obs'][dico_obs[VAR].index(oo)]  , alpha=0.4,zorder=100)

        if VAR not in ['D_OHC_shift','D_Epf_CO2' , 'D_Pg_shift']:
            for ii in np.arange(len(const)):
                if ii==0:
                    plt.errorbar( x=(date_const[ii][0]+date_const[ii][1])/2.,y=const[ii][0],xerr=(date_const[ii][1]-date_const[ii][0])/2.,yerr=np.array(const[ii][1:])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25,label=lbl,color='white',lw=0.75*fac_size*0,zorder=1000 )
                else:
                    plt.errorbar( x=(date_const[ii][0]+date_const[ii][1])/2.,y=const[ii][0],xerr=(date_const[ii][1]-date_const[ii][0])/2.,yerr=np.array(const[ii][1:])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25,color='white',lw=0.75*fac_size*0,zorder=100 )
                plt.scatter( y=const[ii][0] , x=(date_const[ii][0]+date_const[ii][1])/2. , facecolor='k',edgecolor='w',marker='o',s=fac_size*4.*20,zorder=101 )
            del const,date_const
        plt.title( dico_titles[VAR] , size=fac_size*14 * 1.25 )#,fontweight='bold')
        plt.xlim(period[0],period[1])
        # if VAR == 'RF_warm':
        #     plt.ylim(-2,ax.get_ylim()[1])
        if VAR == 'D_Epf_CO2':
            plt.ylim(0,ax.get_ylim()[1])
        if VAR_plot.index(VAR)+1<=9:
            pass
            # ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
        else:
            plt.xlabel( 'Year' , size=fac_size*14 * 1.25 )
        ax.tick_params(labelsize=fac_size*13)
        plt.grid()
        if VAR == 'RF_AERtot':
            plt.legend(loc='lower left',prop={'size':fac_size*13 * 1.25})
        else:
            plt.legend(loc='upper left',prop={'size':fac_size*13 * 1.25})
        box = ax.get_position()
        ax.set_position([box.x0-0.03, box.y0+0.06-0.02*(VAR_plot.index(VAR)//3), box.width*1.0, box.height*1.1])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
        counter += 1

    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/hist.pdf',dpi=600 )
    plt.close(fig)


    
    
    
    
    

    
    

if ('7.11' in option_which_plots):## NEW version
    list_VAR = ['Eff','D_CO2','D_Tg','D_Eluc','D_Epf_CO2','D_Epf_CH4','D_Focean','D_Fland','D_OHC', 'RF_warm','RF_CO2','RF_CH4','RF_N2O','RF_AERtot','RF_O3t','RF_O3s']
    fac_size = 2 * 0.655
    
    ## preparing data for figures
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(1850,2014+1)
        for xp in ['historical', 'esm-hist' , 'piControl' , 'esm-piControl']:
            for setMC in list_setMC:
                print(xp+'/'+str(setMC))
                if xp in ['historical','esm-hist']:
                    out_tmp,for_tmp,Par,mask = func_get( setMC , xp , ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp'] + list_VAR[1:] )
                elif xp in ['piControl' , 'esm-piControl']:
                    out_tmp,for_tmp,Par,mask = func_get( setMC , xp , ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp'] + list_VAR[1:] , option_NeedDiffControl=False )
                for_tmp = eval_compat_emi( ['CO2'], out_tmp,Par,for_tmp )
                for var in list_VAR:
                    if var not in TMP:
                        TMP[var+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                    if var == 'Eff':
                        if xp in ['historical','piControl']:
                            yy = out_tmp['Eff_comp'] * mask
                        else:
                            yy = for_tmp['Eff'].sum('reg_land') * xr.DataArray(mask,dims={'year':for_tmp.year.values,'config':dico_sizesMC[setMC]})
                    elif 'reg_land' in out_tmp[var].dims:yy = for_tmp[var].sum('reg_land') * mask
                    elif 'reg_pf' in out_tmp[var].dims:yy = out_tmp[var].sum('reg_pf') * mask
                    else: yy = out_tmp[var] * mask
                    TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':np.arange(1850,min([2014,out_tmp.year[-1]])+1) }] = yy.sel(year=np.arange(1850,min([2014,out_tmp.year[-1]])+1))
                out_tmp.close()
                for_tmp.close()
            ## adding new variable
            TMP['D_LandNet'+'_'+xp] = TMP['D_Fland'+'_'+xp] - TMP['D_Eluc'+'_'+xp] - TMP['D_Epf_CO2'+'_'+xp] - 1.e-3 * TMP['D_Epf_CH4'+'_'+xp]## not entire oxidation flux of CH4, not everything from Land, eg FF&I
            TMP['D_OHC_shift'+'_'+xp] = TMP['D_OHC'+'_'+xp] - TMP['D_OHC'+'_'+xp].sel(year=np.arange(1971,1971+1)).mean('year')
            TMP['D_Tg_shift'+'_'+xp] = TMP['D_Tg'+'_'+xp] - TMP['D_Tg'+'_'+xp].sel(year=np.arange(1850,1900+1)).mean('year')
            TMP['CO2'+'_'+xp] = TMP['D_CO2'+'_'+xp] + Par['CO2_0']

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_hist.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        #TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_hist.nc' )
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/'+'data_figures_hist.nc' )        

    # statistical values
    OUTPUT = xr.Dataset()
    OUTPUT.coords['stat_value'] = ['mean','std_dev']
    OUTPUT.coords['year'] = np.arange(1850,2014+1)
    for xp in ['piControl' , 'esm-piControl','historical', 'esm-hist']:
        for VAR in list_VAR+['D_LandNet','D_OHC_shift','D_Tg_shift','CO2']:
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]))
            ## constrained
            OUTPUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUTPUT.year.size,OUTPUT.stat_value.size)), dims=('year','stat_value') )
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUTPUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUTPUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUTPUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    ## adding control
    for VAR in list_VAR+['D_LandNet','D_OHC_shift','D_Tg_shift', 'CO2']:
        OUTPUT[VAR+'_'+'historical'].loc[{'stat_value':'mean'}] = OUTPUT[VAR+'_'+'historical'].loc[{'stat_value':'mean'}] + OUTPUT[VAR+'_'+'piControl'].loc[{'stat_value':'mean'}].isel(year=np.arange(-50,-1+1)).mean('year')
        OUTPUT[VAR+'_'+'esm-hist'].loc[{'stat_value':'mean'}] = OUTPUT[VAR+'_'+'esm-hist'].loc[{'stat_value':'mean'}] + OUTPUT[VAR+'_'+'esm-piControl'].loc[{'stat_value':'mean'}].isel(year=np.arange(-50,-1+1)).mean('year')


    ## PLOT
    fig = plt.figure( figsize=(30,20) )
    counter = 0
    period = [1850,2014]
    list_xp = ['historical','esm-hist']
    ## general options
    dico_col = {'historical':CB_color_cycle[1],'esm-hist':CB_color_cycle[2],'obs':CB_color_cycle[4:4+5]}
    dico_ls = {'historical':'-','esm-hist':'--'}
    dico_titles = {'Eff':'Fossil-fuel CO$_2$ emissions (PgC.yr$^{-1}$)','D_CO2':'Change in atmospheric CO$_2$\nwith reference to 1750 (ppm)','D_Eluc':'CO$_2$ emissions from LUC (PgC.yr$^{-1}$)','D_Focean':'Ocean sink of carbon (PgC.yr$^{-1}$)','D_Fland':'Land sink of carbon (PgC.yr$^{-1}$)','D_LandNet':'Net atmosphere to\nland flux of carbon (PgC.yr$^{-1}$)','D_Fland-D_Eluc':'Sum of land sink of carbon\n& CO$_2$ emissions from LUC (PgC.yr$^{-1}$)','D_Epf_CO2':'CO$_2$ emissions from permafrost (PgC.yr$^{-1}$)','D_Tg_shift':'Change in global surface air temperature\nwith reference to 1850-1900 (K)','D_OHC_shift':'Ocean heat content\nwith refence to 1971 (ZJ)','D_Pg_shift':'Change in global mean precipitation\nwith reference to 1979-1989 (mm.yr$^{-1}$)', 'CO2':'Atmospheric CO$_2$ (ppm)',\
                   'RF_AER-rad':'Radiative forcing of\n aerosols-radiation effects (W.m$^{-2}$)',\
                   'RF_warm':'Effective Radiative forcing (W.m$^{-2}$)',\
                   'RF_nonCO2':'Radiative forcing of\nnon-CO2 WMGHG (W.m$^{-2}$)',\
                   'RF_CO2':'Radiative forcing of\nCO$_2$ (W.m$^{-2}$)',\
                   'RF_CH4':'Radiative forcing of\nCH$_4$ (W.m$^{-2}$)',\
                   'RF_N2O':'Radiative forcing of\nN$_2$O (W.m$^{-2}$)',\
                   'ERF_Montreal':'ERF$_{Montreal}$ (W.m$^{-2}$)',\
                   'RF_AERtot':'Radiative forcing of\naerosols (W.m$^{-2}$)',\
                   'RF_O3t':'Radiative forcing of\ntropospheric ozone (W.m$^{-2}$)',\
                   'RF_O3s':'Radiative forcing of\nstratospheric ozone (W.m$^{-2}$)'}
    alpha_plot = 0.3
    dico_obs = {'Eff':['carbon-cycle_GCB'],'D_CO2':['carbon-cycle_GCB'],'D_Eluc':['carbon-cycle_GCB'],'D_Fland-D_Eluc':['carbon-cycle_GCB'],'D_Focean':['carbon-cycle_GCB'],'D_Fland':['carbon-cycle_GCB'],'D_Epf_CO2':[],'RF_AER-rad':['radiative-forcing_AR5'],'RF_O3t':['radiative-forcing_AR5'],'RF_warm':['radiative-forcing_AR5'],'RF_AERtot':['radiative-forcing_AR5'],'D_Tg_shift':['temperature_Cowtan2014'],'RF_nonCO2':['radiative-forcing_AR5'],'D_Pg_shift' :['precipitation_GPCP'],'D_OHC_shift':['ocean-heat_NOAA-NODC']}#'temperature_BerkeleyEarth','temperature_GISTEMP','temperature_HadCRUT4','temperature_NOAA-GlobalTemp'
    ## first line
    VAR_plot = [ 'Eff', 'D_Focean', 'D_LandNet', 'D_CO2' ]
    dico_plot_ObsTt = {'Eff':'Compatible Fossil-fuel emissions of CO2 ', 'D_Focean':'Ocean sink ', 'D_LandNet':'Deduced net land flux '}
    for VAR in VAR_plot:
        ax = plt.subplot(3,len(VAR_plot),VAR_plot.index(VAR)+1)
        ax.set_axisbelow(True)
        for xp in list_xp:func_plot('year',VAR+'_'+xp,OUTPUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],edgecolor=None,ls=dico_ls[xp],label=xp,alpha=alpha_plot,zorder=100,lw=0.75*fac_size*3)
        if VAR in dico_plot_ObsTt.keys():
            list_tt = ['1960-1969', '1970-1979', '1980-1989', '1990-1999', '2000-2009']
            yl = ax.get_ylim()
            for tt in list_tt:
                if dico_plot_ObsTt[VAR] == 'Deduced net land flux ': ## error, did not include that one, not enough time to rerun with it, fast correction
                    vals_plot = {'1960-1969':[np.nan,-0.2-0.9,-0.2,-0.2+0.9,np.nan], '1970-1979':[np.nan,0.8-0.8,0.8,0.8+0.8,np.nan], '1980-1989':[np.nan,0.7-1.0,0.7,0.7+1.0,np.nan], '1990-1999':[np.nan,1.2-1.0,1.2,1.2+1.0,np.nan], '2000-2009':[np.nan,1.5-1.2,1.5,1.5+1.1,np.nan]}[tt]
                else:
                    i = rcmip.indicator.values.tolist().index( dico_plot_ObsTt[VAR] + tt )
                    vals_plot = np.array(  [float(rcmip['very_likely__lower'][i]), float(rcmip['likely__lower'][i]), float(rcmip['central'][i]), float(rcmip['likely__upper'][i]), float(rcmip['very_likely__upper'][i])]  )
                date_const = np.array(str.split(tt,'-'),dtype=np.float32)
                if list_tt.index(tt)==0:
                    plt.errorbar( x=(date_const[0]+date_const[1])/2.,y=vals_plot[2],xerr=(date_const[1]-date_const[0])/2.,yerr=np.array([(vals_plot[2]-vals_plot[1])/1.645,(vals_plot[3]-vals_plot[2])/1.645])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=1000, label = 'GCB 2020' )
                else:
                    plt.errorbar( x=(date_const[0]+date_const[1])/2.,y=vals_plot[2],xerr=(date_const[1]-date_const[0])/2.,yerr=np.array([(vals_plot[2]-vals_plot[1])/1.645,(vals_plot[3]-vals_plot[2])/1.645])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=1000 )
                plt.scatter( y=vals_plot[2] , x=(date_const[0]+date_const[1])/2. , facecolor='k',edgecolor='w',marker='o',s=fac_size*4.*20,zorder=101 )
        elif VAR=='D_CO2':
            i = rcmip.indicator.values.tolist().index( 'Increase Atmospheric Concentrations|CO2 World esm-hist-2011' )
            vals_plot = np.array(  [float(rcmip['very_likely__lower'][i]), float(rcmip['likely__lower'][i]), float(rcmip['central'][i]), float(rcmip['likely__upper'][i]), float(rcmip['very_likely__upper'][i])]  )
            date_const = 2011.
            plt.errorbar( x=date_const,y=vals_plot[2],yerr=np.array([(vals_plot[2]-vals_plot[1])/1.645,(vals_plot[3]-vals_plot[2])/1.645])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=1000, label = 'AR5 WG1 Ch3' )
            plt.scatter( y=vals_plot[2] , x=date_const , facecolor='k',edgecolor='w',marker='o',s=fac_size*4.*20,zorder=101 )
        plt.title( dico_titles[VAR] , size=fac_size*12 * 1.25 )
        plt.xlim(period[0],period[1])
        ax.tick_params(labelsize=fac_size*13)
        plt.xlabel( 'Year' , size=fac_size*14 * 1.25 )
        plt.grid()
        plt.legend(loc='upper left',prop={'size':fac_size*13 * 1.25})
        box = ax.get_position()
        ax.set_position([box.x0-0.03, box.y0+0.07, box.width*1.0, box.height*1.05])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
        counter += 1

    ##----------------------------
    ## second line
    dico_plot_to_obs = {'RF_CO2':'Effective Radiative Forcing|Anthropogenic|CO2 World ',\
                        'RF_CH4':'Effective Radiative Forcing|Anthropogenic|CH4 World ',\
                        'RF_N2O':'Effective Radiative Forcing|Anthropogenic|N2O World ',\
                        'ERF_Montreal':'Effective Radiative Forcing|Anthropogenic|Montreal Gases World ',\
                        'RF_AERtot':'Effective Radiative Forcing|Anthropogenic|Aerosols World ',\
                        'RF_O3t':'Radiative Forcing|Anthropogenic|Tropospheric Ozone World ',\
                        'RF_O3s':'Radiative Forcing|Anthropogenic|Stratospheric Ozone World '
                        }
    list_plot = ['RF_CO2','RF_CH4','RF_N2O','RF_AERtot','RF_O3t','RF_O3s']
    for VAR in list_plot:
        ax = plt.subplot(3,len(list_plot),len(list_plot)+list_plot.index(VAR)+1)
        ax.set_axisbelow(True)
        ## OSCAR
        for xp in list_xp: func_plot('year',VAR+'_'+xp,OUTPUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],edgecolor=None,ls=dico_ls[xp],label=xp,alpha=alpha_plot,zorder=100,lw=0.75*fac_size*3)
        ## observations
        i = rcmip.indicator.values.tolist().index( dico_plot_to_obs[VAR] + 'historical'+'-1750' )
        vals_plot = np.array(  [float(rcmip['very_likely__lower'][i]), float(rcmip['likely__lower'][i]), float(rcmip['central'][i]), float(rcmip['likely__upper'][i]), float(rcmip['very_likely__upper'][i])]  )
        date_const = 2011.
        plt.errorbar( x=date_const,y=vals_plot[2],yerr=np.array([(vals_plot[2]-vals_plot[1])/1.645,(vals_plot[3]-vals_plot[2])/1.645])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=1000, label = 'AR5 WG1 Ch8' )
        plt.scatter( y=vals_plot[2] , x=date_const , facecolor='k',edgecolor='w',marker='o',s=fac_size*4.*20,zorder=101 )
        ## polishing
        plt.title( dico_titles[VAR] , size=fac_size*12 * 1.25 )
        plt.xlim(period[0],period[1])
        ax.tick_params(labelsize=fac_size*13)
        plt.yticks( rotation=45, size=fac_size*12 )
        plt.xlabel( 'Year' , size=fac_size*14 * 1.25 )
        plt.grid()
        plt.legend(loc=0,prop={'size':fac_size*13 * 1.25})
        box = ax.get_position()
        ax.set_position([box.x0-0.03, box.y0+0.02, box.width*1.0, box.height*1.05])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
        counter += 1

    ##----------------------------
    ## third line
    VAR_plot = [ 'RF_warm', 'D_OHC_shift', 'D_Tg_shift']
    for VAR in VAR_plot:
        print(VAR)
        ax = plt.subplot(3,len(VAR_plot),2*len(VAR_plot)+VAR_plot.index(VAR)+1)
        ax.set_axisbelow(True)
        for xp in list_xp:func_plot('year',VAR+'_'+xp,OUTPUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],edgecolor=None,ls=dico_ls[xp],label=xp,alpha=alpha_plot,zorder=100,lw=0.75*fac_size*3)
        if VAR == 'RF_warm':
            date_const = np.array([2011.,2011.])
            vals_plot = np.array([np.nan,1.1,2.3,3.3,np.nan]) ## transformation from 90% range to +/-1 std dev
            plt.errorbar( x=(date_const[0]+date_const[1])/2.,y=vals_plot[2],xerr=(date_const[1]-date_const[0])/2.,yerr=np.array([(vals_plot[2]-vals_plot[1])/1.645,(vals_plot[3]-vals_plot[2])/1.645])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=1000, label = 'AR5 WG1 Ch8' )
            plt.scatter( y=vals_plot[2] , x=(date_const[0]+date_const[1])/2. , facecolor='k',edgecolor='w',marker='o',s=fac_size*4.*20,zorder=101 )
        elif VAR == 'D_OHC_shift':
            i = rcmip.indicator.values.tolist().index( 'Heat Content|Ocean World ssp245 1971-2018' )
            vals_plot = np.array(  [float(rcmip['very_likely__lower'][i]), float(rcmip['likely__lower'][i]), float(rcmip['central'][i]), float(rcmip['likely__upper'][i]), float(rcmip['very_likely__upper'][i])]  )
            date_const = np.array(  [float(rcmip['evaluation_period_start'][i]), float(rcmip['evaluation_period_end'][i])]  )
            plt.errorbar( x=(date_const[0]+date_const[1])/2.,y=vals_plot[2],xerr=(date_const[1]-date_const[0])/2.,yerr=np.array([(vals_plot[2]-vals_plot[1])/1.645,(vals_plot[3]-vals_plot[2])/1.645])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=1000, label = 'Von Schuckmann et al. 2020' )
            plt.scatter( y=vals_plot[2] , x=(date_const[0]+date_const[1])/2. , facecolor='k',edgecolor='w',marker='o',s=fac_size*4.*20,zorder=101 )
            print('could add full Schuckmann')
        elif VAR == 'D_Tg_shift':
            #obs = xr.open_dataset('H:/MyDocuments/Repositories/OSCARv31_CMIP6/input_data/observations/'+ 'temperature_Cowtan2014' +'.nc' )
            obs = xr.open_dataset('observations/'+ 'temperature_Cowtan2014' +'.nc' )
            to_plot = obs['D_Tg'] - obs['D_Tg'].sel(year=np.arange(1850,1900+1)).mean('year')
            plt.plot( obs.year, to_plot ,color='gray',ls='-',lw=0.75*fac_size*2. * 1.25,label='Cowtan et al, 2014' )
            list_tt = [ [1986.,2005.] ]#, [2003.,2012.] ]
            vals_tt = [ [np.nan,0.61 - (0.61-0.55)/1.645,0.61,0.61 + (0.67-0.61)/1.645,np.nan] ]#, [np.nan,0.78-(0.78-0.72)/1.96,0.78,0.78+(0.85-0.78)/1.96,np.nan] ] ## transformed from 90% range to 
            for date_const in list_tt:
                vals_plot = np.array(  vals_tt[list_tt.index(date_const)]  )
                if list_tt.index(date_const)==0:
                    plt.errorbar( x=(date_const[0]+date_const[1])/2.,y=vals_plot[2],xerr=(date_const[1]-date_const[0])/2.,yerr=np.array([vals_plot[2]-vals_plot[1],vals_plot[3]-vals_plot[2]])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=100, label = 'AR5 WG1 Ch2' )
                else:
                    plt.errorbar( x=(date_const[0]+date_const[1])/2.,y=vals_plot[2],xerr=(date_const[1]-date_const[0])/2.,yerr=np.array([vals_plot[2]-vals_plot[1],vals_plot[3]-vals_plot[2]])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=100 )
                plt.errorbar( x=(date_const[0]+date_const[1])/2.,y=vals_plot[2],xerr=(date_const[1]-date_const[0])/2.,yerr=np.array([vals_plot[2]-vals_plot[0],vals_plot[4]-vals_plot[2]])[:,np.newaxis] ,ecolor='black',elinewidth=fac_size*3 * 1.25 / 2.,capthick=fac_size*2 * 1.25,capsize=fac_size*2 * 1.25 / 2.,color='white',lw=0.75*fac_size*0,zorder=100 )
                plt.scatter( y=vals_plot[2] , x=(date_const[0]+date_const[1])/2. , facecolor='k',edgecolor='w',marker='o',s=fac_size*4.*20,zorder=101 )
        plt.title( dico_titles[VAR] , size=fac_size*12 * 1.25 )
        plt.xlim(period[0],2020)#period[1])
        ax.tick_params(labelsize=fac_size*13)
        plt.xlabel( 'Year' , size=fac_size*14 * 1.25 )
        plt.grid()
        plt.legend(loc='upper left',prop={'size':fac_size*13 * 1.25})
        box = ax.get_position()
        ax.set_position([box.x0-0.03, box.y0-0.03, box.width*1.0, box.height*1.1])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
        counter += 1

    print('done')
    ##----------------------------
    
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/'+'hist.pdf',dpi=900 )
    plt.close(fig)
#########################
#########################













#########################
## 7.12. Variant lowNTCF
#########################
if '7.12' in option_which_plots:
    # list_VAR = ['RF_CH4', 'RF_N2O', 'RF_halo','RF_nonCO2']+['RF_H2Os', 'RF_O3s','RF_strat']+['RF_SO4', 'RF_POA', 'RF_NO3', 'RF_SOA', 'RF_dust', 'RF_salt','RF_scatter']+['RF_BC','RF_absorb']+['RF_cloud','RF_AERtot']+['RF_O3t','RF_slcf']+['RF_BCsnow', 'RF_lcc','RF_alb'] + ['RF','D_Tg','D_Pg']
    list_VAR = ['RF_SO4', 'RF_POA', 'RF_NO3', 'RF_SOA', 'RF_dust', 'RF_salt','RF_scatter']+['RF_BC','RF_absorb']+['RF_cloud','RF_AERtot']+['RF_O3t','RF_slcf'] + ['RF','D_Tg','D_Pg']
    VAR_accelerate = ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp']

    ## preparing data for figures
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(1850,2500+1)
        for xp in ['ssp370', 'ssp370-lowNTCF','ssp370ext', 'ssp370-lowNTCFext' , 'historical']:
            name_xp = {'ssp370':'ssp370', 'ssp370-lowNTCF':'ssp370-lowNTCF' , 'ssp370ext':'ssp370', 'ssp370-lowNTCFext':'ssp370-lowNTCF' , 'historical':'historical'}[xp]
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , VAR_accelerate+list_VAR )
                for var in list_VAR:
                    if var+'_'+name_xp not in TMP:
                        TMP[var+'_'+name_xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                    TMP[var+'_'+name_xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = out_tmp[var]
                out_tmp.close()
                for_tmp.close()
        for xp in ['ssp370', 'ssp370-lowNTCF']:
            TMP['RF_warming'+'_'+xp] = TMP['RF_absorb'+'_'+xp] + TMP['RF_O3t'+'_'+xp]
            TMP['RF_cooling'+'_'+xp] = TMP['RF_scatter'+'_'+xp] + TMP['RF_cloud'+'_'+xp]
            TMP['D_Tg_shift'+'_'+xp] = TMP['D_Tg'+'_'+xp] - TMP['D_Tg'+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantlowNTCF.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantlowNTCF.nc')

    # statistical values
    OUT = xr.Dataset()
    OUT.coords['stat_value'] = ['mean','std_dev']
    OUT.coords['year'] = np.arange(1850,2500+1)
    for xp in ['ssp370', 'ssp370-lowNTCF']:
        for VAR in ['RF_warming','RF_cooling','D_Tg_shift']+list_VAR:
            OUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    ## difference
    for VAR in ['RF_warming','RF_cooling','D_Tg_shift']+list_VAR:
        OUT[VAR+'_'+'diff'] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
        val = np.ma.array( TMP[VAR+'_'+'ssp370-lowNTCF'] - TMP[VAR+'_'+'ssp370'] ,mask=np.isnan(TMP[VAR+'_'+'ssp370-lowNTCF'] - TMP[VAR+'_'+'ssp370']))
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        OUT[VAR+'_'+'diff'].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT[VAR+'_'+'diff'].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+'diff'].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data


    ## figure
    option_light = False
    if option_light:
        VAR_plot = ['RF_warming']+['RF_cooling']+['RF']+['D_Tg_shift']+['D_Pg']
        diff_txt = 0
        xshift0,xshift = -0.03,0.0125
    else:
        VAR_plot = ['RF_scatter']+['RF_absorb']+['RF_cloud']+['RF_O3t']+['RF']+['D_Tg_shift','D_Pg']
        diff_txt = 2
        xshift0,xshift = -0.04,0.010
    ## unaffected: ['RF_CH4', 'RF_N2O', 'RF_halo','RF_nonCO2','RF_H2Os', 'RF_O3s','RF_strat', 'RF_dust', 'RF_salt', 'RF_lcc']
    ## scatter only: sum of POA, SO4 ['RF_SO4', 'RF_POA',]
    ## quite small: [, 'RF_NO3', 'RF_SOA']
    ## very small: ['RF_BCsnow','RF_alb']
    ## similar: 'RF_BC'  =  'RFabsorb'
    ## 'RF_AERtot' as sum of three others
    ## 'RF_slcf': from RF_AERtot, RF_start and RF_O3t
    period = [2015,2300]
    dico_col = {'ssp370':(224/255.,0/255.,0/255.), 'ssp370-lowNTCF':CB_color_cycle[3]}#####(224/255.,0/255.,0/255.)
    dico_ls = {'ssp370':'-', 'ssp370-lowNTCF':'--'}
    dico_title_VAR = {'RF_warming':'Warming contribution to RF:\nBC and O$_3^t$ (W.m$^{-2}$)' , 'RF_cooling':'Cooling contribution to RF:\nPOA, SOA, NO$_3$, SO$_4$ and clouds (W.m$^{-2}$)','RF':'Total RF (W.m$^{-2}$)','D_Pg':'Change in global\nprecipitation (mm.yr$^{-1}$)','D_Tg':'Change in global\nmean surface\ntemperature (K)' ,'D_Tg_shift':'Change in global mean\nsurface temperature\nwith reference to 1850-1900 (K)' , 'RF_scatter':'Scattering aerosols:\nPOA, SOA, NO$_3$, SO$_4$ (W.m$^{-2}$)','RF_absorb':'BC (W.m$^{-2}$)','RF_cloud':'Clouds (W.m$^{-2}$)','RF_O3t':'O$_3^t$ (W.m$^{-2}$)'}
    ## scattering: RF_SO4 + Var.RF_POA + Var.RF_NO3 + Var.RF_SOA
    # dico_var_ylim= {'D_Fland':[-2.5,6.5],'D_Eluc':[-0.2,2.5],'LASC':[-0.75,0.3]}
    ## looping on lines for groups of experiments
    fig = plt.figure( figsize=(30,20) )
    counter = 0
    ## looping on variables / subplots
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for xp in ['ssp370', 'ssp370-lowNTCF']:
            func_plot('year',var+'_'+xp,OUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*0.9*3,label=xp)
            date_print = 2100
            mm,ss = OUT[var+'_'+xp].sel(year=date_print,stat_value='mean'), OUT[var+'_'+xp].sel(year=date_print,stat_value='std_dev')
            print('OSCAR: '+var+' on '+xp+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[1])
        # if VAR_plot.index(var) == 0:
        plt.title( dico_title_VAR[var] , size=fac_size*0.9*(13-diff_txt),rotation=0  )#,fontweight='bold'
        if var in ['RF','D_Tg','D_Tg_shift','D_Pg']:
            pass
        elif var in ['RF_warming','RF_cooling']:
            plt.ylim(-2-.75,1.75)
        else:
            plt.ylim(-1.5,1.)
        # plt.text(x=period[0]+0.05*(period[1]-period[0]),y=ax.get_ylim()[1]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=fac_size*0.9*'('+  ['a','b','c','d','e','f','g','h','i'][VAR_plot.index(var)] +')',fontdict={'size':fac_size*0.9*13})
        if var == VAR_plot[0]:
            plt.legend(loc=0,prop={'size':fac_size*0.9*(12-diff_txt)} )#,bbox_to_anchor=(-0.15,0.65))
            plt.ylabel( 'Scenarios' , size=fac_size*0.9*(15-diff_txt)  )#,rotation=0,fontweight='bold'
            ax.yaxis.set_label_coords(-0.40,0.5)
        ax.tick_params(labelsize=fac_size*0.9*(13-diff_txt))
        box = ax.get_position()
        ax.set_position([box.x0+xshift0+xshift*VAR_plot.index(var), box.y0+0.02, box.width*1.0, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*0.9*14})
        counter += 1

    ## plotting differences
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),len(VAR_plot)+VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        func_plot('year',var+'_'+'diff',OUT.sel(year=np.arange(period[0],period[1]+1)),col='k',ls='-',lw=0.75*fac_size*0.9*3,label='ssp370-lowNTCF  -  ssp370')
        date_print = 2100
        mm,ss = OUT[var+'_'+'diff'].sel(year=date_print,stat_value='mean'), OUT[var+'_'+'diff'].sel(year=date_print,stat_value='std_dev')
        print('OSCAR: '+var+' on '+'diff'+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[1])
        if var in ['RF','D_Tg','D_Tg_shift','D_Pg']:
            pass
        elif var in ['RF_warming','RF_cooling']:
            plt.ylim(-0.8,0.8)
        else:
            plt.ylim(-0.5,0.5)
        # plt.text(x=period[0]+0.05*(period[1]-period[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=fac_size*0.9*'('+  ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p'][len(VAR_plot)+VAR_plot.index(var)] +')',fontdict={'size':fac_size*0.9*13})
        if var == VAR_plot[0]:
            # plt.legend(loc=0,prop={'size':fac_size*0.9*(10-diff_txt)})#,bbox_to_anchor=(-0.15,0.65))
            plt.ylabel( 'Differences' , size=fac_size*0.9*(15-diff_txt)  )#,rotation=0,fontweight='bold'
            ax.yaxis.set_label_coords(-0.40,0.5)
        plt.xlabel( 'Year' , size=fac_size*0.9*(15-diff_txt) )
        ax.tick_params(labelsize=fac_size*0.9*(13-diff_txt))
        box = ax.get_position()
        ax.set_position([box.x0+xshift0+xshift*VAR_plot.index(var), box.y0+0.02, box.width*1.0, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*0.9*14})
        counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/variant-lowNTCF.pdf',dpi=300 )
    plt.close(fig)
#########################
#########################




#########################
## 7.13. Number of runs
#########################
if '7.13' in option_which_plots:## former version
    list_xp = ['historical']+['ssp585', 'ssp370-lowNTCF', 'ssp370',  'ssp460', 'ssp245', 'ssp534-over', 'ssp434', 'ssp126', 'ssp119'] + ['abrupt-4xCO2']

    ## preparing data for figures
    if True:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(1850,2500+1)
        for xp in list_xp:
            print(xp)
            if xp not in TMP:
                TMP[xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
            for setMC in list_setMC:
                for ext in ['','ext','-ext','Ext','-Ext']:
                    if os.path.isfile(path_all+'/treated/masks/masknoDV_'+xp+ext+'_'+str(setMC)+'.csv'):
                        with open(path_all+'/treated/masks/masknoDV_'+xp+ext+'_'+str(setMC)+'.csv','r',newline='') as ff:
                            mask = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)
                        out_tmp = xr.open_dataset(path_all+'/'+xp+ext+'_Out-'+str(setMC)+'.nc')
                        years = max( [out_tmp.year.values[0], TMP.year.values[0]] ), min( [out_tmp.year.values[-1], TMP.year.values[-1]] )
                        y0 = out_tmp.year.values[0]
                        TMP[xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':slice(years[0], years[1]) }] = mask[int(years[0]-y0):int(years[1]-y0)+1,:]
                        out_tmp.close()
                        del mask

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_nbruns.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_nbruns.nc' )

    # for graphical reasons / explanations, corrections on historical:
    TMP['historical'].loc[{'year':slice(1850,2014)}] = 1.
        
    ## figure
    period = np.arange(1850,2500+1)
    dico_col = {'historical':'k', 'ssp119':(0/255.,170/255.,208/255.), 'ssp126':(0/255.,52/255.,102/255.), 'ssp245':(239/255.,85/255.,15/255.), 'ssp370':(224/255.,0/255.,0/255.), 'ssp370-lowNTCF':(224/255.,0/255.,0/255.), 'ssp434':(255/255.,169/255.,0/255.), 'ssp460':(196/255.,121/255.,0/255.), 'ssp534-over':(127/255.,0/255.,110/255.), 'ssp585':(153/255.,0/255.,2/255.), 'ssp585':(153/255.,0/255.,2/255.), '1pctCO2':CB_color_cycle[2], 'abrupt-4xCO2':CB_color_cycle[7]}
    dico_ls = {'historical':'-', 'ssp119':'-', 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp370-lowNTCF':'', 'ssp434':'-', 'ssp460':'-', 'ssp534-over':'-', 'ssp585':'-', '1pctCO2':'-', 'abrupt-4xCO2':'-'}
    dico_marker = {'historical':None, 'ssp119':None, 'ssp126':None, 'ssp245':None, 'ssp370':None, 'ssp370-lowNTCF':'x', 'ssp434':None, 'ssp460':None, 'ssp534-over':None, 'ssp585':None, '1pctCO2':None, 'abrupt-4xCO2':None}
    dico_period = {'historical':[1850,2014], 'ssp119':[2014,2500], 'ssp126':[2014,2500], 'ssp245':[2014,2500], 'ssp370':[2014,2500], 'ssp370-lowNTCF':[2014,2500], 'ssp434':[2014,2500], 'ssp460':[2014,2500], 'ssp534-over':[2014,2500], 'ssp585':[2014,2500], '1pctCO2':[1850,2100], 'abrupt-4xCO2':[1850,2500]}
    fig = plt.figure( figsize=(30,20) )
    counter = 0
    ax = plt.subplot(111)
    ## looping on experiments inside a subplot
    for xp in ['historical']+list_xp[1:][::-1]:#list_xp:
        plt.plot( np.arange(dico_period[xp][0],dico_period[xp][1]+1) , 100.*(TMP[xp].sum('all_config') / TMP['all_config'].size).sel(year=np.arange(dico_period[xp][0],dico_period[xp][1]+1)), color=dico_col[xp], ls=dico_ls[xp], marker=dico_marker[xp], markevery=5, ms=10, lw=0.75*fac_size*3, label=xp)
    ## polishing
    plt.grid()
    plt.xlim(period[0],period[-1])
    plt.ylabel( 'Fraction of non-diverging runs (%)' , size=fac_size*14,fontweight='bold'  )
    plt.xlabel( 'Years' , size=fac_size*14,fontweight='bold'  )
    plt.legend(prop={'size':fac_size*14} )#, bbox_to_anchor=(1.6,-0.18))
    ax.tick_params(labelsize=fac_size*14)
    box = ax.get_position()
    # ax.set_position([box.x0-0.06, box.y0+0.05-(VAR_plot.index(var)//3)*0.045, box.width*1.0, box.height*1.1])
    #plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
    #counter += 1
    fig.savefig(path_all+'/treated/OSCAR-CMIP6/plots/pct_exclusions_all.pdf',dpi=300)

    
    
    
if False:
    implicit_mask_weights = lambda setMC: (weights_CMIP6['weights'].sel(all_config=[str(setMC)+'-'+str(cfg) for cfg in range(0,500)]).where( weights_CMIP6.weights!=0. ) / weights_CMIP6['weights'].sel(all_config=[str(setMC)+'-'+str(cfg) for cfg in range(0,500)]).where( weights_CMIP6.weights!=0. )).values

    def func_plot_masks(  list_xp, year_end, label , opt_lw=2, opt_markevery=5, opt_ms=5,force_nomark=False ):
        print('If not 10k members, need to update function')
        if label==None:
            label = list_xp
        for xp in list_xp:
            if type(xp)==list:
                if force_nomark:
                    mark = ''
                else:
                    mark = dico_marker_scenarios[xp[0]]

                period = np.arange( func_start_scen(xp[0]) , np.min([year_end+1,func_start_scen(xp[0])+int(list_noDV[xp[0]][0].shape[0])]) )
                plt.plot( period , np.nansum( [list_noDV[xp[0]][setMC]*implicit_mask_weights(setMC) for setMC in list_setMC] , axis=(0,2))[:len(period)] / (0.01*20*500), marker=mark, color=dico_color_scenarios[xp[0]], mec=dico_color_scenarios[xp[0]], ms=fac_size*opt_ms, markevery=opt_markevery, lw=0.75*fac_size*opt_lw, ls=dico_ls_scenarios[xp[0]], label=label[list_xp.index(xp)] )
                period = np.arange( func_start_scen(xp[1]) , np.min([year_end+1,func_start_scen(xp[1])+int(list_noDV[xp[1]][0].shape[0])]) )
                plt.plot( period , np.nansum( [list_noDV[xp[1]][setMC]*implicit_mask_weights(setMC) for setMC in list_setMC] , axis=(0,2))[:len(period)] / (0.01*20*500), marker=dico_marker_scenarios[xp[0]], color=dico_color_scenarios[xp[0]], mec=dico_color_scenarios[xp[0]], ms=fac_size*opt_ms, markevery=opt_markevery, lw=0.75*fac_size*opt_lw, ls=dico_ls_scenarios[xp[0]] )

            else:
                if force_nomark:
                    mark = ''
                else:
                    mark = dico_marker_scenarios[xp]
                period = np.arange( func_start_scen(xp) , np.min([year_end+1,func_start_scen(xp)+int(list_noDV[xp][0].shape[0])]) )
                if xp in ['1pctCO2','1pctCO2-rad','1pctCO2-bgc','esm-1pctCO2']:
                    plt.plot( period[:-100] , np.nansum( [list_noDV[xp][setMC]*implicit_mask_weights(setMC) for setMC in list_setMC] , axis=(0,2))[:len(period)-100] / (0.01*20*500), marker=mark, color=dico_color_scenarios[xp], mec=dico_color_scenarios[xp], ms=fac_size*opt_ms, markevery=opt_markevery, lw=0.75*fac_size*opt_lw, ls=dico_ls_scenarios[xp] , label=label[list_xp.index(xp)] )
                else:
                    plt.plot( period , np.nansum( [list_noDV[xp][setMC]*implicit_mask_weights(setMC) for setMC in list_setMC] , axis=(0,2))[:len(period)] / (0.01*20*500), marker=mark, color=dico_color_scenarios[xp], mec=dico_color_scenarios[xp], ms=fac_size*opt_ms, markevery=opt_markevery, lw=0.75*fac_size*opt_lw, ls=dico_ls_scenarios[xp] , label=label[list_xp.index(xp)] )
        plt.grid()

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



    if False:## checks
        plt.plot( np.arange(1850,2014+1), np.nansum( np.array([list_noDV['historical'][setMC] for setMC in list_setMC]) , axis=(0,2)) )
        plt.plot( np.arange(2014,2100+1), np.nansum( np.array([list_noDV['ssp585'][setMC] for setMC in list_setMC]) , axis=(0,2)) )
        plt.plot( np.arange(2100,2500+1), np.nansum( np.array([list_noDV['ssp585ext'][setMC] for setMC in list_setMC]) , axis=(0,2)) )



        plt.plot( np.arange(1850,2014+1), np.arange(500)*list_noDV['historical'][setMC] )
        plt.plot( np.arange(2014,2100+1), np.arange(500)*list_noDV['ssp585'][setMC] )
        plt.grid()





if False:## NOT plot paper 
    if False:
        for name_experiment in list_xp:
            print('Loading masks for '+name_experiment)
            list_noDV[name_experiment] = {}
            for setMC in list_setMC:
                with open(path_all+'/treated/masks/masknoDV_'+name_experiment+'_'+str(setMC)+'.csv','r',newline='') as ff:
                    list_noDV[name_experiment][setMC] = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)

    with open('dico_plots_MIPs.csv','r',newline='') as ff:
        tmp = np.array([line for line in csv.reader(ff)])[1:,:]
        dico_color_scenarios = {}
        for line in tmp:
            if line[1]!='':
                dico_color_scenarios[line[0]] = tuple([eval(line[1]),eval(line[2]),eval(line[3])])
        dico_marker_scenarios = {line[0]:line[4] for line in tmp}
        dico_ls_scenarios = {line[0]:line[5] for line in tmp}
        del tmp


    list_noDV = {}
    for name_experiment in  ['1pctCO2-cdr','abrupt-2xCO2'] + \
                            [ 'historical', 'ssp534-over', 'ssp534-over-ext', 'rcp60', 'ssp460', 'ssp460ext', 'ssp370', 'ssp370ext', 'rcp85', 'ssp585', 'ssp585ext' ] + \
                            ['esm-pi-CO2pulse','esm-1pct-brch-1000PgC','esm-bell-1000PgC','esm-abrupt-4xCO2'] + \
                            [ 'esm-hist', 'esm-ssp534-over', 'esm-ssp534-over-ext', 'esm-ssp245', 'esm-ssp245ext', 'esm-rcp60', 'esm-ssp460', 'esm-ssp460ext', 'esm-ssp370', 'esm-ssp370ext', 'esm-rcp85',  'esm-ssp585', 'esm-ssp585ext' ]:
        print('Loading masks for '+name_experiment)
        list_noDV[name_experiment] = {}
        for setMC in list_setMC:
            with open(path_all+'/treated/masks/masknoDV_'+name_experiment+'_'+str(setMC)+'.csv','r',newline='') as ff:
                list_noDV[name_experiment][setMC] = np.array([line for line in csv.reader(ff)] ,dtype=np.float32)



    fig = plt.figure( figsize=(20,10) )
    counter = 0
    ax = plt.subplot(2,2,1)
    # list_xp = [ '1pctCO2', '1pctCO2-4xext', '1pctCO2-bgc', '1pctCO2-cdr', '1pctCO2-rad', 'G2', 'G1', 'abrupt-0p5xCO2', 'abrupt-2xCO2', 'abrupt-4xCO2']
    list_xp = ['1pctCO2-cdr','abrupt-2xCO2']
    label = ['Family 1pctCO2-*','Family abrupt-*']
    func_plot_masks( list_xp, label=label, year_end=2200, opt_lw=2, opt_markevery=10, opt_ms=8 , force_nomark=True)
    plt.xlim(1850,2000)
    plt.ylabel('Fraction of retained configurations (%)',size=fac_size*14)
    ax.tick_params(labelsize=fac_size*13)
    box = ax.get_position()
    # ax.set_position([box.x0-0.04, box.y0+0.02, box.width*0.8, box.height*1.2])
    # ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='center left',bbox_to_anchor=(1.01,0.5))
    ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='lower left')
    plt.ylim(0,105)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
    counter += 1

    ax = plt.subplot(2,2,2)
    # list_xp = [ 'historical', 'historical-CMIP5','rcp26', 'rcp45', 'rcp60', 'rcp85', ['ssp119', 'ssp119ext'], ['ssp126', 'ssp126ext'], 'ssp126-ssp370Lu', ['ssp245', 'ssp245ext'], ['ssp370', 'ssp370ext'], ['ssp370-lowNTCF', 'ssp370-lowNTCFext'], 'ssp370-ssp126Lu', ['ssp434', 'ssp434ext'], ['ssp460', 'ssp460ext'], ['ssp534-over', 'ssp534-over-ext'], ['ssp534-over-bgc', 'ssp534-over-bgcExt'], ['ssp585', 'ssp585ext'], ['ssp585-bgc', 'ssp585-bgcExt'], 'ssp585-ssp126Lu', 'G6solar' ]
    # label = list_xp
    # 'hist-1950HC', 'hist-CO2', 'hist-GHG', 'hist-aer', 'hist-nat', 'hist-noLu', 'hist-piAer', 'hist-piNTCF', 'hist-sol', 'hist-stratO3', 'hist-volc', 'hist-bgc'
    # 'ssp245-CO2', 'ssp245-GHG', 'ssp245-aer', 'ssp245-nat', 'ssp245-sol', 'ssp245-stratO3', 'ssp245-volc'
    # 'yr2010CO2'
    # list_xp = [ 'historical', 'rcp26', 'rcp45', 'rcp60', 'rcp85',    ['ssp119', 'ssp119ext'], ['ssp126', 'ssp126ext'], ['ssp245', 'ssp245ext'], ['ssp370', 'ssp370ext'], ['ssp434', 'ssp434ext'], ['ssp460', 'ssp460ext'], ['ssp534-over', 'ssp534-over-ext'], ['ssp585', 'ssp585ext'] ]
    # label = ['Family historical', 'rcp26', 'rcp45', 'rcp60', 'rcp85','ssp119','Family ssp126','Family ssp245','Family ssp370','ssp434','ssp460','Family ssp534','Family ssp585']
    list_xp = [ 'historical', ['ssp534-over', 'ssp534-over-ext'], 'rcp60', ['ssp460', 'ssp460ext'], ['ssp370', 'ssp370ext'], 'rcp85', ['ssp585', 'ssp585ext'] ]
    label = ['historical','{ssp119,\nrcp26, Family ssp126,\nrcp45, Family ssp245-*,\nssp434, Family ssp534-*}', 'rcp60', 'ssp460', 'Family ssp370-*', 'rcp85', 'Family ssp585-*']
    func_plot_masks( list_xp, label=label, year_end=2500, opt_lw=2, opt_markevery=10, opt_ms=8 )
    plt.xlim(1850,2200)
    ax.tick_params(labelsize=fac_size*13)
    box = ax.get_position()
    # ax.set_position([box.x0-0.01, box.y0+0.02, box.width*0.8, box.height*1.2])
    # ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='center left',bbox_to_anchor=(1.03,0.5))
    ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='lower left')
    plt.ylim(0,105)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
    counter += 1

    ax = plt.subplot(2,2,3)
    # list_xp = [ 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-1pct-brch-750PgC', 'esm-1pctCO2',    'esm-abrupt-4xCO2', 'esm-pi-CO2pulse', 'esm-pi-cdr-pulse',    'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC' ]
    list_xp = ['esm-pi-CO2pulse','esm-1pct-brch-1000PgC','esm-bell-1000PgC','esm-abrupt-4xCO2']
    label = ['Family esm-pi-* pulses','Family esm-1pct-*','Family esm-bell-*','esm-abrupt-4xCO2']
    func_plot_masks( list_xp, label=label, year_end=2000, opt_lw=2, opt_markevery=10, opt_ms=8 , force_nomark=True)
    plt.xlabel('Time (yr)',size=fac_size*14)
    plt.ylabel('Fraction of retained configurations (%)',size=fac_size*14)
    plt.xlim(1850,2000)
    ax.tick_params(labelsize=fac_size*13)
    box = ax.get_position()
    # ax.set_position([box.x0-0.04, box.y0-0.02, box.width*0.8, box.height*1.2])
    # ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='center left',bbox_to_anchor=(1.01,0.5))
    ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='lower left')
    plt.ylim(0,105)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
    counter += 1

    ax = plt.subplot(2,2,4)
    # list_xp = [ 'esm-hist', 'esm-histcmip5', 'esm-rcp26', 'esm-rcp45', 'esm-rcp60', 'esm-rcp85', ['esm-ssp119', 'esm-ssp119ext'], ['esm-ssp126', 'esm-ssp126ext'], ['esm-ssp245', 'esm-ssp245ext'], ['esm-ssp370', 'esm-ssp370ext'], ['esm-ssp370-lowNTCF', 'esm-ssp370-lowNTCFext'], ['esm-ssp434', 'esm-ssp434ext'], ['esm-ssp460', 'esm-ssp460ext'], ['esm-ssp534-over', 'esm-ssp534-over-ext'], ['esm-ssp585', 'esm-ssp585ext'], ['esm-ssp585-ssp126Lu', 'esm-ssp585-ssp126Lu-ext'] ]
    # 'esm-yr2010CO2-CO2pulse', 'esm-yr2010CO2-cdr-pulse', 'esm-yr2010CO2-control', 'esm-yr2010CO2-noemit'
    # list_xp = [ 'esm-hist', 'esm-rcp26', 'esm-rcp45', 'esm-rcp60', 'esm-rcp85', ['esm-ssp119', 'esm-ssp119ext'], ['esm-ssp126', 'esm-ssp126ext'], ['esm-ssp245', 'esm-ssp245ext'], ['esm-ssp370', 'esm-ssp370ext'], ['esm-ssp434', 'esm-ssp434ext'], ['esm-ssp460', 'esm-ssp460ext'], ['esm-ssp534-over', 'esm-ssp534-over-ext'], ['esm-ssp585', 'esm-ssp585ext'] ]
    # label = ['esm-hist', 'esm-rcp26', 'esm-rcp45', 'esm-rcp60', 'esm-rcp85','esm-ssp119','esm-ssp126','esm-ssp245','Family esm-ssp370','esm-ssp434','esm-ssp460','esm-ssp534','Family esm-ssp585']
    list_xp = [ 'esm-hist', ['esm-ssp534-over', 'esm-ssp534-over-ext'], ['esm-ssp245', 'esm-ssp245ext'], 'esm-rcp60', ['esm-ssp460', 'esm-ssp460ext'], ['esm-ssp370', 'esm-ssp370ext'], 'esm-rcp85',  ['esm-ssp585', 'esm-ssp585ext'] ]
    label = ['esm-hist', '{esm-ssp119,\nesm-ssp126, esm-rcp26,\nesm-ssp434, esm-ssp534,\nesm-rcp45}','esm-ssp245', 'esm-rcp60','esm-ssp460','Family esm-ssp370-*', 'esm-rcp85', 'Family esm-ssp585-*']
    func_plot_masks( list_xp, label=label, year_end=2500, opt_lw=2, opt_markevery=10, opt_ms=8 )
    plt.xlabel('Time (yr)',size=fac_size*14)
    plt.xlim(1850,2200)
    ax.tick_params(labelsize=fac_size*13)
    box = ax.get_position()
    # ax.set_position([box.x0-0.01, box.y0-0.02, box.width*0.8, box.height*1.2])
    # ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='center left',bbox_to_anchor=(1.03,0.5))
    ax.legend(prop={'size':fac_size*11.5},ncol=1,loc='lower left')
    plt.ylim(0,105)
    plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
    counter += 1

    fig.savefig(path_all+'/treated/OSCAR-CMIP6/plots/pct_exclusions_all.pdf',dpi=300)
    
#########################
#########################











#########################
## 7.14. Variant G6
#########################
if '7.14' in option_which_plots:
    list_VAR = ['RF_SO4', 'RF_POA', 'RF_NO3', 'RF_SOA', 'RF_dust', 'RF_salt','RF_scatter']+['RF_BC','RF_absorb']+['RF_cloud','RF_AERtot']+['RF_O3t','RF_slcf'] + ['RF','D_Tg','D_Pg'] + ['D_Fland','D_Focean','D_Eluc']
    VAR_accelerate = ['D_Aland','D_csoil1','D_csoil2','D_cveg','D_Csoil1_bk','D_Csoil2_bk','D_Cveg_bk','D_Chwp' , 'D_nbp']
    list_xp = ['ssp245', 'ssp585','G6solar']

    ## preparing data for figures
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        TMP.coords['year'] = np.arange(1850,2500+1)
        for xp in list_xp+['historical']:
            for setMC in list_setMC:
                print(xp+' on '+str(setMC))
                out_tmp,for_tmp,Par,mask = func_get( setMC , xp , VAR_accelerate+list_VAR )
                for var in list_VAR+['D_cLand','D_cOcean']:
                    if var+'_'+xp not in TMP:
                        TMP[var+'_'+xp] = xr.DataArray(  np.full(fill_value=np.nan,shape=(TMP.year.size,TMP.all_config.size)) , dims=('year','all_config')  )
                        if var in ['D_cLand']:
                            ## adding cumulative Land
                            TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = (out_tmp['D_Fland'] - out_tmp['D_Eluc']).cumsum('year') * mask
                        elif var in ['D_cOcean']:
                            ## adding cumulative ocean
                            TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = out_tmp['D_Focean'].cumsum('year') * mask
                        else:
                            TMP[var+'_'+xp].loc[{ 'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])] ,'year':out_tmp.year }] = out_tmp[var]
                out_tmp.close()
                for_tmp.close()
        for xp in list_xp:TMP['D_Tg_shift'+'_'+xp] = TMP['D_Tg'+'_'+xp] - TMP['D_Tg'+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')
        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantG6.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_variantG6.nc' )

    # statistical values
    OUT = xr.Dataset()
    OUT.coords['stat_value'] = ['mean','std_dev']
    OUT.coords['year'] = np.arange(1850,2500+1)
    for xp in list_xp:
        for VAR in list_VAR+['D_cLand','D_cOcean','D_Tg_shift']:
            OUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
    ## difference
    for VAR in list_VAR+['D_cLand','D_cOcean','D_Tg_shift']:
        for diff in ['G6solar - ssp585', 'G6solar - ssp245']:
            OUT[VAR+'_'+'diff'+diff] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array( TMP[VAR+'_'+str.split(diff,' - ')[0]] - TMP[VAR+'_'+str.split(diff,' - ')[1]] ,mask=np.isnan(TMP[VAR+'_'+str.split(diff,' - ')[0]] - TMP[VAR+'_'+str.split(diff,' - ')[1]]))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+'diff'+diff].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+'diff'+diff].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+'diff'+diff].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

    ## figure
    VAR_plot = ['D_Tg_shift','D_cLand','D_cOcean','RF_AERtot','D_Pg']# + ['RF_scatter','RF_absorb','RF_cloud']
    diff_txt = 0
    xshift0,xshift = -0.03,0.0125
    period = [2015,2100]
    dico_col = {'ssp245':(239/255.,85/255.,15/255.), 'ssp585':(153/255.,0/255.,2/255.) , 'G6solar':'k', 'G6solar - ssp585':CB_color_cycle[0], 'G6solar - ssp245':CB_color_cycle[2]}
    dico_ls = {'ssp245':'-', 'ssp585':'-','G6solar':'--' , 'G6solar - ssp585':'-', 'G6solar - ssp245':'-'}
    dico_title_VAR = {'D_cLand':'Change in land\ncarbon stock (PgC)','D_cOcean':'Change in oceanic\ncarbon stock (PgC)','RF_AERtot':'Radiative forcing\nof aerosols (W.m$^{-2}$)','rf_tot':'Radiative forcing (W.m$^{-2}$)', 'RF_solar':'Radiative forcing\nof solar activity (W.m$^{-2}$)' , 'RF_CO2':'Radiative forcing\nof CO$_2$ (W.m$^{-2}$)','D_Pg':'Change in global\nprecipitation (mm.yr$^{-1}$)','D_Tg':'Change in global mean\nsurface temperature (K)','D_Tg_shift':'Change in global mean\nsurface temperature\nwith reference to 1850-1900 (K)' , 'RF_scatter':'Radiative forcing\nof scattering aerosols (W.m$^{-2}$)','RF_absorb':'Radiative forcing\nof absorbing aerosols (W.m$^{-2}$)','RF_cloud':'Radiative forcing\nof aerosols-clouds (W.m$^{-2}$)'}
    ## looping on lines for groups of experiments
    fig = plt.figure( figsize=(30,20) )
    counter = 0
    ## looping on variables / subplots
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for xp in list_xp:
            func_plot('year',var+'_'+xp,OUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[xp],ls=dico_ls[xp],lw=0.8*fac_size*1.0*3,label=xp)
            date_print = 2100
            mm,ss = OUT[var+'_'+xp].sel(year=date_print,stat_value='mean'), OUT[var+'_'+xp].sel(year=date_print,stat_value='std_dev')
            print('OSCAR: '+var+' on '+xp+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[1])
        plt.title( dico_title_VAR[var] , size=fac_size*1.0*(13-diff_txt),rotation=0  )#,fontweight='bold'
        if var == VAR_plot[0]:
            plt.legend(loc=0,prop={'size':fac_size*1.0*(12-diff_txt)} )#,bbox_to_anchor=(-0.15,0.65))
            plt.ylabel( 'Scenarios' , size=fac_size*1.0*(15-diff_txt)  )#,rotation=0,fontweight='bold'
            # ax.yaxis.set_label_coords(-0.40,0.5)
        ax.tick_params(labelsize=fac_size*1.0*(13-diff_txt))
        box = ax.get_position()
        ax.set_position([box.x0+xshift0+xshift*VAR_plot.index(var), box.y0+0.02, box.width*1.0, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*1.0*14})
        counter += 1

    ## plotting differences
    for var in VAR_plot:
        ax = plt.subplot(2,len(VAR_plot),len(VAR_plot)+VAR_plot.index(var)+1)
        ## looping on experiments inside a subplot
        for diff in ['G6solar - ssp585', 'G6solar - ssp245']:
            func_plot('year',var+'_'+'diff'+diff,OUT.sel(year=np.arange(period[0],period[1]+1)),col=dico_col[diff],ls=dico_ls[diff],lw=0.8*fac_size*1.0*3,label=str.split(diff,' - ')[0]+'  -  '+str.split(diff,' - ')[1])
            date_print = 2100
            mm,ss = OUT[var+'_'+'diff'+diff].sel(year=date_print,stat_value='mean'), OUT[var+'_'+'diff'+diff].sel(year=date_print,stat_value='std_dev')
            print('OSCAR: '+var+'_'+'diff'+diff+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        ## polishing
        plt.grid()
        plt.xlim(period[0],period[1])
        if var == VAR_plot[0]:
            plt.legend(loc=0,prop={'size':fac_size*1.0*(10-diff_txt)})#,bbox_to_anchor=(-0.15,0.65))
            plt.ylabel( 'Differences' , size=fac_size*1.0*(15-diff_txt)  )#,rotation=0,fontweight='bold'
            # ax.yaxis.set_label_coords(-0.40,0.5)
        plt.xlabel( 'Year' , size=fac_size*1.0*(15-diff_txt) )
        ax.tick_params(labelsize=fac_size*1.0*(13-diff_txt))
        box = ax.get_position()
        ax.set_position([box.x0+xshift0+xshift*VAR_plot.index(var), box.y0+0.02, box.width*1.0, box.height*1.0])
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*1.0*14})
        counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/variant-G6.pdf',dpi=300 )
    plt.close(fig)
#########################
#########################







#########################
## 7.15. RCPs vs SSPs
#########################
if '7.15' in option_which_plots:
    ## Preparation
    list_xps = [ ['rcp26','ssp126'] ,  ['rcp45', 'ssp245'], ['rcp60', 'ssp460'], ['rcp85', 'ssp585'] ]
    list_var = ['co2','rf_tot','tas','nbp','fgco2']

    ## base data
    OUT = xr.Dataset()
    OUT['year'] = np.arange(1850,2500+1)
    for xps in [['historical','historical-CMIP5']]+list_xps:
        for xp in xps:
            ## variables in CMIP6
            OUT0 = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where(xp==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
            for var in list_var:
                OUT[var+'_'+xp] = OUT0[var].copy(deep=True)
            OUT0.close()
    ## adding control for all var
    OUT0 = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where( 'piControl' ==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
    OUT05 = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where( 'piControl-CMIP5' ==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
    for xps in [['historical','historical-CMIP5']]+list_xps:
        for xp in xps:
            for var in list_var:
                if xp in ['historical','ssp126','ssp245','ssp460','ssp585']:
                    if 'stat_value' in OUT[var+'_'+xp].dims:
                        OUT[var+'_'+xp].loc[{'stat_value':'mean'}] += OUT0[var].sel(stat_value='mean',year=np.arange(1850,1900+1)).mean('year')
                    else:
                        OUT[var+'_'+xp] += OUT0[var].sel(year=np.arange(1850,1900+1)).mean('year')
                elif xp in ['historical-CMIP5','rcp26','rcp45','rcp60','rcp85']:
                    if 'stat_value' in OUT[var+'_'+xp].dims:
                        OUT[var+'_'+xp].loc[{'stat_value':'mean'}] += OUT05[var].sel(stat_value='mean',year=np.arange(1850,1900+1)).mean('year')
                    else:
                        OUT[var+'_'+xp] += OUT05[var].sel(year=np.arange(1850,1900+1)).mean('year')
    OUT0.close()
    OUT05.close()

    ## preparing data for shifting temperature
    if option_overwrite:
        TMP = xr.Dataset()
        TMP.coords['all_config'] = np.array([str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])])
        TMP.coords['year'] = np.arange( 1850,2500+1 )
        for xps in list_xps+[['historical','historical-CMIP5']]:
            for xp in xps:
                var = 'D_Tg_shift'
                if xp[:2+1]=='ssp':
                    list_ext = ['','ext']
                else:
                    list_ext = ['']
                TMP[var+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=[TMP.year.size,TMP.all_config.size]) , dims=['year','all_config'] )
                for ext in list_ext:
                    if xp=='ssp534-over' and ext=='ext': ext='-ext'
                    for setMC in list_setMC:
                        print(xp+ext+'/'+str(setMC))
                        out_tmp,for_tmp,Par,mask = func_get( setMC , xp+ext , ['D_Tg'] )
                        TMP[var+'_'+xp].loc[{'year':out_tmp.year,'all_config':[str(setMC)+'-'+str(cfg) for cfg in np.arange(dico_sizesMC[setMC])]}] = out_tmp["D_Tg"]
                        out_tmp.close()
        ## shifting
        for xps in list_xps+[['historical','historical-CMIP5']]:
            for xp in xps:
                if xp[:2+1]=='rcp':
                    TMP[var+'_'+xp] = TMP[var+'_'+xp] - TMP[var+'_'+'historical-CMIP5'].sel(year=np.arange(1850,1900+1)).mean('year')
                else:
                    TMP[var+'_'+xp] = TMP[var+'_'+xp] - TMP[var+'_'+'historical'].sel(year=np.arange(1850,1900+1)).mean('year')

        TMP.to_netcdf(path_all+'/treated/OSCAR-CMIP6/plots/data_figures_SSP-RCP.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TMP})
    else:
        TMP = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/plots/data_figures_SSP-RCP.nc' )

    ## statistical values
    VAR = 'D_Tg_shift'
    for xps in list_xps+[['historical','historical-CMIP5']]:
        for xp in xps:
            OUT[VAR+'_'+xp] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
            ## calculating mean and std_dev. Using masked arrays to handle NaN values from exclusions.
            val = np.ma.array(TMP[VAR+'_'+xp],mask=np.isnan(TMP[VAR+'_'+xp]))
            ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
            OUT[VAR+'_'+xp].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
            OUT[VAR+'_'+xp].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT[VAR+'_'+xp].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

    for xps in list_xps:
        OUT['D_Tg_shift'+'_'+'diff'+xps[0][-2:]] = xr.DataArray( np.full(fill_value=np.nan,shape=(OUT.year.size,OUT.stat_value.size)), dims=('year','stat_value') )
        val = np.ma.array( (TMP['D_Tg_shift'+'_'+xps[1]]-TMP['D_Tg_shift'+'_'+xps[0]]) / TMP['D_Tg_shift'+'_'+xps[0]] , mask=np.isnan(TMP['D_Tg_shift'+'_'+xps[0]]) * np.isnan(TMP['D_Tg_shift'+'_'+xps[1]]) )
        val.mask[np.isnan(val)] = True
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        OUT['D_Tg_shift'+'_'+'diff'+xps[0][-2:]].loc[{'stat_value':'mean'}] = np.ma.average( a=val , axis=-1 , weights=ww ).data
        OUT['D_Tg_shift'+'_'+'diff'+xps[0][-2:]].loc[{'stat_value':'std_dev'}] = np.ma.sqrt(np.ma.average( (val - np.repeat(OUT['D_Tg_shift'+'_'+'diff'+xps[0][-2:]].sel(stat_value='mean').values[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data
        
        ## printing some values
        date_print = 2300
        mm,ss = OUT['D_Tg_shift'+'_'+'diff'+xps[0][-2:]].sel(year=date_print,stat_value='mean'), OUT['D_Tg_shift'+'_'+'diff'+xps[0][-2:]].sel(year=date_print,stat_value='std_dev')
        print('OSCAR: D_Tg_shift'+'_'+'diff'+xps[0][-2:]+' in '+str(date_print)+': '+str(mm.values)+'+/-'+str(ss.values))
        

    ## avoiding jump 2014-2015 in-between historical and ssps
    for xps in list_xps:
        for xp in xps:
            if xp[:2+1] == 'ssp':
                for VAR in list_var:
                    OUT[VAR+'_'+xp].loc[{'year':2014}] = OUT[VAR+'_'+'historical'].loc[{'year':2014}]

    ## PLOT
    var_plot = ['co2', 'rf_tot','D_Tg_shift','nbp','fgco2']
    period = np.arange(2000,2500+1)
    dico_col = {'historical-CMIP5':'grey' , 'historical':'k' , 'ssp126':(0/255.,52/255.,102/255.),'rcp26':CB_color_cycle[2], 'ssp245':(239/255.,85/255.,15/255.),'rcp45':CB_color_cycle[0], 'ssp460':(196/255.,121/255.,0/255.),'rcp60':CB_color_cycle[9], 'ssp585':(153/255.,0/255.,2/255.),'rcp85':CB_color_cycle[4]}
    dico_ls = {'historical-CMIP5':'-' , 'historical':'-' , 'rcp26':'-', 'ssp126':'-', 'ssp245':'-', 'rcp45':'-', 'rcp60':'-', 'ssp460':'-', 'rcp85':'-', 'ssp585':'-'}
    dico_label_var = {'co2':'Atmospheric CO$_2$ (ppm)', 'erf_tot':'Effective Radiative forcing\nwith reference to 1750 (W.m$^{-2}$)','rf_tot':'Radiative forcing\nwith reference to 1750 (W.m$^{-2}$)', 'tas':'Change in global mean surface temperature\nwith reference to 1850 (K)' , 'D_Tg_shift':'Change in global mean surface temperature\nwith reference to 1850-1900 (K)'  ,  'nbp':'Net land sink (PgC.yr$^{-1}$)'  ,  'fgco2':'Net ocean carbon sink (PgC.yr$^{-1}$)'  ,  'permafrostCO2':'CO$_2$ emissions from permafrost (PgC.yr$^{-1}$)'}

    fig = plt.figure( figsize=(20,30) )
    counter = 0
    # OUT0 = xr.open_dataset( path_all+'/treated/OSCAR-CMIP6/intermediary/'+os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')[np.where('piControl'==np.array([str.split(ff,'_')[0] for ff in os.listdir(path_all+'/treated/OSCAR-CMIP6/intermediary/')]))[0][0]] )
    for var in var_plot:
        for xps in list_xps:
            ax = plt.subplot(len(var_plot), len(list_xps), list_xps.index(xps)+var_plot.index(var)*len(list_xps)+1 )
            ## looping on experiments inside a subplot
            for xp in xps:#['historical-CMIP5','historical']
                if xp in ['historical-CMIP5']:
                    tt = np.arange(1850,2000+1)
                elif xp in ['historical']:
                    tt = np.arange(1850,2014+1)
                elif xp[:2+1] == 'rcp':
                    tt = np.arange(2001,2500+1)
                elif xp[:2+1] == 'ssp':
                    tt = np.arange(2014,2500+1)
                func_plot('year',var+'_'+xp,OUT.sel(year=tt),col=dico_col[xp],ls=dico_ls[xp],lw=0.75*fac_size*0.6*3,label=xp,alpha=0.15)
            ## polishing
            plt.grid()
            plt.xticks( [1850,1900,2000,2100,2200,2300,2400,2500] ,rotation=45 )
            plt.xlim(period[0],period[-1])
            if var=='rf_tot':
                plt.yticks( list(ax.get_yticks())+[float(xp[-2:])*0.1] )
            plt.legend(prop={'size':fac_size*0.6*14} ,loc=0 )
            ax.tick_params(labelsize=fac_size*0.6*14)
            # if var != var_plot[-1]:ax.set_xticklabels(['']*len([item.get_text() for item in ax.get_xticklabels()]))
            if xps == list_xps[0]:plt.ylabel( dico_label_var[var] , size=fac_size*0.6*14 )#,rotation=0  )#,fontweight='bold'
            plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
            counter += 1

    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/RCP_SSP.pdf',dpi=300 )
    plt.close(fig)

    
    if False:
        val = np.ma.array( TMP['D_Tg_shift'+'_'+'ssp585'] - TMP['D_Tg_shift'+'_'+'rcp585'] , mask=np.isnan(TMP['D_Tg_shift'+'_'+'ssp585'] - TMP['D_Tg_shift'+'_'+'rcp585']))
        ww = np.ma.repeat(np.ma.array(weights_CMIP6.weights.sel(all_config=TMP.all_config).values,mask=np.isnan(weights_CMIP6.weights.sel(all_config=TMP.all_config).values))[np.newaxis,:],TMP.year.size,axis=0)
        mm = np.ma.average( a=val , axis=-1 , weights=ww ).data
        ss = np.ma.sqrt(np.ma.average( (val - np.repeat(mm[...,np.newaxis],TMP.all_config.size,axis=-1) )**2. , axis=-1 , weights=ww )).data

#########################
#########################







#########################
## 7.16. PLOT DISTRIB
#########################
if '7.16' in option_which_plots:
    ## plot function
    def func_plot_distrib( ax,VAR,weights,factor_force=1.,n_bins_distrib=100,option_what_plot=['distrib_OSCAR','distrib_obs','line_OSCAR','line_obs'],option_polish=True , alpha_distrib=0.5,color_distrib='b',label_distrib='OSCAR',label_obs=None,lw=0.75*fac_size*5,ls='-',marker_distrib='o', option_print_values=True):
        ## initializing weights
        VALS = xr.Dataset()
        VALS.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        VALS['vals'] = xr.DataArray(  factor_force * (indic['x']*indic['m']).isel(index=indic.indicator.values.tolist().index(VAR)).values , dims=('all_config')  )        
        ind = np.where( ~np.isnan(weights.weights) & ~np.isnan(VALS.vals))[0]

        ## ploting our values
        if 'distrib_OSCAR' in option_what_plot:
            out = plt.hist( x=VALS.vals.isel(all_config=ind).values , bins=n_bins_distrib,density=True,weights=weights.weights.isel(all_config=ind).values , label=label_distrib , alpha=alpha_distrib , color=color_distrib , histtype='step',lw=0.75*fac_size*lw/2.,ls=ls )
        ## adding mean and std_dev of our values
        if 'line_OSCAR' in option_what_plot:
            mm = np.average( VALS.vals.isel(all_config=ind) ,axis=0, weights=weights.weights.isel(all_config=ind) )
            ss = np.sqrt(np.average( (VALS.vals.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights.weights.isel(all_config=ind) ))
            print('OSCAR: '+VAR+': '+str(mm)+'+/-'+str(ss))
            xl,yl,pos = ax.get_xlim(),ax.get_ylim(),option_what_plot[-1]
            plt.axhline( y=(0.90+0.025+pos)*(yl[1]-yl[0]) , xmin=(mm-ss-xl[0])/(xl[1]-xl[0]),xmax=(mm+ss-xl[0])/(xl[1]-xl[0]),color=list(color_distrib)+[1.],lw=0.75*fac_size*lw*1.0,ls=ls,zorder=98 )
            plt.scatter( y=(0.90+0.025+pos)*(yl[1]-yl[0]) , x=mm , facecolor=list(color_distrib)+[1.],edgecolor='k',marker=marker_distrib,s=fac_size*lw*20,zorder=99 )
        ## ploting constraints
        xx = np.linspace(plt.xlim()[0] / factor_force, plt.xlim()[1] / factor_force,500)
        yy = pdf_indic( str(indic['distrib'].isel(index=indic.indicator.values.tolist().index(VAR)).values), xx )
        xx *= factor_force
        if 'distrib_obs' in option_what_plot:
            plt.plot( xx , yy , color='k',lw=0.75*fac_size*lw/2.,ls='-',zorder=100  )
        ## adding mean and std_dev of constraints
        if 'line_obs' in option_what_plot:
            xl,yl = ax.get_xlim(),ax.get_ylim()
            mm,ss = indic_mean_stddev( VAR, xx)#xx_in )
            print('Obs: '+VAR+': '+str(mm)+'+/-'+str(ss))
            plt.axhline( y=(0.875+pos)*(yl[1]-yl[0]) , xmin=(mm-ss-xl[0])/(xl[1]-xl[0]),xmax=(mm+ss-xl[0])/(xl[1]-xl[0]),color='k',lw=0.75*fac_size*lw*1.0,ls='-',zorder=98 )
            plt.scatter( y=(0.875+pos)*(yl[1]-yl[0]) , x=mm , facecolor='k',edgecolor='w',marker=marker_distrib,s=fac_size*lw*25,zorder=99)
            ## ghost plot for label
            plt.plot( xx , np.nan*yy, color='k',lw=0.75*fac_size*lw,ls='-',marker=marker_distrib,ms=fac_size*12.5,mec='white',zorder=100 ,label=label_obs)

        if option_polish:
            ## polishing
            plt.grid()
            plt.legend(loc=0)
            plt.xlabel(VAR)


    fig = plt.figure( figsize=(30,20) )
    counter = 0
    vars_plot_distrib = [   ['Surface Air Ocean Blended Temperature Change World ssp245 2000-2019'],
                            ['Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011'],
                            [   'Cumulative compatible emissions CMIP5 historical-CMIP5',
                                'Cumulative compatible emissions CMIP5 RCP2.6',
                                'Cumulative compatible emissions CMIP5 RCP4.5',
                                'Cumulative compatible emissions CMIP5 RCP6.0',
                                'Cumulative compatible emissions CMIP5 RCP8.5'] ]
    xlabel_text = ['Surface air ocean blended temperature change\nover 2000-2019 with reference to 1961-1990 (K)', 'Cumulative net ocean carbon flux\nover 1750-2011 (PgC)', 'Cumulative compatible emissions (PgC)']
    legend_obs  = { 'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011':'Ciais et al, 2013',\
                    'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019':'Morice et al, 2012',
                    'Cumulative compatible emissions CMIP5 historical-CMIP5':'historical-CMIP5 (Ciais et al, 2013)',
                    'Cumulative compatible emissions CMIP5 RCP2.6':'RCP2.6 (Ciais et al, 2013)',
                    'Cumulative compatible emissions CMIP5 RCP4.5':'RCP4.5 (Ciais et al, 2013)',
                    'Cumulative compatible emissions CMIP5 RCP6.0':'RCP6.0 (Ciais et al, 2013)',
                    'Cumulative compatible emissions CMIP5 RCP8.5':'RCP8.5 (Ciais et al, 2013)'
                    }
    legend_osc  = { 'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011':'OSCAR v3.1',\
                    'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019':'OSCAR v3.1',
                    'Cumulative compatible emissions CMIP5 historical-CMIP5':'historical-CMIP5 OSCAR v3.1',
                    'Cumulative compatible emissions CMIP5 RCP2.6':'RCP2.6 OSCAR v3.1',
                    'Cumulative compatible emissions CMIP5 RCP4.5':'RCP4.5 OSCAR v3.1',
                    'Cumulative compatible emissions CMIP5 RCP6.0':'RCP6.0 OSCAR v3.1',
                    'Cumulative compatible emissions CMIP5 RCP8.5':'RCP8.5 OSCAR v3.1'
                    }
    color_force = { 'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011':CB_color_cycle[0],\
                    'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019':CB_color_cycle[0],
                    'Cumulative compatible emissions CMIP5 historical-CMIP5':CB_color_cycle[0],
                    'Cumulative compatible emissions CMIP5 RCP2.6':CB_color_cycle[2],
                    'Cumulative compatible emissions CMIP5 RCP4.5':CB_color_cycle[5],
                    'Cumulative compatible emissions CMIP5 RCP6.0':CB_color_cycle[1],
                    'Cumulative compatible emissions CMIP5 RCP8.5':CB_color_cycle[3]
                    }
    marker_dist = { 'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011':'o',\
                    'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019':'o',
                    'Cumulative compatible emissions CMIP5 historical-CMIP5':'o',
                    'Cumulative compatible emissions CMIP5 RCP2.6':'v',
                    'Cumulative compatible emissions CMIP5 RCP4.5':'d',
                    'Cumulative compatible emissions CMIP5 RCP6.0':'X',
                    'Cumulative compatible emissions CMIP5 RCP8.5':'^'
                    } 
    xlim_force = [ [0.3, 1.05], [75., 250.], [0., 2300.] ]
    ylim_force = [18.*1.2*0.75*0.70, 0.035*1.2*0.9, 0.013*1.4 ]
    factor_distrib = [1., -1., 1.]

    weights_empty = xr.Dataset()
    weights_empty['weights'] = 1. * xr.ones_like( weights_CMIP6.all_config, dtype=np.float32 )
    for vv in vars_plot_distrib:
        ax = plt.subplot( 1,3,vars_plot_distrib.index(vv)+1 )
        plt.grid()
        plt.xlim(xlim_force[vars_plot_distrib.index(vv)])
        plt.ylim(0,ylim_force[vars_plot_distrib.index(vv)])
        ax.set_axisbelow(True)
        if len(vv)>1:
            tmp = [-0.1-0.05, 0.05-0.05, -0.4-0.05, -0.535-0.03, -0.6]
        else:
            tmp = [-0.035]
        for VAR in vv:
            ## unconstrained
            func_plot_distrib( ax=ax, VAR=VAR, weights=weights_empty, factor_force=factor_distrib[vars_plot_distrib.index(vv)], n_bins_distrib=50,option_what_plot=['distrib_OSCAR','line_OSCAR','line_obs','distrib_obs',0.015+tmp[vv.index(VAR)]], option_polish=False , alpha_distrib=0.8,color_distrib=color_force[VAR],label_distrib=None,label_obs=legend_obs[VAR], lw=0.75*fac_size*4,ls='dotted',marker_distrib=marker_dist[VAR] )
            ## constrained
            func_plot_distrib( ax=ax, VAR=VAR , weights=weights_CMIP6, factor_force=factor_distrib[vars_plot_distrib.index(vv)], n_bins_distrib=50,option_what_plot=['distrib_OSCAR','line_OSCAR',-0.015+tmp[vv.index(VAR)]], option_polish=False , alpha_distrib=0.8,color_distrib=color_force[VAR],label_distrib=legend_osc[VAR],label_obs=legend_obs[VAR],lw=0.75*fac_size*4,ls='-',marker_distrib=marker_dist[VAR] )

        plt.xlabel( xlabel_text[vars_plot_distrib.index(vv)] , size=fac_size*12 )
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False,# labels along the bottom edge are off
            labelsize=fac_size*15)
        ax.tick_params(labelsize=fac_size*13)# ? not working on previous one?
        box = ax.get_position()
        ax.set_position([box.x0, box.y0+0.05-0.03*(vars_plot_distrib.index(vv)//3), box.width*1.1, box.height*1.0])#-(vars_plot_distrib.index(vv)//2)*0.03
        plt.legend(loc='upper right',prop={'size':fac_size*9.5},ncol=1)##,bbox_to_anchor=(-0.66,-0.20)
        plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
        counter += 1
    fig.savefig( path_all+'/treated/OSCAR-CMIP6/plots/effect-constraints.pdf',dpi=300 )
    plt.close(fig)

    
#########################
#########################








#########################
# PLOT FOR CONFIGURATION
#########################
if False:
    def func_load_all_exclusions(name_experiment):
        # Loading
        Out = xr.open_mfdataset([path_runs+'/'+name_experiment+'_Out-'+str(setMC)+'.nc' for setMC in list_setMC], combine='nested', concat_dim='config')
        Out = Out.assign_coords(config=np.arange(len(Out.config)))
        Out = Out.transpose(*(['year'] + [dim for dim in list(Out.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
        print('loaded outputs of '+name_experiment)
        Par = xr.open_mfdataset([path_runs + '/' + 'Par-' + str(setMC) + '.nc' for setMC in list_setMC], combine='nested', concat_dim='config')
        Par = Par.assign_coords(config=np.arange(len(Par.config)))
        print('loaded parameters of '+name_experiment)
        For = xr.open_mfdataset([path_runs+ '/' + name_experiment + '_For-' + str(setMC) + '.nc' for setMC in list_setMC], combine='nested', concat_dim='config')
        For = For.assign_coords(config=np.arange(len(For.config)))
        For = For.transpose(*(['year'] + [dim for dim in list(For.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
        print('loaded inputs of '+name_experiment)

        ## calculating variables that may be used
        for VAR in ['D_Eluc'] + (name_experiment[:len('land-')]!='land-')*['D_Focean','D_Epf_CO2'] + ['D_Fland']:#,'D_Cosurf','dic_0'
            Out[VAR] = OSCAR[VAR](Out, Par, For,recursive=True)
            print('calculated '+VAR+' of '+name_experiment)
        return Out

    # PREPARING
    dico_VAR = {'D_Eluc':'CO$_2$ emissions\nfrom LUC (PgC.yr$^{-1}$)','D_Focean':'Ocean sink of\ncarbon (PgC.yr$^{-1}$)','D_Fland':'Land sink of\ncarbon (PgC.yr$^{-1}$)', 'D_Epf_CO2':'CO$_2$ emissions from\n permafrost (PgC.yr$^{-1}$)'}
    xps_excl = ['ssp370', 'ssp585', 'abrupt-4xCO2', '1pctCO2' ]
    #dico_col_xps = {'ssp585':CB_color_cycle[1], 'ssp370':CB_color_cycle[4], '1pctCO2':CB_color_cycle[5], 'abrupt-4xCO2':CB_color_cycle[3]}
    dico_col_xps = {'ssp585':CB_color_cycle[3], 'ssp370':CB_color_cycle[3], '1pctCO2':CB_color_cycle[3], 'abrupt-4xCO2':CB_color_cycle[3]}
    
    # LOADING
    OUT = {}
    for name_experiment in xps_excl:
        OUT[name_experiment] = func_load_all_exclusions(name_experiment)
        
    print('identifying configurations')
    inds_all = {}
    # looping on xp
    for name_experiment in xps_excl:
        inds_all[name_experiment] = {}
        # preparing experiment
        if name_experiment in ['abrupt-4xCO2']:
            list_thres = [ ['D_Focean',-20,20], ['D_Fland',-20,20], ['D_Eluc',-20,20], ['D_Epf_CO2',-20,20] ]
        elif name_experiment in ['1pctCO2']:
            list_thres = [ ['D_Focean',-20,20], ['D_Fland',-20,20], ['D_Eluc',-20,20], ['D_Epf_CO2',-20,20] ] # derivate of D_Focean < -0.01, 1%
        else:
            list_thres = [ ['D_Focean',0,20], ['D_Fland',-20,20], ['D_Eluc',-20,20], ['D_Epf_CO2',-20,20] ]
            
        # going through thresholds
        for th in list_thres:
            # exclusion
            VAR, th_min, th_max = th
                    
            if name_experiment in ['abrupt-4xCO2']:
                ind_yr = np.where(  (OUT[name_experiment][VAR].isel(year=np.arange(-50,-1+1)) > th_max)  |  (OUT[name_experiment][VAR].isel(year=np.arange(-50,-1+1)) < th_min) |  np.isnan(OUT[name_experiment][VAR].isel(year=np.arange(-50,-1+1)))  )
            else:
                ind_yr = np.where(  (OUT[name_experiment][VAR] > th_max)  |  (OUT[name_experiment][VAR] < th_min) |  np.isnan(OUT[name_experiment][VAR])  )
            inds_all[name_experiment][VAR] = set(ind_yr[1])
            
            
            
    # PLOTING
    ft_sz = 18
    lw_gl = 2
    fig = plt.figure(figsize=(20,20))
    # looping on xp
    counter = 0
    for name_experiment in xps_excl:
        inds_tmp = set()
        if name_experiment in ['abrupt-4xCO2']:
            list_thres = [ ['D_Focean',-20,20], ['D_Fland',-20,20], ['D_Eluc',-20,20], ['D_Epf_CO2',-20,20] ]
        elif name_experiment in ['1pctCO2']:
            list_thres = [ ['D_Focean',-20,20], ['D_Fland',-20,20], ['D_Eluc',-20,20], ['D_Epf_CO2',-20,20] ] # derivate of D_Focean < -0.01, 1%
        else:
            list_thres = [ ['D_Focean',0,20], ['D_Fland',-20,20], ['D_Eluc',-20,20], ['D_Epf_CO2',-20,20] ]

        for i_th,th in enumerate(list_thres):
            print('plotting '+name_experiment+' on '+VAR)
            # exclusion
            VAR, th_min, th_max = th
            
            inds_here = inds_all[name_experiment][VAR] - inds_all[name_experiment][VAR].intersection(inds_tmp)
            inds_tmp = inds_tmp.union(inds_all[name_experiment][VAR])
        
            # prep
            to_exclude_here = np.nan * np.ones(OUT[name_experiment].config.size)
            to_exclude_here[ list(inds_here) ] = 1
            not_excluded = np.ones(OUT[name_experiment].config.size)
            not_excluded[ list(inds_tmp) ] = np.nan
            txt = str(np.round(np.nansum(not_excluded) / len(not_excluded) * 100,2))+'%'# str(int(np.nansum(not_excluded)))+'/'+str(int(len(not_excluded)))
            
            # prep VAR
            if VAR in ['D_Epf_CO2']:
                to_plot = OUT[name_experiment][VAR].sum('reg_pf')
            else:
                to_plot = OUT[name_experiment][VAR]
            
            # plotting
            ax = plt.subplot( len(xps_excl)+1, len(list_thres), xps_excl.index(name_experiment)*len(list_thres) + i_th+1 )
            plt.plot( OUT[name_experiment].year, to_plot, color='grey', lw=lw_gl/2,zorder=0 )
            plt.plot( OUT[name_experiment].year, to_plot * to_exclude_here, color=dico_col_xps[name_experiment], lw=lw_gl, ls='--' )
            plt.plot( OUT[name_experiment].year, to_plot * not_excluded, color=CB_color_cycle[2], lw=lw_gl )
            
            # polish
            if i_th == 0:
                plt.ylabel( name_experiment, fontsize=ft_sz )
            if xps_excl.index(name_experiment) == 0:
                plt.title( dico_VAR[VAR], fontsize=ft_sz, color='k' )
            plt.xticks( size=0.8*ft_sz, rotation=30 )
            plt.yticks( size=0.8*ft_sz )
            if name_experiment in ['abrupt-4xCO2']:
                plt.hlines( y=[th_min,th_max], xmin=OUT[name_experiment].year[-1]-50, xmax=OUT[name_experiment].year[-1], color='black', lw=lw_gl*1.5, ls='--', zorder=100  )
                plt.vlines( x=OUT[name_experiment].year[-1]-50, ymin=th_min, ymax=th_max, color='black', lw=3, ls='--', zorder=100 )
            else:
                plt.hlines( y=[th_min,th_max], xmin=OUT[name_experiment].year[0], xmax=OUT[name_experiment].year[-1], color='black', lw=lw_gl*1.5, ls='--', zorder=100  )
            if name_experiment in ['1pctCO2']:
                plt.vlines( x=OUT[name_experiment].year[0]+150, ymin=th_min, ymax=th_max, color='black', lw=lw_gl*1.5, ls='--', zorder=100 )
            # ghost plot for txt
            plt.hlines( y=th_min, xmin=OUT[name_experiment].year[-1], xmax=OUT[name_experiment].year[-1], color=None, lw=0, ls=':', zorder=100 , label=txt  )
            yl = -35,35
            xl = OUT[name_experiment].year[0], OUT[name_experiment].year[-1] # - 100 * (name_experiment in ['1pctCO2'])
            plt.ylim( yl )
            plt.xlim( xl )
            plt.text(x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter],fontdict={'size':0.9*fac_size*14})
            counter += 1
            plt.grid()
            plt.legend(loc='upper left', prop={'size':ft_sz} )
            
    # SAVING
    #fig.savefig( 'tmp_figures/OSCARv31_exclusions-process.pdf' )
    fig.savefig( 'tmp_figures/OSCARv31_exclusions-process.png',dpi=500 )
    plt.close(fig)
    print('done')
    
#########################
#########################

##################################################
##################################################




















##################################################
## 8. PLOTS FOR FURTHER PROJECT
##################################################

#########################
## 8.1. PLOT DISTRIB for CONSTRAIN
#########################
if '8.1' in option_which_plots:
    def load_Out(exp, Nset=20):
        Out = xr.open_mfdataset(['/landclim/yquilcaille/OSCARv31_CMIP6/results/CMIP6_v3.1/' + exp + '_Out-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
        Out = Out.assign_coords(config=np.arange(len(Out.config)))
        Out = Out.transpose(*(['year'] + [dim for dim in list(Out.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
        return Out
            
    # preparing data
    list_scens = ['ssp585', 'ssp370',  'ssp460', 'ssp245', 'ssp534-over', 'ssp434', 'ssp126', 'ssp119']
    dico_scens = { 'ssp585':'SSP5-8.5', 'ssp370':'SSP3-7.0', 'ssp460':'SSP4-6.0', 'ssp534-over':'SSP5-3.4-OS', 'ssp434':'SSP4-3.4', 'ssp245':'SSP2-4.5', 'ssp126':'SSP1-2.6', 'ssp119':'SSP1-1.9' }
    list_periods = [ [2021,2031], [2031,2041], [2021,2041] ]
    to_plot = xr.Dataset()
    
    # preparing coords
    to_plot.coords['config'] = indic.config.values
    tmp = []
    for scen in list_scens:
        for period in list_periods:
            tmp.append( 'D_Tg_'+scen+'_'+str(period[0])+'-'+str(period[1]) )
    to_plot.coords['values'] = tmp
    
    # preparing values
    to_plot['x'] = xr.DataArray( np.nan, coords={'config':to_plot['config'].values, 'values':to_plot['values'].values}, dims=('values', 'config') )
    
    # adding mask
    to_plot['m'] = xr.DataArray( indic['m'].isel(index=0).drop('index').values, coords={'config':to_plot['config'].values}, dims=('config') )
    

    # loading reference
    ref = load_Out( 'historical' )['D_Tg'].sel(year=range(1850,1900)).sel(year=range(1850,1900)).mean('year')
    
    # loading scens
    for scen in list_scens:
        tmp = load_Out( scen )['D_Tg']
        for period in list_periods:
            to_plot['x'].loc[{'values':'D_Tg_'+scen+'_'+str(period[0])+'-'+str(period[1])}] = tmp.sel(year=range(period[0],period[1])).mean('year') - ref

            
    ## plot function
    def func_plot_distrib( ax, VAR, weights, factor_force=1.,n_bins_distrib=100, option_what_plot=['distrib_OSCAR','line_OSCAR'], option_polish=True, alpha_distrib=0.5, color_distrib='b', label_distrib='OSCAR', lw=0.75*fac_size*5,ls='-',marker_distrib='o', zord=0):
        
        ## initializing weights
        VALS = xr.Dataset()
        VALS.coords['all_config'] = [ str(setMC)+'-'+str(cfg) for setMC in list_setMC for cfg in np.arange(dico_sizesMC[setMC])]
        VALS['vals'] = xr.DataArray(  factor_force * (to_plot['x']*to_plot['m']).sel( values=VAR ).values , dims=('all_config')  )        
        ind = np.where( ~np.isnan(weights.weights) & ~np.isnan(VALS.vals))[0]

        ## ploting our values
        if 'distrib_OSCAR' in option_what_plot:
            out = plt.hist( x=VALS.vals.isel(all_config=ind).values , bins=n_bins_distrib,density=True,weights=weights.weights.isel(all_config=ind).values , alpha=alpha_distrib , color=color_distrib , histtype='step',lw=0.75*fac_size*lw/2.,ls=ls, orientation="horizontal", zorder=zord )
            
        ## adding mean and std_dev of our values
        if 'line_OSCAR' in option_what_plot:
            mm = np.average( VALS.vals.isel(all_config=ind) ,axis=0, weights=weights.weights.isel(all_config=ind) )
            ss = np.sqrt(np.average( (VALS.vals.isel(all_config=ind) - mm)**2. ,axis=0, weights=weights.weights.isel(all_config=ind) ))
            xl,yl,pos = ax.get_xlim(),ax.get_ylim(),option_what_plot[-1]
            plt.axvline( x=(0.90+0.025+pos)*(xl[1]-xl[0]) , ymin=(mm-ss-yl[0])/(yl[1]-yl[0]), ymax=(mm+ss-yl[0])/(yl[1]-yl[0]), color=list(color_distrib)+[1.],lw=0.75*fac_size*lw*1.0,ls=ls,zorder=98, label=label_distrib+': '+str(np.round(mm,2))+'$\pm$'+str(np.round(ss,2))+'$^\circ$C' )
            plt.scatter( x=(0.90+0.025+pos)*(xl[1]-xl[0]) , y=mm , facecolor=list(color_distrib)+[1.],edgecolor='k',marker=marker_distrib,s=fac_size*lw*6,zorder=99 )

        if option_polish:
            ## polishing
            plt.grid()
            plt.legend(loc=0)
            plt.xlabel(VAR)

    weights_empty = xr.Dataset()
    weights_empty['weights'] = 1. * xr.ones_like( weights_CMIP6.all_config, dtype=np.float32 )
            
    # PLOTING
    fac_size= 1.2
    fig = plt.figure( figsize=(19.0*(1/2.54), 23.0*(1/2.54)) )
    xlim_force = [ 0,6. ]
    ylim_force = [ 0.5, 2.5 ]
    for scen in list_scens:
        for period in list_periods:
            ax = plt.subplot( len(list_scens),len(list_periods), 1+list_scens.index(scen)*len(list_periods) + list_periods.index(period) )
            plt.grid()
            plt.xlim(xlim_force)
            plt.ylim(ylim_force)
            #ax.set_axisbelow(True)
            
            VAR = 'D_Tg_'+scen+'_'+str(period[0])+'-'+str(period[1])
            
            ## unconstrained
            func_plot_distrib( ax=ax, VAR=VAR, weights=weights_empty, factor_force = 0.75/0.89, n_bins_distrib=100, option_what_plot=['distrib_OSCAR','line_OSCAR',  0.05], option_polish=False , alpha_distrib=0.8, color_distrib=CB_color_cycle[1], label_distrib='raw', lw=0.75*fac_size*4, ls='-', marker_distrib='o', zord=10 )
            
            ## constrained
            func_plot_distrib( ax=ax, VAR=VAR , weights=weights_CMIP6, factor_force = 0.75/0.89, n_bins_distrib=100, option_what_plot=['distrib_OSCAR','line_OSCAR', -0.05], option_polish=False , alpha_distrib=0.8, color_distrib=CB_color_cycle[0], label_distrib='const.', lw=0.75*fac_size*4,ls='-', marker_distrib='o', zord=0 )

            ax.tick_params(labelsize=fac_size*8)# ? not working on previous one?
            ax.tick_params(axis='x',label1On=False)
            plt.yticks( [0.5, 1.0, 1.5, 2.0, 2.5] )
            plt.tick_params(bottom = False)

            if period == list_periods[0]:
                plt.ylabel( dico_scens[scen] , size=fac_size*9, rotation=90 )
            else:
                ax.tick_params(axis='y',label1On=False)
            if scen == list_scens[0]:
                plt.title( str(period[0])+'-'+str(period[1]) , size=fac_size*9 )
            
            box = ax.get_position()
            ax.set_position([box.x0, box.y0+0.02-0.005*list_scens.index(scen), box.width*1.1, box.height*1.05])
            plt.legend(loc='upper right',prop={'size':fac_size*5},ncol=1)##,bbox_to_anchor=(-0.66,-0.20)
    plt.suptitle( 'Surface air ocean blended temperature change\nwith reference to 1850-1900 (K)' , size=fac_size*10 )
        
    fig.savefig( 'tmp_figures/OSCARv31_figure-CONSTRAIN.pdf',dpi=300 )
    fig.savefig( 'tmp_figures/OSCARv31_figure-CONSTRAIN.png',dpi=400 )
    plt.close(fig)

    # Preparing data for saving
    TO_SAVE = xr.Dataset()
    # identifying configurations that are not excluded
    ind_cfg = np.where(~np.isnan(to_plot['m'].values))[0]
    # id config
    TO_SAVE.coords['index_member'] = ind_cfg
    # period
    TO_SAVE.coords['periods'] = [str(period[0])+'-'+str(period[1]) for period in list_periods]
    # scenarios
    TO_SAVE.coords['scenarios'] = list_scens
    # variables
    vals = weights_empty['weights'].values[ind_cfg]
    vals /= np.sum(vals)
    TO_SAVE['weights_unconstrained'] = xr.DataArray( vals , coords={'index_member':TO_SAVE.index_member.values}, dims=('index_member') )
    TO_SAVE['weights_unconstrained'].attrs['info'] = 'Weights have been renormalized by their sum'
    vals = weights_CMIP6['weights'].values[ind_cfg]
    # renormalizing
    vals /= np.sum(vals)
    TO_SAVE['weights_constrained'] = xr.DataArray( vals , coords={'index_member':TO_SAVE.index_member.values}, dims=('index_member') )
    TO_SAVE['weights_constrained'].attrs['info'] = 'Weights have been renormalized by their sum'
    # saving the important one
    TO_SAVE['D_Tg'] = xr.DataArray( np.nan , coords={'scenarios':TO_SAVE.scenarios.values, 'periods':TO_SAVE.periods.values, 'index_member':TO_SAVE.index_member.values}, dims=('scenarios', 'periods', 'index_member') )
    for scen in list_scens:
        for period in list_periods:
            VAR = 'D_Tg_'+scen+'_'+str(period[0])+'-'+str(period[1])
            TO_SAVE['D_Tg'].loc[{'scenarios':scen, 'periods':str(period[0])+'-'+str(period[1])}] = to_plot['x'].sel(values=VAR, config=ind_cfg).values
    # correction for temperature
    TO_SAVE['D_Tg'] *= 0.75 / 0.89
    TO_SAVE['D_Tg'].attrs['info'] = 'Change in surface air ocean blended temperature change, derived from the change in global mean surface temperature of OSCAR v3.1'
    TO_SAVE['D_Tg'].attrs['unit'] = 'K'
    TO_SAVE['D_Tg'].attrs['reference period'] = '1850-1900'    
    # adding attributes to the global dataset
    TO_SAVE.attrs['info'] = 'Change in surface air ocean blended temperature change from SSP projections of OSCAR v3.1, as used in https://doi.org/10.5194/gmd-2021-412'
    TO_SAVE.attrs['warning'] = 'All calculations using the provided temperatures have to be used along the provided weights.'
    TO_SAVE.attrs['contact'] = 'Yann Quilcaille <yann.quilcaille@env.ethz.ch>, Thomas Gasser <gasser@iiasa.ac.at>'
    TO_SAVE.to_netcdf( 'tmp_figures/OSCARv31_values-CONSTRAIN.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in TO_SAVE})
#########################
#########################

##################################################
##################################################



print('done')










