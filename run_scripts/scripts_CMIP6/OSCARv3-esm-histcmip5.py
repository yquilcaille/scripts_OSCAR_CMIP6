import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from core_fct.fct_loadD import load_all_hist
from core_fct.fct_process import OSCAR


##################################################
## 1. OPTIONS
##################################################
## options for CMIP6
mod_region = 'RCP_5reg'
folder = 'CMIP6_v3.0'
type_LCC = 'gross'                        # gross  |  net
nt_run = 4

## options for this experiment
name_experiment = 'esm-histcmip5'
year_PI = 1850
year_start = 1850
year_end = 2000
nMC = 1000
setMC = 1

try:## script run under the script 'RUN-ALL_OSCARv3-CMIP6.py', overwrite the options where required.
    for key in forced_options_CMIP6.keys(): exec(key+" = forced_options_CMIP6[key]")
except NameError: pass ## script not run under the script 'RUN-ALL_OSCARv3-CMIP6.py', will use the options defined in section.
##################################################
##################################################



##################################################
## 2. PARAMETERS
##################################################
with xr.open_dataset('results/'+folder+'/Par-'+str(setMC)+'.nc') as TMP: Par = TMP.load()
print("Parameters done")
##################################################
##################################################



##################################################
## 3. INITIALIZATION
##################################################
## Using initialization from last year of spin-up
out_init = xr.open_dataset('results/'+folder+'/esm-spinup-CMIP5_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'esm-histcmip5':
## - Concentrations for CH4, N2O and halo from 'concentrations_Meinshausen2011'
## - Emissions for all except CO2, CH4, N2O and halo from 'emissions_ACCMIP'
## - LULCC from reference scenario of LUH1
## - RF for solar and volcanoes as reference AR5
## - RF for contrails: 0
## - Emissions for FF, N2O, CH4 and Xhalo are taken as 0. The run is concentrations-driven, these emissions are prescribed only to allow the computation to run.

## Loading all drivers, with correct regional/sectoral aggregation
For0 = load_all_hist(mod_region, LCC=type_LCC)

## Preparing dataset
For = xr.Dataset()
for cc in For0.coords:
    if (cc[:len('data_')] != 'data_'):
        For.coords[cc] = For0[cc]
## Cutting years
For.coords['year'] = np.arange(year_start,year_end+1)
## Adding coordinates for config
For.coords['config'] = np.arange(nMC)

## Concentrations
with xr.open_dataset('input_data/drivers/concentrations_Meinshausen_2011.nc') as TMP:
    for var in ['CH4','N2O']:
        For['D_'+var] = TMP[var].loc[{'year':np.arange(year_start,year_end+1),'scen':'historical'}]-Par[var+'_0']
    For['D_Xhalo'] = TMP['Xhalo'].loc[{'year':np.arange(year_start,year_end+1),'scen':'historical'}]-Par['Xhalo_0']
    For = For.drop('scen')

## Emissions
for var in ['BC','CO','NH3','VOC','NOX','OC','SO2']:
    For['E_'+var] = For0['E_'+var].loc[{'year':np.arange(year_start,year_end+1),'data_E_'+var:'ACCMIP'}]
    For = For.drop('data_E_'+var)
## Emissions (will not impact results, only here to allow the computation)
for var in ['Eff','E_CH4','E_N2O']:
    For[var] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
For['E_Xhalo'] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )

## Land-Use
for var in ['d_Ashift','d_Acover','d_Hwood']:
    For[var] = For0[var].loc[{'year':np.arange(year_start,year_end+1),'data_LULCC':'LUH1'}]
For = For.drop('data_LULCC')

## RF solar and volc
with xr.open_dataset('input_data/drivers/radiative-forcing_Meinshausen_2011.nc') as TMP:
    for tp_rf in ['volc','solar']:
        For['RF_'+tp_rf] = TMP['RF_'+tp_rf].loc[{'scen':'historical','year':np.arange(year_start,year_end+1)}]
    For = For.drop('scen')

## RF contr
For['RF_contr'] = xr.DataArray( np.zeros((year_end-year_start+1)), dims=('year') )

## CORRECTION OF 'Par.Aland_0': accounting for different starting year
for_init = xr.open_dataset('results/'+folder+'/esm-spinup-CMIP5_For-'+str(setMC)+'.nc' )
## Passing new value for preindustrial lands to forcing AND parameters 
Par['Aland_0'] = for_init['Aland_0']
For['Aland_0'] = for_init['Aland_0']
print("Corrected Aland_0. New values passed to parameters used, and saved ONLY in corresponding forcings.")

## Saving forcings
For.to_netcdf('results/'+folder+'/'+name_experiment+'_For-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in For})

print("Forcings done")
##################################################
##################################################



##################################################
## 5. RUN
##################################################
Out = OSCAR(Ini, Par, For , nt=nt_run)
Out.to_netcdf('results/'+folder+'/'+name_experiment+'_Out-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out})
print("Experiment "+name_experiment+" done")
##################################################
##################################################
