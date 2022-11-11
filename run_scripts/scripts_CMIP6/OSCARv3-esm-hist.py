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
name_experiment = 'esm-hist'
year_PI = 1850
year_start = 1850
year_end = 2014
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
out_init = xr.open_dataset('results/'+folder+'/esm-spinup_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'esm-hist':
## - Concentrations for CH4, N2O and halo from 'concentrations_CMIP6'
## - Emissions for all except CH4, N2O and halo from 'emissions_CEDS'. Emissions of CO2 fall in this category.
## - LULCC from reference scenario of LUH2
## - RF for solar and volcanoes as reference CMIP6
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
with xr.open_dataset('input_data/observations/concentrations_CMIP6.nc') as TMP:
    for var in ['CH4','N2O']:
        For['D_'+var] = TMP[var].loc[{'year':np.arange(year_start,year_end+1),'region':'Globe'}]-Par[var+'_0']
    For['D_Xhalo'] = TMP['Xhalo'].loc[{'year':np.arange(year_start,year_end+1),'region':'Globe'}]-Par['Xhalo_0']
    For = For.drop('region')

## Emissions
for var in ['Eff','E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2']:
    For[var] = For0[var].loc[{'year':np.arange(year_start,year_end+1),'data_'+var:'CEDS'}]
    For = For.drop('data_'+var)
## Emissions (will not impact results, only here to allow the computation)
for var in ['E_CH4','E_N2O']:
    For[var] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
For['E_Xhalo'] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )

## Land-Use
for var in ['d_Ashift','d_Acover','d_Hwood']:
    For[var] = For0[var].loc[{'year':np.arange(year_start,year_end+1),'data_LULCC':'LUH2'}]
For = For.drop('data_LULCC')

## RF solar and volc
for tp_rf in ['volc','solar']:
    For['RF_'+tp_rf] = For0['RF_'+tp_rf].loc[{'year':np.arange(year_start,year_end+1),'data_RF_'+tp_rf:'CMIP6'}]
    For = For.drop('data_RF_'+tp_rf)

## RF contr
For['RF_contr'] = xr.DataArray( np.zeros((year_end-year_start+1)), dims=('year') )

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
