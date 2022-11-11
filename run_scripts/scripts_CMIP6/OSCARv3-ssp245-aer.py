import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from core_fct.fct_loadD import load_all_hist,load_emissions_scen,load_landuse_scen,load_RFdrivers_scen  # bug with 'load_all_scen'
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
name_experiment = 'ssp245-aer'
year_PI = 1850
year_start = 2020
year_end = 2100
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
out_init = xr.open_dataset('results/'+folder+'/hist-aer_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'ssp245-aer':
## - Concentrations for CO2, CH4, N2O and halo from 'concentrations_CMIP6', year 1850
## - Emissions for all except GhG from 'emissions_CEDS' and 'emissions_ScenarioMIP', SSP2-4.5 over 2021-2100
## - LULCC from reference scenario of LUH2, year 1850
## - RF for solar and volcanoes as reference CMIP6, year 1850, similar to piControl.
## - RF for contrails: 0
## - Emissions for FF, N2O, CH4 and Xhalo are taken as 0. The run is concentrations-driven, these emissions are prescribed only to allow the computation to run.

## Loading all drivers, with correct regional/sectoral aggregation
For0 = load_all_hist(mod_region, LCC=type_LCC)
For0E = load_emissions_scen(mod_region,datasets=['ScenarioMIP'])
# For0L = load_landuse_scen(mod_region,datasets=['LUH2'],LCC=type_LCC)

## Preparing dataset
For = xr.Dataset()
for cc in For0.coords:
    if (cc[:len('data_')] != 'data_'):
        For.coords[cc] = For0[cc]
## Cutting years
For.coords['year'] = np.arange(year_start,year_end+1)
## Adding coordinates for config
For.coords['config'] = np.arange(nMC)

## Using dataset from forcings to prepare those from this experiment
for_runs_hist = xr.open_dataset('results/'+folder+'/hist-aer_For-'+str(setMC)+'.nc' )

## Preparing variables
for var in for_runs_hist.variables:
    if var not in for_runs_hist.coords:
        For[var] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1]+list(for_runs_hist[var].shape[1:])), dims=['year']+list(for_runs_hist[var].dims[1:]) )
        For[var].loc[dict(year=year_start)] = for_runs_hist[var].sel(year=year_start)

## Concentrations
with xr.open_dataset('input_data/observations/concentrations_CMIP6.nc') as TMP:
    for var in ['CO2','CH4','N2O']:
        For['D_'+var] = For['D_'+var].fillna( TMP[var].loc[{'year':year_PI,'region':'Globe'}] - Par[var+'_0'] )
    For['D_Xhalo'] = For['D_Xhalo'].fillna( TMP['Xhalo'].loc[{'year':year_PI,'region':'Globe'}] - Par['Xhalo_0'] )
    For = For.drop('region')

## Emissions
for var in ['E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2']:
    For[var] = For0E[var].loc[{'year':np.arange(year_start,year_end+1),'scen_'+var:'SSP2-4.5'}]
    For = For.drop('scen_'+var)
## Emissions (will not impact results, only here to allow the computation)
for var in ['Eff','E_CH4','E_N2O','E_Xhalo']:
    For[var] = For[var].fillna(0.)

## Land-Use
For['d_Hwood'] = For['d_Hwood'].fillna( For0['d_Hwood'].loc[{'year':year_PI,'data_LULCC':'LUH2'}] )
For['d_Ashift'] = For['d_Ashift'].fillna( For0['d_Ashift'].loc[{'year':year_PI,'data_LULCC':'LUH2'}] )
For['d_Acover'] = For['d_Acover'].fillna(0.)
For = For.drop('data_LULCC')

## RF solar and volc
For['RF_solar'] = xr.DataArray( np.full( fill_value=For0['RF_solar'].loc[{'year':np.arange(1850,1873),'data_RF_solar':'CMIP6'}].mean('year') , shape=(year_end-year_start+1)), dims=('year') )
For['RF_volc'] = xr.DataArray( np.full( fill_value=For0['RF_solar'].loc[{'year':np.arange(1850,2014+1),'data_RF_solar':'CMIP6'}].mean('year') , shape=(year_end-year_start+1)), dims=('year') )

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
