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
name_experiment = 'yr2010CO2'
year_PI = 1850
year_start = 1850
year_end = 2010+1000
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
out_init = xr.open_dataset('results/'+folder+'/spinup_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'yr2010CO2':
## - Concentrations for CO2, CH4, N2O and halo from 'concentrations_CMIP6'
## - Emissions for all except CO2, CH4, N2O and halo from 'emissions_CEDS'
## - LULCC from reference scenario of LUH2
## - RF for solar and volcanoes as reference CMIP6
## - RF for contrails: 0
## --> all of these forcings follow the historical drivers. From 2010, all of these drivers are constant, with the exception of transitions in LUC that cannot remain constant over +1kyr.

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
    for var in ['CO2','CH4','N2O']:
        For['D_'+var] = xr.DataArray( np.full(fill_value=np.nan , shape=(year_end-year_start+1)), dims=('year') )
        For['D_'+var].loc[{'year':np.arange(year_start,2010+1)}] = TMP[var].loc[{'year':np.arange(year_start,2010+1),'region':'Globe'}]-Par[var+'_0']
        For['D_'+var] = For['D_'+var].fillna( For['D_'+var].sel(year=2010) )
    For['D_Xhalo'] = xr.DataArray( np.full(fill_value=np.nan , shape=(year_end-year_start+1,For.spc_halo.size)), dims=('year','spc_halo') )
    val = TMP['Xhalo'].loc[{'year':np.arange(year_start,2010+1),'region':'Globe'}]-Par['Xhalo_0']
    For['D_Xhalo'].loc[{'year':np.arange(year_start,2010+1),'spc_halo':val.spc_halo}] = val
    For['D_Xhalo'] = For['D_Xhalo'].fillna( For['D_Xhalo'].sel(year=2010) )

## Emissions
for var in ['BC','CO','NH3','VOC','NOX','OC','SO2']:
    For['E_'+var] = xr.DataArray( np.full(fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
    For['E_'+var].loc[{'year':np.arange(year_start,2010+1)}] = For0['E_'+var].loc[{'year':np.arange(year_start,2010+1),'data_E_'+var:'CEDS'}]
    For['E_'+var].loc[{'year':np.arange(year_start,2010+1),'reg_land':0}] = For['E_'+var].loc[{'year':np.arange(year_start,2010+1),'reg_land':0}].fillna(0.)
    For['E_'+var] = For['E_'+var].fillna( For['E_'+var].sel(year=2010) )
## Emissions (will not impact results, only here to allow the computation)
for var in ['Eff','E_CH4','E_N2O']:
    For[var] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
For['E_Xhalo'] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )

## Land-Use
For['d_Ashift'] = xr.DataArray( np.full(fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size)), dims=('year','reg_land','bio_from','bio_to') )
For['d_Ashift'].loc[{'year':np.arange(year_start,2010+1)}] = For0['d_Ashift'].loc[{'year':np.arange(year_start,2010+1),'data_LULCC':'LUH2'}]
For['d_Ashift'] = For['d_Ashift'].fillna( For['d_Ashift'].sel(year=2010) )
For['d_Acover'] = xr.DataArray( np.full(fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size)), dims=('year','reg_land','bio_from','bio_to') )
For['d_Acover'].loc[{'year':np.arange(year_start,2010+1)}] = For0['d_Acover'].loc[{'year':np.arange(year_start,2010+1),'data_LULCC':'LUH2'}]
For['d_Acover'] = For['d_Acover'].fillna( 0. ) # Stopping transitions, to avoid over-deforestation
For['d_Hwood'] = xr.DataArray( np.full(fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_land.size)), dims=('year','reg_land','bio_land') )
For['d_Hwood'].loc[{'year':np.arange(year_start,2010+1)}] = For0['d_Hwood'].loc[{'year':np.arange(year_start,2010+1),'data_LULCC':'LUH2'}]
For['d_Hwood'] = For['d_Hwood'].fillna( For['d_Hwood'].sel(year=2010) )

## RF solar and volc
for tp_rf in ['volc','solar']:
    For['RF_'+tp_rf] = xr.DataArray( np.full(fill_value=np.nan , shape=(year_end-year_start+1)), dims=('year') )
    For['RF_'+tp_rf].loc[{'year':np.arange(year_start,2010+1)}] = For0['RF_'+tp_rf].loc[{'year':np.arange(year_start,2010+1),'data_RF_'+tp_rf:'CMIP6'}]
    For['RF_'+tp_rf] = For['RF_'+tp_rf].fillna( For['RF_'+tp_rf].sel(year=2010) )

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
