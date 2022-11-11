import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from core_fct.fct_loadD import load_all_hist,load_emissions_scen,load_landuse_scen  # bug with 'load_all_scen'
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
name_experiment = 'esm-rcp45'
year_PI = 1850
year_start = 2000
year_end = 2500
nMC = 500
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
## Using initialization from last year of historical
out_init = xr.open_dataset('results/'+folder+'/esm-histcmip5_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'esm-rcp45':
## - Concentrations for CH4, N2O and halo: from 'concentrations_Meinshausen2011', RCP4.5
## - Emissions for all from 'emissions_RCPdb', RCP4.5  (FF, N2O, CH4 and Xhalo: not used, but here to allow the computation to run)
## - LULCC from LUH1, RCP4.5.
## - RF for solar and volcanoes: are ramped down from the value in 2014 to the average of 1850-2014 reached in 2024 as reference scenario for CMIP6
## - RF for contrails: 0


## Loading all drivers, with correct regional/sectoral aggregation
For0E = load_emissions_scen(mod_region,datasets=['RCPdb'])
For0E2 = xr.open_dataset('input_data/drivers/emissions_Meinshausen_2011.nc')
For0L = load_landuse_scen(mod_region,datasets=['LUH1'],LCC=type_LCC)
## Loading all drivers, with correct regional/sectoral aggregation
For0 = load_all_hist(mod_region, LCC=type_LCC)

## Using dataset from forcings to prepare those from this experiment
for_runs_hist = xr.open_dataset('results/'+folder+'/esm-histcmip5_For-'+str(setMC)+'.nc')

## Preparing dataset
For = xr.Dataset()
for cc in for_runs_hist.coords:
    For.coords[cc] = for_runs_hist[cc]
## Correcting years
For.coords['year'] = np.arange(year_start,year_end+1)

## Preparing variables
for var in for_runs_hist.variables:
    if (var not in for_runs_hist.coords) and (var != 'Aland_0'):
        For[var] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1]+list(for_runs_hist[var].shape[1:])), dims=['year']+list(for_runs_hist[var].dims[1:]) )
        For[var].loc[dict(year=year_start)] = for_runs_hist[var].sel(year=year_start)

## Concentrations
with xr.open_dataset('input_data/drivers/concentrations_Meinshausen_2011.nc') as TMP:
    ## CH4, N2O and Xhalo
    for var in ['CH4','N2O','Xhalo']:
        valh = TMP[var].loc[{'year':np.arange(year_start+1,2005+1),'scen':'historical'}] - Par[var+'_0']
        vals = TMP[var].loc[{'year':np.arange(2005+1,year_end+1),'scen':'RCP4.5'}] - Par[var+'_0']
        if 'spc_halo' in valh.dims:
            For['D_'+var].loc[{'year':np.arange(year_start+1,2005+1),'spc_halo':valh.spc_halo.values}] = valh
            For['D_'+var].loc[{'year':np.arange(2005+1,year_end+1),'spc_halo':valh.spc_halo.values}] = vals
        else:
            For['D_'+var].loc[{'year':np.arange(year_start+1,2005+1)}] = valh
            For['D_'+var].loc[{'year':np.arange(2005+1,year_end+1)}] = vals

## Emissions
for var in ['E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2'] + ['Eff','E_CH4','E_N2O']:
    For[var].loc[{'year':np.arange(year_start+1,2100+1)}] = For0E[var].loc[{'year':np.arange(year_start+1,2100+1),'scen_'+var:'RCP4.5'}]
    if var == 'Eff':
        For[var].loc[{'year':np.arange(2100+1,year_end+1),'reg_land':0}] = For0E2[var].loc[{'year':np.arange(2100+1,year_end+1),'scen':'RCP4.5'}]
    else:
        For[var].loc[{'year':np.arange(2100+1,year_end+1)}] = For[var].loc[{'year':np.arange(2100+1,year_end+1)}].fillna( For[var].loc[{'year':2100}] )
## Emissions (will not impact results, only here to allow the computation)
For['E_Xhalo'].loc[{'year':np.arange(year_start+1,2100+1),'spc_halo':For0E.spc_halo}] = For0E['E_Xhalo'].loc[{'year':np.arange(year_start+1,2100+1),'scen_E_Xhalo':'RCP4.5'}]
For['E_Xhalo'].loc[{'year':np.arange(2100+1,year_end+1),'spc_halo':For0E2.spc_halo,'reg_land':0}] = For0E2['E_Xhalo'].loc[{'year':np.arange(2100+1,year_end+1),'scen':'RCP4.5'}]

## Land-Use: LUH2 provides values up to 2099: taking 2100 as 2099.
For['d_Hwood'].loc[{'year':np.arange(year_start+1,2004+1),'bio_land':For0.bio_land}] = For0['d_Hwood'].loc[{'year':np.arange(year_start+1,2004+1),'data_LULCC':'LUH1'}]
For['d_Hwood'].loc[{'year':np.arange(2004+1,2099+1),'bio_land':For0L.bio_land}] = For0L['d_Hwood'].loc[{'year':np.arange(2004+1,2099+1),'scen_LULCC':'RCP4.5'}]
For['d_Hwood'].loc[dict(year=2100)] = For['d_Hwood'].sel(year=2099) # 2100 as 2099
for var in ['d_Ashift','d_Acover']:
    For[var].loc[{'year':np.arange(year_start+1,2004+1),'bio_from':For0.bio_from,'bio_to':For0.bio_to}] = For0[var].loc[{'year':np.arange(year_start+1,2004+1),'data_LULCC':'LUH1'}]
    For[var].loc[{'year':np.arange(2004+1,2099+1),'bio_from':For0L.bio_from,'bio_to':For0L.bio_to}] = For0L[var].loc[{'year':np.arange(2004+1,2099+1),'scen_LULCC':'RCP4.5'}]
    For[var].loc[dict(year=2100)] = For[var].sel(year=2099) # 2100 as 2099
## Freeze over 2100-2500
For['d_Hwood'].loc[dict(year=np.arange(2101,year_end+1))] = For['d_Hwood'].sel(year=2100)
For['d_Ashift'].loc[dict(year=np.arange(2101,year_end+1))] = For['d_Ashift'].sel(year=2100)
For['d_Acover'].loc[dict(year=np.arange(2101,year_end+1))] = 0.

## RF solar and volc
with xr.open_dataset('input_data/drivers/radiative-forcing_Meinshausen_2011.nc') as TMP:
    for tp_rf in ['volc','solar']:
        For['RF_'+tp_rf] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1]), dims=['year'] )
        For['RF_'+tp_rf].loc[{'year':np.arange(year_start,2005+1)}] = TMP['RF_'+tp_rf].loc[{'scen':'historical','year':np.arange(year_start,2005+1)}]
        For['RF_'+tp_rf].loc[{'year':np.arange(2005+1,year_end+1)}] = TMP['RF_'+tp_rf].loc[{'scen':'RCP4.5','year':np.arange(2005+1,year_end+1)}]
    # For = For.drop('scen')

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
