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
name_experiment = 'ssp119'
year_PI = 1850
year_start = 2014
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
## Using initialization from last year of historical
out_init = xr.open_dataset('results/'+folder+'/historical_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'ssp119':
## - Concentrations for CO2, CH4, N2O and halo: from 'concentrations_ScenarioMIP', SSP1-1.9
## - Emissions for all from 'emissions_ScenarioMIP', SSP1-1.9  (FF, N2O, CH4 and Xhalo: not used, but here to allow the computation to run)
## - LULCC from LUH2, SSP1-1.9
## - RF for solar and volcanoes: are ramped down from the value in 2014 to the average of 1850-2014 reached in 2024 as reference scenario for CMIP6
## - RF for contrails: 0

## Loading all drivers, with correct regional/sectoral aggregation
For0 = load_all_hist(mod_region, LCC=type_LCC)
For0E = load_emissions_scen(mod_region,datasets=['ScenarioMIP'])
For0L = load_landuse_scen(mod_region,datasets=['LUH2'],LCC=type_LCC)
For0R = load_RFdrivers_scen()

## Using dataset from forcings to prepare those from this experiment
for_runs_hist = xr.open_dataset('results/'+folder+'/historical_For-'+str(setMC)+'.nc')

## Preparing dataset
For = xr.Dataset()
for cc in for_runs_hist.coords:
    For.coords[cc] = for_runs_hist[cc]
## Correcting years
For.coords['year'] = np.arange(year_start,year_end+1)

## Preparing variables
for var in for_runs_hist.variables:
    if var not in for_runs_hist.coords:
        For[var] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1]+list(for_runs_hist[var].shape[1:])), dims=['year']+list(for_runs_hist[var].dims[1:]) )
        For[var].loc[dict(year=year_start)] = for_runs_hist[var].sel(year=year_start)

## Concentrations
with xr.open_dataset('input_data/drivers/concentrations_ScenarioMIP.nc') as TMP:
    ## CO2, CH4, N2O and Xhalo
    for var in ['CO2','CH4','N2O','Xhalo']:
        val = TMP[var].loc[{'year':np.arange(year_start+1,year_end+1),'scen':'SSP1-1.9'}] - Par[var+'_0']
        if 'spc_halo' in val.dims:
            For['D_'+var].loc[{'year':np.arange(year_start+1,year_end+1),'spc_halo':val.spc_halo.values}] = val
        else:
            For['D_'+var].loc[{'year':np.arange(year_start+1,year_end+1)}] = val

## Emissions
for var in ['E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2'] + ['Eff','E_CH4','E_N2O']:
    For[var].loc[{'year':np.arange(year_start+1,year_end+1)}] = For0E[var].loc[{'year':np.arange(year_start+1,year_end+1),'scen_'+var:'SSP1-1.9'}]
## Emissions (will not impact results, only here to allow the computation)
For['E_Xhalo'].loc[{'year':np.arange(year_start+1,year_end+1)}] = xr.DataArray( np.zeros((year_end-year_start+1-1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )

## Land-Use: LUH2 provides values up to 2099: taking 2100 as 2099
For['d_Hwood'].loc[{'year':np.arange(year_start+1,year_end),'bio_land':For0L.bio_land}] = For0L['d_Hwood'].loc[{'year':np.arange(year_start+1,year_end),'scen_LULCC':'SSP1-1.9'}]
For['d_Hwood'].loc[dict(year=year_end)] = For['d_Hwood'].sel(year=year_end-1) # 2100 as 2099
for var in ['d_Ashift','d_Acover']:
    For[var].loc[{'year':np.arange(year_start+1,year_end),'bio_from':For0L.bio_from,'bio_to':For0L.bio_to}] = For0L[var].loc[{'year':np.arange(year_start+1,year_end),'scen_LULCC':'SSP1-1.9'}]
    For[var].loc[dict(year=year_end)] = For[var].sel(year=year_end-1) # 2100 as 2099

## RF volc
tmp_start = for_runs_hist['RF_volc'].loc[{'year':2014}]
tmp_end = for_runs_hist['RF_volc'].loc[{'year':np.arange(1850,2014+1)}].mean('year')
For['RF_volc'].loc[dict(year=np.arange(year_start,year_start+10+1))] = np.linspace(tmp_start,tmp_end,10+1)  # ramp over 10 years
For['RF_volc'] = For['RF_volc'].fillna( tmp_end )

## RF solar
For['RF_solar'].loc[dict(year=np.arange(year_start,year_end+1))] = For0R.RF_solar.sel(scen_RF_solar='CMIP6',year=np.arange(year_start,year_end+1))  *  For0.RF_solar.sel(data_RF_solar='CMIP6',year=np.arange(2015-11+1,2015+1)).mean('year') / For0R.RF_solar.sel(scen_RF_solar='CMIP6',year=np.arange(2015-11+1,2015+1)).mean('year')

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
