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
name_experiment = 'G6solar'
METHOD_BALANCE = 'prescribed_RF_solar'      # forced_to_RF_ssp245  |  prescribed_RF_solar
year_PI = 1850
year_start = 2014
year_end = 2100
nMC = 500
setMC = 1

try:## script run under the script 'RUN-ALL_OSCARv3-CMIP6.py', overwrite the options where required.
    for key in forced_options_CMIP6.keys(): exec(key+" = forced_options_CMIP6[key]")
except NameError: pass ## script not run under the script 'RUN-ALL_OSCARv3-CMIP6.py', will use the options defined in section .
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
## Checking consistency of years
if out_init.year[-1] != year_start:raise Exception("Warning, check the definition of the period/initialization.")
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'G6solar':
## - Concentrations for CO2, CH4, N2O and halo: from 'concentrations_ScenarioMIP', SSP5-8.5
## - Emissions for all from 'emissions_ScenarioMIP', SSP5-8.5  (FF, N2O, CH4 and Xhalo: not used, but here to allow the computation to run)
## - LULCC from LUH2, SSP5-8.5
## - RF for volcanoes: ramped down from the value in 2014 to the average of 1850-2014 reached in 2024 as reference scenario for CMIP6
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

## Preparing first year, 2014, that will be skipped. Taken as last year of the forcings from historical
for var in for_runs_hist.variables:
    if var not in for_runs_hist.coords:
        For[var] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1]+list(for_runs_hist[var].shape[1:])), dims=['year']+list(for_runs_hist[var].dims[1:]) )
        For[var].loc[{'year':year_start}] = for_runs_hist[var].loc[{'year':year_start}]

## Concentrations
with xr.open_dataset('input_data/drivers/concentrations_ScenarioMIP.nc') as TMP:
    ## CO2, CH4, N2O and Xhalo
    for var in ['CO2','CH4','N2O','Xhalo']:
        val = TMP[var].loc[{'year':np.arange(year_start+1,year_end+1),'scen':'SSP5-8.5'}] - Par[var+'_0']
        if 'spc_halo' in val.dims:
            For['D_'+var].loc[{'year':np.arange(year_start+1,year_end+1),'spc_halo':val.spc_halo.values}] = val
        else:
            For['D_'+var].loc[{'year':np.arange(year_start+1,year_end+1)}] = val

## Emissions
for var in ['E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2'] + ['Eff','E_CH4','E_N2O']:
    For[var].loc[{'year':np.arange(year_start+1,year_end+1)}] = For0E[var].loc[{'year':np.arange(year_start+1,year_end+1),'scen_'+var:'SSP5-8.5'}]
## Emissions (will not impact results, only here to allow the computation)
For['E_Xhalo'].loc[{'year':np.arange(year_start+1,year_end+1)}] = xr.DataArray( np.zeros((year_end-year_start+1-1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )

## Land-Use: LUH2 provides values up to 2099: taking 2100 as 2099
For['d_Hwood'].loc[{'year':np.arange(year_start+1,year_end),'bio_land':For0L.bio_land}] = For0L['d_Hwood'].loc[{'year':np.arange(year_start+1,year_end),'scen_LULCC':'SSP5-8.5'}]
For['d_Hwood'].loc[dict(year=year_end)] = For['d_Hwood'].sel(year=year_end-1) # 2100 as 2099
for var in ['d_Ashift','d_Acover']:
    For[var].loc[{'year':np.arange(year_start+1,year_end),'bio_from':For0L.bio_from,'bio_to':For0L.bio_to}] = For0L[var].loc[{'year':np.arange(year_start+1,year_end),'scen_LULCC':'SSP5-8.5'}]
    For[var].loc[dict(year=year_end)] = For[var].sel(year=year_end-1) # 2100 as 2099

## RF volc
tmp_start = for_runs_hist['RF_volc'].loc[{'year':2014}]
tmp_end = for_runs_hist['RF_volc'].loc[{'year':np.arange(1850,2014+1)}].mean('year')
For['RF_volc'].loc[dict(year=np.arange(year_start+1,year_start+10+1))] = np.linspace(tmp_start,tmp_end,10+1)[1:]  # ramp over 10 years
For['RF_volc'] = For['RF_volc'].fillna( tmp_end )

## RF solar
if METHOD_BALANCE=='prescribed_RF_solar':
    For['RF_solar'] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1,nMC]), dims=['year','config'] )
    For['RF_solar'] = For['RF_solar'].fillna( For0R.RF_solar.sel(scen_RF_solar='CMIP6',year=np.arange(year_start,year_end+1))  *  For0.RF_solar.sel(data_RF_solar='CMIP6',year=np.arange(2015-11+1,2015+1)).mean('year') / For0R.RF_solar.sel(scen_RF_solar='CMIP6',year=np.arange(2015-11+1,2015+1)).mean('year') )

    ## Compensating increase in RF_warm by a decrease in the solar constant from 2020
    tmp_out = xr.open_dataset('results/'+folder+'/ssp585_Out-'+str(setMC)+'.nc')
    tmp_for = xr.open_dataset('results/'+folder+'/ssp585_For-'+str(setMC)+'.nc')
    tmp_out['RF_warm'] = OSCAR['RF_warm'](tmp_out, Par, tmp_for,recursive=True) # ERF, not RF
    For['RF_solar'].loc[{'year':np.arange(2020,year_end+1)}] -= tmp_out['RF_warm'].loc[{'year':np.arange(2020,year_end+1)}] ## removes SSP5-8.5 ERF

    tmp_out = xr.open_dataset('results/'+folder+'/ssp245_Out-'+str(setMC)+'.nc')
    tmp_for = xr.open_dataset('results/'+folder+'/ssp245_For-'+str(setMC)+'.nc')
    tmp_out['RF_warm'] = OSCAR['RF_warm'](tmp_out, Par, tmp_for,recursive=True) # ERF, not RF
    For['RF_solar'].loc[{'year':np.arange(2020,year_end+1)}] += tmp_out['RF_warm'].loc[{'year':np.arange(2020,year_end+1)}] ## adds SSP2-4.5 ERF
elif METHOD_BALANCE=='forced_to_RF_ssp245':
    For['RF_solar'] = xr.DataArray( np.full(fill_value=0. , shape=[year_end-year_start+1,nMC]), dims=['year','config'],attrs={'Warning':'These values were not prescribed to the model, the RF of SSP2-4.5 has been forced to the model.'} ) # does not matter
    tmp_out = xr.open_dataset('results/'+folder+'/ssp245_Out-'+str(setMC)+'.nc')
    tmp_for = xr.open_dataset('results/'+folder+'/ssp245_For-'+str(setMC)+'.nc')
    tmp_out['RF_warm'] = OSCAR['RF_warm'](tmp_out, Par, tmp_for,recursive=True) # ERF, not RF
    # Forcing the total RF of the model by the RF of SSP2-4.5. Includes the ramp
    For['RF_warm'] = tmp_out['RF_warm']

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

## Recalculating the RF_solar that would have been prescribed to obtain these result:
# For['RF_solar']
if METHOD_BALANCE=='forced_to_RF_ssp245':
    for VAR in ['RF_warm','RF_wmghg','RF_slcf','RF_snow','RF_lcc']:
        Out[VAR] = OSCAR[VAR](Out, Par, For,recursive=True)
    For['RF_solar'] = Out.RF_warm - (Out.RF_wmghg + Out.RF_slcf + Par.w_warm_snow * Out.RF_snow + Par.w_warm_lcc * Out.RF_lcc + Par.w_warm_volc * For.RF_volc + For.RF_contr)
    ## Re-saving forcings
    For.to_netcdf('results/'+folder+'/'+name_experiment+'_For-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in For})
print("Experiment "+name_experiment+" done")
##################################################
##################################################




if False:
    plt.figure()
    plt.subplot(211)
    plt.plot( Out.year, Out.D_Tg )
    plt.subplot(212)
    plt.plot( Out.year, Out.D_Pg )

