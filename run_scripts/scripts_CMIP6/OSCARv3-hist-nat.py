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
name_experiment = 'hist-nat'
year_PI = 1850
year_start = 1850
year_end = 2020
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
## Forcings for 'hist-nat':
## - Concentrations for CO2, CH4, N2O and halo from 'concentrations_CMIP6', year 1850
## - Emissions for all except GhG from 'emissions_CEDS', year 1850
## - Emissions of GhG: set to 0, doesnt matter
## - LULCC from reference scenario of LUH2, year 1850
## - RF for solar and volcanoes as reference CMIP6 over 1850-2014 + the ramp over 10 years from ssp245
## - RF for contrails: 0

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
        For['D_'+var] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1)), dims=('year') )
        For['D_'+var] = For['D_'+var].fillna( TMP[var].loc[{'year':year_PI,'region':'Globe'}] - Par[var+'_0'] )
    For['D_Xhalo'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,len(For.spc_halo))), dims=('year','spc_halo') )
    For['D_Xhalo'] = For['D_Xhalo'].fillna( TMP['Xhalo'].loc[{'year':year_PI,'region':'Globe'}] - Par['Xhalo_0'] )
    For = For.drop('region')

## Emissions
for var in ['E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2']:
    For[var] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
    For[var] = For[var].fillna( For0[var].loc[{'year':year_PI,'data_'+var:'CEDS'}] )
    For = For.drop('data_'+var)
## Emissions (will not impact results, only here to allow the computation)
for var in ['Eff','E_CH4','E_N2O']:
    For[var] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
For['E_Xhalo'] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )

## Land-Use
For['d_Hwood'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_land.size)), dims=('year', 'reg_land', 'bio_land') )
For['d_Hwood'] = For['d_Hwood'].fillna( For0['d_Hwood'].loc[{'year':year_PI,'data_LULCC':'LUH2'}] )
For['d_Ashift'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size)), dims=('year', 'reg_land', 'bio_from', 'bio_to') )
For['d_Ashift'] = For['d_Ashift'].fillna( For0['d_Ashift'].loc[{'year':year_PI,'data_LULCC':'LUH2'}] )
For['d_Acover'] = xr.DataArray( np.zeros( (year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size) ), dims=('year', 'reg_land', 'bio_from', 'bio_to') )
For = For.drop('data_LULCC')

## RF solar and volc
for tp_rf in ['volc','solar']:
    For['RF_'+tp_rf] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1]), dims=['year'] )
    ## historical evolutions up to 2014
    For['RF_'+tp_rf].loc[{'year':np.arange(year_start,2014+1)}] = For0['RF_'+tp_rf].loc[{'year':np.arange(year_start,2014+1),'data_RF_'+tp_rf:'CMIP6'}]
    ## ramp down over 10 years to background level, as prescribed in ssp245
    tmp_start = For0['RF_'+tp_rf].loc[{'year':2014,'data_RF_'+tp_rf:'CMIP6'}]
    tmp_end = For0['RF_'+tp_rf].loc[{'year':np.arange(1850,2014+1),'data_RF_'+tp_rf:'CMIP6'}].mean('year')
    For['RF_'+tp_rf].loc[dict(year=np.arange(2014+1,min([2014+10,year_end])+1))] = np.linspace(tmp_start,tmp_end,10+1)[1:min([2014+10,year_end+1])-2014]  # ramp over 10 years
    For['RF_'+tp_rf] = For['RF_'+tp_rf].fillna( tmp_end )

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
