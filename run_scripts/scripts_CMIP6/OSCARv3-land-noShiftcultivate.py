import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from core_fct.fct_misc import aggreg_region
from core_fct.fct_loadD import load_all_hist
from core_fct.fct_process import OSCAR_landC    ## using ONLY the land carbon cycle, nothing more.



##################################################
## 1. OPTIONS
##################################################
## options for CMIP6
mod_region = 'RCP_5reg'
folder = 'CMIP6_v3.0'
type_LCC = 'gross'                        # gross  |  net
nt_run = 4

## options for this experiment
name_experiment = 'land-noShiftcultivate'
year_PI = 1850
year_start = 1850
year_end = 2010
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
## Using initialization from mathing year of land-spinup
out_init = xr.open_dataset('results/'+folder+'/land-spinup_Out-'+str(setMC)+'.nc' )
## Because of recycling over 1850-1900 of the period 1901-1920, and matching, need the last equivalent of a 1910
if out_init.year.size % 20 == 0:
    ind_ini = out_init.year.size - 10 - 1
else:
    ind_ini = out_init.year.size - (out_init.year.size % 20)
    if (out_init.year.size % 20) >= 10:
        ind_ini += 10-1
    else:
        ind_ini -= 10+1
## taking corresponding values
Ini = out_init.isel(year=ind_ini, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR_landC.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'land-noShiftcultivate':
## - Concentration for CO2: concentrations CMIP6
## - Climatology (local temperatures and precipitations): GSWP3 recycled over 1901-1920 for 1850-1900, then GSWP3 1901-2010
## - Land-use: LUH2 historical, but shifting frozen at 1850 level

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

## Climatology GSWP3
with xr.open_dataset('input_data/observations/climatology_GSWP3.nc') as TMP:
    vals = aggreg_region(ds_in=TMP , mod_region=mod_region, weight_var={'Tl':'area','Pl':'area'})
    for var in ['Tl','Pl']:
        For['D_'+var] = xr.full_like(other=0.*For.year + For.reg_land, fill_value=np.nan)
        ## defining preindustrial as mean over 1901-1920
        val = (vals[var] - vals[var].sel(year=np.arange(1901,1920+1)).mean('year'))
        ## recycling 1901-1920 over 1850-1900
        if (year_start != 1850) or (year_end != 2010):
            raise Exception("This script has been prepared for a period 1850-2010.")
        For['D_'+var].loc[{'year':np.arange(1881,1900+1)}] = val.sel(year=np.arange(1901,1920+1)).values
        For['D_'+var].loc[{'year':np.arange(1861,1880+1)}] = val.sel(year=np.arange(1901,1920+1)).values
        For['D_'+var].loc[{'year':np.arange(1850,1860+1)}] = val.sel(year=np.arange(1910,1920+1)).values
        For['D_'+var].loc[{'year':np.arange(1901,2010+1)}] = val

## Concentrations
with xr.open_dataset('input_data/observations/concentrations_CMIP6.nc') as TMP:
    for var in ['CO2']:
        For['D_'+var] = TMP[var].loc[{'year':np.arange(year_start,year_end+1),'region':'Globe'}]-Par[var+'_0']
    For = For.drop('region')

## Land-Use
for var in ['d_Acover','d_Hwood']:
    For[var] = For0[var].loc[{'year':np.arange(year_start,year_end+1),'data_LULCC':'LUH2'}]
For['d_Ashift'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size)), dims=('year', 'reg_land', 'bio_from', 'bio_to') )
For['d_Ashift'] = For['d_Ashift'].fillna( For0['d_Ashift'].loc[{'year':year_PI,'data_LULCC':'LUH2'}] )
For = For.drop('data_LULCC')

## Saving forcings
For.to_netcdf('results/'+folder+'/'+name_experiment+'_For-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in For})

print("Forcings done")
##################################################
##################################################







##################################################
## 4. RUN
##################################################
Out = OSCAR_landC(Ini, Par, For , nt=nt_run)
Out.to_netcdf('results/'+folder+'/'+name_experiment+'_Out-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out})
print("Experiment "+name_experiment+" done")
##################################################
##################################################

