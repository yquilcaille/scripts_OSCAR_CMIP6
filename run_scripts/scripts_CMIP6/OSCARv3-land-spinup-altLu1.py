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
name_experiment = 'land-spinup-altLu1'
year_PI = 1850
year_start = 1850
year_end = 1850+1000
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
## 3. DRIVERS / FORCINGS
##################################################
## Forcings for 'land-spinup-altLu1':
## - Concentration for CO2: concentrations CMIP6, 1850 value
## - Climatology (local temperatures and precipitations): GSWP3 recycled over 1901-1920
## - Land-use: LUH2 historical "LUH2-High", fixed at 1850 values

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
        if year_start != 1850:
            raise Exception("Dates incorrect.")
        ## recycling 1901-1920
        For['D_'+var].loc[{'year':np.arange(year_start,year_start+10)}] = val.sel(year=np.arange(1911,1920+1)).values
        For['D_'+var].loc[{'year':np.arange(year_start+10,year_start+10+int((year_end-year_start-10)/20.)*20)}] = np.stack( list(val.sel(year=np.arange(1901,1920+1)).values) * int((year_end-year_start-10)/20.) )
        For['D_'+var].loc[{'year':np.arange(year_start+10+int((year_end-year_start-10)/20.)*20,year_end+1)}] = val.isel(year=np.arange(year_end-year_start-10-int((year_end-year_start-10)/20.)*20+1)).values

## Concentrations
with xr.open_dataset('input_data/observations/concentrations_CMIP6.nc') as TMP:
    for var in ['CO2']:
        For['D_'+var] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1)), dims=('year') )
        For['D_'+var] = For['D_'+var].fillna( TMP[var].loc[{'year':year_PI,'region':'Globe'}] - Par[var+'_0'] )
    For = For.drop('region')

## Land-Use
For['d_Hwood'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_land.size)), dims=('year', 'reg_land', 'bio_land') )
For['d_Hwood'] = For['d_Hwood'].fillna( For0['d_Hwood'].loc[{'year':year_PI,'data_LULCC':'LUH2-High'}] )
For['d_Ashift'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size)), dims=('year', 'reg_land', 'bio_from', 'bio_to') )
For['d_Ashift'] = For['d_Ashift'].fillna( For0['d_Ashift'].loc[{'year':year_PI,'data_LULCC':'LUH2-High'}] )
For['d_Acover'] = xr.DataArray( np.zeros( (year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size) ), dims=('year', 'reg_land', 'bio_from', 'bio_to') )
For = For.drop('data_LULCC')

## CORRECTION OF 'Par.Aland_0': accounting for different driving LU
For0['Aland'] = For0['d_Acover'].sum('bio_from', min_count=1).rename({'bio_to':'bio_land'}).cumsum('year') - For0['d_Acover'].sum('bio_to', min_count=1).cumsum('year').rename({'bio_from':'bio_land'})
For0['Aland'] += For0.Aland_0 - For0['Aland'].sel(year=For0.Aland_0.year,drop=True)
## Passing new value for preindustrial lands to forcing AND parameters 
For['Aland_0'] = For0.Aland.sel(year=year_PI,data_LULCC='LUH2-High')
Par['Aland_0'] = For['Aland_0']
For = For.drop('data_LULCC')
Par = Par.drop('data_LULCC')
print("Corrected Aland_0. New values passed to parameters used, and saved ONLY in corresponding forcings.")

## Saving forcings
For.to_netcdf('results/'+folder+'/'+name_experiment+'_For-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in For})

print("Forcings done")
##################################################
##################################################




##################################################
## 4. INITIALIZATION
##################################################
Ini = xr.Dataset()
for VAR in list(OSCAR_landC.var_prog):
    if len(OSCAR_landC[VAR].core_dims) == 0: Ini[VAR] = xr.DataArray(0.)
    else: Ini[VAR] = sum([xr.zeros_like(Par[dim], dtype=float) if dim in Par.coords else xr.zeros_like(For0[dim], dtype=float) for dim in OSCAR_landC[VAR].core_dims])
print("Initialization done")
##################################################
##################################################



##################################################
## 5. RUN
##################################################
Out = OSCAR_landC(Ini, Par, For , nt=nt_run)
Out.to_netcdf('results/'+folder+'/'+name_experiment+'_Out-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out})
print("Experiment "+name_experiment+" done")
##################################################
##################################################

