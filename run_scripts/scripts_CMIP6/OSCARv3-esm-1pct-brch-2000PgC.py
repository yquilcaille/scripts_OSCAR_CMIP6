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
name_experiment = 'esm-1pct-brch-2000PgC'
year_PI = 1850
year_start = 1850
year_end = 1850+150+1000
nMC = 1000
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
## Using initialization from last year of spin-up
out_init = xr.open_dataset('results/'+folder+'/spinup_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'esm-1pct-brch-2000PgC':
## - Concentrations for CH4, N2O and halo from 'concentrations_CMIP6', year 1850
## - Emissions of CO2 as compatible emissions, but cut when 2000 PgC have been emitted
## - Emissions for all except CH4, N2O and halo from 'emissions_CEDS', year 1850
## - LULCC from reference scenario of LUH2, states of year 1850
## - RF for solar and volcanoes as reference CMIP6, year 1850
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
    ## no CO2
    ## CH4 and N2O
    for var in ['CH4','N2O']:
        For['D_'+var] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1)), dims=('year') )
        For['D_'+var] = For['D_'+var].fillna( TMP[var].loc[{'year':year_PI,'region':'Globe'}] - Par[var+'_0'] )
    ## Xhalo
    For['D_Xhalo'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,len(For.spc_halo))), dims=('year','spc_halo') )
    For['D_Xhalo'] = For['D_Xhalo'].fillna( TMP['Xhalo'].loc[{'year':year_PI,'region':'Globe'}] - Par['Xhalo_0'] )
    For = For.drop('region')

## Emissions
for var in ['BC','CO','NH3','VOC','NOX','OC','SO2']:
    For['E_'+var] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
    For['E_'+var] = For['E_'+var].fillna( For0['E_'+var].loc[{'year':year_PI,'data_E_'+var:'CEDS'}] )
    For = For.drop('data_E_'+var)
## Emissions (will not impact results, only here to allow the computation)
for var in ['E_CH4','E_N2O']:
    For[var] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size)), dims=('year','reg_land') )
For['E_Xhalo'] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )

## Eff
For['Eff'] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size,nMC)), dims=('year','reg_land','config') )
## producing required variables to calculate compatible emissions, using the experiment '1pctCO2'
out_conc = xr.open_dataset('results/'+folder+'/1pctCO2_Out-'+str(setMC)+'.nc' )
for_conc = xr.open_dataset('results/'+folder+'/1pctCO2_For-'+str(setMC)+'.nc' )
for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4']:
    out_conc[VAR] = OSCAR[VAR](out_conc, Par, for_conc,recursive=True)
## calculating compatible emissions
val = - out_conc.D_Eluc  -  out_conc.D_Epf_CO2.sum('reg_pf',min_count=1)  +  out_conc.D_Fland  +  out_conc.D_Focean  -  out_conc.D_Foxi_CH4
For['Eff'].loc[{'reg_land':0,'year':np.arange(year_start+1,for_conc.year.isel(year=-1)+1)}] = Par.a_CO2 * out_conc.D_CO2.diff(dim='year') + 0.5*( val + val.shift(year=1) )
## calculating year exceeding required cumulated compatible emissions
cum = For.Eff.sel(reg_land=0).cumsum('year')
yy = For.year.isel(year=(cum<2000.).argmin('year')) # last year where exceeds this threshold. It is 0 if does not reach it.
## removing Eff and year_breach for members that dont reach the required cumulated emissions
configs_reached = For.config.isel(config=np.where(yy!=1850)[0]).values
## saving years where breached
For['year_breach'] = xr.DataArray( np.full(fill_value=np.nan,shape=(nMC)) , dims='config' )
For.year_breach.attrs['note'] = 'Each member leads to different compatible emissions, thus different timing for the breach.'
For.year_breach.attrs['warning'] = 'If a given member cannot reach the required cumulated emissions, this member is excluded.'
For['year_breach'].loc[{'config':For.config.isel(config=configs_reached)}]  =  yy.loc[{'config':For.config.isel(config=configs_reached)}] # nan for non-reached members
## Eff for non-reached members become full nan
For['Eff'].loc[{'config':[ic for ic in np.arange(nMC) if ic not in configs_reached]}] = np.nan
## correction of Eff for used members
for ic in configs_reached:
    # ## Eff become 0 after breach
    For['Eff'].loc[{'config':For.config.isel(config=ic),'year':np.arange(For['year_breach'].isel(config=ic),year_end+1)}] = 0
    ## Eff over last year so that reach exactly required cumulated compatible emissions
    For['Eff'].loc[{'reg_land':0,'year':For['year_breach'].isel(config=ic),'config':For.config.isel(config=ic)}] = 2000. - cum.loc[{'year':For['year_breach'].isel(config=ic)-1,'config':For.config.isel(config=ic)}]
## Checking:
if False:
    plt.plot( For.year, For.Eff.sel(reg_land=0).cumsum('year') )

## Land-Use
For['d_Hwood'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_land.size)), dims=('year', 'reg_land', 'bio_land') )
For['d_Hwood'] = For['d_Hwood'].fillna( For0['d_Hwood'].loc[{'year':year_PI,'data_LULCC':'LUH2'}] )
For['d_Ashift'] = xr.DataArray( np.full( fill_value=np.nan , shape=(year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size)), dims=('year', 'reg_land', 'bio_from', 'bio_to') )
For['d_Ashift'] = For['d_Ashift'].fillna( For0['d_Ashift'].loc[{'year':year_PI,'data_LULCC':'LUH2'}] )
For['d_Acover'] = xr.DataArray( np.zeros( (year_end-year_start+1,For.reg_land.size,For.bio_from.size,For.bio_to.size) ), dims=('year', 'reg_land', 'bio_from', 'bio_to') )
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
