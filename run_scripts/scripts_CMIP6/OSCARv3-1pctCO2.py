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
name_experiment = '1pctCO2'
year_PI = 1850
year_start = 1850
year_end = 1850+150+100
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
## Forcings for '1pctCO2':
## - Concentrations for CO2: from the value in 1850, increase by 1%/yr of the atmospheric CO2 (1850: CO2_1850, 1851:*1.01, 1852: *1.01**2.,...). Lasts for at least 150 years (beyond x4), with 1%/yr applied throughout.
## - Concentrations for CH4, N2O and halo from 'concentrations_CMIP6', year 1850
## - Emissions for all except CO2, CH4, N2O and halo from 'emissions_CEDS', year 1850
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
    ## CO2
    For['D_CO2'] = xr.DataArray( np.full(fill_value=np.nan,shape=(year_end-year_start+1)) , dims=('year'))
    For['D_CO2'].loc[{'year':year_PI}] = TMP['CO2'].loc[{'year':year_PI,'region':'Globe'}] - Par['CO2_0']
    For['D_CO2'].loc[{'year':np.arange(year_PI+1,year_end+1)}] = xr.DataArray( np.cumprod(np.repeat(1.01,year_end-year_start)),coords={'year':np.arange(year_start+1,year_end+1)},dims=('year')) * TMP['CO2'].loc[{'year':year_PI,'region':'Globe'}] - Par['CO2_0']
    print("******************************")
    print("!WARNING!")
    print('This experiment is meant to last for at least 150 years (beyond x4), with 1%/yr applied throughout.')
    print('Here, '+str(year_end-year_start)+' years chosen ('+str(np.round(1.01**(year_end-year_start),1))+'*CO2_1850='+str(np.round((For.D_CO2.sel(year=year_end)+Par['CO2_0']).values,1))+'ppm), might change dates for this experiment if too high...')
    print("******************************")
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

if False:
    ## selecting runs where 'pCO2_is_Pade' is False
    ind = np.where( Par.pCO2_is_Pade==0. )[0]

    Out['D_Focean'] = OSCAR['D_Focean'](Out, Par, For,recursive=True)

    Out_before5to7 = xr.open_dataset('results/'+folder+'/'+name_experiment+'_Out-'+str(setMC)+'.nc')
    Out_before5to7['D_Focean'] = OSCAR['D_Focean'](Out_before5to7, Par, For,recursive=True)

    plt.plot( Out.year, Out.D_Focean.sel(config=ind).transpose() ,'k', label='change in PowerLaw')
    plt.plot( Out_before5to7.year, Out_before5to7.D_Focean.sel(config=ind).transpose(),'r',label='no change' )


    diverged = np.where(   (Out.D_Focean.diff('year').isel(year=-1) * Out.D_Focean.diff('year').isel(year=-2) <= -1.)     )[0]
    tooNeg = np.where(   (Out.D_Focean.isel(year=-1) < -100)     )[0]
    list_negFocean = list(set(list(diverged)+list(tooNeg)))


    list_par_ocean = ['t_circ','p_circ','v_fg','mld_0' , 'p_mld' , 'g_mld' , 'pCO2_is_Pade', 'To_0','A_ocean']
    for VAR in list_par_ocean:
        plt.figure(VAR)
        if 'box_osurf' in Par[VAR].dims:
            plt.hist( Par[VAR].sel(box_osurf=3) , label='unrestrained')
            plt.hist( Par[VAR].sel(box_osurf=3,config=list_negFocean) , label='restrained to aberrant D_Focean')
        else:
            plt.hist( Par[VAR] , label='unrestrained')
            plt.hist( Par[VAR].sel(config=list_negFocean) , label='restrained to aberrant D_Focean')
        plt.legend(loc=0)
        plt.grid()
