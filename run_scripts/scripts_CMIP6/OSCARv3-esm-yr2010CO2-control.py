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
name_experiment = 'esm-yr2010CO2-control'
year_PI = 1850
year_start = 1850
year_end = 2010+1000
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
## Forcings for 'esm-yr2010CO2-control':
## - Concentrations of CH4, N2O and halo are still driven using 'concentrations_CMIP6'
## - Emissions
##      - CO2 from compatible emissions from the experiment 'yr2010CO2'
##      - CH4, N2O and halo: nothing to prescribe
##      - all others from 'emissions_CEDS'
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
    for var in ['CH4','N2O']: # CO2 is emissions-driven
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

## initializing GhG emissions
for var in ['Eff','E_N2O','E_CH4']:
    For[var] = xr.DataArray( np.full(fill_value=0. , shape=(year_end-year_start+1,For.reg_land.size,nMC)), dims=('year','reg_land','config') )
For['E_Xhalo'] = xr.DataArray( np.full(fill_value=0. , shape=(For.spc_halo.size,year_end-year_start+1,For.reg_land.size,nMC)), dims=('spc_halo','year','reg_land','config') )

## producing required variables to calculate compatible emissions, using the experiment 'yr2010CO2'
out_conc = xr.open_dataset('results/'+folder+'/yr2010CO2_Out-'+str(setMC)+'.nc' )
for_conc = xr.open_dataset('results/'+folder+'/yr2010CO2_For-'+str(setMC)+'.nc' )
for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4'] + ['D_Ebb','D_Fsink_N2O'] + ['D_Ewet','D_Ebb','D_Epf_CH4','D_Fsink_CH4'] + ['D_Fsink_Xhalo']:
    out_conc[VAR] = OSCAR[VAR](out_conc, Par, for_conc,recursive=True)

## compatible emissions
val = - out_conc.D_Eluc  -  out_conc.D_Epf_CO2.sum('reg_pf',min_count=1)  +  out_conc.D_Fland  +  out_conc.D_Focean  -  out_conc.D_Foxi_CH4
For['Eff'].loc[{'reg_land':0,'year':np.arange(year_start+1,year_end+1)}] = Par.a_CO2 * out_conc.D_CO2.diff(dim='year') + 0.5*( val + val.shift(year=1) )
## N2O, CH4 and halo are concentrations-driven, not emissions-driven.
# val = - out_conc.D_Ebb.sel({'spc_bb':'N2O'}).sum('bio_land',min_count=1).sum('reg_land',min_count=1)  +  out_conc.D_Fsink_N2O
# For['E_N2O'].loc[{'reg_land':0,'year':np.arange(year_start+1,year_end+1)}] = Par.a_N2O * out_conc.D_N2O.diff(dim='year') + 0.5*( val + val.shift(year=1) )
# val = - out_conc.D_Ewet.sum('reg_land',min_count=1) - out_conc.D_Ebb.sel({'spc_bb':'CH4'}).sum('bio_land',min_count=1).sum('reg_land',min_count=1)  -  out_conc.D_Epf_CH4.sum('reg_pf',min_count=1)  +  out_conc.D_Fsink_CH4
# For['E_CH4'].loc[{'reg_land':0,'year':np.arange(year_start+1,year_end+1)}] = Par.a_CH4 * out_conc.D_CH4.diff(dim='year') + 0.5*( val + val.shift(year=1) )
# val = out_conc.D_Fsink_Xhalo
# For['E_Xhalo'].loc[{'reg_land':0,'year':np.arange(year_start+1,year_end+1)}] = Par.a_Xhalo * out_conc.D_Xhalo.diff(dim='year')  + 0.5*( val + val.shift(year=1) )

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
# from core_fct.fct_ancillary import Int_imex as Int_test
# Out = OSCAR(Ini, Par, For , Int=Int_test)
Out = OSCAR(Ini, Par, For , nt=nt_run)
Out.to_netcdf('results/'+folder+'/'+name_experiment+'_Out-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out})
print("Experiment "+name_experiment+" done")
##################################################
##################################################




if False:
    ## systematic check if identical forcings
    # for_esmyr = xr.open_dataset('results/'+folder+'/'+name_experiment+'_For-'+str(setMC)+'.nc' )
    # for_yr2010 = xr.open_dataset('results/'+folder+'/yr2010CO2_For-'+str(setMC)+'.nc' )
    # for var in for_esmyr.variables:
    #     if var not in for_yr2010.variables:
    #         print(var+' not in yr2010CO2 forcings')
    #     if (for_esmyr[var] == for_yr2010[var]).all():
    #         pass # ok
    #     else:
    #         print(var+' are different.')

    ## comparing concentrations: IN of yr2010CO2 and OUT of esm-yr2010CO2-control
    for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4'] + ['D_Ebb','D_Fsink_N2O'] + ['D_Ewet','D_Ebb','D_Epf_CH4','D_Fsink_CH4'] + ['D_Fsink_Xhalo']:
        Out[VAR] = OSCAR[VAR](Out, Par, For,recursive=True)
    for VAR in ['D_CO2','D_CH4','D_N2O']:
        plt.subplot(4,1,1+['D_CO2','D_CH4','D_N2O'].index(VAR))
        plt.plot( Out.year, Out[VAR] , color= 'gray' )
        # plt.plot( Out.year, Out[VAR].sel(config=0) , color= 'k', lw=2, label=name_experiment )
        plt.plot( Out.year, Out[VAR].mean('config') , color= 'k', lw=2, label=name_experiment )
        plt.plot( for_conc.year, for_conc[VAR] , label='forced to yr2010CO2' )
        plt.grid()
        plt.legend(loc=0)
        plt.title(VAR+' (for 1 config)')
    plt.subplot(4,1,4)
    # plt.plot( Out.year, Out['D_Xhalo'].sel(spc_halo='HFC-23') , color= 'gray' )
    plt.plot( Out.year, Out['D_Xhalo'].sel(spc_halo='HFC-23').sel(config=0) , color= 'k', lw=2, label=name_experiment )
    plt.plot(for_conc.year, for_conc['D_Xhalo'].sel(spc_halo='HFC-23') , label='forced to yr2010CO2' )
    plt.grid()
    plt.legend(loc=0)
    plt.title('D_Xhalo'+' (for 1 config)')

    ## comparing outputs:
    list_var = ['D_CO2','D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4']
    # list_var = ['D_Ewet','D_Ebb','D_Epf_CH4','D_Fsink_CH4']
    for VAR in list_var:
        plt.subplot( len(list_var),1,list_var.index(VAR)+1 )
        # emi-driven
        if VAR in ['D_Epf_CO2','D_Epf_CH4']:
            plt.plot( Out.year, Out[VAR].sel(config=0).sum('reg_pf',min_count=1) , color= 'k', lw=2, label=name_experiment )
        elif VAR=='D_Ebb':
            plt.plot( Out.year, Out[VAR].sel({'spc_bb':'CH4','config':0}).sum('bio_land',min_count=1).sum('reg_land',min_count=1) , color= 'k', lw=2, label=name_experiment )
        elif VAR=='D_Ewet':
            plt.plot( Out.year, Out[VAR].sel(config=0).sum('reg_land',min_count=1) , color= 'k', lw=2, label=name_experiment )
        else:
            plt.plot( Out.year, Out[VAR].sel(config=0) , color= 'k', lw=2, label=name_experiment )        
        # conc-driven
        if VAR in ['D_Epf_CO2','D_Epf_CH4']:
            to_plot = out_conc[VAR].sum('reg_pf',min_count=1)
        elif VAR=='D_Ebb':
            to_plot = out_conc[VAR].sel({'spc_bb':'CH4'}).sum('bio_land',min_count=1).sum('reg_land',min_count=1)
        elif VAR=='D_Ewet':
            to_plot = out_conc[VAR].sum('reg_land',min_count=1)
        else:
            to_plot = out_conc[VAR]
        if 'config' in out_conc[VAR].dims:
            to_plot = to_plot.sel(config=0)
        else:
            to_plot = to_plot
        plt.plot( out_conc.year, to_plot , color= 'r', lw=2, label='yr2010CO2' )
        plt.grid()
        plt.legend(loc=0)
        plt.title(VAR+' (for 1 config)')





