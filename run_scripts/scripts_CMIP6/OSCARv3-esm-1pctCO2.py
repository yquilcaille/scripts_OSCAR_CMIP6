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
name_experiment = 'esm-1pctCO2'
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
## Forcings for 'esm-1pctCO2':
## - Concentrations for CH4, N2O and halo from 'concentrations_CMIP6', year 1850
## - Emissions of CO2 from compatible emissions of '1pctCO2'
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
    ## NOT CO2!
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

## Compatible emissions for Eff:
For['Eff'] = xr.DataArray( np.zeros((year_end-year_start+1,For.reg_land.size,nMC)), dims=('year','reg_land','config') )
## producing required variables to calculate compatible emissions, using the experiment '1pctCO2'
out_conc = xr.open_dataset('results/'+folder+'/1pctCO2_Out-'+str(setMC)+'.nc' )
for_conc = xr.open_dataset('results/'+folder+'/1pctCO2_For-'+str(setMC)+'.nc' )
for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4']:
    out_conc[VAR] = OSCAR[VAR](out_conc, Par, for_conc,recursive=True)
## calculating them
val = - out_conc.D_Eluc  -  out_conc.D_Epf_CO2.sum('reg_pf',min_count=1)  +  out_conc.D_Fland  +  out_conc.D_Focean  -  out_conc.D_Foxi_CH4
For['Eff'].loc[{'reg_land':0,'year':np.arange(year_start+1,year_end+1)}] = Par.a_CO2 * out_conc.D_CO2.diff(dim='year') + 0.5*( val + val.shift(year=1) )

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
    plt.plot( Out.D_CO2 )
    # two blocks
    plt.plot( Out.D_CO2.sel(config=0) ) # ok
    plt.plot( Out.D_CO2.sel(config=1) ) # negative?
    plt.plot( Out.D_CO2.sel(config=3) ) # nothing, error with parameters, ok.

    plt.plot( For.Eff.sum('reg_land').sel(config=1) )

    for VAR in ['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4']:
        plt.subplot( 5,1,1+['D_Eluc','D_Epf_CO2','D_Fland','D_Focean','D_Foxi_CH4'].index(VAR) )
        plt.title(VAR)
        for i_config in [0,1]:
            if VAR=='D_Epf_CO2':
                plt.plot( out_conc[VAR].sel(config=i_config).sum('reg_pf',min_count=1) , label=str(i_config) )
            else:
                plt.plot( out_conc[VAR].sel(config=i_config) , label=str(i_config) )
        plt.legend(loc=0)
        plt.grid()
     ## --> ocean!!

    ## list of those being problems
    plt.plot( Out.D_CO2 , 'k' )
    # list_negFocean = np.where( (out_conc.D_Focean.diff('year').isel(year=-1) < -10)  |  (out_conc.D_Focean.isel(year=-1) < -100) )[0]
    # list_negFocean = np.where(   (np.sign(out_conc.D_Focean.diff('year').isel(year=-1)) != np.sign(out_conc.D_Focean.diff('year').isel(year=-2)))   |   (out_conc.D_Focean.isel(year=-1) < -100)   )[0]
    diverged = np.where(   (out_conc.D_Focean.diff('year').isel(year=-1) * out_conc.D_Focean.diff('year').isel(year=-2) <= -1.)     )[0]
    tooNeg = np.where(   (out_conc.D_Focean.isel(year=-1) < -100)     )[0]
    list_negFocean = list(set(list(diverged)+list(tooNeg)))
    # list_negFocean = tooNeg
    plt.plot( Out.D_CO2.sel(config=list_negFocean) , 'r' ) # ok for selection

    plt.plot( out_conc.year, out_conc.D_Focean.transpose() , 'k' )
    plt.plot( out_conc.year, out_conc.D_Focean.sel(config=list_negFocean).transpose() , 'r' ) # ok for selection
    # plt.plot( out_conc.year, out_conc.D_Focean.sel(config=[ii for ii in out_conc.config.values if ii not in list_negFocean]).transpose() , 'r' ) # ok for selection

    # slection imperfect... but already not bad.



    ## trying identifying simple pattern
    list_par_ocean = ['t_circ','p_circ','v_fg','mld_0' , 'p_mld' , 'g_mld' , 'pCO2_is_Pade', 'To_0','A_ocean','a_dic']
    for VAR in list_par_ocean: # v_fg, p_circ?
        plt.figure(VAR)
        if 'box_osurf' in Par[VAR].dims:
            plt.hist( Par[VAR].sel(box_osurf=3) , label='unrestrained')
            plt.hist( Par[VAR].sel(box_osurf=3,config=list_negFocean) , label='restrained to aberrant D_Focean')
        else:
            plt.hist( Par[VAR] , label='unrestrained')
            plt.hist( Par[VAR].sel(config=list_negFocean) , label='restrained to aberrant D_Focean')
        plt.legend(loc=0)
        plt.grid()

    ## trying identifying distributions
    from core_fct.fct_loadP import load_all_param
    Par0 = load_all_param(mod_region)
    ## initiliazing counter
    distrib = xr.Dataset()
    for VAR in list_par_ocean:
        for cc in Par0[VAR].coords:
            if (cc[:len('mod_')]=='mod_')  and  cc not in distrib.coords:
                distrib.coords[cc] = Par0[cc]
    distrib['counter'] = xr.DataArray( np.full( fill_value=0. , shape=[distrib[cc].size for cc in distrib.coords]), dims=[cc for cc in distrib.coords] )
    ## filling in counter
    for i_config in list_negFocean:
        ## initializing dictionary with names chosen for this config
        dico_config = {}
        ## looping on dimensions
        for cc in distrib.coords:
            ## finding a parameter that has this 'mod_' as input
            ind_VAR = 0
            while cc not in Par0[list_par_ocean[ind_VAR]].coords:
                ind_VAR += 1
            ## identifying which value for 'mod_' has been used in Par
            ## due to numerical precision, values are slightly changed after afectation, then cannot do 'where(==)', instead using argmin
            if 'box_osurf' in Par[list_par_ocean[ind_VAR]].coords:
                ii = np.argmin( np.abs(Par[list_par_ocean[ind_VAR]].isel(config=i_config,box_osurf=2)  -  Par0[list_par_ocean[ind_VAR]].sel(box_osurf=2)) )
            else:
                ii = np.argmin( np.abs(Par[list_par_ocean[ind_VAR]].isel(config=i_config)  -  Par0[list_par_ocean[ind_VAR]]) )
            dico_config[cc] = Par0.coords[cc].values[ii]
        ## incrementing counter
        distrib['counter'].loc[dico_config] += 1

    ####
    ## evaluating counter
    distrib.mod_Focean_struct
    ## HILDA, Princeton-3D: ok
    distrib.counter.sel(mod_Focean_struct='Princeton-2D')
    ## mean_CMIP5: ok
    distrib.counter.sel(mod_Focean_trans='MPI-ESM-LR')

    distrib.counter.sel(mod_Focean_struct=['box-diffusion','Princeton-2D'],mod_Focean_trans=['CESM1-BGC', 'IPSL-CM5A-LR', 'MPI-ESM-LR'] ).sel(mod_Focean_chem='CO2Sys-Pade') # 92 values
    ## vs
    distrib.counter.sel(mod_Focean_struct=['box-diffusion','Princeton-2D'],mod_Focean_trans=['CESM1-BGC', 'IPSL-CM5A-LR', 'MPI-ESM-LR'] ).sel(mod_Focean_chem='CO2Sys-Power') # 275 values
    ## 3 times more values in CO2Sys-Power than in CO2Sys-Pade


