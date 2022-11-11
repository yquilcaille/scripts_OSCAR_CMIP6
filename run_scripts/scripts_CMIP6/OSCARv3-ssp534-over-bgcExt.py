import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from core_fct.fct_loadD import load_emissions_scen,load_landuse_scen,load_RFdrivers_scen  # bug with 'load_all_scen'
from core_fct.fct_process import OSCAR
from core_fct.fct_misc import aggreg_region

##################################################
## 1. OPTIONS
##################################################
## options for CMIP6
mod_region = 'RCP_5reg'
folder = 'CMIP6_v3.0'
type_LCC = 'gross'                        # gross  |  net
nt_run = 4

## options for this experiment
name_experiment = 'ssp534-over-bgcExt'
SOURCE_EXTENSION = 'Meinshausen2019'           # ONeill2017   | Meinshausen2019
year_PI = 1850
year_start = 2100
year_end = 2500
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
## Using initialization from last year of ssp534-over-bgc
out_init = xr.open_dataset('results/'+folder+'/ssp534-over-bgc_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################




##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'ssp534-over-bgcExt':
## - Concentrations for CO2, CH4, N2O and halo: from 'concentrations_ScenarioMIP', SSP5-3.4-OS
##      - only the carbon cycle responds to the increase in atm CO2
##      - the RF CO2 DOES NOT respond to the increase in atm CO2
## *********************************************************
## If the source for the extensions is ONeill et al, 2017:
## - Emissions used from the extensions described in its update, presented in the poster of Meinshausen at ScenarioForum2019
##      - CO2 emissions from FF&I remain constant over 2100-2140, then increase linearly to 0 over 2140-2190
##      - here, CO2 emissions from LU are not prescribed directly, LU is prescribed instead.
##      - non-CO2 emissions from LU AND FF&I stay constant after 2100
##      (FF, N2O, CH4 and Xhalo: actually not used, but here to allow the computation to run)
## *********************************************************
## If the source for the extensions is Meinshausen et al, 2019:
## - Emissions used from the extensions described in its update, presented in the poster of Meinshausen at ScenarioForum2019
##      - CO2 emissions from FF&I remain constant over 2100-2140, then increase linearly to 0 over 2140-2170 (not 2190)
##      - here, CO2 emissions from LU is not prescribed directly, LU is prescribed instead.
##      - non-CO2 emissions from FF&I decrease linearly to  0 over 2100-2250
##      - non-CO2 emissions from LU stay constant after 2100
##      (FF, N2O, CH4 and Xhalo: actually not used, but here to allow the computation to run)
nonCO2_ignored_sectors = ['Forest Burning', 'Grassland Burning', 'Peat Burning']
sectors_LU = ['Agricultural Waste Burning','Agriculture']
## *********************************************************
## - LULCC from LUH2, extension for SSP5-3.4 (NB: not called '-OS')
## - RF for solar and volcanoes as average of the reference scenario for CMIP6 over 1850-2014
## - RF for contrails: 0

## Loading all drivers, with correct regional/sectoral aggregation
For0E = load_emissions_scen(mod_region,datasets=['ScenarioMIP'])
For0L = load_landuse_scen(mod_region,datasets=['LUH2'],LCC=type_LCC)
For0R = load_RFdrivers_scen()

## Using dataset from forcings to prepare those from this experiment
for_runs_hist = xr.open_dataset('results/'+folder+'/historical_For-'+str(setMC)+'.nc')

## Using dataset from forcings to prepare those from this experiment
for_runs_scen = xr.open_dataset('results/'+folder+'/ssp534-over-bgc_For-'+str(setMC)+'.nc')

## Preparing dataset
For = xr.Dataset()
for cc in for_runs_scen.coords:
    For.coords[cc] = for_runs_scen[cc]
## Correcting years
For.coords['year'] = np.arange(year_start,year_end+1)

## Preparing variables
for var in for_runs_scen.variables:
    if var not in for_runs_scen.coords:
        For[var] = xr.DataArray( np.full(fill_value=np.nan , shape=[year_end-year_start+1]+list(for_runs_scen[var].shape[1:])), dims=['year']+list(for_runs_scen[var].dims[1:]) )
        For[var].loc[dict(year=year_start)] = for_runs_scen[var].sel(year=year_start)

## Concentrations
with xr.open_dataset('input_data/drivers/concentrations_ScenarioMIP.nc') as TMP:
    ## CO2, CH4, N2O and Xhalo
    for var in ['CO2','CH4','N2O','Xhalo']:
        val = TMP[var].loc[{'year':np.arange(year_start+1,year_end+1),'scen':'SSP5-3.4-OS'}] - Par[var+'_0']
        if 'spc_halo' in val.dims:
            For['D_'+var].loc[{'year':np.arange(year_start+1,year_end+1),'spc_halo':val.spc_halo.values}] = val
        else:
            For['D_'+var].loc[{'year':np.arange(year_start+1,year_end+1)}] = val

## Emissions
TMP = xr.open_dataset('input_data/drivers/emissions_ScenarioMIP.nc')
TMP = TMP.interp({'year':np.arange(int(TMP.year[0]), int(TMP.year[-1])+1, 1)})
TMP = aggreg_region(TMP, mod_region, old_axis='region', dataset='ScenarioMIP')
if SOURCE_EXTENSION=='ONeill2017': # O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016)
    print("******************************")
    print("!WARNING!")
    print("The extensions of the emissions of this scenario are those described in O'Neill et al, 2017, NOT the update in the poster of Malte Meinshausen in the ScenarioForum2019.")
    print("******************************")
    for var in ['E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2'] + ['E_CH4']:
        ## non-CO2 emissions from LU AND FF&I stay constant after 2100
        cst_emi = TMP[var].loc[dict(scen='SSP5-3.4-OS',year=2100,sector=[sec for sec in TMP.sector.values if sec not in nonCO2_ignored_sectors])].sum('sector')
        For[var] = For[var].fillna( cst_emi )
        For[var].attrs['warning'] = "Warning: the extensions for these emissions are the ones presented in O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), not those described in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
    ## N2O is aggregated in a single sector in the database.
    cst_emi = TMP['E_N2O'].loc[dict(scen='SSP5-3.4-OS',year=2100)]
    For['E_N2O'] = For[var].fillna( cst_emi )
    For['E_N2O'].attrs['warning'] = "Warning: the extensions for these emissions are the ones presented in O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), not those described in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
    For = For.drop('scen')
    ## CO2 emissions from FF&I remain constant over 2100-2140, then increase linearly to 0 over 2140-2190
    if TMP['Eff'].loc[dict(scen='SSP5-3.4-OS',year=2100,sector=[sec for sec in TMP.sector.values if sec in sectors_LU])].sum() > 0.: raise Exception("Check definition of the sectors.")
    todec_emiFF = TMP['Eff'].loc[dict(scen='SSP5-3.4-OS',year=2100,sector=[sec for sec in TMP.sector.values if sec not in nonCO2_ignored_sectors])].sum('sector')
    For['Eff'].loc[dict(year=np.arange(year_start+1,2140+1))] = todec_emiFF # constant over 2100-2140
    For['Eff'].loc[dict(year=np.arange(2140+1,2190+1))] = np.linspace(todec_emiFF,xr.DataArray(np.zeros(shape=todec_emiFF.shape),dims=todec_emiFF.dims),2190-2140+1)[1:]  # linear ramp over 2140-2190
    For['Eff'].loc[dict(year=np.arange(2190+1,year_end+1))] = 0.  # 0. afterwards
    For['Eff'].attrs['warning'] = "Warning: the extensions for these emissions are the ones presented in O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), not those described in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
elif SOURCE_EXTENSION=='Meinshausen2019': # (update in poster of Meinshausen at ScenarioForum2019)
    print("******************************")
    print("!WARNING!")
    print("The extensions of the emissions of this scenario are not those described in O'Neill et al, 2017, we use instead the update in the poster of Malte Meinshausen in the ScenarioForum2019.")
    print("******************************")
    for var in ['E_BC','E_CO','E_NH3','E_VOC','E_NOX','E_OC','E_SO2'] + ['E_CH4']:
        ## non-CO2 emissions from LU stay constant after 2100
        cst_emiLU = TMP[var].loc[dict(scen='SSP5-3.4-OS',year=2100,sector=sectors_LU)].sum('sector')
        For[var] = For[var].fillna( cst_emiLU )
        ## non-CO2 emissions from FF&I decrease linearly to  0 over 2100-2250
        todec_emiFF = TMP[var].loc[dict(scen='SSP5-3.4-OS',year=2100,sector=[sec for sec in TMP.sector.values if sec not in nonCO2_ignored_sectors+sectors_LU])].sum('sector')
        For[var].loc[dict(year=np.arange(year_start+1,2250+1))] += np.linspace(todec_emiFF,xr.DataArray( np.zeros(shape=todec_emiFF.shape), dims=todec_emiFF.dims ),2250-2100+1)[1:]  # linear ramp over 2100-2250 years
        For[var].attrs['warning'] = "Warning: the extensions for these emissions are not those of O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), but the ones presented in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
    ## N2O is aggregated in a single sector in the database. However, it will not hamper the run which is driven in concentration of N2O.
    todec_emiall = TMP['E_N2O'].loc[dict(scen='SSP5-3.4-OS',year=2100)]
    For['E_N2O'].loc[dict(year=np.arange(year_start+1,2250+1))] += np.linspace(todec_emiall,xr.DataArray( np.zeros(shape=todec_emiall.shape), dims=todec_emiall.dims ),2250-2100+1)[1:]  # linear ramp over 2100-2250 years
    For['E_N2O'].attrs['warning'] = "Warning: the extensions for these emissions are not those of O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), but the ones presented in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
    For['E_N2O'].attrs['warning n2'] = "The emissions used for N2O are the harmonized ones available on https://tntcat.iiasa.ac.at/SspDb/dsd, for which no sectoral detail is available. Yet, the experiment is concentrations-driven in N2O."
    For = For.drop('scen')
    ## CO2 emissions from FF&I remain constant over 2100-2140, then increase linearly to 0 over 2140-2170
    if TMP['Eff'].loc[dict(scen='SSP5-3.4-OS',year=2100,sector=[sec for sec in TMP.sector.values if sec in sectors_LU])].sum() > 0.: raise Exception("Check definition of the sectors.")
    todec_emiFF = TMP['Eff'].loc[dict(scen='SSP5-3.4-OS',year=2100,sector=[sec for sec in TMP.sector.values if sec not in nonCO2_ignored_sectors])].sum('sector')
    dico_dates_end_plateau = {'SSP1-1.9':2140, 'SSP1-2.6':2140, 'SSP2-4.5':2100, 'SSP3-7.0':2100, 'SSP3-7.0-LowNTCF':2100, 'SSP4-3.4':2140, 'SSP4-6.0':2100, 'SSP5-3.4-OS':2140, 'SSP5-8.5':2100}
    dico_dates_end_ramp = { 'SSP1-1.9':2190, 'SSP1-2.6':2190, 'SSP2-4.5':2250, 'SSP3-7.0':2250, 'SSP3-7.0-LowNTCF':2250, 'SSP4-3.4':2190, 'SSP4-6.0':2250, 'SSP5-3.4-OS':2170, 'SSP5-8.5':2250 }
    For['Eff'].loc[dict(year=range(year_start,dico_dates_end_plateau['SSP5-3.4-OS']+1))] = todec_emiFF # constant over 2100-2140
    For['Eff'].loc[dict(year=np.arange(dico_dates_end_plateau['SSP5-3.4-OS'],dico_dates_end_ramp['SSP5-3.4-OS']+1))] = np.linspace( todec_emiFF , 0.*todec_emiFF , dico_dates_end_ramp['SSP5-3.4-OS']-dico_dates_end_plateau['SSP5-3.4-OS']+1)  # linear ramp over 2140-2190
    For['Eff'].loc[dict(year=np.arange(dico_dates_end_ramp['SSP5-3.4-OS']+1,year_end+1))] = 0. # 0 afterwards
    For['Eff'].attrs['warning'] = "Warning: the extensions for these emissions are not those of O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), but the ones presented in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
## No Xhalo in the downloaded database. However, it will not hamper the run which is driven in concentration of Xhalo.
For['E_Xhalo'].loc[dict(year=np.arange(year_start+1,year_end+1))] = xr.DataArray( np.zeros((year_end-year_start+1-1,For.reg_land.size,For.spc_halo.size)), dims=('year','reg_land','spc_halo') )
For['E_Xhalo'].attrs['warning'] = "No relevant emissions for halogenated compounds are available on https://tntcat.iiasa.ac.at/SspDb/dsd. Yet, the experiment is concentrations-driven in halogenated compounds."

## Land-Use: LUH2 provides values over 2101-2299: taking 2300 as 2299
for var in ['d_Hwood','d_Ashift','d_Acover']:
    For[var].loc[{'year':np.arange(year_start,2299+1)}] = For0L[var].loc[{'year':np.arange(year_start,2299+1),'scen_LULCC':'SSP5-3.4'}]
    For[var].loc[dict(year=2300)] = For[var].sel(year=2299) # 2300 as 2299
    if var in ['d_Hwood','d_Ashift']:
        For[var].loc[{'year':np.arange(2301,year_end+1)}] = For[var].sel(year=2300) # constant extension
    else:
        For[var].loc[{'year':np.arange(2301,year_end+1)}] = 0. # LUC frozen

## RF volc
For['RF_volc'].loc[{'year':np.arange(year_start+1,year_end+1)}] = for_runs_scen['RF_volc'].isel(year=-1)

## RF solar
For['RF_solar'].loc[dict(year=np.arange(year_start+1,2299+1))] = For0R.RF_solar.sel(scen_RF_solar='CMIP6',year=np.arange(year_start+1,2299+1))  *  for_runs_scen.RF_solar.sel(year=np.arange(year_start-11+1,year_start+1)).mean('year') / For0R.RF_solar.sel(scen_RF_solar='CMIP6',year=np.arange(year_start-11+1,year_start+1)).mean('year') # rescale on 2089-2100, equivalent to historical 2003-2014
For['RF_solar'].loc[dict(year=np.arange(2300,year_end+1))] = For['RF_solar'].sel(year=np.arange(2299-11+1,2299+1)).mean()

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
## Updating parameters so that a constant atm CO2 will be prescribed to the RF CO2
Par['D_CO2_rad'] = for_runs_hist.D_CO2.sel(year=year_PI)
Out = OSCAR(Ini, Par, For , nt=nt_run)
if SOURCE_EXTENSION=='ONeill2017': # O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016)
    Out.attrs['warning'] = "Warning: the extensions for the emissions are the ones presented in O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), not those described in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
elif SOURCE_EXTENSION=='Meinshausen2019': # (update in poster of Meinshausen at ScenarioForum2019)
    Out.attrs['warning'] = "Warning: the extensions for the emissions are not those of O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), but the ones presented in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
Out.to_netcdf('results/'+folder+'/'+name_experiment+'_Out-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out})
print("Experiment "+name_experiment+" done")
##################################################
##################################################
