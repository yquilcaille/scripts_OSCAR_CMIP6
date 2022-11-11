import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import csv 

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
name_experiment = 'ssp370-lowNTCFext-gidden'
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
## Using initialization from last year of ssp370
out_init = xr.open_dataset('results/'+folder+'/ssp370-lowNTCF_Out-'+str(setMC)+'.nc' )
Ini = out_init.isel(year=-1, drop=True).drop([VAR for VAR in out_init if VAR not in list(OSCAR.var_prog)])
print("Initialization done")
##################################################
##################################################



##################################################
## 4. DRIVERS / FORCINGS
##################################################
## Forcings for 'ssp370-lowNTCFext':
## - Concentrations for CO2, CH4, N2O and halo: from 'concentrations_ScenarioMIP', SSP3-7.0-lowNTCFext
## *********************************************************
## If the source for the extensions is ONeill et al, 2017:
## - Emissions used from the extensions described in its update, presented in the poster of Meinshausen at ScenarioForum2019
##      - CO2 emissions from FF&I remain constant over 2100-2140, then increase linearly to 0 over 2140-2185
##      - CO2 emissions from LU is not prescribed, and its extension not explicitely defined...
##      - non-CO2 emissions from LU AND FF&I stay constant after 2100
##      (FF, N2O, CH4 and Xhalo: actually not used, but here to allow the computation to run)
## *********************************************************
## If the source for the extensions is Meinshausen et al, 2019:
## - Emissions used from the extensions described in its update, presented in the poster of Meinshausen at ScenarioForum2019
##      - CO2 emissions from FF&I remain constant over 2100-2140, then increase linearly to 0 over 2140-2190
##      - CO2 emissions from LU decrease linearly to 0 over 2100-2150
##      - non-CO2 emissions from FF&I decrease linearly to  0 over 2100-2250
##      - non-CO2 emissions from LU stay constant after 2100
##      (FF, N2O, CH4 and Xhalo: actually not used, but here to allow the computation to run)
nonCO2_ignored_sectors = ['Forest Burning', 'Grassland Burning', 'Peat Burning']
sectors_LU = ['Agricultural Waste Burning','Agriculture']
## *********************************************************
## - LULCC from LUH2, extension for SSP3-7.0
## - RF for solar and volcanoes as reference scenario for CMIP6
## - RF for contrails: 0

## Loading all drivers, with correct regional/sectoral aggregation
For0E = load_emissions_scen(mod_region,datasets=['ScenarioMIP'])
For0L = load_landuse_scen(mod_region,datasets=['LUH2'],LCC=type_LCC)
For0R = load_RFdrivers_scen()

## Using dataset from forcings to prepare those from this experiment
for_runs_scen = xr.open_dataset('results/'+folder+'/ssp370-lowNTCF_For-'+str(setMC)+'.nc')

## Preparing dataset
For = xr.Dataset()
for cc in for_runs_scen.coords:
    For.coords[cc] = for_runs_scen[cc]
## Correcting years
For.coords['year'] = np.arange(year_start,year_end+1)

## Preparing variables
for var in for_runs_scen.variables:
    if var not in for_runs_scen.coords:
        For[var] = xr.DataArray( np.full(fill_value=0. , shape=[year_end-year_start+1]+list(for_runs_scen[var].shape[1:])), dims=['year']+list(for_runs_scen[var].dims[1:]) )
        For[var].loc[dict(year=year_start)] = for_runs_scen[var].sel(year=year_start)

dico_spc_halo = {'C2F6':'C2F6', 'C3F8':'C3F8', 'C4F10':'C4F10', 'C5F12':'C5F12', 'C6F14':'C6F14', 'C7F16':'C7F16', 'CCl4':'CCl4', 'CF4':'CF4',\
                'CFC11':'CFC-11', 'CFC113':'CFC-113', 'CFC114':'CFC-114', 'CFC115':'CFC-115', 'CFC12':'CFC-12', 'CH3Br':'CH3Br','CH3CCl3':'CH3CCl3',\
                'CH3Cl':'CH3Cl', 'HCFC141b':'HCFC-141b', 'HCFC142b':'HCFC-142b', 'HCFC22':'HCFC-22', 'HFC125':'HFC-125', 'HFC134a':'HFC-134a',\
                'HFC143a':'HFC-143a', 'HFC152a':'HFC-152a', 'HFC227ea':'HFC-227ea', 'HFC23':'HFC-23', 'HFC236fa':'HFC-236fa', 'HFC245fa':'HFC-245fa',\
                'HFC32':'HFC-32', 'HFC365mfc':'HFC-365mfc', 'HFC4310mee':'HFC-43-10mee', 'Halon1202':'Halon-1202', 'Halon1211':'Halon-1211',\
                'Halon1301':'Halon-1301', 'Halon2402':'Halon-2402', 'NF3':'NF3', 'SF6':'SF6', 'cC4F8':'c-C4F8'}

## Concentrations
with open('extra_data/RCMIP/rcmip-concentrations-annual-means-v2-0-0.csv','r',newline='') as ff:
    TMP_RCMIP = np.array([line for line in csv.reader(ff)])
    head = list(TMP_RCMIP[0,:])
for ii in np.where( (TMP_RCMIP[:,head.index('Scenario')]=='ssp370-lowNTCF') & (TMP_RCMIP[:,head.index('Region')]=='World') )[0]:
    cp , val = str.split(TMP_RCMIP[ii,head.index('Variable')],'|')[-1] , np.array( TMP_RCMIP[ii,head.index(str(year_start+1)):head.index(str(year_end))+1] , dtype=np.float32)
    if cp in ['CH4','N2O','CO2']:
        For['D_'+cp].loc[{'year':np.arange(year_start+1,year_end+1)}] = val - Par[cp+'_0'].values
    elif cp in ['C8F18', 'SO2F2','CH2Cl2','CHCl3']:## not in OSCAR
        pass
    elif cp in dico_spc_halo.keys():
        For['D_Xhalo'].loc[{'year':np.arange(year_start+1,year_end+1),'spc_halo':dico_spc_halo[cp]}] = val - Par['Xhalo_0'].sel(spc_halo=dico_spc_halo[cp]).values
    else:
        raise Exception("Variable not recognized")

## Emissions
dico_regions = {'World':0,'World|R5.2ASIA':1,'World|R5.2LAM':2,'World|R5.2MAF':3,'World|R5.2OECD':4,'World|R5.2REF':5} # will have special case for 'World'
dico_emi_names = {'VOC':'E_VOC','Sulfur':'E_SO2','OC':'E_OC','NOx':'E_NOX','NH3':'E_NH3','N2O':'E_N2O','CO':'E_CO','CO2':'Eff','CH4':'E_CH4','BC':'E_BC'}
dico_factor_units = {'E_BC':1,'E_CO':12./28,'E_N2O':(28./44.)*1.e-3,'E_NH3':14/17.,'E_NOX':14/46.,'E_OC':1.,'E_SO2':32./64,'E_VOC':1.,'Eff':(12/44.)*1.e-3,'E_CH4':12/16.}
with open('extra_data/RCMIP/rcmip-emissions-annual-means-v2-0-0.csv','r',newline='') as ff:
    TMP_RCMIP = np.array([line for line in csv.reader(ff)])
    head = list(TMP_RCMIP[0,:])
for ii in np.where( (TMP_RCMIP[:,head.index('Scenario')]=='ssp370-lowNTCF') )[0]:
    sector = TMP_RCMIP[ii,head.index('Variable')][len('Emissions|'+str.split(TMP_RCMIP[ii,head.index('Variable')],'|')[1]+'|'):]
    if str.split(TMP_RCMIP[ii,head.index('Variable')],'|')[1] in ['F-Gases','Montreal Gases']:
        cp = str.split(TMP_RCMIP[ii,head.index('Variable')],'|')[-1]
    else:
        cp = dico_emi_names[str.split(TMP_RCMIP[ii,head.index('Variable')],'|')[1]]
    if cp in ['E_BC','E_CH4','E_CO','Eff','E_NH3','E_NOX','E_OC','E_SO2','E_VOC']:
        list_sectors_kept = ['MAGICC AFOLU|Agricultural Waste Burning', 'MAGICC AFOLU|Agriculture', 'MAGICC Fossil and Industrial']
    elif cp in ['E_N2O']:
        list_sectors_kept = ['MAGICC Fossil and Industrial','MAGICC AFOLU']
    else:## halogenated
        pass
    if (sector in list_sectors_kept)  or  (cp not in ['E_BC','E_CH4','E_CO','Eff','E_NH3','E_NOX','E_OC','E_SO2','E_VOC']+['E_N2O']):
        val0 = TMP_RCMIP[ii,head.index('1750'):head.index('2500')+1]
        val0 = np.interp(x=np.arange(1750,2500+1) , xp=np.array( TMP_RCMIP[0,head.index('1750')+np.where(val0!='')[0]],dtype=np.float32 ) , fp=np.array( val0[np.where(val0!='')[0]],dtype=np.float32 ) )
        val = val0[head.index(str(year_start+1))-head.index(str(1750)):head.index(str(year_end))-head.index(str(1750))+1]
        if cp in dico_spc_halo.keys():
            For['E_Xhalo'].loc[{'year':np.arange(year_start+1,year_end+1),'spc_halo':dico_spc_halo[cp],'reg_land':dico_regions[TMP_RCMIP[ii,head.index('Region')]]}] += val*1.
            if TMP_RCMIP[ii,head.index('Region')] != 'World': ## converting from World to Unknown
                For['E_Xhalo'].loc[{'year':np.arange(year_start+1,year_end+1),'spc_halo':dico_spc_halo[cp],'reg_land':0}] -= val*1.
        elif cp in ['C8F18', 'SO2F2','CH2Cl2','CHCl3']:## not in OSCAR
            pass
        else:
            For[cp].loc[{'year':np.arange(year_start+1,year_end+1),'reg_land':dico_regions[TMP_RCMIP[ii,head.index('Region')]]}] += val*dico_factor_units[cp]
            if TMP_RCMIP[ii,head.index('Region')] != 'World': ## converting from World to Unknown
                For[cp].loc[{'year':np.arange(year_start+1,year_end+1),'reg_land':0}] -= val*dico_factor_units[cp]
                
## Land-Use:
for var in ['d_Hwood','d_Ashift',]:
    For[var].loc[{'year':np.arange(year_start,year_end+1)}] = For0L[var].loc[{'year':2099,'scen_LULCC':'SSP3-7.0'}] # constant extension
For['d_Acover'].loc[{'year':np.arange(year_start,year_end+1)}] = 0. # LUC frozen

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
Out = OSCAR(Ini, Par, For , nt=nt_run)
if SOURCE_EXTENSION=='ONeill2017': # O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016)
    Out.attrs['warning'] = "Warning: the extensions for the emissions are the ones presented in O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), not those described in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
elif SOURCE_EXTENSION=='Meinshausen2019': # (update in poster of Meinshausen at ScenarioForum2019)
    Out.attrs['warning'] = "Warning: the extensions for the emissions are not those of O'Neill et al, 2017 (doi:10.5194/gmd-9-3461-2016), but the ones presented in Malte Meinshausen's poster at ScenarioForum2019 ('The xWG scenario process, dimensions of integration, extensions and colorcodes.')."
Out.to_netcdf('results/'+folder+'/'+name_experiment+'_Out-'+str(setMC)+'.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out})
print("Experiment "+name_experiment+" done")
##################################################
##################################################
