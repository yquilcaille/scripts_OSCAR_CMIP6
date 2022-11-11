import os
import sys
import time
sys.path.append("H:\MyDocuments\Repositories\OSCARv31_CMIP6") ## line required for run on server ebro
time0 = time.process_time()



## Running all experiments CMIP6, taking into account dependencies




##################################################
## 0. OPTIONS
##################################################
## Options for this script
TO_RUN = ['CMIP5']            # list of blocks to run. If 'ALL', all blocks will be run. Otherwise, elements must be strings in: 'BASIS', 'SCENARIOS', 'ESM-SCENARIOS', 'BGC', 'IDEALIZED', 'VAR_HISTSCEN', 'CDRMIP', 'STAND_ALONE', 'CMIP5', 'LAND'
OPTION_OVERWRITE = False    # if True, will run experiment even outputs have already been saved. Otherwise, will not.
OPTION_PLOT = False

## Options that will be forced in each experiment, for homogeneity among all experiments. 
## List of possible options to force: setMC, folder, mod_region, type_LCC, nt_run, nMC.
## The options that are not written will not be forced, and the values of every script will be used locally.
## Two arguments MUST be written in this dictionnary to run this script:   setMC   and   folder

forced_options_CMIP6 = {
    'setMC':19,
    'folder':'CMIP6_v3.1',
    'mod_region':'RCP_5reg',
    'type_LCC':'gross',
    'nt_run':4,
    'nMC':500,
}
##################################################
##################################################






##################################################
## 1. BASIS
##################################################
if OPTION_OVERWRITE:
    print("Will overwrite previous files!")
if ('ALL' in TO_RUN)  or  ('BASIS' in TO_RUN):
    ## These experiment are the absolute basis for all other experiments. Have to be run before any other.
    for MDL in ['spinup','esm-spinup']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")

    ## These experiments are used as basis/init in several blocks (see below). Has to be run them before the blocks.
    for MDL in ['historical'] + ['hist-bgc'] + ['esm-hist']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")
##################################################
##################################################





##################################################
## 2. BLOCKS
##################################################
## Each "block" corresponds to experiments that have dependencies. Each block may be run independently, but the order of experiment inside of each list in each block must not changed.

## Block scenarios and extensions
if ('ALL' in TO_RUN)  or  ('SCENARIOS' in TO_RUN):
    for MDL in ['ssp119','ssp119ext'] + ['ssp126','ssp126ext'] + ['ssp126-ssp370Lu'] + ['ssp245','ssp585','ssp585ext','G6solar','ssp245ext'] + ['ssp370','ssp370ext'] + ['ssp370-lowNTCF','ssp370-lowNTCFext'] + ['ssp370-lowNTCF-gidden','ssp370-lowNTCFext-gidden'] + ['ssp370-ssp126Lu'] + ['ssp434','ssp434ext'] + ['ssp460','ssp460ext'] + ['ssp534-over','ssp534-over-ext'] + ['ssp585-ssp126Lu']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")

## Block ESM-scenarios and extensions
if ('ALL' in TO_RUN)  or  ('ESM-SCENARIOS' in TO_RUN):
    for MDL in ['esm-ssp585','esm-ssp585ext'] + ['esm-ssp534-over','esm-ssp534-over-ext'] + ['esm-ssp585-ssp126Lu','esm-ssp585-ssp126Lu-ext'] + ['esm-ssp119','esm-ssp119ext'] + ['esm-ssp126','esm-ssp126ext'] + ['esm-ssp245','esm-ssp245ext'] + ['esm-ssp370','esm-ssp370ext'] + ['esm-ssp370-lowNTCF','esm-ssp370-lowNTCFext'] + ['esm-ssp370-lowNTCF-gidden','esm-ssp370-lowNTCFext-gidden'] + ['esm-ssp434','esm-ssp434ext'] + ['esm-ssp460','esm-ssp460ext']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")


## Block idealized
if ('ALL' in TO_RUN)  or  ('IDEALIZED' in TO_RUN):
    for MDL in ['1pctCO2','esm-1pctCO2','esm-1pct-brch-1000PgC','esm-1pct-brch-750PgC','esm-1pct-brch-2000PgC']+['esm-pi-cdr-pulse']+['esm-pi-CO2pulse']+['1pctCO2-bgc']+['1pctCO2-rad']+['1pctCO2-cdr']+['1pctCO2-4xext']+['abrupt-4xCO2','esm-abrupt-4xCO2']+['abrupt-2xCO2']+['abrupt-0p5xCO2']+['G1']+['G2']+['esm-bell-1000PgC']+['esm-bell-750PgC']+['esm-bell-2000PgC']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")

## Block BGC
if ('ALL' in TO_RUN)  or  ('BGC' in TO_RUN):
    for MDL in ['ssp534-over-bgc','ssp534-over-bgcExt']+['ssp585-bgc','ssp585-bgcExt']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")

## Block variations from historical and scenarios
if ('ALL' in TO_RUN)  or  ('VAR_HISTSCEN' in TO_RUN):
    for MDL in ['hist-noLu'] + ['hist-aer','ssp245-aer'] + ['hist-CO2','ssp245-CO2'] + ['hist-GHG','ssp245-GHG'] + ['hist-nat','ssp245-nat'] + ['hist-sol','ssp245-sol'] + ['hist-stratO3','ssp245-stratO3'] + ['hist-volc','ssp245-volc']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")

## Block of stand-alones experiments
if ('ALL' in TO_RUN)  or  ('STAND_ALONE' in TO_RUN):
    for MDL in ['piControl'] + ['esm-piControl'] + ['hist-1950HC'] + ['hist-piAer'] + ['hist-piNTCF']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")

## Block CDR-MIP
if ('ALL' in TO_RUN)  or  ('CDRMIP' in TO_RUN):
    for MDL in ['yr2010CO2','esm-yr2010CO2-control','esm-yr2010CO2-noemit','esm-yr2010CO2-cdr-pulse','esm-yr2010CO2-CO2pulse']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")

## Block CMIP5
if ('ALL' in TO_RUN)  or  ('CMIP5' in TO_RUN):
    for MDL in ['spinup-CMIP5','historical-CMIP5','rcp26','rcp45','rcp60','rcp85','piControl-CMIP5'] + ['esm-spinup-CMIP5','esm-histcmip5','esm-rcp26','esm-rcp45','esm-rcp60','esm-rcp85','esm-piControl-CMIP5']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")


## Block LAND
if ('ALL' in TO_RUN)  or  ('LAND' in TO_RUN):
    for MDL in ['land-spinup','land-hist','land-piControl','land-cClim','land-cCO2','land-noLu','land-noWoodHarv','land-noShiftcultivate','land-crop-grass'] + ['land-spinup-altStartYear','land-hist-altStartYear','land-piControl-altStartYear'] + ['land-spinup-altLu1','land-hist-altLu1','land-piControl-altLu1'] + ['land-spinup-altLu2','land-hist-altLu2','land-piControl-altLu2']:
        if OPTION_OVERWRITE or os.path.isfile('results/'+forced_options_CMIP6['folder']+'/'+MDL+'_Out-'+str(forced_options_CMIP6['setMC'])+'.nc')==False:
            print("Running experiment "+MDL+" on set "+str(forced_options_CMIP6['setMC']))
            exec(open('run_scripts/scripts_CMIP6/OSCARv3-'+MDL+'.py').read())
            if OPTION_PLOT:
                plt.figure()
                plt.plot( Out.D_Tg.mean('config') )
                plt.title(MDL)
            del Out,For,Ini,Par,setMC,nMC,mod_region,folder,type_LCC,nt_run
        else:
            print(MDL+" already done.")
        print(" ")
##################################################
##################################################



print("########################################")
print("CPU time of this set:")
print( time.process_time() - time0 )
print("########################################")




