import numpy as np
import xarray as xr

from core_fct.fct_process import OSCAR



## this script loads outputs, calculates required variables and saves them in folder_extra
## NB: script working here, but has to be adapted it for parameters if add experiments -bgc, -rad, or in CMIP5/LAND blocks.


##################################################
##################################################

## mode: adding variables to existing files or new one?
add_only_variables = False

## info
folder_raw = 'results/CMIP6_v3.1/'
folder_extra = 'results/CMIP6_v3.1_extra/'

## number of sets
Nset = 20

## experiments
list_exp = ['piControl', '1pctCO2', 'abrupt-4xCO2', 'historical', 
            'ssp119', 'ssp119ext',
            'ssp126', 'ssp126ext',
            'ssp245', 'ssp245ext',
            'ssp370', 'ssp370ext',
            'ssp434', 'ssp434ext',
            'ssp460', 'ssp460ext',
            'ssp534-over', 'ssp534-over-ext',
            'ssp585', 'ssp585ext',
            'esm-piControl','esm-hist','abrupt-2xCO2'] + ['piControl-CMIP5', 'historical-CMIP5', 'rcp26', 'rcp45', 'rcp60', 'rcp85'] + ['abrupt-0p5xCO2'] + \
           ['esm-ssp119', 'esm-ssp119ext',
            'esm-ssp126', 'esm-ssp126ext',
            'esm-ssp245', 'esm-ssp245ext',
            'esm-ssp370', 'esm-ssp370ext',
            'esm-ssp434', 'esm-ssp434ext',
            'esm-ssp460', 'esm-ssp460ext',
            'esm-ssp534-over', 'esm-ssp534-over-ext',
            'esm-ssp585', 'esm-ssp585ext'] + ['1pctCO2-cdr', 'esm-1pctCO2', 'esm-1pct-brch-750PgC', 'esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-pi-CO2pulse', 'esm-pi-cdr-pulse'] + \
            ['land-hist-altStartYear', 'land-piControl-altStartYear']

list_exp = ['ssp126ext']

## variables
list_var = ['D_Focean', 'D_Fland', 'D_Eluc', 'D_Epf_CO2',  
    'D_Ewet', 'D_Ebb', 'D_Epf_CH4', 'D_Fsink_CH4',
    'D_Fsink_N2O', 'D_Fsink_Xhalo',
    'RF_warm', 'RF', 'RF_AERtot', 'RF_cloud2', 'RF_SO4', 'RF_BC',  'RF_O3t', 'RF_O3s', 'RF_lcc', 'RF_CO2', 'RF_CH4', 'RF_N2O', 'RF_Xhalo', 'RF_wmghg',
    'd_OHC'] + ['RF_alb', 'RF_BCsnow', 'tau_N2O', 'tau_CH4', 'RF_H2Os', 'd_CO2']

to_add = ['RF_lcc']

if add_only_variables:
    list_var = to_add
else:
    list_var = list_var

## sort list of variables to speed up
list_var_new = []
proc_levels = OSCAR.proc_levels()
for n in np.sort(list(proc_levels.keys())):
    for var in list_var:
        if var in proc_levels[n] and var not in list_var_new:
            list_var_new.append(var)
list_var = list_var_new


##################################################
##################################################

for exp in list_exp:
    for nset in range(Nset):
        print (exp + ' | ' + str(nset))

        ## checking for OSCAR_landC
        test_OSCAR_landC = exp in ['land-hist-altStartYear', 'land-piControl-altStartYear']

        ## load 
        with xr.open_dataset(folder_raw + 'Par-' + str(nset) + '.nc') as TMP: Par = TMP.load()
        with xr.open_dataset(folder_raw + exp + '_For-' + str(nset) + '.nc') as TMP: For = TMP.load()
        with xr.open_dataset(folder_raw + exp + '_Out-' + str(nset) + '.nc') as TMP: Out = TMP.load()

        ## adapting parameters: Aland_0
        if 'Aland_0' in For:Par['Aland_0'] = For['Aland_0']
        ## adapting parameters
        if exp in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc']:
            with xr.open_dataset(folder_raw + 'historical' + '_For-' + str(nset) + '.nc') as TMP: for_tmp = TMP.load()
            Par['D_CO2_rad'] = for_tmp.D_CO2.sel(year=1850)
        elif exp in ['1pctCO2-rad']:
            with xr.open_dataset(folder_raw + 'historical' + '_For-' + str(nset) + '.nc') as TMP: for_tmp = TMP.load()
            Par['D_CO2_bgc'] = for_tmp.D_CO2.sel(year=1850)

        if add_only_variables:
            ## load previous calculations
            with xr.open_dataset(folder_extra + exp + '_Out2-' + str(nset) + '.nc') as TMP: Out2 = TMP.load().copy(deep=True)
        else:
            Out2 = xr.Dataset()
        Out = For.update( Out ) # line added
        ## get extra variables
        for var in list_var:

            if (var not in Out2)   and   (var not in ['d_CO2']):
                print(var)
                Out2[var] = OSCAR[var](xr.merge([Out, Out2]), Par, For, recursive=True)

for var in Out2:
    if ('config' in Out2[var].dims) and (np.any(np.isnan( Out2[var].sel(config=396) ))):
        raise Exception

for var in Out:
    if ('config' in Out[var].dims) and (np.any(np.isnan( Out[var].sel(config=396) ))):
        raise Exception


        ## compatible emissions
        if (add_only_variables==False)  and  (test_OSCAR_landC==False):
            ## Xhalo
            Out2['E_Xhalo_comp'] = (Par.a_Xhalo * Out.D_Xhalo.differentiate('year')
                + Out2.D_Fsink_Xhalo)
            ## N2O
            Out2['E_N2O_comp'] = (Par.a_N2O * Out.D_N2O.differentiate('year') 
                - Out2.D_Ebb.sel({'spc_bb':'N2O'}, drop=True).sum('bio_land', min_count=1).sum('reg_land', min_count=1)
                + Out2.D_Fsink_N2O)
            ## CH4
            Out2['E_CH4_comp'] = (Par.a_CH4 * Out.D_CH4.differentiate('year') 
                - Out2.D_Ewet.sum('reg_land', min_count=1)
                - Out2.D_Ebb.sel({'spc_bb':'CH4'}, drop=True).sum('bio_land', min_count=1).sum('reg_land', min_count=1)
                - Out2.D_Epf_CH4.sum('reg_pf', min_count=1) 
                + Out2.D_Fsink_CH4)
            ## Foxi!
            Out2['D_Foxi_CH4'] = 1E-3 * (Par.p_CH4geo * Out2.E_CH4_comp
                + Out2.D_Epf_CH4.sum('reg_pf', min_count=1) 
                - Par.a_CH4 * Out.D_CH4.differentiate('year'))
            ## FF CO2
            Out2['Eff_comp'] = (Par.a_CO2 * Out.D_CO2.differentiate('year') 
                - Out2.D_Eluc
                - Out2.D_Epf_CO2.sum('reg_pf', min_count=1) 
                + Out2.D_Fland
                + Out2.D_Focean
                - Out2.D_Foxi_CH4)

        ## correction of specific variables
        if 'd_CO2' in list_var:
            print('d_CO2')
            # Out2 = Out2.drop('d_CO2')
            Out2['d_CO2'] = Out['D_CO2'].differentiate('year')
            # For['Eff'] = xr.zeros_like( For['E_CH4'] * Out2['D_Fland'] )
            # For['Eff'].loc[{'reg_land':0}] = Out2['Eff_comp']
            # Out2['d_CO2'] = OSCAR['d_CO2'](xr.merge([Out, Out2]), Par, For, recursive=True)

        ## save
        Out2.to_netcdf(folder_extra + exp + '_Out2-' + str(nset) + '.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out2})

        ## empty memory
        del Par, For, Out, Out2

