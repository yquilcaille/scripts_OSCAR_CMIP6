import os
import csv
import math
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy.optimize import fmin

import sys
sys.path.append("H:/MyDocuments/Repositories/OSCARv31_CMIP6") ## line required for run on server ebro
from core_fct.fct_process import OSCAR, OSCAR_landC
from core_fct.fct_loadD import load_all_hist
from run_scripts.RCMIP_phase2.weighted_quantile import weighted_quantile
from run_scripts.RCMIP_phase2.more_distribs import st_gennorm2, st_extskewnorm, st_skewgennorm, st_flexgenskewnorm3, st_flexgenskewnorm5


# 'C:\Users\quilcail\AppData\Roaming\Python\Python37\Scripts' #rcmip.exe in this folder. Had to do a pip --user.
import pyrcmip as pyr
import scmdata as scm

##################################################
##################################################

## info
folder_raw = 'results/CMIP6_v3.1/'
folder_extra = 'results/CMIP6_v3.1_extra/'
folder_rcmip = 'results/RCMIP_phase2/'

## 
option_mask = 'mask_unique' ## mask_unique | mask_all | mask_select | mask_indiv
#mask_all_exp = True
get_indicators = True
option_full_configs = True

## option
sigma_list = [-2., -1., 0., 1., 2.]
option_overwrite = False



##################################################
##   RCMIP INDICATORS
##################################################

## load RCMIP indicators
file_ranges = 'run_scripts/RCMIP_phase2/RCMIP-phase2_assessed-ranges-v2-2-0.csv'
rcmip = xr.Dataset.from_dataframe(pd.read_csv(file_ranges))
## complying to IPCC script
rcmip = rcmip.rename({'RCMIP name':'indicator'})
## load as well IPCC indicators for consistent weighting
file_ranges_ipcc = 'run_scripts/RCMIP_phase2/ipcc-wg1-emulator-assessed-ranges_v2+.csv'
ipcc = xr.Dataset.from_dataframe(pd.read_csv(file_ranges_ipcc))
## those in common will be rewritten: [ind for ind in rcmip['indicator'].values if ind in ipcc['indicator'].values]
## load as well extra indicators for testing
file_ranges_extra = 'extra_data/extra_assessed-ranges.csv'
extra = xr.Dataset.from_dataframe(pd.read_csv(file_ranges_extra)).rename({'RCMIP name':'indicator'})
## those in common will be rewritten: [ind for ind in extra['indicator'].values if ind in ipcc['indicator'].values]

## manual merging of rcmip using ipcc to increment index while not adding indicators already in rcmip
ipcc = ipcc.sel( index=[ind for ind in ipcc.index.values if ipcc['indicator'].sel(index=ind).values not in rcmip['indicator']] )
ipcc.coords['index'] = np.arange( rcmip.index[-1]+1, rcmip.index[-1]+1+ipcc.index.size )
extra = extra.sel( index=[ind for ind in extra.index.values if extra['indicator'].sel(index=ind).values not in rcmip['indicator']] )
extra.coords['index'] = np.arange( ipcc.index[-1]+1, ipcc.index[-1]+1+extra.index.size )
rcmip = xr.merge( [rcmip,ipcc,extra] )

## indicators to use for weighting OSCAR by Yann Quilcaille for RCMIP-phase 2 and CMIP6
ind_list = [#'Equilibrium Climate Sensitivity',
            #'Transient Climate Response',
            #'Transient Climate Response to Emissions',
            'Cumulative Net Land to Atmosphere Flux|CO2 World esm-hist-2011',
            # 'Net Land to Atmosphere Flux|CO2 World esm-hist-1980',
            # 'Net Land to Atmosphere Flux|CO2 World esm-hist-1990',
            # 'Net Land to Atmosphere Flux|CO2 World esm-hist-2000',
            # 'Net Land to Atmosphere Flux|CO2 World esm-hist-2002',
            'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011',
            # 'Net Ocean to Atmosphere Flux|CO2 World esm-hist-1980',
            # 'Net Ocean to Atmosphere Flux|CO2 World esm-hist-1990',
            # 'Net Ocean to Atmosphere Flux|CO2 World esm-hist-2000',
            # 'Net Ocean to Atmosphere Flux|CO2 World esm-hist-2002',
            # 'Carbon uptake|Land World 1pctCO2 1850-1990',
            # 'Carbon uptake|Ocean World 1pctCO2 1850-1990',
            # 'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019',
            # 'Increase Atmospheric Concentrations|CO2 World esm-hist-2011',
            'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1980',
            'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1990',
            'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2000',
            'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2002',
            # 'Atmospheric Lifetime|CH4 World historical-2005',
            # 'Atmospheric Lifetime|N2O World historical-2005',
            ]

## choose indicators to use with OSCAR
oscar_indicators = {'indicator': ('variable', 'operator', 'factor', 'units'),
    'Carbon uptake|Land World ssp245 1750-2018':  ('D_Fland', 'cumsum', 1, 'PgC'),
    'Carbon uptake|Ocean World ssp245 1750-2018': ('D_Focean', 'cumsum', 1, 'PgC'), 
    'Carbon uptake|Land World 1pctCO2 1850-1990': ('D_Fland', 'cumsum', 1, 'PgC'),
    'Carbon uptake|Ocean World 1pctCO2 1850-1990': ('D_Focean', 'cumsum', 1, 'PgC'),
    'Carbon uptake rate|Land World ssp245 1980-1989': ('D_Fland', 'mean', 1, 'PgC'),
    'Carbon uptake rate|Land World ssp245 1990-1999': ('D_Fland', 'mean', 1, 'PgC'),
    'Carbon uptake rate|Land World ssp245 2000-2009': ('D_Fland', 'mean', 1, 'PgC'),
    'Carbon uptake rate|Land World ssp245 2009-2018': ('D_Fland', 'mean', 1, 'PgC'),
    'Carbon uptake rate|Ocean World ssp245 1980-1989': ('D_Focean', 'mean', 1, 'PgC'),
    'Carbon uptake rate|Ocean World ssp245 1990-1999': ('D_Focean', 'mean', 1, 'PgC'),
    'Carbon uptake rate|Ocean World ssp245 2000-2009': ('D_Focean', 'mean', 1, 'PgC'),
    'Carbon uptake rate|Ocean World ssp245 2009-2018': ('D_Focean', 'mean', 1, 'PgC'),
    'Cumulative Net Land to Atmosphere Flux|CO2 World esm-hist-2011': ('Cum LandToAtm', 'custom', 1., 'PgC'),
    'Net Land to Atmosphere Flux|CO2 World esm-hist-1980': ('LandToAtm', 'custom', 1., 'PgC yr-1'),
    'Net Land to Atmosphere Flux|CO2 World esm-hist-1990': ('LandToAtm', 'custom', 1., 'PgC yr-1'),
    'Net Land to Atmosphere Flux|CO2 World esm-hist-2000': ('LandToAtm', 'custom', 1., 'PgC yr-1'),
    'Net Land to Atmosphere Flux|CO2 World esm-hist-2002': ('LandToAtm', 'custom', 1., 'PgC yr-1'),
    'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011': ('Cum D_Focean', 'custom', -1., 'PgC'),
    'Net Ocean to Atmosphere Flux|CO2 World esm-hist-1980': ('D_Focean', 'mean', -1., 'PgC yr-1'),
    'Net Ocean to Atmosphere Flux|CO2 World esm-hist-1990': ('D_Focean', 'mean', -1., 'PgC yr-1'),
    'Net Ocean to Atmosphere Flux|CO2 World esm-hist-2000': ('D_Focean', 'mean', -1., 'PgC yr-1'),
    'Net Ocean to Atmosphere Flux|CO2 World esm-hist-2002': ('D_Focean', 'mean', -1., 'PgC yr-1'),
    'ECS_FROM_ABRUPT4XCO2_EXPT': ('D_Tg', 'sel_year=-1', 0.5, 'K'),
    'Equilibrium Climate Sensitivity': ('ECS', 'custom', 1., 'K'),
    'Effective Radiative Forcing|Tropospheric Ozone ssp245 2018': ('RF_O3t', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Tropospheric Ozone World historical-1750':('RF_O3t', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Stratospheric Ozone ssp245 2018': ('RF_O3s', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Stratospheric Ozone World historical-1750':('RF_O3s', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|CH4 ssp245 2018': ('RF_CH4', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|CH4 World historical-1750':('RF_CH4', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|CO2 ssp245 2018': ('RF_CO2', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|CO2 World historical-1750':('RF_CO2', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|N2O ssp245 2018': ('RF_N2O', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|N2O World historical-1750':('RF_N2O', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Aerosols|Indirect Effect ssp245 2018': ('RF_cloud2', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Aerosols ssp245 2018': ('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|Aerosols World historical-1750':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Greenhouse Gases': ('RF_wmghg', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Albedo Change World historical-1750':('RF_alb', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Other|BC on Snow World historical-1750':('RF_BCsnow', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|Montreal Gases World historical-1750':('Montreal', 'custom', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|F-Gases World historical-1750':('F-Gases', 'custom', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Other|CH4 Oxidation Stratospheric H2O World historical-1750':('RF_H2Os', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O ssp245 2018': ('RF_H2Os', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|Other|Land cover change ssp245 2018': ('RF_lcc', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|Other|BC on Snow ssp245 2018': ('ERF_BCsnow', 'custom', 1, 'W m-2'),
    'Effective Radiative Forcing|Halogens ssp245 2018': ('RF_halo', 'custom', 1, 'W m-2'),
    'Effective Radiative Forcing': ('RF_warm', 'mean', 1, 'W m-2'),
    'Heat gain|Ocean ssp245 1971-2018': ('D_OHC', 'mean', 1, 'ZJ'),
    'Heat gain|Ocean ssp245 2006-2018': ('D_OHC', 'mean', 1, 'ZJ'),
    'Heat Content|Ocean World ssp245 1971-2018': ('D_OHC', 'mean', 1, 'ZJ'),
    'Transient Climate Response': ('D_Tg', 'sel_year=70', 1, 'K'),
    'Transient Climate Response to Emissions': ('f_TCRE', 'custom', 1, 'K EgC-1'),
    'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019': ('D_Tg', 'mean', 0.75/0.89, 'K'), #TODO: decide on coeff!
    'Surface Air Temperature Change World ssp245 2009-2018': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Ocean Blended Temperature Change World ssp245 2009-2018': ('D_Tg', 'mean', 0.86/0.99, 'K'), #TODO: decide on coeff!
    'Surface Air Temperature Change World ssp245 1995-2014': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Ocean Blended Temperature Change World ssp245 1995-2014': ('D_Tg', 'mean', 0.75/0.89, 'K'), #TODO: decide on coeff!
    'Surface Air Temperature Change World ssp119 2021-2040': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp119 2041-2060': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp119 2081-2100': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp126 2021-2040': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp126 2041-2060': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp126 2081-2100': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp245 2021-2040': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp245 2041-2060': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp245 2081-2100': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp370 2021-2040': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp370 2041-2060': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp370 2081-2100': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp585 2021-2040': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp585 2041-2060': ('D_Tg', 'mean', 1, 'K'),
    'Surface Air Temperature Change World ssp585 2081-2100': ('D_Tg', 'mean', 1, 'K'),
    'Airborne Fraction|CO2 World 1pctCO2 1850-1920': ('f_AF', 'custom', 1, '--'),
    'Airborne Fraction|CO2 World 1pctCO2 1850-1990': ('f_AF', 'custom', 1, '--'),
    'Airborne Fraction|CO2 World ssp245 1997-2018': ('f_AF', 'custom', 1, '--'),
    'Fossil-only Airborne Fraction|CO2 World ssp245 1997-2018': ('f_AF_ff', 'custom', 1, '--'),
    'Fossil-only Airborne Fraction|CO2 World ssp245 1997-2018': ('f_AF_ff', 'custom', 1, '--'),
    'Increase Atmospheric Concentrations|CO2 World esm-hist-2011':('D_CO2', 'mean', 1, 'ppm'),
    'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1980':('d_CO2', 'mean', 1, 'ppm yr-1'),
    'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1990':('d_CO2', 'mean', 1, 'ppm yr-1'),
    'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2000':('d_CO2', 'mean', 1, 'ppm yr-1'),
    'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2002':('d_CO2', 'mean', 1, 'ppm yr-1'),
    'Atmospheric Lifetime|CH4 World historical-2005':('tau_CH4', 'mean', 1, 'yr'),
    'Atmospheric Lifetime|N2O World historical-2005':('tau_N2O', 'mean', 1, 'yr'),
    'Compatible Fossil-fuel emissions of CO2 1960-1969':('Eff_comp', 'mean', 1, 'PgC yr-1'),
    'Compatible Fossil-fuel emissions of CO2 1970-1979':('Eff_comp', 'mean', 1, 'PgC yr-1'),
    'Compatible Fossil-fuel emissions of CO2 1980-1989':('Eff_comp', 'mean', 1, 'PgC yr-1'),
    'Compatible Fossil-fuel emissions of CO2 1990-1999':('Eff_comp', 'mean', 1, 'PgC yr-1'),
    'Compatible Fossil-fuel emissions of CO2 2000-2009':('Eff_comp', 'mean', 1, 'PgC yr-1'),
    'Compatible Fossil-fuel emissions of CO2 2009-2018':('Eff_comp', 'mean', 1, 'PgC yr-1'),
    # 'Land-use change CO2 emissions 1960-1969',
    # 'Land-use change CO2 emissions 1970-1979',
    # 'Land-use change CO2 emissions 1980-1989',
    # 'Land-use change CO2 emissions 1990-1999',
    # 'Land-use change CO2 emissions 2000-2009',
    # 'Land-use change CO2 emissions 2009-2018',
    # 'Total CO2 emissions 1960-1969',
    # 'Total CO2 emissions 1970-1979',
    # 'Total CO2 emissions 1980-1989',
    # 'Total CO2 emissions 1990-1999',
    # 'Total CO2 emissions 2000-2009',
    # 'Total CO2 emissions 2009-2018',
    'Growth rate in atmospheric CO2 1960-1969':('d_CO2', 'mean', 2.1179152, 'PgC yr-1'),
    'Growth rate in atmospheric CO2 1970-1979':('d_CO2', 'mean', 2.1179152, 'PgC yr-1'),
    'Growth rate in atmospheric CO2 1980-1989':('d_CO2', 'mean', 2.1179152, 'PgC yr-1'),
    'Growth rate in atmospheric CO2 1990-1999':('d_CO2', 'mean', 2.1179152, 'PgC yr-1'),
    'Growth rate in atmospheric CO2 2000-2009':('d_CO2', 'mean', 2.1179152, 'PgC yr-1'),
    'Growth rate in atmospheric CO2 2009-2018':('d_CO2', 'mean', 2.1179152, 'PgC yr-1'),
    'Ocean sink 1960-1969':('D_Focean', 'mean', 1, 'PgC yr-1'),
    'Ocean sink 1970-1979':('D_Focean', 'mean', 1, 'PgC yr-1'),
    'Ocean sink 1980-1989':('D_Focean', 'mean', 1, 'PgC yr-1'),
    'Ocean sink 1990-1999':('D_Focean', 'mean', 1, 'PgC yr-1'),
    'Ocean sink 2000-2009':('D_Focean', 'mean', 1, 'PgC yr-1'),
    'Ocean sink 2009-2018':('D_Focean', 'mean', 1, 'PgC yr-1'),
    'Terrestrial sink 1960-1969':('D_Fland', 'mean', 1., 'PgC yr-1'),
    'Terrestrial sink 1970-1979':('D_Fland', 'mean', 1., 'PgC yr-1'),
    'Terrestrial sink 1980-1989':('D_Fland', 'mean', 1., 'PgC yr-1'),
    'Terrestrial sink 1990-1999':('D_Fland', 'mean', 1., 'PgC yr-1'),
    'Terrestrial sink 2000-2009':('D_Fland', 'mean', 1., 'PgC yr-1'),
    'Terrestrial sink 2009-2018':('D_Fland', 'mean', 1., 'PgC yr-1'),
    'Cumulative Fossil-fuel CO2 emissions 1750-2018':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative Fossil-fuel CO2 emissions 1850-2014':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative Fossil-fuel CO2 emissions 1959-2018':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative Fossil-fuel CO2 emissions 1850-2018':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative Fossil-fuel CO2 emissions 1850-2019':('Eff_comp', 'cumsum', 1, 'PgC'),
    # 'Cumulative Land-use change CO2 emissions 1750-2018',
    # 'Cumulative Land-use change CO2 emissions 1850-2014',
    # 'Cumulative Land-use change CO2 emissions 1959-2018',
    # 'Cumulative Land-use change CO2 emissions 1850-2018',
    # 'Cumulative Land-use change CO2 emissions 1850-2019',
    'Cumulative growth rate in atmospheric CO2 1750-2018':('d_CO2', 'cumsum', 2.1179152, 'PgC'),
    'Cumulative growth rate in atmospheric CO2 1850-2014':('d_CO2', 'cumsum', 2.1179152, 'PgC'),
    'Cumulative growth rate in atmospheric CO2 1959-2018':('d_CO2', 'cumsum', 2.1179152, 'PgC'),
    'Cumulative growth rate in atmospheric CO2 1850-2018':('d_CO2', 'cumsum', 2.1179152, 'PgC'),
    'Cumulative growth rate in atmospheric CO2 1850-2019':('d_CO2', 'cumsum', 2.1179152, 'PgC'),
    'Cumulative ocean sink 1750-2018':('D_Focean', 'cumsum', 1, 'PgC'),
    'Cumulative ocean sink 1850-2014':('D_Focean', 'cumsum', 1, 'PgC'),
    'Cumulative ocean sink 1959-2018':('D_Focean', 'cumsum', 1, 'PgC'),
    'Cumulative ocean sink 1959-2018':('D_Focean', 'cumsum', 1, 'PgC'),
    'Cumulative ocean sink 1850-2018':('D_Focean', 'cumsum', 1, 'PgC'),
    'Cumulative ocean sink 1850-2019':('D_Focean', 'cumsum', 1, 'PgC'),
    'Cumulative terrestrial sink 1750-2018':('D_Fland', 'cumsum', 1, 'PgC'),
    'Cumulative terrestrial sink 1850-2014':('D_Fland', 'cumsum', 1, 'PgC'),
    'Cumulative terrestrial sink 1959-2018':('D_Fland', 'cumsum', 1, 'PgC'),
    'Cumulative terrestrial sink 1850-2018':('D_Fland', 'cumsum', 1, 'PgC'),
    'Cumulative terrestrial sink 1850-2019':('D_Fland', 'cumsum', 1, 'PgC'),
    'Transient Climate Response CMIP6': ('ECS', 'sel_year=70', 1, 'K'),
    'Transient Climate Response to Emissions CMIP6': ('f_TCRE', 'custom', 1, 'K EgC-1'),
    'Cumulative diagnosed emissions 1pctCO2 CMIP6':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative diagnosed emissions 1pctCO2 CMIP5':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Carbon-climate feedback of land gamma_land CMIP6 4xCO2':('gamma_land', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of land beta_land CMIP6 4xCO2':('beta_land', 'custom', 1, 'PgC ppm-1'),
    'Carbon-climate feedback of ocean gamma_ocean CMIP6 4xCO2':('gamma_ocean', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of ocean beta_ocean CMIP6 4xCO2':('beta_ocean', 'custom', 1, 'PgC ppm-1'),
    'Transient climate linear sensitivity CMIP6 4xCO2':('clim_sensi', 'custom', 1, 'K ppm-1'),
    'Carbon-climate feedback of land gamma_land CMIP6 2xCO2':('gamma_land', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of land beta_land CMIP6 2xCO2':('beta_land', 'custom', 1, 'PgC ppm-1'),
    'Carbon-climate feedback of ocean gamma_ocean CMIP6 2xCO2':('gamma_ocean', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of ocean beta_ocean CMIP6 2xCO2':('beta_ocean', 'custom', 1, 'PgC ppm-1'),
    'Transient climate linear sensitivity CMIP6 2xCO2':('clim_sensi', 'custom', 1, 'K ppm-1'),
    'Carbon-climate feedback of land gamma_land CMIP5 4xCO2':('gamma_land', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of land beta_land CMIP5 4xCO2':('beta_land', 'custom', 1, 'PgC ppm-1'),
    'Carbon-climate feedback of ocean gamma_ocean CMIP5 4xCO2':('gamma_ocean', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of ocean beta_ocean CMIP5 4xCO2':('beta_ocean', 'custom', 1, 'PgC ppm-1'),
    'Transient climate linear sensitivity CMIP5 4xCO2':('clim_sensi', 'custom', 1, 'K ppm-1'),
    'Carbon-climate feedback of land gamma_land CMIP5 2xCO2':('gamma_land', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of land beta_land CMIP5 2xCO2':('beta_land', 'custom', 1, 'PgC ppm-1'),
    'Carbon-climate feedback of ocean gamma_ocean CMIP5 2xCO2':('gamma_ocean', 'custom', 1, 'PgC K-1'),
    'Carbon-concentration feedback of ocean beta_ocean CMIP5 2xCO2':('beta_ocean', 'custom', 1, 'PgC ppm-1'),
    'Transient climate linear sensitivity CMIP6 2xCO2':('clim_sensi', 'custom', 1, 'K ppm-1'),
    'Atmospheric Concentration of CO2 of esm-ssp585 in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp585 in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp370 in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp370 in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp460 in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp460 in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp245 in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp245 in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp534-over in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp534-over in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp434 in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp434 in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp126 in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp126 in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp119 in 2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-ssp119 in 2300':('CO2atm', 'custom', 1, 'ppm'),
    'Increase in global mean temperature in ssp585, mid-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp585, mid-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp126, mid-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp126, mid-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp585, end-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp585, end-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp126, end-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp126, end-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp370, mid-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp370, mid-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp245, mid-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp245, mid-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp370, end-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp370, end-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp245, end-century (unconstrained)':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp245, end-century (constrained)':('D_Tg', 'mean', 1, 'K'),
    'Cumulative compatible emissions CMIP5 historical-CMIP5':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative compatible emissions CMIP5 RCP2.6':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative compatible emissions CMIP5 RCP4.5':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative compatible emissions CMIP5 RCP6.0':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Cumulative compatible emissions CMIP5 RCP8.5':('Eff_comp', 'cumsum', 1, 'PgC'),
    'Land carbon changes CMIP5 historical-CMIP5':('Cum LandToAtm', 'custom', -1, 'PgC'),
    'Land carbon changes CMIP5 RCP2.6':('Cum LandToAtm', 'custom', -1, 'PgC'),
    'Land carbon changes CMIP5 RCP4.5':('Cum LandToAtm', 'custom', -1, 'PgC'),
    'Land carbon changes CMIP5 RCP6.0':('Cum LandToAtm', 'custom', -1, 'PgC'),
    'Land carbon changes CMIP5 RCP8.5':('Cum LandToAtm', 'custom', -1, 'PgC'),
    'Ocean carbon changes CMIP5 historical-CMIP5':('D_Focean', 'cumsum', 1, 'PgC'),
    'Ocean carbon changes CMIP5 RCP2.6':('D_Focean', 'cumsum', 1, 'PgC'),
    'Ocean carbon changes CMIP5 RCP4.5':('D_Focean', 'cumsum', 1, 'PgC'),
    'Ocean carbon changes CMIP5 RCP6.0':('D_Focean', 'cumsum', 1, 'PgC'),
    'Ocean carbon changes CMIP5 RCP8.5':('D_Focean', 'cumsum', 1, 'PgC'),
    'Year of peak in GSAT SSP1-1.9':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in GSAT SSP1-2.6':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in GSAT SSP4-3.4':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in GSAT SSP5-3.4-OS':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT SSP2-4.5':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT SSP4-6.0':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT SSP3-7.0':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT SSP5-8.5':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in ERF SSP1-1.9':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in ERF SSP1-2.6':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in ERF SSP4-3.4':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in ERF SSP5-3.4-OS':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF SSP2-4.5':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF SSP4-6.0':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF SSP3-7.0':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF SSP5-8.5':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 SSP1-1.9':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 SSP1-2.6':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 SSP4-3.4':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 SSP5-3.4-OS':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 SSP2-4.5':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 SSP4-6.0':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 SSP3-7.0':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 SSP5-8.5':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Year of peak in GSAT esm-SSP1-1.9':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in GSAT esm-SSP1-2.6':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in GSAT esm-SSP4-3.4':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in GSAT esm-SSP5-3.4-OS':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT esm-SSP2-4.5':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT esm-SSP4-6.0':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT esm-SSP3-7.0':('year_peak_GSAT', 'custom', 1, 'yr'),
    # 'Year of peak in GSAT esm-SSP5-8.5':('year_peak_GSAT', 'custom', 1, 'yr'),
    'Year of peak in ERF esm-SSP1-1.9':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in ERF esm-SSP1-2.6':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in ERF esm-SSP4-3.4':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in ERF esm-SSP5-3.4-OS':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF esm-SSP2-4.5':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF esm-SSP4-6.0':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF esm-SSP3-7.0':('year_peak_ERF', 'custom', 1, 'yr'),
    # 'Year of peak in ERF esm-SSP5-8.5':('year_peak_ERF', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 esm-SSP1-1.9':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 esm-SSP1-2.6':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 esm-SSP4-3.4':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Year of peak in atmCO2 esm-SSP5-3.4-OS':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 esm-SSP2-4.5':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 esm-SSP4-6.0':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 esm-SSP3-7.0':('year_peak_atmCO2', 'custom', 1, 'yr'),
    # 'Year of peak in atmCO2 esm-SSP5-8.5':('year_peak_atmCO2', 'custom', 1, 'yr'),
    'Increase in global mean temperature in ssp119, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp126, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp434, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp534-over, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp245, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp460, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp370, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp585, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp119, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp126, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp434, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp534-over, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp245, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp460, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp370, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp585, 2081-2100 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp119, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp126, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp434, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp534-over, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp245, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp460, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp370, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in ssp585, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp119, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp126, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp434, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp534-over, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp245, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp460, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp370, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Increase in global mean temperature in esm-ssp585, 2250-2300 with reference to 1995-2014':('D_Tg', 'mean', 1, 'K'),
    'Peak in GSAT SSP1-1.9 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in GSAT SSP1-2.6 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in GSAT SSP4-3.4 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in GSAT SSP5-3.4-OS with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT SSP2-4.5 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT SSP4-6.0 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT SSP3-7.0 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT SSP5-8.5 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in ERF SSP1-1.9':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in ERF SSP1-2.6':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in ERF SSP4-3.4':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in ERF SSP5-3.4-OS':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF SSP2-4.5':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF SSP4-6.0':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF SSP3-7.0':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF SSP5-8.5':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in GSAT esm-SSP1-1.9 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in GSAT esm-SSP1-2.6 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in GSAT esm-SSP4-3.4 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in GSAT esm-SSP5-3.4-OS with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT esm-SSP2-4.5 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT esm-SSP4-6.0 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT esm-SSP3-7.0 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    # 'Peak in GSAT esm-SSP5-8.5 with reference to 1995-2014':('value_peak_GSAT', 'custom', 1, 'K'),
    'Peak in ERF esm-SSP1-1.9':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in ERF esm-SSP1-2.6':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in ERF esm-SSP4-3.4':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in ERF esm-SSP5-3.4-OS':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF esm-SSP2-4.5':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF esm-SSP4-6.0':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF esm-SSP3-7.0':('value_peak_ERF', 'custom', 1, 'W m-2'),
    # 'Peak in ERF esm-SSP5-8.5':('value_peak_ERF', 'custom', 1, 'W m-2'),
    'Peak in atmCO2 SSP1-1.9':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    'Peak in atmCO2 SSP1-2.6':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    'Peak in atmCO2 SSP4-3.4':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    'Peak in atmCO2 SSP5-3.4-OS':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    # 'Peak in atmCO2 SSP2-4.5':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    # 'Peak in atmCO2 SSP4-6.0':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    # 'Peak in atmCO2 SSP3-7.0':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    # 'Peak in atmCO2 SSP5-8.5':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    'Peak in atmCO2 esm-SSP1-1.9':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    'Peak in atmCO2 esm-SSP1-2.6':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    'Peak in atmCO2 esm-SSP4-3.4':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    'Peak in atmCO2 esm-SSP5-3.4-OS':('value_peak_atmCO2', 'custom', 1, 'ppm'),
    # 'Peak in atmCO2 esm-SSP2-4.5':('value_peak_atmCO2', 'custom', 1, 'K'),
    # 'Peak in atmCO2 esm-SSP4-6.0':('value_peak_atmCO2', 'custom', 1, 'K'),
    # 'Peak in atmCO2 esm-SSP3-7.0':('value_peak_atmCO2', 'custom', 1, 'K'),
    # 'Peak in atmCO2 esm-SSP5-8.5':('value_peak_atmCO2', 'custom', 1, 'K'),
    'Atmospheric Concentration of CO2 of SSP1-1.9 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP1-2.6 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP4-3.4 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP5-3.4-OS 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP2-4.5 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP4-6.0 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP3-7.0 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP5-8.5 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP1-1.9 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP1-2.6 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP4-3.4 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP5-3.4-OS 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP2-4.5 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP4-6.0 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP3-7.0 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP5-8.5 2081-2100':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP1-1.9 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP1-2.6 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP4-3.4 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP5-3.4-OS 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP2-4.5 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP4-6.0 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP3-7.0 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of SSP5-8.5 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP1-1.9 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP1-2.6 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP4-3.4 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP5-3.4-OS 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP2-4.5 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP4-6.0 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP3-7.0 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Atmospheric Concentration of CO2 of esm-SSP5-8.5 2250-2300':('CO2atm', 'custom', 1, 'ppm'),
    'Effective Radiative Forcing SSP1-1.9 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP1-2.6 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP4-3.4 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP5-3.4-OS 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP2-4.5 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP4-6.0 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP3-7.0 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP5-8.5 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP1-1.9 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP1-2.6 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP4-3.4 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP5-3.4-OS 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP2-4.5 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP4-6.0 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP3-7.0 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP5-8.5 2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP1-1.9 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP1-2.6 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP4-3.4 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP5-3.4-OS 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP2-4.5 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP4-6.0 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP3-7.0 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP5-8.5 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP1-1.9 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP1-2.6 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP4-3.4 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP5-3.4-OS 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP2-4.5 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP4-6.0 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP3-7.0 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP5-8.5 2081-2100':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP1-1.9 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP1-2.6 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP4-3.4 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP5-3.4-OS 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP2-4.5 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP4-6.0 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP3-7.0 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing SSP5-8.5 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP1-1.9 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP1-2.6 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP4-3.4 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP5-3.4-OS 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP2-4.5 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP4-6.0 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP3-7.0 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP5-8.5 2250-2300':('RF_warm', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP1-1.9 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP1-2.6 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP4-3.4 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP5-3.4-OS 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP2-4.5 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP4-6.0 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP3-7.0 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP5-8.5 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP1-1.9 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP1-2.6 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP4-3.4 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP5-3.4-OS 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP2-4.5 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP4-6.0 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP3-7.0 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP5-8.5 2081-2100':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP1-1.9 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP1-2.6 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP4-3.4 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP5-3.4-OS 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP2-4.5 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP4-6.0 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP3-7.0 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols SSP5-8.5 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP1-1.9 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP1-2.6 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing esm-SSP4-3.4 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP5-3.4-OS 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP2-4.5 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP4-6.0 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP3-7.0 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing of Aerosols esm-SSP5-8.5 2250-2300':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Deduced net land flux 1960-1969':('LandToAtm', 'custom', -1., 'PgC yr-1'),
    'Deduced net land flux 1970-1979':('LandToAtm', 'custom', -1., 'PgC yr-1'),
    'Deduced net land flux 1980-1989':('LandToAtm', 'custom', -1., 'PgC yr-1'),
    'Deduced net land flux 1990-1999':('LandToAtm', 'custom', -1., 'PgC yr-1'),
    'Deduced net land flux 2000-2009':('LandToAtm', 'custom', -1., 'PgC yr-1'),
    'Deduced net land flux 2009-2018':('LandToAtm', 'custom', -1., 'PgC yr-1'),
    'Deduced cumulative net land flux 1750-2018':('Cum LandToAtm', 'custom', -1., 'PgC'),
    'Deduced cumulative net land flux 1850-2014':('Cum LandToAtm', 'custom', -1., 'PgC'),
    'Deduced cumulative net land flux 1959-2018':('Cum LandToAtm', 'custom', -1., 'PgC'),
    'Deduced cumulative net land flux 1850-2018':('Cum LandToAtm', 'custom', -1., 'PgC'),
    'Deduced cumulative net land flux 1850-2019':('Cum LandToAtm', 'custom', -1., 'PgC'),
    'Effective Radiative Forcing|Anthropogenic|CH4 World esm-hist-1750':('RF_CH4', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|CO2 World esm-hist-1750':('RF_CO2', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|N2O World esm-hist-1750':('RF_N2O', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|Aerosols World esm-hist-1750':('RF_AERtot', 'mean', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|Montreal Gases World esm-hist-1750':('Montreal', 'custom', 1, 'W m-2'),
    'Effective Radiative Forcing|Anthropogenic|F-Gases World esm-hist-1750':('F-Gases', 'custom', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Tropospheric Ozone World esm-hist-1750':('RF_O3t', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Stratospheric Ozone World esm-hist-1750':('RF_O3s', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Albedo Change World esm-hist-1750':('RF_alb', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Other|BC on Snow World esm-hist-1750':('RF_BCsnow', 'mean', 1, 'W m-2'),
    'Radiative Forcing|Anthropogenic|Other|CH4 Oxidation Stratospheric H2O World esm-hist-1750':('RF_H2Os', 'mean', 1, 'W m-2'),
    }

##################################################
##################################################





##################################################
##   LOADING FUNCTIONS
##################################################

def load_Par(Nset=20):
    Par = xr.open_mfdataset([folder_raw + 'Par-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    Par = Par.assign_coords(config=np.arange(len(Par.config)))
    return Par


def load_For(exp, Nset=20):
    For = xr.open_mfdataset([folder_raw + exp + '_For-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    For = For.assign_coords(config=np.arange(len(For.config)))
    For = For.transpose(*(['year'] + [dim for dim in list(For.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
    return For


def load_Out(exp, Nset=20):
    Out = xr.open_mfdataset([folder_raw + exp + '_Out-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    Out = Out.assign_coords(config=np.arange(len(Out.config)))
    Out = Out.transpose(*(['year'] + [dim for dim in list(Out.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
    Out2 = xr.open_mfdataset([folder_extra + exp + '_Out2-' + str(n) + '.nc' for n in range(Nset)], combine='nested', concat_dim='config')
    Out2 = Out2.assign_coords(config=np.arange(len(Out2.config)))
    Out2 = Out2.transpose(*(['year'] + [dim for dim in list(Out2.dims)[::-1] if dim not in ['year', 'config']] + ['config']))
    return xr.merge([Out, Out2])


def load_mask(exp, Nset=20):
    for n in range(Nset):
        with open(folder_raw + 'treated/masks/masknoDV_' + exp + '_' + str(n) + '.csv', 'r') as f: 
            TMP = np.array([line for line in csv.reader(f)], dtype=float)
        if n==0: mask = TMP.copy()
        else: mask = np.append(mask, TMP, axis=1)
    mask = xr.DataArray(mask, coords={'year': 1850 + np.arange(len(mask)), 'config': np.arange(len(mask.T))}, dims=['year', 'config'])
    return mask#.notnull()


def get_var(var, exp):
    return OSCAR[var](load_Out(exp), load_Par(), load_For(exp), recursive=True)
##################################################
##################################################




##################################################
##   DISTRIBUTIONS
##################################################

## dict for distribs
distrib_dict = {
    'N': st.norm,
    'GN1': st.gennorm, 
    'SN': st.skewnorm,
    'PN': st.powernorm, 
    'EN': st.exponnorm,
    'GL': st.genlogistic,
    'NIG': st.norminvgauss,
    'LN': st.lognorm,
    'PLN': st.powerlognorm,
    'IW': st.invweibull,
    'LL': st.loglaplace,
    'GX': st.genextreme,
    'GN2': st_gennorm2,
    'ESN': st_extskewnorm,
    'SGN': st_skewgennorm,
    'FGSN3': st_flexgenskewnorm3,
    'FGSN5': st_flexgenskewnorm5,
    }

## function to fit distrib parameters on a few CDF points
def fit_distrib(cdf, xy_list, x_val):

    def err(param, trans=None):
        if trans is not None:
            param = [f_tr(par) for par, f_tr in zip(param, trans)]
        dist = 0.
        for xy in xy_list:
            dist += (cdf(xy[0], *param) - xy[1])**2
        return dist

    ## normal distrib
    if cdf == st.norm.cdf:
        trans = [lambda x: x, abs]
        param_0 = [x_val[0], x_val[1]]

    ## generalized normal distrib
    elif cdf == st.gennorm.cdf:
        trans = [lambda x: 1+abs(x), lambda x: x, abs]
        param_0 = [1, x_val[0], x_val[1]]

    ## power-normal distrib
    elif cdf == st.powernorm.cdf:
        trans = [abs, lambda x: x, abs]
        param_0 = [1, x_val[0], x_val[1]]

    ## exponentially modified normal distrib
    elif cdf == st.exponnorm.cdf:
        trans = [abs, lambda x: x, abs]
        param_0 = [1, x_val[0], x_val[1]]

    ## generalized logistic distrib
    elif cdf == st.genlogistic.cdf:
        trans = [abs, lambda x: x, abs]
        param_0 = [1, x_val[0], x_val[1]]

    ## normal inverse Gaussian distrib
    elif cdf == st.norminvgauss.cdf:
        trans = [abs, lambda x: x, lambda x: x, abs] # warning! extra condition not implemented: abs(b) <= a
        param_0 = [1, 1, x_val[0], x_val[1]]

    ##--------------------
    ## log-normal distrib
    elif cdf == st.lognorm.cdf:
        trans = [abs, lambda x: x, abs]
        param_0 = [1, x_val[0], x_val[1]]

    ## power-log-normal distrib
    elif cdf == st.powerlognorm.cdf:
        trans = [abs, abs, lambda x: x, abs]
        param_0 = [1, 1, x_val[0], x_val[1]]

    ## inverse Weibull distrib
    elif cdf == st.invweibull.cdf:
        trans = [abs, lambda x: x, abs]
        param_0 = [1, x_val[0], x_val[1]]

    ## log-Laplace distrib
    elif cdf == st.loglaplace.cdf:
        trans = [abs, lambda x: x, abs]
        param_0 = [1, x_val[0], x_val[1]]

    ## generalized extreme distrib
    elif cdf == st.genextreme.cdf:
        trans = [abs, lambda x: x, abs]
        param_0 = [0, x_val[0], x_val[1]]

    ## generalized extreme distrib
    elif cdf == st_gennorm2.cdf:
        trans = [lambda x: x, lambda x: x, abs]
        param_0 = [0, x_val[0], x_val[1]]

    ##--------------------
    ## skew-normal distrib
    elif cdf == st.skewnorm.cdf:
        trans = [lambda x: x, lambda x: x, abs]
        param_0 = [0, x_val[0], x_val[1]]

    ## extended skew-normal distrib
    elif cdf == st_extskewnorm.cdf:
        trans = [lambda x: x, lambda x: x, lambda x: x, abs]
        param_0 = [0, 0, x_val[0], x_val[1]]

    ## skew-generalized normal distrib
    elif cdf == st_skewgennorm.cdf:
        trans = [lambda x: x, abs, lambda x: x, abs]
        param_0 = [0, 0, x_val[0], x_val[1]]

    ## flexible generalized skew-normal distrib (3rd degree)
    elif cdf == st_flexgenskewnorm3.cdf:
        trans = [lambda x: x, lambda x: x, lambda x: x, abs]
        param_0 = [0, 0, x_val[0], x_val[1]]

    ## flexible generalized skew-normal distrib (5th degree)
    elif cdf == st_flexgenskewnorm5.cdf:
        trans = [lambda x: x, lambda x: x, lambda x: x, lambda x: x, abs]
        param_0 = [0, 0, 0, x_val[0], x_val[1]]

    ## fit an transform
    param, _, _, _, fmin_flag = fmin(err, param_0, args=(trans,), xtol=1E-9, ftol=1E-9, full_output=True, disp=False)
    param = [f_tr(par) for par, f_tr in zip(param, trans)]

    ## return
    return param, fmin_flag==0


## function to test closeness of fit
def test_fit(val_list, ref_list=[0.05, 0.17, 0.50, 0.83, 0.95], rtol=0.00, atol=0.01):
    #is_close = [math.isclose(val, ref, rel_tol=rel_tol, abs_tol=abs_tol) for val, ref in zip(val_list, ref_list)]
    is_close = np.isclose(val_list, ref_list, rtol=rtol, atol=atol)
    return np.where(np.isnan(val_list), 5*[True], is_close)
##################################################
##################################################



##################################################
##   ADDING INFORMATIONS ON EXPERIMENTS
##################################################
list_experiments = set([zou.split('_')[1] for zou in os.listdir(folder_raw + 'treated/masks/') if '.csv' in zou])
list_experiments.remove('all')

## Experiments preceding
dico_experiments_before = {}
for xp in list_experiments:
    if xp[-4:] in ['-ext','-Ext']: xp2 = xp[:-4]
    elif (xp[-3:] in ['ext','Ext']) and (xp not in ['1pctCO2-4xext']): xp2 = xp[:-3]
    elif xp in ['ssp245-GHG', 'ssp245-CO2'] + ['ssp245-aer'] + ['ssp245-nat', 'ssp245-sol', 'ssp245-volc']  + ['ssp245-stratO3']:xp2='hist-'+str.split(xp,'-')[1]
    elif xp in ['ssp534-over-bgc','ssp585-bgc']:xp2 = 'hist-bgc'
    elif xp[:3]=='ssp': xp2 = 'historical'
    elif xp[:7]=='esm-ssp': xp2 = 'esm-hist'
    elif xp[:3]=='rcp': xp2 = 'historical-CMIP5'
    elif xp[:7]=='esm-rcp': xp2 = 'esm-histcmip5'
    elif xp=='G6solar': xp2 = 'historical'
    else: xp2 = None
    if (xp2 in list_experiments) or xp2==None:dico_experiments_before[xp] = xp2
    else: raise Exception("Correct the name of this experiment")

## Control for each experiment
dico_Xp_Control = { 'piControl':['1pctCO2-4xext', '1pctCO2-bgc', '1pctCO2-cdr', '1pctCO2-rad', '1pctCO2', 'abrupt-0p5xCO2', 'abrupt-2xCO2', 'abrupt-4xCO2', 'G1', 'G2', 'G6solar', 'hist-1950HC', 'hist-aer', 'hist-bgc', 'hist-CO2', 'hist-GHG', 'hist-nat', 'hist-piAer', 'hist-piNTCF', 'hist-sol', 'hist-stratO3', 'hist-volc', 'historical', 'hist-noLu', 'ssp119', 'ssp119ext', 'ssp126-ssp370Lu', 'ssp126', 'ssp126ext', 'ssp245-aer', 'ssp245-CO2', 'ssp245-GHG', 'ssp245-nat', 'ssp245-sol', 'ssp245-stratO3', 'ssp245-volc', 'ssp245', 'ssp245ext', 'ssp370-lowNTCF', 'ssp370-lowNTCFext', 'ssp370-lowNTCF-gidden', 'ssp370-lowNTCFext-gidden', 'ssp370-ssp126Lu', 'ssp370', 'ssp370ext', 'ssp434', 'ssp434ext', 'ssp460', 'ssp460ext', 'ssp534-over-bgc', 'ssp534-over-bgcExt', 'ssp534-over-ext', 'ssp534-over', 'ssp585-bgc', 'ssp585-bgcExt', 'ssp585-ssp126Lu', 'ssp585', 'ssp585ext', 'yr2010CO2'] ,
                    'esm-piControl':['esm-1pct-brch-1000PgC', 'esm-1pct-brch-2000PgC', 'esm-1pct-brch-750PgC', 'esm-1pctCO2', 'esm-abrupt-4xCO2', 'esm-bell-1000PgC', 'esm-bell-2000PgC', 'esm-bell-750PgC', 'esm-hist', 'esm-pi-cdr-pulse', 'esm-pi-CO2pulse', 'esm-ssp119', 'esm-ssp119ext', 'esm-ssp126', 'esm-ssp126ext', 'esm-ssp245', 'esm-ssp245ext', 'esm-ssp370-lowNTCF', 'esm-ssp370-lowNTCFext', 'esm-ssp370-lowNTCF-gidden', 'esm-ssp370-lowNTCFext-gidden', 'esm-ssp370', 'esm-ssp370ext', 'esm-ssp460', 'esm-ssp460ext', 'esm-ssp434', 'esm-ssp434ext', 'esm-ssp534-over-ext', 'esm-ssp534-over', 'esm-ssp585-ssp126Lu-ext', 'esm-ssp585-ssp126Lu', 'esm-ssp585', 'esm-ssp585ext', 'esm-yr2010CO2-cdr-pulse', 'esm-yr2010CO2-CO2pulse', 'esm-yr2010CO2-control', 'esm-yr2010CO2-noemit'],
                    'esm-piControl-CMIP5':['esm-histcmip5','esm-rcp26', 'esm-rcp45', 'esm-rcp60', 'esm-rcp85'],
                    'piControl-CMIP5':['historical-CMIP5', 'rcp26', 'rcp45', 'rcp60', 'rcp85'],
                    'land-piControl':['land-cClim', 'land-cCO2', 'land-crop-grass', 'land-hist', 'land-noLu', 'land-noShiftcultivate', 'land-noWoodHarv'],
                    'land-piControl-altLu1':['land-hist-altLu1'],
                    'land-piControl-altLu2':['land-hist-altLu2'],
                    'land-piControl-altStartYear':['land-hist-altStartYear'],
                    'spinup':[],'esm-spinup':[],'esm-spinup-CMIP5':[],'spinup-CMIP5':[],'land-spinup':[],'land-spinup-altLu1':[],'land-spinup-altLu2':[],'land-spinup-altStartYear':[] }
dico_controls = dict(  [(vv,kk) for kk in dico_Xp_Control.keys() for vv in dico_Xp_Control[kk]] )
##################################################
##################################################





##################################################
##   GET MASK
##################################################
if option_mask=='mask_all':
    if os.path.isfile(folder_rcmip + 'mask_all_exp.nc'):
        mask_all = xr.open_dataarray(folder_rcmip + 'mask_all_exp.nc')
    else:
        mask_all = load_mask('piControl').all('year')
        for exp in list_experiments:
            if exp in ['1pctCO2','1pctCO2-bgc','1pctCO2-cdr','1pctCO2-rad','G2'] + ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC', 'esm-1pctCO2']:
                mask_exp = load_mask(exp).isel(year=slice(0,150)).all('year')
            else:
                mask_exp = load_mask(exp).all('year')
            mask_all *= mask_exp
            print(exp, mask_exp.sum().values)
        mask_all.to_netcdf(folder_rcmip + 'mask_all_exp.nc')
    mask_all_exp = {}
    for exp in list_experiments:mask_all_exp[exp] = mask_all

elif option_mask=='mask_select':
    #raise Exception() ???
    if os.path.isfile(folder_rcmip + 'mask_sel_exp.nc'):
        mask_all = xr.open_dataarray(folder_rcmip + 'mask_sel_exp.nc')
    else:
        mask_all = load_mask('piControl').all('year')
        for exp in ['1pctCO2', 'abrupt-4xCO2', 'ssp585ext']:
            if exp in ['1pctCO2','1pctCO2-bgc','1pctCO2-cdr','1pctCO2-rad','G2'] + ['esm-1pct-brch-750PgC','esm-1pct-brch-1000PgC','esm-1pct-brch-2000PgC', 'esm-1pctCO2']:
                mask_exp = load_mask(exp).isel(year=slice(0,150)).all('year')
            else:
                mask_exp = load_mask(exp).all('year')
            mask_all *= mask_exp
        mask_all.to_netcdf(folder_rcmip + 'mask_sel_exp.nc')
    mask_all_exp = {}
    for exp in list_experiments:mask_all_exp[exp] = mask_all

elif option_mask=='mask_indiv':
    ## loading masks for every experiment. Making an array "Year x Config x Scen" takes too much memory, instead we make within a dataset arrays for each experiment
    mask_all_exp = {}
    ## looping on experiments
    for exp in list_experiments:
        print("loading masks: " + str(np.round(100.*(list(list_experiments).index(exp)+1)/len(list_experiments)))+'%',end='\r' )
        masks = []
        masks.append( load_mask(exp) )
        ## adding previous ones for continuity
        ss = dico_experiments_before[exp]
        while ss != None:
            masks.append(  load_mask(ss)  )
            ss = dico_experiments_before[ss]
        ## correcting mistake on years in function load_mask
        tmp = np.concatenate( [masks[-1].values[0,np.newaxis,:]] + [np.array(mm.values[1:,:],dtype=np.float32) for mm in masks[::-1]] )
        ## setting as NaN rather than False
        tmp[np.where(tmp==0.)] = np.nan
        ## creating the array
        mask_all_exp[exp] = xr.DataArray( tmp, coords={'year':1850+np.arange(len(tmp)), 'config':np.arange(len(tmp.T)) }, dims=['year', 'config'])

elif option_mask=='mask_unique':
    # nan much better than True / False to multiply directly variables.
    # mask_all = load_mask('piControl').all('year')
    mask_all = load_mask('piControl').prod('year')
    for setMC in range(20):
        ## loading unique mask for this set
        with open(folder_raw+'/treated/masks/mask_all_exp_'+str(setMC)+'.csv','r',newline='') as ff:
            mask_all.loc[{ 'config':setMC*500 + np.where( np.isnan(np.array([line for line in csv.reader(ff)] ,dtype=np.float32)[:,0]) )[0] }] = np.nan # nan much better than True / False to multiply directly variables.
    mask_all_exp = {}
    for exp in list_experiments:mask_all_exp[exp] = mask_all

else:
    raise Exception('Option for masks not known.')
##################################################
##################################################









##################################################
##   CALCULATE INDICATORS
##################################################
if option_overwrite:
    if __name__ == '__main__':

        ## create empty indicator dataset
        if option_mask == 'mask_indiv':mask_all = mask_all_exp['1pctCO2']
        oscar = xr.Dataset()
        oscar.coords['index'] = rcmip.index.sel(index=[i for i in rcmip.index.values if str(rcmip.indicator[i].values) in oscar_indicators.keys()  if  i>-1])
        missed = [str(rcmip.indicator[i].values) for i in rcmip.index.values if str(rcmip.indicator[i].values) not in oscar_indicators.keys()]

        if option_full_configs:
            oscar.coords['config'] = mask_all.config
        else:
            oscar.coords['config'] = mask_all.where(~np.isnan(mask_all)).dropna('config').config
        oscar['x'] = xr.zeros_like(oscar.index, dtype=float) + xr.zeros_like(oscar.config, dtype=float)
        oscar['w'] = xr.zeros_like(oscar.index, dtype=float) + xr.zeros_like(oscar.config, dtype=float)
        oscar['m'] = xr.zeros_like(oscar.index, dtype=float) + xr.zeros_like(oscar.config, dtype=float)
        # oscar['m'] = xr.zeros_like(oscar.index, dtype=bool) + xr.zeros_like(oscar.config, dtype=bool)
        oscar['distrib'] = xr.zeros_like(oscar.index, dtype=object)

        ## loop on all indices
        NN,mem_scen,mem_treat,mem_var = 0,'','',''
        for i in oscar.index.values:
            ind = str(rcmip.indicator[i].values)
        
            ## check if chosen
            if ind in oscar_indicators.keys():
                print(ind)
                
                ## get info on indicator
                assert str(rcmip['RCMIP region'][i].values) in ['World', 'nan']
                scen = str(rcmip['RCMIP scenario'][i].values)
                eval_year = (float(rcmip['evaluation_period_start'][i]), float(rcmip['evaluation_period_end'][i]))
                norm_year = (float(rcmip['norm_period_start'][i]), float(rcmip['norm_period_end'][i]))
                x_vl = float(rcmip['very_likely__lower'][i])
                x_l = float(rcmip['likely__lower'][i])
                x_c = float(rcmip['central'][i])
                x_u = float(rcmip['likely__upper'][i])
                x_vu = float(rcmip['very_likely__upper'][i])

                ## get info on OSCAR
                var, treat, fact, units = oscar_indicators[ind]

                ## checking if will need to reload the data  ----> changed to always False because of changes in mask/periods
                no_reload  =  False # (scen == mem_scen)    and    ( ((mem_treat=='custom') and (treat=='custom'))  or  ((mem_treat!='custom') and (treat!='custom') and (mem_var==var)) )
                mem_scen,mem_treat,mem_var = scen,treat,var

                ## CALCULATE!
                if True:

                    ## standard cases
                    if treat != 'custom':
                        if scen == "['1pctCO2','1pctCO2-bgc']":raise Exception("Only betas and gammas require these 2 experiments at once, and they are calculated in custom.")

                        ## concat historical and scenario if relevant
                        if no_reload:
                            print("Not reloading here, same scenario and variable")
                        else:
                            if ('ssp' in scen) or ('rcp' in scen) or (scen in ['G6solar']):
                                Var0 = load_Out(  dico_experiments_before[scen]  )[var].sel(config=oscar.config)
                                Var1 = load_Out(scen)[var].sel(config=oscar.config)
                                For0 = load_For(  dico_experiments_before[scen]  ).sel(config=oscar.config)
                                For1 = load_For(scen).sel(config=oscar.config)
                                if scen+'ext' in list_experiments: ## extension if any
                                    Var2 = load_Out(scen+'ext')[var].sel(config=oscar.config)
                                    For2 = load_For(scen+'ext').sel(config=oscar.config)
                                    mask_tmp = mask_all_exp[scen+'ext'].sel(config=oscar.config)
                                    if option_mask == 'mask_indiv':
                                        if np.isnan(eval_year).all()==False:mask_tmp = mask_tmp.sel(year=eval_year[1]).drop('year')
                                        elif 'sel_year' in treat:mask_tmp = mask_tmp.isel(year=int(treat.split('sel_year=')[1])).drop('year')
                                        else:raise Exception('Period unknown.')
                                elif scen+'-ext' in list_experiments:
                                    Var2 = load_Out(scen+'-ext')[var].sel(config=oscar.config)
                                    For2 = load_For(scen+'-ext').sel(config=oscar.config)
                                    mask_tmp = mask_all_exp[scen+'-ext'].sel(config=oscar.config)
                                    if option_mask == 'mask_indiv':
                                        if np.isnan(eval_year).all()==False:mask_tmp = mask_tmp.sel(year=eval_year[1]).drop('year')
                                        elif 'sel_year' in treat:mask_tmp = mask_tmp.isel(year=int(treat.split('sel_year=')[1])).drop('year')
                                        else:raise Exception('Period unknown.')
                                else:
                                    mask_tmp = mask_all_exp[scen].sel(config=oscar.config)
                                    if option_mask == 'mask_indiv':
                                        if np.isnan(eval_year).all()==False:mask_tmp = mask_tmp.sel(year=eval_year[1]).drop('year')
                                        elif 'sel_year' in treat:mask_tmp = mask_tmp.isel(year=int(treat.split('sel_year=')[1])).drop('year')
                                        else:raise Exception('Period unknown.')
                                if (scen+'ext' in list_experiments)  or  (scen+'-ext' in list_experiments):
                                    Var = xr.concat([Var0, Var1.sel(year=slice(Var0.year[-1].values + 1, None, None)), Var2.sel(year=slice(Var1.year[-1].values + 1, None, None))], dim='year')
                                    del Var0, Var1, Var2
                                    For = xr.concat([For0, For1.sel(year=slice(For0.year[-1].values + 1, None, None)), For2.sel(year=slice(For1.year[-1].values + 1, None, None))], dim='year')
                                    del For0, For1, For2
                                else:
                                    Var = xr.concat([Var0, Var1.sel(year=slice(Var0.year[-1].values + 1, None, None))], dim='year')
                                    del Var0, Var1
                                    For = xr.concat([For0, For1.sel(year=slice(For0.year[-1].values + 1, None, None))], dim='year')
                                    del For0, For1
                            else:
                                Var = load_Out(scen)[var].sel(config=oscar.config)
                                For = load_For(scen).sel(config=oscar.config)
                                mask_tmp = mask_all_exp[scen].sel(config=oscar.config)
                                if option_mask == 'mask_indiv':
                                    if np.isnan(eval_year).all()==False:mask_tmp = mask_tmp.sel(year=eval_year[1]).drop('year')
                                    elif 'sel_year' in treat:mask_tmp = mask_tmp.isel(year=int(treat.split('sel_year=')[1])).drop('year')
                                    else:raise Exception('Period unknown.')
                                    
                            ## correct drift using control
                            VarPi = load_Out(  dico_controls[scen]  )[var].sel(config=oscar.config)
                            ForPi = load_For(  dico_controls[scen]  ).sel(config=oscar.config)
                            Var = Var - VarPi + VarPi.mean('year')
                            # del VarPi

                        ## get values
                        if treat == 'cumsum' and all([year >= 1850 for year in norm_year]):
                            val = Var.cumsum('year').sel(year=slice(*eval_year)).mean('year') - Var.cumsum('year').sel(year=slice(*norm_year)).mean('year')
                        elif treat == 'cumsum' and norm_year == (1750., 1750.):
                            val = Var.cumsum('year').sel(year=slice(*eval_year)).mean('year')
                        elif treat == 'mean' and all([year >= 1850 for year in norm_year]):
                            val = Var.sel(year=slice(*eval_year)).mean('year') - Var.sel(year=slice(*norm_year)).mean('year')
                        elif treat == 'mean' and (norm_year == (1750., 1750.)  or  np.all(np.isnan(norm_year))):
                            val = Var.sel(year=slice(*eval_year)).mean('year')
                        elif 'sel_year' in treat and np.isnan(norm_year).all():
                            val = Var.isel(year=int(treat.split('sel_year=')[1]), drop=True) - Var.isel(year=0, drop=True)
                        else: 
                            raise RuntimeError('dunno what to do!')

                    ## special cases
                    else:

                        if no_reload:
                            print("Not reloading here, same scenario")
                        else:
                            if scen == "['1pctCO2','1pctCO2-bgc']":l_scen = ['1pctCO2','1pctCO2-bgc']
                            else:l_scen = [scen]

                            ## get all variables
                            Var,For,VarPi,ForPi,Par,mask,mask_tmp = {},{},{},{},{},{},1.*xr.ones_like(oscar.config)
                            for scen in l_scen:
                                if var != 'ECS':
                                    if ('ssp' in scen) or ('rcp' in scen) or (scen in ['G6solar']):
                                        Var0 = load_Out( dico_experiments_before[scen] ).sel(config=oscar.config)
                                        Var1 = load_Out(scen).sel(config=oscar.config)
                                        For0 = load_For(  dico_experiments_before[scen]  ).sel(config=oscar.config)
                                        For1 = load_For(scen).sel(config=oscar.config)
                                        if scen+'ext' in list_experiments: ## extension if any
                                            Var2 = load_Out(scen+'ext').sel(config=oscar.config)
                                            For2 = load_For(scen+'ext').sel(config=oscar.config)
                                            mask[scen] = mask_all_exp[scen+'ext'].sel(config=oscar.config)
                                        elif scen+'-ext' in list_experiments:
                                            Var2 = load_Out(scen+'-ext').sel(config=oscar.config)
                                            For2 = load_For(scen+'-ext').sel(config=oscar.config)
                                            mask[scen] = mask_all_exp[scen+'-ext'].sel(config=oscar.config)
                                        else:
                                            mask[scen] = mask_all_exp[scen].sel(config=oscar.config)
                                        if (scen+'ext' in list_experiments)  or  (scen+'-ext' in list_experiments):
                                            Var[scen] = xr.concat([Var0, Var1.sel(year=slice(Var0.year[-1].values + 1, None, None)), Var2.sel(year=slice(Var1.year[-1].values + 1, None, None))], dim='year')
                                            del Var0, Var1, Var2
                                            For[scen] = xr.concat([For0, For1.sel(year=slice(For0.year[-1].values + 1, None, None)), For2.sel(year=slice(For1.year[-1].values + 1, None, None))], dim='year')
                                            del For0, For1, For2
                                        else:
                                            Var[scen] = xr.concat([Var0, Var1.sel(year=slice(Var0.year[-1].values + 1, None, None))], dim='year')
                                            del Var0, Var1
                                            For[scen] = xr.concat([For0, For1.sel(year=slice(For0.year[-1].values + 1, None, None))], dim='year')
                                            del For0, For1
                                    else:
                                        Var[scen] = load_Out(scen).sel(config=oscar.config)
                                        For[scen] = load_For(scen).sel(config=oscar.config)
                                        mask[scen] = mask_all_exp[scen].sel(config=oscar.config)
                                    ## control
                                    VarPi[scen] = load_Out( dico_controls[scen] ).sel(config=oscar.config)
                                    ForPi[scen] = load_For(  dico_controls[scen]  ).sel(config=oscar.config)
                                    Var[scen] = Var[scen] - VarPi[scen] + VarPi[scen].mean('year')
                                ## specific case of ECS
                                Par[scen] = load_Par()
                                ## qdqpting parameters
                                if var != 'ECS':
                                    ## adapting parameters: Aland_0
                                    if 'Aland_0' in For[scen]:Par[scen]['Aland_0'] = For[scen]['Aland_0']
                                    ## adapting parameters
                                    if scen in ['1pctCO2-bgc','hist-bgc','ssp534-over-bgc','ssp585-bgc']:
                                        Par[scen]['D_CO2_rad'] = For[scen].D_CO2.sel(year=1850)
                                    elif scen in ['1pctCO2-rad']:
                                        Par[scen]['D_CO2_bgc'] = For[scen].D_CO2.sel(year=1850)
                            ## masks
                            if option_mask == 'mask_indiv':
                                if var == 'ECS':
                                    mask_tmp = mask_all_exp['abrupt-2xCO2'].sel(config=oscar.config).isel(year=-1).drop('year')
                                elif var == 'f_TCRE':
                                    mask_tmp = mask_all_exp['1pctCO2'].sel(config=oscar.config).isel(year=70).drop('year')
                                elif ('year_peak' in var)  or  ('value_peak' in var):
                                    mask_tmp = mask[scen]
                                elif 'sel_year' in treat:
                                    for scen in l_scen:mask_tmp *= mask[scen].isel(year=int(treat.split('sel_year=')[1])).drop('year')
                                elif np.isnan(eval_year).all()==False:
                                    for scen in l_scen:mask_tmp *= mask[scen].sel(year=eval_year[1]).drop('year')
                                else:raise Exception('Period unknown.')

                        ## ECS
                        if var == 'ECS':
                            if 'lambda_corr' in Par[scen]:
                                val = (Par[scen]['lambda_0'] * Par[scen]['lambda_corr'] * Par[scen]['rf_CO2'] * np.log(2)).sel(config=oscar.config)
                            else:
                                val = (Par[scen]['lambda_0'] * Par[scen]['rf_CO2'] * np.log(2)).sel(config=oscar.config)

                        ## RF halo
                        if var == 'RF_halo':
                            val = Var[scen]['RF_Xhalo'].sum('spc_halo', min_count=1).sel(year=slice(*eval_year)).mean('year')

                        ## ERFs
                        # if var =='ERF': ## ---> using RF_wmghg instead
                        #     val = (Var[scen]['RF'] + (Par[scen]['w_warm_bcsnow'] - 1) * Var[scen]['RF_BCsnow']).sel(year=slice(*eval_year)).mean('year')
                        if var == 'ERF_BCsnow':
                            val = Par[scen]['w_warm_bcsnow'] * Var[scen]['RF_BCsnow'].sel(year=slice(*eval_year)).mean('year')
                        if var == 'ERF_lcc':
                            val = Par[scen]['w_warm_lcc'] * Var[scen]['RF_lcc'].sel(year=slice(*eval_year)).mean('year')

                        ## TCRE
                        if var == 'f_TCRE':
                            T = Var[scen]['D_Tg'].isel(year=70, drop=True) \
                            - Var[scen]['D_Tg'].isel(year=0, drop=True)
                            E = (Var[scen]['Eff_comp'] + Var[scen]['D_Eluc'] + Var[scen]['D_Epf_CO2'].sum('reg_pf') + Var[scen]['D_Foxi_CH4']).cumsum('year').isel(year=70, drop=True) \
                            - (Var[scen]['Eff_comp'] + Var[scen]['D_Eluc'] + Var[scen]['D_Epf_CO2'].sum('reg_pf') + Var[scen]['D_Foxi_CH4']).cumsum('year').isel(year=0, drop=True)
                            val = 1E3 * T / E

                        ## AF
                        if var == 'f_AF':
                            S = (Var[scen]['D_Focean'] + Var[scen]['D_Fland']).cumsum('year').sel(year=slice(*eval_year)).mean('year') \
                            - (Var[scen]['D_Focean'] + Var[scen]['D_Fland']).cumsum('year').sel(year=slice(*norm_year)).mean('year')
                            E = (Var[scen]['Eff_comp'] + Var[scen]['D_Eluc'] + Var[scen]['D_Epf_CO2'].sum('reg_pf') + Var[scen]['D_Foxi_CH4']).cumsum('year').sel(year=slice(*eval_year)).mean('year') \
                            - (Var[scen]['Eff_comp'] + Var[scen]['D_Eluc'] + Var[scen]['D_Epf_CO2'].sum('reg_pf') + Var[scen]['D_Foxi_CH4']).cumsum('year').sel(year=slice(*norm_year)).mean('year')
                            val = 1 - S / E

                        ## ff-only AF
                        if var == 'f_AF_ff':
                            S = (Var[scen]['D_Focean'] + Var[scen]['D_Fland'] - Var[scen]['D_Eluc']).cumsum('year').sel(year=slice(*eval_year)).mean('year') \
                            - (Var[scen]['D_Focean'] + Var[scen]['D_Fland'] - Var[scen]['D_Eluc']).cumsum('year').sel(year=slice(*norm_year)).mean('year')
                            E = (Var[scen]['Eff_comp'] + Var[scen]['D_Epf_CO2'].sum('reg_pf') + Var[scen]['D_Foxi_CH4']).cumsum('year').sel(year=slice(*eval_year)).mean('year') \
                            - (Var[scen]['Eff_comp'] + Var[scen]['D_Epf_CO2'].sum('reg_pf') + Var[scen]['D_Foxi_CH4']).cumsum('year').sel(year=slice(*norm_year)).mean('year')
                            val = 1 - S / E

                        ## rate D_CO2
                        if var == 'rate_D_CO2':
                            xx = Var[scen]['D_CO2'].sel(year=slice(*eval_year)) ## would need to update xarray for xr.polyfit
                            val = np.polyfit( x=xx.year,y=xx.values , deg=1 )[0]

                        ## full land to atmospher carbon sink
                        if var in ['LandToAtm','Cum LandToAtm']:
                            FF = -1. * (Var[scen]['D_Fland']  - Var[scen]['D_Eluc']  - Var[scen]['D_Epf_CO2'].sum('reg_pf')  - 1.e-3 * Var[scen]['D_Epf_CH4'].sum('reg_pf')) ## assuming directly oxidation of CH4
                            if var == 'Cum LandToAtm':
                                if norm_year == (1750, 1750):
                                    val = FF.cumsum('year').sel(year=slice(*eval_year)).mean('year')
                                    ## estimating 1750-1850
                                    For0 = load_all_hist('RCP_5reg', LCC='gross')
                                    if For0.Aland_0.year==1750:
                                        Aland_1750 = For0.Aland_0.sel(data_LULCC='LUH2').drop('data_LULCC') ## reference of LUH2 in 1750
                                    else:
                                        raise Exception('Aland_0 of LUH2 is not in 1750')
                                    ## calculating equilibrium of OSCAR, defined in 1750
                                    for vv in ['cveg_0', 'csoil1_0', 'csoil2_0']:
                                        VarPi[scen][vv] = OSCAR[vv](VarPi[scen], Par[scen], ForPi[scen], recursive=True)
                                    ## calculating the difference in-between the 2 equilibriums
                                    diff =  (VarPi[scen]['cveg_0'] + VarPi[scen]['csoil1_0'] + VarPi[scen]['csoil2_0'] + VarPi[scen]['D_cveg'] + VarPi[scen]['D_csoil1'] + VarPi[scen]['D_csoil2']) * Par[scen]['Aland_0'] -\
                                            (VarPi[scen]['cveg_0'] + VarPi[scen]['csoil1_0'] + VarPi[scen]['csoil2_0']) * Aland_1750
                                    dd = diff.sum( ('bio_land','reg_land') ).mean('year')
                                    ## adding for budget over 1750-1850
                                    val = val - dd ## Land TO Atmosphere

                                else:
                                    val = FF.cumsum('year').sel(year=slice(*eval_year)).mean('year') - FF.cumsum('year').sel(year=slice(*norm_year)).mean('year')

                            elif var == 'LandToAtm':
                                if norm_year == (1750, 1750):
                                    val = FF.sel(year=slice(*eval_year)).mean('year')
                                else:
                                    val = FF.sel(year=slice(*eval_year)).mean('year') - FF.sel(year=slice(*norm_year)).mean('year')

                        if var == 'Cum D_Focean':
                            ## contribution after 1850
                            val = Var[scen]['D_Focean'].cumsum('year').sel(year=slice(*eval_year)).mean('year')
                            ## contribution over 1750-1850
                            for vv in ['D_Cosurf']:
                                VarPi[scen][vv] = OSCAR[vv](VarPi[scen], Par[scen], ForPi[scen], recursive=True)
                            val = val + VarPi[scen]['D_Cosurf'].sum('box_osurf').mean('year') ## Atmosphere TO Ocean, fact is -1.

                        ## RF of F-Gases or Montreal Gases
                        if var in ['F-Gases','Montreal']:
                            if var == 'F-Gases':
                                set_spc = [spc for spc in Par[scen].spc_halo.values if spc not in Par[scen].p_fracrel.dropna('spc_halo').spc_halo]
                            elif var == 'Montreal':
                                set_spc = [spc for spc in Par[scen].spc_halo.values if spc in Par[scen].p_fracrel.dropna('spc_halo').spc_halo]
                            val = Var[scen]['RF_Xhalo'].sel( spc_halo=set_spc ).sum('spc_halo').sel(year=slice(*eval_year)).mean('year')
                            del set_spc

                        ## beta, gamma
                        if var in ['gamma_land','beta_land','gamma_ocean','beta_ocean']:
                            if 'land' in var:
                                c_cou = (Var['1pctCO2']['D_Fland']-Var['1pctCO2']['D_Eluc']-Var['1pctCO2']['D_Epf_CO2'].sum('reg_pf',min_count=1)-Var['1pctCO2']['D_Epf_CH4'].sum('reg_pf',min_count=1)*1.e-3).cumsum('year')
                                c_bgc = (Var['1pctCO2-bgc']['D_Fland']-Var['1pctCO2']['D_Eluc']-Var['1pctCO2']['D_Epf_CO2'].sum('reg_pf',min_count=1)-Var['1pctCO2']['D_Epf_CH4'].sum('reg_pf',min_count=1)*1.e-3).cumsum('year')
                            elif 'ocean' in var:
                                c_cou = (Var['1pctCO2']['D_Focean']).cumsum('year')
                                c_bgc = (Var['1pctCO2-bgc']['D_Focean']).cumsum('year')
                            else:raise Exception('Domain?')
                            if 'gamma' in var:
                                val = (  (c_cou - c_bgc) / (Var['1pctCO2']['D_Tg'] - Var['1pctCO2-bgc']['D_Tg'])  ).sel(year=slice(*eval_year)).mean('year')
                            elif 'beta' in var:
                                val = (  (1. / Var['1pctCO2']['D_CO2']) * (c_bgc*Var['1pctCO2']['D_Tg'] - c_cou*Var['1pctCO2-bgc']['D_Tg']) / (Var['1pctCO2']['D_Tg'] - Var['1pctCO2-bgc']['D_Tg'])  ).sel(year=slice(*eval_year)).mean('year')
                            else:raise Exception("indic?")

                        ## transient climate linear sensitivity
                        if var == 'clim_sensi':
                            val = (  Var[scen]['D_Tg'] / Var[scen]['D_CO2']  ).sel(year=slice(*eval_year)).mean('year')

                        ## total atmospheric CO2
                        if var == 'CO2atm':
                            val = (Par[scen]['CO2_0'] + Var[scen]['D_CO2']).sel(year=slice(*eval_year)).mean('year')

                        ## year of peak
                        if 'year_peak' in var:
                            xx = Var[scen][ {'GSAT':'D_Tg', 'ERF':'RF_warm', 'atmCO2':'D_CO2'}[var[len('year_peak_'):]] ]*mask_tmp
                            ## applying running mean
                            xx = xx.rolling(year=21,center=True).mean()
                            if np.isnan(norm_year).all():
                                val = 2015. + ( xx.sel(year=slice(2015, None, None)) ).argmax(dim='year')
                            else:
                                val = 2015. + ( xx.sel(year=slice(2015, None, None)) - xx.sel(year=slice(*norm_year)).mean('year') ).argmax(dim='year')
                            if option_mask == 'mask_indiv':
                                val_tmp = val.values## to avoid to spam command shell during "val.sel(config=cfg).values+1"
                                mask_tmp = xr.concat( [mask_tmp.sel(config=cfg,year=min([mask_tmp.year[-1],val_tmp[list(mask_tmp.config.values).index(cfg)]+1])) for cfg in mask_tmp.config.values] , dim='config' )

                        ## value of peak
                        if 'value_peak' in var:
                            xx = (Var[scen][{'GSAT':'D_Tg', 'ERF':'RF_warm', 'atmCO2':'D_CO2'}[var[len('value_peak_'):]]]  +  (var[len('value_peak_'):]=='atmCO2')*Par[scen]['CO2_0'] )  *  mask_tmp
                            ## applying running mean
                            xx = xx.rolling(year=21,center=True).mean()
                            if np.isnan(norm_year).all():
                                yy = 2015. + ( xx.sel(year=slice(2015, None, None)) ).argmax(dim='year')
                                val = ( xx.sel(year=slice(2015, None, None)) ).max(dim='year')
                            else:
                                yy = 2015. + ( xx.sel(year=slice(2015, None, None)) - xx.sel(year=slice(*norm_year)).mean('year') ).argmax(dim='year')
                                val = ( xx.sel(year=slice(2015, None, None)) - xx.sel(year=slice(*norm_year)).mean('year') ).max(dim='year')
                            if option_mask == 'mask_indiv':
                                yy_tmp = yy.values## to avoid to spam command shell during "yy.sel(config=cfg).values+1"
                                mask_tmp = xr.concat( [mask_tmp.sel(config=cfg,year=min([mask_tmp.year.values[-1],yy_tmp[list(mask_tmp.config.values).index(cfg)]+1])) for cfg in mask_tmp.config.values] , dim='config' )

                    ## not forcing mask!! will force it AFTER, for calculation such as indicators, but weights must not be masked. Otherwise, masked values are counted as 1 by xarray.
                    ##***************
                    # ## forcing mask
                    # val *= mask_tmp
                    ##***************

                    ## add to dataset
                    oscar['x'].loc[{'index':i}][:] = fact * val.compute()
                    oscar['m'].loc[{'index':i}][:] = mask_tmp.copy()
                    del val

                ## FIT DISTRIB!
                if True:

                    ## check whether symetrical and keep only actual values
                    if np.isnan([x_vl, x_l, x_c, x_u, x_vu]).sum() == 0:
                        symetrical = math.isclose(abs(x_c - x_u), abs(x_c - x_l)) & math.isclose(abs(x_c - x_vu), abs(x_c - x_vl))
                        xy_list = [(x_vl, 0.05), (x_l, 0.17), (x_c, 0.50), (x_u, 0.83), (x_vu, 0.95)]
                        x_val = [x_c, (x_u - x_l) / 2.]
                    elif np.isnan([x_l, x_u]).sum() == 2:
                        symetrical = math.isclose(abs(x_c - x_vu), abs(x_c - x_vl))
                        xy_list = [(x_vl, 0.05), (x_c, 0.50), (x_vu, 0.95)]
                        x_val = [x_c, (x_vu - x_vl) / 3.]
                    elif np.isnan([x_vl, x_vu]).sum() == 2:
                        symetrical = math.isclose(abs(x_c - x_u), abs(x_c - x_l))
                        xy_list = [(x_l, 0.17), (x_c, 0.50), (x_u, 0.83)]
                        x_val = [x_c, (x_u - x_l) / 2.]
                    else: raise RuntimeError('weird range provided!')

                    ## set list of distrib to test (in order)
                    if symetrical: distrib_list = ['N', 'GN1']
                    else: distrib_list = ['PN', 'EN', 'GL', 'NIG'] #+ ['LN', 'PLN', 'IW', 'LL', 'GX', 'GN2'] #+ ['SN', 'ESN', 'SGN', 'FGSN3', 'FGSN5']

                    ## loop to fit on distrib
                    for distrib in distrib_list:

                        ## actual fit
                        dist = distrib_dict[distrib]
                        param, is_okay = fit_distrib(dist.cdf, xy_list, x_val)

                        ## break loop if good enough
                        is_close = test_fit(dist.cdf([x_vl, x_l, x_c, x_u, x_vu], *param)).all()
                        if is_okay and is_close: break

                    ## add distrib and weights to dataset
                    oscar.distrib.loc[{'index':i}] = distrib + str(param)
                    oscar['w'].loc[{'index':i}][:] = dist.pdf(oscar['x'].loc[{'index':i}], *param)

                    ## distrib plot
                    if True:
                        xx = np.linspace(np.nanmin([x_vl - abs(x_c), x_l - 2*abs(x_c)]), np.nanmax([x_vu + abs(x_c), x_u + 2*abs(x_c)]), 2000)
                        n_sub = np.ceil(max(np.roots([1, 1, -oscar.index.size])))
                        plt.subplot(int(n_sub), int(n_sub)+1, NN+1); NN += 1
                        plt.plot(xx, dist.cdf(xx, *param), label='cdf')
                        plt.plot(xx, dist.pdf(xx, *param) / np.nanmax(dist.pdf(xx, *param)), ls='--', label='pdf')
                        plt.plot([x_vl, x_l, x_c, x_u, x_vu], [0.05, 0.17, 0.50, 0.83, 0.95], ls='none', marker='+', ms=8, mew=2, label='fit:\n'+ dist.name + '\n'+str(is_close))
                        plt.legend(loc=0, fontsize='xx-small')
                        plt.title(ind, fontsize='x-small')
                        plt.xticks(fontsize='x-small')
                        plt.yticks(fontsize='x-small')
                        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.20, hspace=0.40)

        ## save
        tmp = oscar.copy(deep=True)#.drop('distrib')
        ## adding a line for names of index
        tmp.coords['indicator'] = rcmip['indicator'].sel(index=[i for i in rcmip.index.values if str(rcmip.indicator[i].values) in oscar_indicators.keys()])
        for var in ['RCMIP variable', 'RCMIP region', 'RCMIP scenario', 'evaluation_period_start', 'evaluation_period_end', 'norm_period_start', 'norm_period_end', 'unit']:
            tmp.coords[var] = rcmip[var].sel(index=[i for i in rcmip.index.values if str(rcmip.indicator[i].values) in oscar_indicators.keys()])
        if option_full_configs:
            tmp.to_netcdf(folder_rcmip + 'oscar_indicators_full-configs_'+option_mask+'.nc', encoding={var:{'zlib':True, 'dtype':{'x':np.float32,'w':np.float32,'distrib':str}[var]} for var in ['x','w','distrib']})
        else:
            tmp.to_netcdf(folder_rcmip + 'oscar_indicators.nc', encoding={var:{'zlib':True, 'dtype':{'x':np.float32,'w':np.float32,'distrib':str}[var]} for var in ['x','w','distrib']})
##################################################
##################################################








##################################################
##   TEST FUNCTIONS
##################################################

## test weights with pct
def test_w_pct(ind_list=ind_list, w_corr=1., pct_list=[0.05, 0.17, 0.50, 0.83, 0.95]):
    wi_list = [rcmip.indicator.values.tolist().index(ind) for ind in ind_list]
    print('==========================================')
    print('indicator', 50*' ', 'vll    ', 'll    ', 'c    ', 'lu    ', 'vlu    ', 'vll (abs)', 'll (abs)', 'c (abs)', 'lu (abs)', 'vlu (abs)', 'units', sep=5*' ')
    for i in oscar.index.values:
        x_vl, x_l, x_c, x_u, x_vu = [float(rcmip[var][i]) for var in ['very_likely__lower', 'likely__lower', 'central', 'likely__upper', 'very_likely__upper']]
        name = str(rcmip.indicator[i].values)
        abs_val = [weighted_quantile(oscar.x.sel(index=i).values, pct, oscar.w.sel(index=wi_list).prod('index').values) for pct in pct_list]
        abs_diff = [val - x for val, x in zip(abs_val, [x_vl, x_l, x_c, x_u, x_vu])]
        rel_diff = [val / x - 1 for val, x in zip(abs_val, [x_vl, x_l, x_c, x_u, x_vu])]
        units = oscar_indicators[str(rcmip.indicator[i].values)][-1]
        print(name + ' '*(70-len(name)), *['{: 4.0f}'.format(100*val)+'%     ' for val in rel_diff], ' | ', *['{: 4.2f}'.format(val)+'     ' for val in abs_diff], '('+units+')')
    print('==========================================')


## test function with mean/std
def test_w_sigma(ind_list=ind_list, w_corr=1., sigma_list=[-1.645, -0.954, 0., 0.954, 1.645]):
    wi_list = [rcmip.indicator.values.tolist().index(ind) for ind in ind_list]
    print('==========================================')
    print('indicator', 50*' ', 'vll    ', 'll    ', 'c    ', 'lu    ', 'vlu    ', 'vll (abs)', 'll (abs)', 'c (abs)', 'lu (abs)', 'vlu (abs)', 'units', sep=5*' ')
    for i in oscar.index.values:
        x_vl, x_l, x_c, x_u, x_vu = [float(rcmip[var][i]) for var in ['very_likely__lower', 'likely__lower', 'central', 'likely__upper', 'very_likely__upper']]
        name = str(rcmip.indicator[i].values)
        mean = np.average(oscar.x.sel(index=i).values, weights=oscar.w.sel(index=wi_list).prod('index').values * w_corr)
        std = np.sqrt(np.cov(oscar.x.sel(index=i).values, aweights=oscar.w.sel(index=wi_list).prod('index').values * w_corr))
        abs_val = [(mean + sigma * std) for sigma in sigma_list]
        abs_diff = [val - x for val, x in zip(abs_val, [x_vl, x_l, x_c, x_u, x_vu])]
        rel_diff = [val / x - 1 for val, x in zip(abs_val, [x_vl, x_l, x_c, x_u, x_vu])]
        units = oscar_indicators[str(rcmip.indicator[i].values)][-1]
        print(name + ' '*(70-len(name)), *['{: 4.0f}'.format(100*val)+'%     ' for val in rel_diff], ' | ', *['{: 4.2f}'.format(val)+'     ' for val in abs_diff], '('+units+')')
    print('==========================================')


def weighted_quantiles_adhoc(INPUT,weights,quantiles):
    hist,edges = np.histogram( a=INPUT, bins=weights.size, weights=weights, density=True )
    hist /= np.sum( hist * np.diff(edges) )
    cumhist = np.cumsum( hist * np.diff(edges) )
    return np.interp(x=quantiles, xp=cumhist, fp=0.5*(edges[1:]+edges[:-1]) )


# ## indicators to use for weighting OSCAR by Yann Quilcaille for RCMIP-Phase 2
# ind_list = ['Cumulative Net Land to Atmosphere Flux|CO2 World esm-hist-2011',
#             'Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1980',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-1990',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2000',
#             'Rate Increase Atmospheric Concentrations|CO2 World esm-hist-2002',
#             ]


## indicators to use for weighting OSCAR by Yann Quilcaille for CMIP6
ind_list = ['Cumulative Net Ocean to Atmosphere Flux|CO2 World esm-hist-2011',
            'Surface Air Ocean Blended Temperature Change World ssp245 2000-2019',
            'Cumulative compatible emissions CMIP5 historical-CMIP5',
            'Cumulative compatible emissions CMIP5 RCP2.6',
            'Cumulative compatible emissions CMIP5 RCP4.5',
            'Cumulative compatible emissions CMIP5 RCP6.0',
            'Cumulative compatible emissions CMIP5 RCP8.5',
            ]




## idea, add few lines to sort the order in which indicators are calculated, so that they make use of the same scenarios/variables, hence not reload several time the same data.

##################################################
##   EXTRACT INDICATORS
##################################################
if __name__ == '__main__':

    ## load if already calculated
    if not get_indicators:
        if option_full_configs:
            with xr.open_dataset(folder_rcmip + 'oscar_indicators_full-configs_'+option_mask+'.nc') as TMP: oscar = TMP.load()            
        else:
            with xr.open_dataset(folder_rcmip + 'oscar_indicators.nc') as TMP: oscar = TMP.load()

    ## plot to check distrib
    if False:
        plt.figure()
        n_sub = np.ceil(max(np.roots([1, 1, -len(ind_list)]))); NN = 0
        for i in oscar.index.values:
            ind = str(rcmip.indicator[i].values)
            if ind in ind_list:
                x_vl, x_l, x_c, x_u, x_vu = [float(rcmip[var][i]) for var in ['very_likely__lower', 'likely__lower', 'central', 'likely__upper', 'very_likely__upper']]
                xx = np.linspace(np.nanmin([x_vl - abs(x_c), x_l - 2*abs(x_c)]), np.nanmax([x_vu + abs(x_c), x_u + 2*abs(x_c)]), 2000)
                distrib = oscar.distrib.sel(index=i).values.tolist().split('[')[0]
                param = [float(par) for par in oscar.distrib.sel(index=i).values.tolist().split('[')[1][:-1].split(', ')]
                dist = distrib_dict[distrib]
                plt.subplot(int(n_sub), int(n_sub)+1, NN+1); NN += 1
                plt.hist(oscar.x.sel(index=i), density=True, bins=50, histtype='step', ls=':', label='OSCAR (uncon.)')
                plt.hist(oscar.x.sel(index=i), density=True, weights=oscar.w.sel(index=i), bins=50, histtype='step', ls='--', label='OSCAR (1 con.)')
                plt.hist(oscar.x.sel(index=i), density=True, weights=oscar.w.prod('index'), bins=50, histtype='step', ls='-', label='OSCAR (all con.)')
                plt.plot(xx, dist.pdf(xx, *param), lw=2, label='RCMIP')
                plt.legend(loc=0, fontsize='xx-small')
                plt.title(ind, fontsize='x-small')
                plt.xticks(fontsize='x-small')
                plt.yticks(fontsize='x-small')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.20, hspace=0.40)

    ## select list of final indicators (optional)
    if ind_list is None: ind_list = [rcmip.indicator.values[i] for i in oscar.index]
    wi_list = [rcmip.indicator.values.tolist().index(ind) for ind in ind_list]

    ## write csv with pct
    option_csv_full = True
    if option_csv_full:
        writer = csv.writer(open(folder_rcmip + 'OSCAR_rcmip_' + file_ranges.split('.csv')[0].split('_')[-1] + '_weighted_pct-differences_'+option_mask+'.csv', 'w', newline=''))
        writer.writerow(['indicator', 'vll (oscar)', 'll (oscar)', 'c (oscar)', 'lu (oscar)', 'vlu (oscar)', 'vll (indic)', 'll (indic)', 'c (indic)', 'lu (indic)', 'vlu (indic)', 'vll (diff rel)', 'll (diff rel)', 'c (diff rel)', 'lu (diff rel)', 'vlu (diff rel)', 'vll (diff abs)', 'll (diff abs)', 'c (diff abs)', 'lu (diff abs)', 'vlu (diff abs)'])
    else:
        writer = csv.writer(open(folder_rcmip + 'OSCAR_rcmip_' + file_ranges.split('.csv')[0].split('_')[-1] + '_weighted_pct.csv', 'w', newline=''))
        writer.writerow(['indicator', 'vll', 'll', 'c', 'lu', 'vlu'])
    for i in oscar.index.values:
        x_vl, x_l, x_c, x_u, x_vu = [float(rcmip[var][i]) for var in ['very_likely__lower', 'likely__lower', 'central', 'likely__upper', 'very_likely__upper']]
        name = [str(rcmip.indicator[i].values)]
        # cfg = np.where( (~np.isnan(oscar.x).sel(index=i).values)  &  (oscar.m.sel(index=i).values) )[0]
        cfg = np.where( (~np.isnan(oscar.x).sel(index=i).values)  &  ~np.isnan(oscar.m.sel(index=i).values) )[0]
        abs_val = [weighted_quantile(oscar['x'].sel(index=i).isel(config=cfg).values, pct, oscar.w.sel(index=wi_list).isel(config=cfg).prod('index').values) for pct in [0.05, 0.17, 0.50, 0.83, 0.95]]
        if option_csv_full:
            abs_diff = [val - x for val, x in zip(abs_val, [x_vl, x_l, x_c, x_u, x_vu])]
            rel_diff = [val / x - 1 for val, x in zip(abs_val, [x_vl, x_l, x_c, x_u, x_vu])]
            units = [oscar_indicators[str(rcmip.indicator[i].values)][-1]]
            writer.writerow(name + abs_val + [x_vl, x_l, x_c, x_u, x_vu] + rel_diff + abs_diff + units)
        else:
            writer.writerow(name + abs_val)
    writer = csv.writer(open(folder_rcmip + 'empty', 'wb'))



##################################################
##   PLOTS
##################################################
if __name__ == '__main__':

    ## weights
    #ind_list = ['Surface Air Temperature Change World ssp119 2041-2060', 'Surface Air Temperature Change World ssp119 2081-2100']
    if ind_list is None: ind_list = [rcmip.indicator.values[i] for i in oscar.index]
    wi_list = [rcmip.indicator.values.tolist().index(ind) for ind in ind_list]
    w = oscar.w.sel(index=wi_list).prod('index')
    #w = xr.full_like(w, 1)

    ## TEMPERATURE TIMESERIES
    plt.figure()

    ## historical
    plt.subplot(2, 3, 1)
    
    VAR = 'D_Tg'
    Var = load_Out('historical')[VAR].sel(config=w.config)
    Var0 = load_Out('piControl')[VAR].sel(config=w.config)
    Var = Var - Var0 + Var0.mean('year')
    Var = Var - Var.sel(year=slice(1850, 1900)).mean('year')
    Var = oscar_indicators['Surface Air Ocean Blended Temperature Change World ssp245 1995-2014'][2] * Var.compute(); del Var0

    median = np.array([weighted_quantile(Var.sel(year=year).values, 0.50, w) for year in Var.year])
    lower = np.array([weighted_quantile(Var.sel(year=year).values, 0.05, w) for year in Var.year])
    upper = np.array([weighted_quantile(Var.sel(year=year).values, 0.95, w) for year in Var.year])
    mean = np.average(Var, axis=-1, weights=w)
    std = np.sqrt(np.average((Var - mean[...,np.newaxis])**2, axis=-1, weights=w))

    plt.fill_between(Var.year, lower, upper, color='k', alpha=.5)
    plt.plot(Var.year, median, color='k', lw=2, label='historical')
    plt.plot(Var.year, mean, color='k', lw=1, ls='--')
    plt.plot(Var.year, mean - 1.645*std, color='k', lw=0.5, ls=':')
    plt.plot(Var.year, mean + 1.645*std, color='k', lw=0.5, ls=':')

    for i in rcmip.index:
        ind = str(rcmip.indicator.sel(index=i).values)
        if 'Surface Air Ocean Blended Temperature Change World' in ind and 'ssp245' in ind and rcmip.sel(index=i).evaluation_period_start < 2014:
            plt.fill_between([rcmip.sel(index=i).evaluation_period_start, rcmip.sel(index=i).evaluation_period_end], [rcmip.sel(index=i).very_likely__upper, rcmip.sel(index=i).very_likely__upper], [rcmip.sel(index=i).very_likely__lower, rcmip.sel(index=i).very_likely__lower], color='r', alpha=0.5)
            plt.plot([rcmip.sel(index=i).evaluation_period_start, rcmip.sel(index=i).evaluation_period_end], [rcmip.sel(index=i).central, rcmip.sel(index=i).central], color='r', lw=2, label='RCMIP')

    plt.title('Surface Air Ocean Blended Temperature Change World ' + 'historical')

    ## SSPs
    for n,ssp in enumerate(['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']):
        ax = plt.subplot(2, 3, n+2)

        Var = load_Out(ssp)[VAR].sel(config=w.config)
        Var1 = load_Out('historical')[VAR].sel(config=w.config)
        Var = xr.concat([Var1, Var.sel(year=slice(Var1.year[-1].values + 1, None, None))], dim='year')
        Var0 = load_Out('piControl')[VAR].sel(config=w.config)
        Var = Var - Var0 + Var0.mean('year')
        Var = Var - Var.sel(year=slice(1995, 2014)).mean('year')
        Var = Var.compute(); del Var0, Var1

        median = np.array([weighted_quantile(Var.sel(year=year).values, 0.50, w) for year in Var.year])
        lower = np.array([weighted_quantile(Var.sel(year=year).values, 0.05, w) for year in Var.year])
        upper = np.array([weighted_quantile(Var.sel(year=year).values, 0.95, w) for year in Var.year])
        mean = np.average(Var, axis=-1, weights=w)
        std = np.sqrt(np.average((Var - mean[...,np.newaxis])**2, axis=-1, weights=w))

        plt.fill_between(Var.year[150:], lower[150:], upper[150:], color='k', alpha=.5)
        plt.plot(Var.year[150:], median[150:], color='k', lw=2, label=ssp)
        plt.plot(Var.year[150:], mean[150:], color='k', lw=1, ls='--')
        plt.plot(Var.year[150:], mean[150:] - 1.645*std[150:], color='k', lw=0.5, ls=':')
        plt.plot(Var.year[150:], mean[150:] + 1.645*std[150:], color='k', lw=0.5, ls=':')

        for i in rcmip.index:
            ind = str(rcmip.indicator.sel(index=i).values)
            if 'Surface Air Temperature Change World' in ind and ssp in ind and rcmip.sel(index=i).evaluation_period_start > 2014:
                plt.fill_between([rcmip.sel(index=i).evaluation_period_start, rcmip.sel(index=i).evaluation_period_end], [rcmip.sel(index=i).very_likely__upper, rcmip.sel(index=i).very_likely__upper], [rcmip.sel(index=i).very_likely__lower, rcmip.sel(index=i).very_likely__lower], color='r', alpha=0.5)
                plt.plot([rcmip.sel(index=i).evaluation_period_start, rcmip.sel(index=i).evaluation_period_end], [rcmip.sel(index=i).central, rcmip.sel(index=i).central], color='r', lw=2, label='RCMIP')

        plt.title('Surface Air Temperature Change World ' + ssp)


    ## SOME OTHER METRICS
    plt.figure()
    for n, ind in enumerate(['Transient Climate Response', 'Transient Climate Response to Emissions', 'Equilibrium Climate Sensitivity', 'Airborne Fraction|CO2 World 1pctCO2 1850-1990']):
        plt.subplot(2, 2, n+1)
        i = int(rcmip.index.where(rcmip.indicator==ind).dropna('index').values[0])
        
        plt.plot([weighted_quantile(oscar.x.sel(index=i), pct, w) for pct in np.arange(0, 1.01, 0.01)], np.arange(0, 1.01, 0.01), 'k', label='OSCAR')
        plt.plot([rcmip[val].sel(index=i) for val in ['very_likely__lower', 'likely__lower', 'central', 'likely__upper', 'very_likely__upper']], [0.05, 0.17, 0.50, 0.83, 0.95], ls='none', marker='+', ms=9, mew=3, color='r')
        
        plt.xlabel(str(rcmip.unit.sel(index=i).values))
        plt.ylabel('CDF')
        plt.title(ind)





##################################################
##   CREATING EXTRA FILE FOR RCMIP phase 2 PLOTS
##################################################
if True:
    n_sample = 10

    ## load if already calculated
    if not get_indicators:
        if option_full_configs:
            with xr.open_dataset(folder_rcmip + 'oscar_indicators_full-configs_'+option_mask+'.nc') as TMP: oscar = TMP.load()            
        else:
            with xr.open_dataset(folder_rcmip + 'oscar_indicators.nc') as TMP: oscar = TMP.load()

    ## select list of final indicators (optional)
    if ind_list is None: ind_list = [rcmip.indicator.values[i] for i in oscar.index]
    wi_list = [rcmip.indicator.values.tolist().index(ind) for ind in ind_list]

    ## write csv with pct
    quantiles = np.arange(0.,1.+1.e-10,0.01)
    writer = csv.writer(open(folder_rcmip + 'OSCAR_rcmip_complement-distributions.csv', 'w', newline=''))
    writer.writerow( ['indicator','units'] + ['value_'+str(ii) for ii in range(n_sample * quantiles.size)] )
    for i in oscar.index.values:
        name = [str(rcmip.indicator[i].values)]
        cfg = np.where( ~np.isnan((oscar.x * oscar.m).sel(index=i).values) )[0]
        ## deducing from the distribution an ensemble of values by placing n_sample in every bin
        abs_val = np.array([weighted_quantile((oscar.x * oscar.m).sel(index=i).isel(config=cfg).values, pct, oscar.w.sel(index=wi_list).isel(config=cfg).prod('index').values) for pct in quantiles])
        hist_val = np.hstack(abs_val[:-1,np.newaxis] * np.repeat(1.,n_sample) + np.diff(abs_val)[:,np.newaxis] * np.arange(0,1.,1./n_sample))
        ## checking
        if False:
            plt.plot( 0.5*(abs_val[:-1]+abs_val[1:]), np.diff(quantiles) / np.diff(abs_val), ls='', marker='*')
            out = plt.hist( hist_val,bins=n_sample * quantiles.size , density=True, alpha=0.25)
            out = plt.hist( hist_val,bins=quantiles.size , density=True, alpha=0.25)
        ## adding units
        units = [oscar_indicators[str(rcmip.indicator[i].values)][-1]]
        writer.writerow(name + units + list(hist_val))
    writer = csv.writer(open(folder_rcmip + 'empty', 'wb'))




