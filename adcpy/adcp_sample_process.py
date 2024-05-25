# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:51:43 2014

@author: blsaenz
"""


import numpy as np
import sys, os
import datetime
from matplotlib.dates import num2date,date2num
sys.path.append(r'C:\\svn\\Code\\Python\\rma\\ADCPy\\adcpy')
from . import adcpy
from . import adcpy_recipes
os.chdir(r'Y:\temp\adcp_sample_data')


xy_projection = r'EPSG:26910'
adcp_face_depth = 0.4  # m
regrid_h_resolution = 0.5 # m in horizontal
regrid_v_resolution = 0.5 # m in horizontal


files = ['GEO4thRelease653r.000',
         'GEO4thRelease624r.000']

mcm = None
for f in files:

    f_base,ext = os.path.splitext(f)

    a = adcpy.open_adcp(f,'ADCPRdiWorkhorseData',adcp_depth=adcp_face_depth)
    a.lonlat_to_xy(xy_srs=xy_projection)
    fig = adcpy.plot.plot_uvw_velocity(a,'uvw',ures=0.01,vres=0.01,vmin=-0.4,vmax=0.4)
    fig.savefig(f_base + '.uvw_raw.png')

    a.remove_sidelobes()
    a.sd_drop()
    a.sd_drop(sd_axis='ensemble')
    a.xy_regrid(dxy=regrid_h_resolution,
                    dz=regrid_v_resolution)
    a.kernel_smooth(kernel_size = 3)
    r = adcpy_recipes.transect_rotate(a,'no transverse flow')
    
    r.extrapolate_boundaries()
    
    r.write_nc(f_base +'.nc',True)
    
    fig = adcpy.plot.plot_uvw_velocity(r,'uvw',ures=0.01,vres=0.01,vmin=-0.4,vmax=0.4)
    fig.savefig(f_base + '.uvw.png')
    fig2 = adcpy.plot.plot_ensemble_mean_vectors(r,n_vectors = 50)
    fig2.savefig(f_base + '.mean_vecs.png')

    
    # other things to play with
    #adcpy_recipes.write_csv_velocity_array - write matlab-readable text files
    #adcpy_recipes.write_csv_velocity_db - write column data for database import
