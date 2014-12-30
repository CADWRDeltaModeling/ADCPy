# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:56:12 2014

@author: blsaenz
"""

import os
import adcpy
import adcpy_recipes
import transect_preprocessor

transects, outpath = \
  transect_preprocessor.transect_preprocessor('trn_pre_input_RIO.py','000','ADCPRdiWorkhorseData')
for t in transects:

    t.average_ensembles(10)
    t.kernel_smooth(3)
    t.extrapolate_boundaries()
    t = adcpy_recipes.transect_rotate(t,'no transverse flow')
    Uflows, Umean, survey_area, streamwise_area = t.calc_crossproduct_flow()
    print t.source, "--", t.date_time_str()[0], "--", Umean, 'm^3/s'
    adcpy.plot.plot_flow_summmary(t,title=t.source,ures=0.1,vres=0.1,
                                  use_grid_flows=False)

#    t.xy_regrid(dxy=5,dz=0.25)
#    t.kernel_smooth(3)
#    t.extrapolate_boundaries()
#    t = adcpy_recipes. transect_rotate(t,'no transverse flow')
#    scalar_mean_velocity,depth_ave_velocity,flow,survey_area = \
#      adcpy_recipes.calc_transect_flows_from_uniform_velocity_grid(t,use_grid_only=True)
#    print t.source, "--", t.date_time_str()[0], "--", flow[0], 'm^3/s'
#    adcpy.plot.plot_flow_summmary(t,title=t.source,ures=0.1,vres=0.1,
#                                  use_grid_flows=True)

    adcpy.plot.plt.savefig(os.path.join(outpath,t.source+'.png'))
