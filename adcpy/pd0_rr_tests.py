# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 09:54:28 2014

@author: blsaenz
"""
import numpy as np
import sys, os
import datetime
import matplotlib.animation as animation
from matplotlib.dates import num2date,date2num
sys.path.append(r'C:\\svn\\Code\\Python\\rma\\ADCPy\\adcpy')
from . import adcpy
#a = adcpy.open_adcp(r'Z:\Downloads\riverray\Montezuma_National_Steel\2013\NSL072213\NSL072213_010.PD0')
#a = adcpy.open_adcp(r'Y:\temp\channelmaster\BDT080613.000','ADCPRdiChannelmasterData')#,num_av=1)

a = adcpy.open_adcp(r'Z:\Downloads\riverray\Montezuma_National_Steel\2014\NSL082114_0\NSL082114_0_001.PD0','ADCPRdiRiverRayData')#,num_av=1)
a.lonlat_to_xy(r'EPSG:26910')
adcpy.plot.plot_uvw_velocity(a,ures=0.01,vres=0.01,wres=0.01,)
adcpy.plot.show()

print(a.xy)

#(a,b) = a.split_by_ensemble((70,))

#fig = adcpy.plot.plot_uvw_velocity(a,'uv',ures=0.01,vres=0.01,match_scales=False)
#adcpy.plot.plot_ensemble_uv(a,0,fig=None,title=None,n_vectors=50,return_panel=False)
#ani = adcpy.plot.animate_plot_ensemble_uv(a,100,span=3)#,interval=1000,span=None,fig=None,title=None,n_vectors=50)
#animation.MovieWriterRegistry.list()
#matplotlib.animation.AVConvWriter(fps=5, codec=None, bitrate=None, extra_args=None, metadata=None)
#writer = animation.writers['avconv'](fps=4)
#ani.save('demo.mp4',writer=writer,dpi=100)

#adcpy.plot.plt.show()

