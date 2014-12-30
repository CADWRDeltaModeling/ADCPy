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
import adcpy
a = adcpy.open_adcp(r'Y:\temp\channelmaster\BDT080613.000','ADCPRdiChannelmasterData')#,num_av=1)

(b,c) = a.split_by_ensemble((70,))

fig = adcpy.plot.plot_uvw_velocity(b,'uv',ures=0.01,vres=0.01,match_scales=False)
#adcpy.plot.plot_ensemble_uv(b,0,fig=None,title=None,n_vectors=50,return_panel=False)
ani = adcpy.plot.animate_plot_ensemble_uv(a,100,span=3)#,interval=1000,span=None,fig=None,title=None,n_vectors=50)
writer = animation.writers['avconv'](fps=4)
ani.save('demo.mp4',writer=writer,dpi=100)
#adcpy.plot.plt.show()

