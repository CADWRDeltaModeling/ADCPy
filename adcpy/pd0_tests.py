# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 09:54:28 2014

@author: blsaenz
"""
# import required python modules
import numpy as np
import sys, os
import datetime
import matplotlib.animation as animation
from matplotlib.dates import num2date,date2num

# temporarily add the path to ADCPy to the path in order to import it
sys.path.append(r'C:\\svn\\Code\\Python\\rma\\ADCPy\\adcpy')
from . import adcpy

# read the raw Channelmaster data file into an ADCPData subclass 
#(ADCPRdiChannelmasterData) object, returned as variable ‘a’.
a = adcpy.open_adcp(r'Y:\temp\channelmaster\BDT080613.000',   
                    'ADCPRdiChannelmasterData')#,num_av=1)

# split the ADCP data into two new ADCPData objects, with ‘b’ containing the 
# first 70 ensembles, and ‘c’ containing the rest
(b,c) = a.split_by_ensemble((70,))

# generate a velocity profile plot, showing the first 70 ensemble u,v 
# velocities vs. time
fig = adcpy.plot.plot_uvw_velocity(b,'uv',ures=0.01,vres=0.01,match_scales=False)
adcpy.plot.plt.savefig(r'Y:\temp\channelmaster\demo_70_ens.png')

# generate an animation of ensemble velocity over time, using 100 frames, 
# with each frame showing every 3rd ensemble
ani = adcpy.plot.animate_plot_ensemble_uv(a,100,span=3)

# save the animation to an MP4 movie file at 4 fps, using the independently-
# installed avconv program
writer = animation.writers['avconv'](fps=4)
ani.save(r'Y:\temp\channelmaster\demo.mp4',writer=writer,dpi=100)

#adcpy.plot.plt.show()

