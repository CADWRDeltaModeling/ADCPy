# -*- coding: utf-8 -*-
"""Production example script that averages repeated tranects, resolving and visualizing secondary circulation
Driver script that is designed to average repeated ADCP transect observations
in an effort of reduce measurement error and better resolve non-steamwise and 
velocities and secondary circulation features.  A summary of script functions:
    1) Assess an input directory for ADCP observations (raw files) that match 
       in space and time.
    2) Group matchcing ADCP observations into groups of a maxium number for 
       averaging.
    3) Pre-process raw ADCP observations before averaging as appropriate.
    4) Bin-average pre-processed ADCP observation velcotities
    5) Generate netcdf and/or CSV output files of bin-average velocities
    6) Generate various plots of streamwise, depth averaged, 3D velocities
       and secondary circulation features.
       
The script options are listed and described immediatly below this comment 
block.

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""
## START script options #######################################################

# bin-averaging paramterters
avg_dxy                        = 2.0          # Horizontal resolution of averaging bins {m}
avg_dz                         = 0.25         # Vertical resolution of averaging bins {m}
avg_max_gap_m                  = 30.0         # Maximum distance allowed between ADCP observations when averaging {m}
avg_max_gap_minutes            = 20.0         # Maximum time allowed between ADCP observations when averaging {m}
avg_max_group_size             = 6            # Maximum number of ADCP observations to average {m}
avg_bin_sd_drop                = 3            # Maximum number of ADCP observations to average {m}
avg_normal_to_flow             = False

# post-average options
avg_rotation                   = 'Rozovski'   # One of ['Rozovski','no transverse flow','principal flow','normal',None]
avg_std_drop                   = 3.0          # Standard deviation of velocity, above which samples are dropped from analysis {0.0=no dropping, 2.0-3.0 typically [number of standard deviations]}
avg_std_interp                 = True         # Perform interpolation of holes in velocity profiles left by high standard deviation removal {typically True with std_drop > 0.0}
avg_smooth_kernel              = 5            # Smooth velocity data using a square kernel box-filter, with side dimension = avg_smooth_kernel.  0 = no kernel smoothing
avg_save_netcdf                = True         # Save bin-averaged velocities as an ADCPData netcdf file
avg_save_csv                   = True         # Save bin-averaged velocities as a CSV text file
avg_plot_xy                    = True         # Generate a composite plot of survey location(s) of original ADCP ensembles
avg_plot_avg_n_sd              = True         # Generate image plots of bin-averaged U,V,W velocities, and the number and standard deviation of bin averages
avg_plot_mean_vectors          = True         # Generate an arrow plot of bin-averaged U-V mean velocities in the x-y plane
avg_plot_secondary_circulation = True         # Generate an image plot of 2D bin-averaged steamwise (u) velocities, overlain by an arrow plot showing secondary circulation in the V-W plane
avg_plot_uvw_velocity_array    = True         # Generate a 3-panel image plot showing bin-averaged U,V,W velocities in the V-W plane
avg_plot_flow_summmary         = True         # Generate a summary plot showing image plots of U,V bin-averaged velocities, an arrow plot of bin-averaged U-V mean velocities, and flow/discharge calculations
avg_save_plots                 = True         # Save the plots to disk 
avg_show_plots                 = False        # Print plots to screen (pauses execution until plots are manually closed)


## END script options #########################################################

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import transect_preprocessor
reload(transect_preprocessor)
import adcpy
reload(adcpy)
from adcpy_recipes import *

def transect_average(pre_process_input_file=None):
    """
    Received a list of ADCPTransectData objects from transect_preprocessor.py,
    and then groups and bin-averages the transects.  Group average ADCPData 
    objects, velocties, and plots are optionally output to the outpath supplied
    by transect_preprocessor().
    Inputs:
        pre_process_input_file = path to a transect_preprocessor input file,
          or None to use the default file.
    """

    (transects,outpath) = transect_preprocessor.transect_preprocessor(pre_process_input_file)
    print 'total transects loaded:',len(transects)    
    grps_to_average = group_adcp_obs_by_spacetime(transects,
                                                  max_gap_m=avg_max_gap_m,
                                                  max_gap_minutes=avg_max_gap_minutes,
                                                  max_group_size=avg_max_group_size)
    print 'total groups to average',len(grps_to_average)
    #write_csv_velocity(transects[0],os.path.join(outpath,'temp.csv'),no_header=True)
    #grps_to_average = grps_to_average[1:]

    grp_num = 0
    track_fig=101
    for grp in grps_to_average:
        if avg_plot_xy:
            adcpy.plot.plot_obs_group_xy_lines(grp,fig=track_fig,title='Group%03i Source Observations'%grp_num)

        avg = average_transects(grp,
                                dxy=avg_dxy,
                                dz=avg_dz,
                                return_adcpy=True,
                                plotline_from_flow=avg_normal_to_flow,
                                sd_drop=avg_bin_sd_drop)

        if avg_plot_xy:
            adcpy.plot.get_fig(fig=track_fig)
            plt.plot(avg.xy[:,0],avg.xy[:,1],label='average projection')
            plt.legend(prop={'size':10})
            if avg_save_plots:
                plt.savefig(os.path.join(outpath,"group%03i_xy_lines.png"%grp_num))              
        
        if avg_rotation is not None:
            avg = transect_rotate(avg,avg_rotation)
        if avg_std_drop > 0:
            avg.sd_drop(sd=3.0,
                      sd_axis='elevation',
                      interp_holes=avg_std_interp)
            avg.sd_drop(sd=3.0,
                      sd_axis='ensemble',
                      interp_holes=avg_std_interp)
        if avg_smooth_kernel > 2:
            avg.kernel_smooth(kernel_size = 3)
        
        if avg_save_csv:
            write_csv_velocity_array(avg,os.path.join(outpath,'group%03i_velocity.csv'%grp_num),no_header=True)
            write_csv_velocity_db(avg,os.path.join(outpath,'group%03i_velocity_db.csv'%grp_num),no_header=False)
            write_ensemble_mean_velocity_db(avg,os.path.join(outpath,'group%03i_velocity_depth_means.csv'%grp_num),
                                            no_header=False,range_from_velocities=True)

        if avg_save_netcdf:
            fname = os.path.join(outpath,'group%03i.nc'%grp_num)
            avg.write_nc(fname,zlib=True)
        
        if avg_plot_avg_n_sd:
            uvw = 'uvw'
            for i in range(3):
                plot_avg_n_sd(avg,i,0.05)
                if avg_save_plots:
                    plt.savefig(os.path.join(outpath,"group%03i_%s_avg_n_sd.png"%(grp_num,uvw[i])))
        
        if avg_plot_mean_vectors:
            fig3 = adcpy.plot.plot_ensemble_mean_vectors(avg,title='Group%03i Mean Velocity [m/s]'%grp_num)
            if avg_save_plots:
                plt.savefig(os.path.join(outpath,"group%03i_mean_velocity.png"%grp_num))

        if avg_plot_secondary_circulation:
            fig4 = adcpy.plot.plot_secondary_circulation(avg,u_vecs=30,v_vecs=30,
                                                         title='Group%03i Cross-Stream Velocity [m/s] and Secondary Circulation Vectors'%grp_num)
            if avg_save_plots:
                plt.savefig(os.path.join(outpath,"group%03i_secondary_circulation.png"%grp_num))

        if avg_plot_uvw_velocity_array:
            fig5 = adcpy.plot.plot_uvw_velocity_array(avg.velocity,
                                                      title='Group%03i Velocity [m/s]'%grp_num,
                                                      ures=0.1,vres=0.1,wres=0.05)
            if avg_save_plots:
                plt.savefig(os.path.join(outpath,"group%03i_uvw_velocity.png"%grp_num))

        if avg_plot_flow_summmary:
            fig6 = adcpy.plot.plot_flow_summmary(avg,title='Group%03i Streawise Summary'%grp_num,
                                                 ures=0.1,vres=0.1,use_grid_flows=True)
            if avg_save_plots:
                plt.savefig(os.path.join(outpath,"group%03i_flow_summary.png"%grp_num))

        if avg_show_plots:
            plt.show()
        plt.close('all')
        grp_num += 1

    print 'ADCP processing complete!'


def plot_avg_n_sd(avg,uvw,resolution=0.1):
    """
    Generates a vertical three-panel plot, showing images of a bin-averaged
    velocity, the number of velociy measurements in each bin, and the bin standard
    deviation velocity.  Desinged to be used in conjuction with 
    transect_average() output.
    Inputs:
        avg = ADCPData object, with extra velocity_n and velocity_sd data 
          fields produced by transect_average()
        uvw = python string, either 'U','V', or 'W' to select which velocity
          compontent to plot.
        resolution = optional value to round the plot velocity scales up toward
    """

    if uvw == 0:
        v_str = 'U'
    elif uvw == 1:
        v_str = 'V'
    else:
        v_str = 'W'
    
    inv = 1/resolution
    xx,yy,dd,pline = adcpy.util.find_projection_distances(avg.xy)
    mtest = np.floor(avg.velocity[...,uvw]*inv)
    minv = np.nanmin(np.nanmin(mtest))*resolution
    mtest = np.ceil(avg.velocity[...,uvw]*inv)    
    maxv = np.nanmax(np.nanmax(mtest))*resolution
    avg_panel = adcpy.plot.IPanel(velocity = avg.velocity[:,:,uvw],
                             x = dd,
                             y = avg.bin_center_elevation,
                             minv = minv,
                             maxv = maxv,
                             xlabel = 'm',
                             ylabel = 'm',
                             units = 'm/s',
                             use_pcolormesh = True,
                             title='%s Averaged Velocity [m/s]'%v_str)
    maxv = np.nanmax(np.nanmax(avg.velocity_n[...,uvw]))
    n_panel = adcpy.plot.IPanel(velocity = avg.velocity_n[:,:,uvw],
                             x = dd,
                             y = avg.bin_center_elevation,
                             minv = 0,
                             maxv = maxv,
                             xlabel = 'm',
                             ylabel = 'm',
                             units = 'number',
                             use_pcolormesh = True,
                             title='n Samples')
    mtest = np.floor(avg.velocity_sd[...,uvw]*inv)
    minv = np.nanmin(np.nanmin(mtest))*resolution
    mtest = np.ceil(avg.velocity_sd[...,uvw]*inv)    
    maxv = np.nanmax(np.nanmax(mtest))*resolution
    sd_panel = adcpy.plot.IPanel(velocity = avg.velocity_sd[:,:,uvw],
                             x = dd,
                             y = avg.bin_center_elevation,
                             minv = 0,
                             maxv = maxv,
                             xlabel = 'm',
                             ylabel = 'm',
                             units = 'm/s',
                             use_pcolormesh = True,
                             title='Standard Deviation [m/s]')
    fig = adcpy.plot.plot_vertical_panels((avg_panel,n_panel,sd_panel))
    return fig


def main():
    import sys
    prepro_input = sys.argv[1]
    transect_average(prepro_input)
    

# run myself
if __name__ == "__main__":
    main()

