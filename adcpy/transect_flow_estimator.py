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

# processing paramterters
flow_regrid                     = True         
flow_regrid_dxy                 = 2.0          # Horizontal resolution of averaging bins {m}
flow_regrid_dz                  = 0.25         # Vertical resolution of averaging bins {m}
flow_regrid_bin_sd_drop         = 3          # Maximum number of ADCP observations to average {m}
flow_regrid_normal_to_flow      = False
flow_crossprod_ens_num_ave      = 15
flow_sd_drop                    = 3            # Maximum number of ADCP observations to average {m}
flow_sd_drop_interp             = True            # Maximum number of ADCP observations to average {m}
flow_smooth_kernel              = 3            # Smooth velocity data using a square kernel box-filter, with side dimension = avg_smooth_kernel.  0 = no kernel smoothing
flow_extrapolate_boundaries     = True            # Maximum number of ADCP observations to average {m}
flow_rotation                   = "no transverse flow"           
flow_p1_lon_lat                  = None
flow_p2_lon_lat                  = None
# post-average options
flow_save_processed_netcdf         = True
flow_netcdf_data_compression       = True
flow_plot_timeseries_values        = True
flow_write_timeseries_flows_to_csv = True
flow_plot_mean_vectors          = True         # Generate an arrow plot of bin-averaged U-V mean velocities in the x-y plane
flow_plot_flow_summmary         = True         # Generate a summary plot showing image plots of U,V bin-averaged velocities, an arrow plot of bin-averaged U-V mean velocities, and flow/discharge calculations
flow_save_plots                 = True         # Save the plots to disk 
flow_show_plots                 = False        # Print plots to screen (pauses execution until plots are manually closed)


## END script options #########################################################

import os
import numpy as np
import csv
import scipy.stats.stats as sp
import matplotlib.pyplot as plt
import transect_preprocessor
reload(transect_preprocessor)
import adcpy
reload(adcpy)
import adcpy_recipes

def transect_flow_estimator(pre_process_input_file=None):
    """
    Receives a list of ADCPTransectData objects from transect_preprocessor.py,
    and processes to find average velocities acording to options, and finally 
    outputs flows. Flows can be written to CSV or and plotted across time, and
    invidual plot produced for each transect.  Output fils=es are save to
    the outpath supplied by transect_preprocessor().
    Inputs:
        pre_process_input_file = path to a transect_preprocessor input file,
          or None to use the default file.
    """


    # initialize collector lists for time series plots
    mtimes = []
    mean_u = []
    mean_v = []
    align_angle = []
    total_flow = []
    ustars = []
    kxs = []
    kys = []
    data_file_num = -1

    # setup compression option
    if flow_netcdf_data_compression:
        zlib = True
    else:
        zlib = None
        
    # prepare plot line, if any
    if flow_p1_lon_lat is None or flow_p2_lon_lat is None:
        ll_pline = None
    else:
        ll_pline = np.array([flow_p1_lon_lat,flow_p2_lon_lat])

    (transects,outpath) = transect_preprocessor.transect_preprocessor(pre_process_input_file)

    if flow_write_timeseries_flows_to_csv:
        logfname = os.path.join(outpath,'transect_flow_log.csv')
        lf = open(logfname,'wb')
        logfile = csv.writer(lf)
        logfile.writerow(('filename','start_date','end_date','water_mode',
                         'bottom_mode'                        
                         'mean_velocity_U [m/s]',
                         'mean_velocity V [m/s]',
                         'flow_volume_U [m^3/s]','flow_volume_V [m^3/s]',
                         'sampling_area [m^2]',
                         'alignment_angle [degree]',
                         'notes'))
    
    for t in transects:
        
        fname, ext = os.path.splitext(t.source)
        outname = os.path.join(outpath,fname)
        
        if flow_regrid:

            if ll_pline is not None:
                if flow_regrid_normal_to_flow:
                    print "Warning! Regridding plotline given, but options also ask for a flow-based plotline."
                    print "Ignoring flow-based plotline option..."
                if t.xy_srs is not None:
                    ll_srs = t.lonlat_srs
                    if ll_srs is None:
                        ll_srs = t.default_lonlat_srs            
                    pline = adcpy.util.coordinate_transform(ll_pline,ll_srs,t.xy_srs)
                else:
                    print "ADCPData must be projected to use transect_flow_estimator"
                    exit()
            else:
                if flow_regrid_normal_to_flow:
                    flows = t.calc_ensemble_flows(range_from_velocities=True)
                    pline = adcpy.util.map_flow_to_line(t.xy,flows[:,0],flows[:,1])
                else:
                    pline = adcpy.util.map_xy_to_line(t.xy)                    
            t.xy_regrid(dxy=flow_regrid_dxy,dz=flow_regrid_dz,
                        pline=pline,
                        sd_drop=flow_regrid_bin_sd_drop,
                        mtime_regrid=True)
        
        else:
            t = t.average_ensembles(flow_crossprod_ens_num_ave)
            
        if flow_sd_drop > 0:
            t.sd_drop(sd=flow_sd_drop,
                      sd_axis='elevation',
                      interp_holes=flow_sd_drop_interp)
            t.sd_drop(sd=flow_sd_drop,
                      sd_axis='ensemble',
                      interp_holes=flow_sd_drop_interp)
        if flow_smooth_kernel > 2:
            t.kernel_smooth(kernel_size = flow_smooth_kernel)
        if flow_extrapolate_boundaries:
            t.extrapolate_boundaries()
        if flow_rotation is not None:
            t = adcpy_recipes.transect_rotate(t,flow_rotation)
            
        if flow_regrid:
            UVW,UVWens,flow,survey_area = \
              adcpy_recipes.calc_transect_flows_from_uniform_velocity_grid(t,use_grid_only=False)
            Uflow = flow[0]
            Vflow = flow[1]
            U = UVW[0]
            V = UVW[1]
        else:
            U,Uflow,total_area,survey_area = t.calc_crossproduct_flow()
            V, Vflow = (np.nan,np.nan)
            
        if flow_plot_mean_vectors:
            fig3 = adcpy.plot.plot_ensemble_mean_vectors(t,title='Mean Velocity [m/s]')
            if flow_save_plots:
                adcpy.plot.plt.savefig("%s_mean_velocity.png"%outname)
            
        if flow_plot_flow_summmary:
            fig6 = adcpy.plot.plot_flow_summmary(t,title='Streamwise Summary',
                                                 ures=0.1,vres=0.1,
                                                 use_grid_flows=flow_regrid)
            if flow_save_plots:
                adcpy.plot.plt.savefig("%s_flow_summary.png"%outname)

        if flow_show_plots:
            adcpy.plot.show()
        adcpy.plot.plt.close('all')
        
        if (flow_save_processed_netcdf):
            fname = outname + '.flow_processed.nc'
            t.write_nc(fname,zlib=zlib)

        if flow_plot_timeseries_values or flow_write_timeseries_flows_to_csv:
            
            data_file_num += 1

            # must fit to line to calc dispersion
            xy_line = adcpy.util.map_xy_to_line(t.xy)
            xd,yd,dd,xy_line = adcpy.util.find_projection_distances(t.xy,xy_line)
            ustar, kx, ky = adcpy.util.calcKxKy(t.velocity[:,:,0],t.velocity[:,:,1],
                                     dd,t.bin_center_elevation,t.bt_depth)
                                     
            if t.rotation_angle is not None:
                r_angle = t.rotation_angle*180.0/np.pi
            else:
                r_angle = 0.0
                    
            if flow_write_timeseries_flows_to_csv:
                times = t.date_time_str(filter_missing=True)
                try:
                    w_mode_str = '%s'%t.raw_adcp.config.prof_mode
                    bm = t.raw_adcp.bt_mode.tolist()
                    b_mode = [];
                    for bm1 in bm:
                        bmi = int(bm1)
                        if (bmi not in b_mode):
                            b_mode.append(bmi)   
                    b_mode_str = ''
                    for bm1 in b_mode:                           
                        if (b_mode_str == ''):
                            b_mode_str = '%i'%bm1                               
                        else:
                            b_mode_str = '%s/%i'%(b_mode_str,bm1)
                except:
                    w_mode_str = 'Unknown'
                    b_mode_str = 'Unknown'
                logfile.writerow((t.source,
                                 times[0],
                                 times[-1],
                                 w_mode_str,
                                 b_mode_str,
                                 '%7.4f'%U,
                                 '%7.4f'%V,
                                 '%10.2f'%Uflow,
                                 '%10.2f'%Vflow,
                                 '%10.2f'%survey_area,
                                 '%5.2f'%r_angle,
                                 t.history))
        
                if flow_plot_timeseries_values:
                    mtimes.append(sp.nanmedian(t.mtime))
                    mean_u.append(U)
                    mean_v.append(V)
                    total_flow.append(Uflow)
                    align_angle.append(r_angle)
                    ustars.append(ustar)
                    kxs.append(kx)
                    kys.append(ky)

    # plot timeseries data after all files have been processed
    if flow_plot_timeseries_values and data_file_num>0:

        # sort by mtime
        mtimes = np.array(mtimes)        
        nn = np.argsort(mtimes)
        mtimes = mtimes[nn]
        mean_u = np.array(mean_u)[nn]
        mean_v = np.array(mean_v)[nn]
        total_flow = np.array(total_flow)[nn]
        align_angle = np.array(align_angle)[nn]
        ustars = np.array(ustars)[nn]
        kxs = np.array(kxs)[nn]
        kys = np.array(kys)[nn]

        # plot timeseries figures
        fig_handle = plt.figure()
        plt.subplot(311)
        align_angle= np.array(align_angle)
        mtimes = np.array(mtimes)
        mean_u = np.array(mean_u)
        mean_v = np.array(mean_v)
        total_flow = np.array(total_flow)
        aa = -align_angle*np.pi/180.0
        if np.isnan(mean_v[0]):
             # probably crossproduct flows were calculated, which has zero v flow
            mean_v[:] = 0.0
        uq = np.cos(aa)*mean_u + np.sin(aa)*mean_v
        vq = -np.sin(aa)*mean_u + np.cos(aa)*mean_v
        v_mag = np.sqrt(uq**2 + vq**2)
        vScale = np.max(v_mag)
        vScale = max(vScale,0.126)
        qk_value = np.round(vScale*4)/4
        Q = plt.quiver(mtimes,np.zeros(len(mtimes)),uq,vq,
                       width=0.003,
                       headlength=10,
                       headwidth=7,
                       scale = 10*vScale,   #scale = 0.005,
                       scale_units = 'width'
                       )
        qk = plt.quiverkey(Q, 0.5, 0.85, qk_value, 
                       r'%3.2f '%qk_value + r'$ \frac{m}{s}$', labelpos='W',
                       )
        plt.title('Time series data: %s'%outpath)
        ax = plt.gca()
        ax.xaxis_date()
        plt.gcf().autofmt_xdate()
        ax.yaxis.set_visible(False)
        ax.set_xticklabels([])
        plt.subplot(312)
        plt.plot(mtimes,total_flow)
        plt.ylabel('m^3/s')
        ax = plt.gca()
        ax.xaxis_date()
        plt.gcf().autofmt_xdate()
        ax.set_xticklabels([])
        plt.subplot(313)
        plt.plot(mtimes,align_angle,'bo')
        plt.ylabel('rotation angle')
        ax = plt.gca()
        ax.xaxis_date()
        plt.gcf().autofmt_xdate()
        ts_plot = os.path.join(outpath,'time_series_plots.png')
        fig_handle.savefig(ts_plot)

        fig_handle = plt.figure(1111)
        plt.subplot(311)
        plt.plot(mtimes,ustars)
        plt.ylabel('u* m/s')
        ax = plt.gca()
        ax.xaxis_date()
        #plt.gcf().autofmt_xdate()
        ax.set_xticklabels([])
        plt.title('Dispersion Coefficients: %s'%outpath)
        plt.subplot(312)
        plt.plot(mtimes,kxs)
        plt.ylabel('Kx m^2/s')
        ax = plt.gca()
        ax.xaxis_date()
        #plt.gcf().autofmt_xdate()
        ax.set_xticklabels([])
        plt.subplot(313)
        plt.plot(mtimes,kys,'b')
        plt.ylabel('Ky m^2/s')
        ax = plt.gca()
        ax.xaxis_date()
        plt.gcf().autofmt_xdate()
        ts_plot = os.path.join(outpath,'time_series_dispersion.png')
        fig_handle.savefig(ts_plot)
        plt.close('all')
    
    if flow_write_timeseries_flows_to_csv:
        lf.close()
        
    print 'transect_flow_estimator completed!'


def main():
    import sys
    prepro_input = sys.argv[1]
    transect_flow_estimator(prepro_input)
    

# run myself
if __name__ == "__main__":
    main()

