# -*- coding: utf-8 -*-
"""
transect_preprocess

Driver script that is designed find and load raw ADCP observations from a
designated directory, and perform certain processing task on them, optionally
saving the reuslts to ADCPData netcdf format, and/or passing a list of
ADCPData python objects to a parent script.

This script requires an options file.  If no path to a valid options file is
passed as an argument to transect_preprocessor(), it look for the default
file 'transect_preprocessor_input.py' in the local run directory.

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""
import glob
import os
import numpy as np
#import scipy.stats.stats as sp
import scipy.stats.morestats as ssm

import transect_preprocessor_input
reload(transect_preprocessor_input)
from transect_preprocessor_input import *
import shutil
import adcpy
reload(adcpy)
from matplotlib.dates import num2date#,date2num,

# Common formatting for datenums:
def fmt_dnum(dn):
    return num2date(dn).strftime('%c')



default_option_file = r"transect_preprocessor_input.py"

def transect_preprocessor(option_file=None):
    """
    The method finds, loads, pre-preprocess, and returns ADCPData ojects
    for ADCPRdiWorkhorseData compatible raw/netcdf files.  It returns
    a list of ADCPData objects.
    
    See the default options file 'transect_preprocessor_input.py' for the 
    input options.
    """
    np.seterr(all='ignore')
    
    data_files = None
    data_files_nc = None
    
    if option_file is None:
        option_file = default_option_file
    try:
        options, fileExtension = os.path.splitext(option_file)
        exec('import %s'%options)
        exec('reload(%s)'%options)
        exec("from %s import *"%options)
    except:
        print "Could not load options file: %s"%option_file
        raise

    path_or_file = working_directory
    
    if os.path.exists(path_or_file):
        if (os.path.isdir(path_or_file)):
            data_files = glob.glob(os.path.join(path_or_file,'*[rR].000'))
            data_files += glob.glob(os.path.join(path_or_file,'*.nc'))
            
            data_path = path_or_file
        else:
            try:
                fileName, fileExtension = os.path.splitext(path_or_file)
                if (('R.000' in path_or_file and fileExtension is '000') or
                    ('r.000' in path_or_file and fileExtension is '000')):
                    data_files = ( path_or_file, )
                    data_files_nc = None;
                elif fileExtension is '.nc':
                    data_files_nc = ( path_or_file, )
                else:
                    print "Filename (%s) does not appear to be a valid raw (*r.000) or netcdf (*.nc) file - exiting."%path_or_file
                data_path, fname = os.path.split(path_or_file)
            except:
                print "Could not interpret filename (%s) as a raw (*r.000) or netcdf (*.nc) file - exiting."%path_or_file        
    else:
        print "Path or file (%s) not found - exiting."%path_or_file
        exit()
    
    outpath = os.path.join(data_path,'ADCPy')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
   
    # copy processing options
    shutil.copyfile(option_file, os.path.join(outpath,option_file))

    hc_count = 0
    if head_correct_spanning and (len(data_files) > 1):
        # bin data rquired for head_correct form input files
        mtime_a = None
        mtime_b = None
        reference_heading = None
        is_a = list()
        print 'Gathering data from data_files for head_correct spanning...'
        for data_file in data_files:
            path, fname = os.path.split(data_file)
            
#            try:
            a = ADCPy.open_adcp(data_file,
                            file_type="ADCPRdiWorkhorseData",
                            num_av=1,
                            adcp_depth=adcp_depth)
            m1,h1,bt1,xy1 = a.copy_head_correct_vars(xy_srs=xy_projection)
            
            if reference_heading is None:
                reference_heading = ssm.circmean(h1*np.pi/180.)*180./np.pi

            current_heading = ssm.circmean(h1*np.pi/180.)*180./np.pi
                        
            if mtime_a is None:
                mtime_a = m1
                heading_a = h1
                bt_vel_a = bt1
                xy_a = xy1
            else:
                mtime_a = np.concatenate((mtime_a,m1))
                heading_a = np.concatenate((heading_a,h1))
                bt_vel_a = np.row_stack((bt_vel_a,bt1))
                xy_a = np.row_stack((xy_a,xy1))
                    
            print '+',fname #, 'Heading Group A:', is_a[-1]
            print fmt_dnum(a.mtime[0])
            
            
            if debug_stop_after_n_transects:
                hc_count += 1
                if hc_count >= debug_stop_after_n_transects:
                    break           
#            except:
                
#                print 'Failure reading %s for head_correct spanning!'%fname                    
#                is_a.append('True')
               
        print 'Number direction a headings:',np.shape(mtime_a)

        # this method is independent of self/a
        #try: 
        heading_correction_a = ADCPy.util.fit_head_correct(mtime_in=mtime_a,
                                   hdg_in=heading_a,
                                   bt_vel_in=bt_vel_a,
                                   xy_in=xy_a,
                                   u_min_bt=u_min_bt,
                                   hdg_bin_size=hdg_bin_size,
                                   hdg_bin_min_samples=hdg_bin_min_samples)
#        except:
#                print 'Failure fitting head_correct spanning!'
#                if mag_declination is not None:
#                    print 'Using simple magnetic declination correction instead'
#                    heading_correction_a = None
#                else:
#                    print 'No magnetic declination value found - head_correct failure.'
#                    print 'exiting'
#                    exit()
                  
    # setup compression option
    if use_netcdf_data_compression:
        zlib = True
    else:
        zlib = None
   
    # begin cycling/processing input files 
    adcp_preprocessed = []
    for data_file in data_files:    
 #       try:
        a = adcpy.open_adcp(data_file,
                            file_type='ADCPRdiWorkhorseData',
                            num_av=1,
                            adcp_depth=adcp_depth)
        path, fname = os.path.split(data_file)
        fname, ext = os.path.splitext(fname)
        outname = os.path.join(outpath,fname)
        
        print 'Processing data_file:', outname
        
        if save_raw_data_to_netcdf:
            fname = outname + '.nc'
            a.write_nc(fname,zlib=zlib)

        # setup for heading correction based
        if head_correct_spanning and (len(data_files) > 1):        
            heading_cf = heading_correction_a
        else:
            heading_cf = None
        
        a.lonlat_to_xy(xy_srs=xy_projection)
        a.heading_correct(cf=heading_cf,
                          u_min_bt=u_min_bt,
                          hdg_bin_size=hdg_bin_size,
                          hdg_bin_min_samples=hdg_bin_min_samples,
                          mag_dec=mag_declination)
        if sidelobe_drop != 0:
            a.remove_sidelobes(fsidelobe=sidelobe_drop)
        if std_drop > 0:
            a.sd_drop(sd=std_drop,
                      sd_axis='elevation',
                      interp_holes=True)
            a.sd_drop(sd=std_drop,
                      sd_axis='ensemble',
                      interp_holes=True)
        if smooth_kernel > 2:
            a.kernel_smooth(kernel_size = smooth_kernel)
        if extrap_boundaries:
            a.extrapolate_boundaries()
        if average_ens > 1:
            a = a.average_ensembles(ens_to_avg=average_ens)
        if regrid_horiz_m is not None:
            a.xy_regrid(dxy=regrid_horiz_m,
                        dz=regrid_vert_m,
                        xy_srs=xy_projection,
                        pline=None,
                        sort=False)

        if (save_preprocessed_data_to_netcdf):
            fname = outname + '.preprocessed.nc'
            a.write_nc(fname,zlib=zlib)
            
        adcp_preprocessed.append(a)
        
        if debug_stop_after_n_transects:
            if len(adcp_preprocessed) >= debug_stop_after_n_transects:
                return (adcp_preprocessed,outpath)

    return (adcp_preprocessed,outpath)

# run myself
if __name__ == "__main__":
    transect_preprocessor()

