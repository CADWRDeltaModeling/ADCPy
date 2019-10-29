# -*- coding: utf-8 -*-
"""Batch processing and logic for organizing and binning multiple ADCPData objects.
Tools and methods for categorizing/manipulating/visualizing data in ADCPy/ADCPData
format.  This module is dependent upon adcpy. 

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""
from __future__ import print_function

import numpy as np  # numpy 1.7 
import glob
import os
import csv
import scipy.stats as sp
#import scipy.signal as sps
import scipy.stats.morestats as ssm
from matplotlib.dates import num2date#,date2num,
import datetime

from . import adcpy
from . import adcpy_utilities as util


def average_transects(transects,dxy,dz,plotline=None,return_adcpy=True,
                      stats=True,plotline_from_flow=False,sd_drop=0,
                      plotline_orientation='default'):
    """ 
    This method takes a list of input ADCPy transect objects, and:
    1) Projects and re-grids each transect to either the input plotline, or a best
       fit of available projected xy locations;
    2) Bin-averages the re-gridded U,V, and W velocities of input ADCPTransectData
       objects
    Inputs:
        transects = list of ADCPTransectData objects
        dxy = new grid spacing in the xy (or plotline) direction
        dz = new regular grid spacing in the z direction (downward for transects)
        plotline = optional designated line in the xy plane for projecting ensembles onto
        plotline_orientaion=how to resolve ambiguity in plotline orientation.  'default'
          uses (probably) the min x point as the origin.  'river' makes the average flow
          positive, putting river left at 0 and river right at >0.  This is ignored
          if plotline or plotline_from_flow are given.
        return_adcpy = True: returns an ADCPData object containing averaged velocities
                       False: returns a 3D numpy array containing U,V,W gridded veloctiy
    """
    n_transects = len(transects)
    avg = transects[0].copy_minimum_data()
    if n_transects > 1:
        # ugly brute-force method ot find furthest points;
        # ConvexHull-type approach is only available in more recent scipy
        max_dist = 0.0
        centers = [adcpy.util.centroid(a.xy) for a in transects]
        for c1 in centers:
            for c2 in centers:
                max_dist = max(max_dist,adcpy.util.find_line_distance(c1,c2))
        print("max dist:",max_dist)
        if max_dist > 30.0:
            print('WARNING: averaging transects with maximum centroid distance of %f m!'%max_dist)

    # gather position data for new grid generation
    xy_data = np.vstack([transects[i].xy for i in range(n_transects)])
    z_data = np.hstack([transects[i].bin_center_elevation for i in range(n_transects)])

    # find common grid
    if plotline is None:    
        if plotline_from_flow:
            flows = transects[0].calc_ensemble_flow(range_from_velocities=False)
            xy_line = adcpy.util.map_flow_to_line(xy_data,flows[:,0],flows[:,1])
        else:
            xy_line = adcpy.util.map_xy_to_line(xy_data)
            if plotline_orientation=='river':
                flows = transects[0].calc_ensemble_flow(range_from_velocities=False)
                Qmean=flows[:,:2].mean(axis=0) # [Qx,Qy]
                tran_tangent=xy_line[1,:] - xy_line[0,:]
                if tran_tangent[0]*Qmean[1] - tran_tangent[1]*Qmean[0] < 0:
                    xy_line=xy_line[::-1,:]
    else:
        xy_line = plotline
    # NEED logic around determining whether original data was negative down, positive up, etc
    z_mm = np.array([np.max(z_data),np.min(z_data)])
    (dd,xy_new_range,xy_new,z_new) = adcpy.util.new_xy_grid(xy_data,z_mm,dxy,dz,xy_line,True)

    # initialize arrays
    xy_bins = adcpy.util.find_regular_bin_edges_from_centers(xy_new_range)
    z_bins = adcpy.util.find_regular_bin_edges_from_centers(z_new)
    new_shape = [len(xy_new_range),len(z_new),3]
    avg.velocity = np.empty(new_shape)
    if stats:
        avg.velocity_n = np.empty(new_shape)
        avg.velocity_sd = np.empty(new_shape)


    # generate linear xy,z,velocties for bin averaging, perform bin averaging
    for i in range(3):
        bin_ave_inputs = []
        mtimes = []
        for t in transects:
            xx,yy,xy_range,xy_line = adcpy.util.find_projection_distances(t.xy,xy_line)
            bin_ave_inputs.append(adcpy.util.xy_z_linearize_array(xy_range,
                                              t.bin_center_elevation,
                                              t.velocity[...,i]))
        xy = np.hstack([bin_ave_inputs[j][0] for j in range(n_transects)])
        z = np.hstack([bin_ave_inputs[j][1] for j in range(n_transects)])
        values = np.hstack([bin_ave_inputs[j][2] for j in range(n_transects)])
        bin_ave = adcpy.util.bin_average(xy,xy_bins,values,z,z_bins,return_stats=stats,sd_drop=sd_drop)
        bin_ave = adcpy.util.un_flip_bin_average(xy_new_range,z_new,bin_ave)
        if stats:
            (avg.velocity[...,i],
             avg.velocity_n[...,i],
             avg.velocity_sd[...,i]) = bin_ave
        else:
            avg.velocity[...,i] = bin_ave[0]
    
    # update adcpData object
    avg.xy = xy_new
    avg.bin_center_elevation = z_new
    avg.n_ensembles = new_shape[0]
    avg.n_bins = new_shape[1]

    # report back
    if return_adcpy:
        avg.xy_srs = transects[0].xy_srs
        sources = [transects[i].source for i in range(n_transects)]
        avg.source = "\n".join(sources)
        mtimes = [util.nanmedian(transects[i].mtime) for i in range(n_transects)]
        # list() to force realization
        mtimes = np.array(list(filter(None,mtimes)))
        if mtimes.any():
            avg.mtime = np.ones(new_shape[0],np.float64) * util.nanmean(mtimes)
        if plotline is not None:
            plotlinestr = "[%f,%f],[%f,%f]"%(plotline[0,0],
                                             plotline[0,1],
                                             plotline[1,0],
                                             plotline[1,1])
        else:
            plotlinestr='None'
        avg.history_append('average_transects(dxy=%f,dz=%f,plotline=%s)'%(dxy,dz,plotlinestr))
        avg.xy_line=xy_line
        return avg
    else:
        return avg.velocity
    

def write_csv_velocity_array(adcp,csv_filename,no_header=False,un_rotate_velocties=True):
    """ 
    Writes comma-delimited u,v,w velocties to a text file.
    Inputs:
        ADCP = ADCPData object
        csv_filename = file path of output file
        no_header = boolean, True = don't write header line 
    Returns:
        nothing
    """    
    # direct dump of numpy array - opps required numpy v1.8
    #np.savetext(csv_filename,adcp.velocity,delimiter=",")
    if un_rotate_velocties:
        v = adcp.get_unrotated_velocity()
    else:
        v = adcp.velocity
    with open(csv_filename, 'wb') as csvfile:
        arraywriter = csv.writer(csvfile, delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL)
        if not no_header:
            if adcp.xy is not None:
                arraywriter.writerow(['x']+[adcp.xy_srs])
                arraywriter.writerow(adcp.xy[:,0].tolist())
                arraywriter.writerow(['y'])
                arraywriter.writerow(adcp.xy[:,1].tolist())
            elif adcp.lonlat is not None:
                arraywriter.writerow(['longitude'])
                arraywriter.writerow(adcp.lonlat[:,0].tolist())
                arraywriter.writerow(['latitude'])
                arraywriter.writerow(adcp.lonlat[:,1].tolist())
            arraywriter.writerow(['bin_center_elevation'])
            arraywriter.writerow(adcp.bin_center_elevation.tolist())
        arraywriter.writerow(['U'])            
        for i in range(adcp.n_ensembles):
            arraywriter.writerow(v[i,:,0].tolist())
        arraywriter.writerow(['V'])            
        for i in range(adcp.n_ensembles):
            arraywriter.writerow(v[i,:,1].tolist())
        arraywriter.writerow(['W'])            
        for i in range(adcp.n_ensembles):
            arraywriter.writerow(v[i,:,2].tolist())


def write_csv_velocity_db(adcp,csv_filename,no_header=False,un_rotate_velocties=True):
    """ 
    Writes comma-delimited ensemble-mean U,V
    Inputs:
        ADCP = ADCPData object
        csv_filename = file path of output file
        no_header = boolean, True = don'r write position data, False = write position data
    Returns:
        nothing
    """    
    # direct dump of numpy array - opps required numpy v1.8
    #np.savetext(csv_filename,adcp.velocity,delimiter=",")
    if un_rotate_velocties:
        v = adcp.get_unrotated_velocity()
    else:
        v = adcp.velocity
    with open(csv_filename, 'wb') as csvfile:
        arraywriter = csv.writer(csvfile, delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL)        
        if not no_header:
            if adcp.xy is not None:
                header = ['x [%s]'%adcp.xy_srs,'y [%s]'%adcp.xy_srs]
            elif adcp.lonlat is not None:
                header = ['longitude [degE]','latitude [degN]']
            else:
                print('Error, input adcp has no position data - no file written')
                return
            header.extend(['z [m]','datetime','u [m/s]','v [m/s]','w [m/s]',])                
            arraywriter.writerow(header)
        for i in range(adcp.n_ensembles):
            for j in range(adcp.n_bins):
                if adcp.mtime is None:
                    rec_time = 'None'
                elif adcp.mtime[i] is None or np.isnan(adcp.mtime[i]):
                    rec_time = 'None'
                else:
                    rec_time = num2date(adcp.mtime[i]).strftime('%c')
                if adcp.xy is not None:
                    db_record = [adcp.xy[i,0],adcp.xy[i,1]]
                else:
                    db_record = [adcp.lonlat[i,0], adcp.lonlat[i,1]]
                db_record = db_record + [adcp.bin_center_elevation[j],
                                         rec_time,
                                         v[i,j,0],
                                         v[i,j,1],
                                         v[i,j,2]]
                arraywriter.writerow(db_record)
                

def write_ensemble_mean_velocity_db(adcp,csv_filename,no_header=False,
                                    un_rotate_velocties=True,elev_line=None,
                                    range_from_velocities=False):
    """ 
    Writes comma-delimited velocties to a text file, optionally with xy-positions
    or lon-lat positions, and bin_center_elveations.  The write order for a 2D
    velocity aray is the first (leftmost) axis is written horizontally.
    Inputs:
        ADCP = ADCPData object
        csv_filename = file path of output file
        no_header = boolean, True = don'r write position data, False = write position data
    Returns:
        nothing
    """    
    # direct dump of numpy array - opps required numpy v1.8
    #np.savetext(csv_filename,adcp.velocity,delimiter=",")
    if un_rotate_velocties and adcp.rotation_angle is not None:
        r_axis = adcp.rotation_axes
        r_angle = adcp.rotation_angle
        adcp.set_rotation(None)
        UVW = adcp.ensemble_mean_velocity(elev_line=elev_line,
                                      range_from_velocities=range_from_velocities)
        adcp.set_rotation(r_angle,r_axis)
    else:
        UVW = adcp.ensemble_mean_velocity(elev_line=elev_line,
                                      range_from_velocities=range_from_velocities)
    with open(csv_filename, 'wb') as csvfile:
        arraywriter = csv.writer(csvfile, delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL)        
        if not no_header:
            if adcp.xy is not None:
                header = ['x [%s]'%adcp.xy_srs,'y [%s]'%adcp.xy_srs]
            elif adcp.lonlat is not None:
                header = ['longitude [degE]','latitude [degN]']
            else:
                print('Error, input adcp has no position data - no file written')
                return
            header.extend(['datetime','U [m/s]','V [m/s]'])                
            arraywriter.writerow(header)
        for i in range(adcp.n_ensembles):
            if adcp.mtime is None:
                rec_time = 'None'
            elif adcp.mtime[i] is None or np.isnan(adcp.mtime[i]):
                rec_time = 'None'
            else:
                rec_time = num2date(adcp.mtime[i]).strftime('%c')
            if adcp.xy is not None:
                db_record = [adcp.xy[i,0],adcp.xy[i,1]]
            else:
                db_record = [adcp.lonlat[i,0], adcp.lonlat[i,1]]
                
            db_record = db_record + [rec_time,
                                     UVW[i,0],
                                     UVW[i,1]]
            arraywriter.writerow(db_record)



#def split_repeat_survey_into_transects(adcp_survey):
#    
#    if adcp_survey.xy is None:
#        raise Exception,'ADCP data must have an XY projection for trasect detection and splitting'
#    velocity_change = np.abs(adcp_survey.xy[1:-1,0]-adcp_survey.xy[0:-2,0]) + \
#    np.abs(adcp_survey.xy[1:-1,1]-adcp_survey.xy[0:-2,1])
#    sps.find_peaks_cwt(velocity_change, np.arange(1,10), wavelet=None, max_distances=None, gap_thresh=None, min_length=None, min_snr=1, noise_perc=10)

def group_adcp_obs_by_spacetime(adcp_obs,max_gap_m=30.0,
                                max_gap_minutes=20.0,max_group_size=6):
    """ 
    Sorts ADCPData objects first into groups by closeness in terms of location, 
    and then further sorts location groups by time.  Groups of ADCPData objects
    must first be within max_gap_m from each other, and then be within max_gap_minutes
    of each other.
    Inputs:
        adcp_obs = list of ADCPTransectData objects
        max_gap_m = maximum distance allowed between ADCP observations when grouping
        max_gap_minutes = maximum time allowed between ADCP observations when grouping
        max_group_size = maximum number of ADCPData objects per group
    Returns:
        List of lists that contain groups of input ADCPData objects
    """    
    space_groups = group_adcp_obs_within_space(adcp_obs,max_gap_m)
    for i in range(len(space_groups)):
        print('space group',i,'- ',len(space_groups[i]), 'observations')
    spacetime_groups = []
    for grp in space_groups:
        (sub_groups, gaps) = group_adcp_obs_within_time(grp,
                                                        max_gap_minutes,
                                                        max_group_size)
        spacetime_groups.extend(sub_groups)
    for i in range(len(spacetime_groups)):
        print('spacetime group',i,'- ',len(spacetime_groups[i]), 'observations')
    return spacetime_groups


def group_adcp_obs_within_space(adcp_obs,max_gap_m=30.0):
    """ 
    Sorts ADCPData objects  into groups by closeness in space, in an 
    ordered-walk/brute force manner.  Distances between all ADCO_Data observation
    centroids are found, and then starting with the first ADCO_data, the remaining
    ADCPData objects are evaluated for distance to the first.  If within 'max_gap_m'
    they are grouped and marked as 'picked' so they will not assigned to a group
    more than once.
    Inputs:
        adcp_obs = list of ADCPTransectData objects
        max_gap_m = maximum distance allowed between ADCP observations when grouping
    Returns:
        List of lists that contain groups of input ADCPData objects
    """    

    n_obs = len(adcp_obs)
    (centers,distances) = find_centroid_distance_matrix(adcp_obs)
    picked = np.zeros(n_obs,np.int)
    groups = []
    for i in range(n_obs):
        if not picked[i] and ~np.isnan(centers[i][0,0]):
            sub_group = [adcp_obs[i],]
            picked[i] = 1
            my_dist = distances[i,:]
            nn = np.argsort(my_dist)
            for n in nn:
                if not picked[n] and ~np.isnan(centers[n][0,0]):
                    if my_dist[n] < max_gap_m:
                        sub_group.append(adcp_obs[n])
                        picked[n] = 1
            groups.append(sub_group)
    return groups


def group_adcp_obs_within_time(adcp_obs,max_gap_minutes=20.0,max_group_size=6):
    """ 
    Sorts ADCPData objects into groups by closeness in time, with groups being
    separated by more than 'max_gap_minutes'.This method first sorts the group by
    start time, and then splits the observations where they are more than
    'max_gap_minutes' apart.
    Inputs:
        adcp_obs = list of ADCPTransectData objects
        max_gap_minutes = maximum Time allowed between ADCP observations when grouping
        max_group_size = maximum number of ADCPData objects per group
    Returns:
        List of lists that contain groups of input ADCPData objects
    """    
    if len(adcp_obs) == 1:
        return ([adcp_obs,], [None,])
    else:
        start_times = list()
        for a in adcp_obs:
            if a.mtime is not None:
                start_times.append(a.mtime[0])
            else:    
                start_times.append(None)
    
        if start_times:
            gaps, nn, nnan = find_start_time_gaps(start_times)
            adcp_obs_sorted = [ adcp_obs[i] for i in nn ]
            # convert nnan boolean list to integer index
            nnan_i = nnan * range(len(nnan))
            adcp_obs_sorted = [ adcp_obs_sorted[i] for i in nnan_i ]
            return group_according_to_gap(adcp_obs_sorted,gaps,max_gap_minutes,max_group_size=6)
        else:      
            raise Exception("find_transects_within_minimum_time_gap(): No valid times found in input files!")



def find_adcp_files_within_period(working_directory,max_gap=20.0,max_group_size=6):
    """ 
    Sorts a directory of ADCPRdiWorkHorseData raw files into groups by 
    closeness in time, with groups being separated by more than 
    'max_gap_minutes'. This method first sorts the files by start time, and 
    then splits the observations where they are more than
    'max_gap_minutes' apart.
    Inputs:
        working_directory = directory path containing ADCP raw or netcdf files
        max_gap = maximum time allowed between ADCP observations when grouping (minutes)
        max_group_size = maximum number of ADCPData objects per group
    Returns:
        List of lists that contain groups of input ADCPData objects
    """    
    if os.path.exists(working_directory):
        data_files = glob.glob(os.path.join(working_directory,'*[rR].000'))
        data_files.extend(glob.glob(os.path.join(working_directory,'*.nc')))
    else:
        print("Path (%s) not found - exiting."%working_directory)
        exit()
    
    start_times = list()
    for data_file in data_files:
        try:
            a = adcpy.open_adcp(data_file,
                            file_type="ADCPRdiWorkhorseData",
                            num_av=1)
            start_times.append(a.mtime[0])
        except:
            start_times.append(None)

    if start_times:
        gaps, nn, nnan = find_start_time_gaps(start_times)
        data_files_sorted = [ data_files[i] for i in nn ]
        # convert nnan boolean list to integer index
        nnan_i = nnan * range(len(nnan))
        data_files_sorted = [ data_files_sorted[i] for i in nnan_i ]
    return group_according_to_gap(data_files_sorted,gaps,max_gap,max_group_size)       
    

def find_start_time_gaps(start_times_list):
    """ 
    Find the time difference in minutes between datenum elements in a list 
    Sorts, removed nans, adn turns remaing datnum values in 'start_times_list'
    ino datetime objects, finds the timedelta objects between then, and
    converts to minutes.
    Inputs:
        start_times_list = numpy 1D array of matplotlib datenum values
    Returns:
        time_gaps_minutes = gaps between sorted times in start_times_list {minutes}
        nn = sort index for start_times_list
        nnan = boolean index of start_times_list[nn] where True is is non-nan
    """    
   
    # sort, remove unknowns, convert to datetime object
    start_times = np.array(start_times_list, dtype=np.float64)
    nn = np.argsort(start_times)
    start_times_sorted = start_times[nn]
    nnan = ~np.isnan(start_times_sorted)
    start_times_sorted = num2date(start_times_sorted[nnan])     
    # returns datetime.timedelta objects
    time_gaps_minutes = np.zeros(len(start_times_sorted)-1,np.float64)
    for i in range(len(start_times_sorted)-1):
        t_delta = start_times_sorted[i+1]-start_times_sorted[i]
        # timedelta objects only have days/seconds
        time_gaps_minutes[i] = t_delta.total_seconds()/60.0
    return (time_gaps_minutes, nn, nnan)


def group_according_to_gap(flat_list,gaps,max_gap,max_group_size):
    """ 
    Splits a python list into groups by their gaps in time, using a list of
    gaps between them.
    Inputs:
        flat_list = python list, shape [n]
        gaps = numeric list, shape [n-1], descibing gaps between elements of flat_list
        max_gap = maximum gap allowed between list elements
        max_group_size = maximum number of list elements per group
    Returns:
        List of lists that contain groups of input list elements
    """    
    within_gap = gaps <= max_gap
    groups = list()
    group_gaps = list()
    sub_group = list()
    sub_gaps = list()
    sub_group.append(flat_list[0])
    for i in range(len(gaps)):
        if ~within_gap[i] or len(sub_group) >= max_group_size:
            groups.append(sub_group)
            if not sub_gaps:
                sub_gaps.append((None,))
            group_gaps.append(sub_gaps)
            sub_group = []
            sub_gaps = []
        else:
            sub_gaps.append(gaps[i])
        sub_group.append(flat_list[i+1])
    groups.append(sub_group)
    if not sub_gaps:         
        sub_gaps.append((None,))
    group_gaps.append(sub_gaps)
    # returning (list of file lists, list of gap time lists)
    return (groups, group_gaps)


def calc_transect_flows_from_uniform_velocity_grid(adcp,depths=None,use_grid_only=False,
                                                   xy_line=None):
    """ 
    Calculates the cross-sectional area of the ADCP profiles from projection
    data, and multiplies it by the velocities to calculate flows
    and mean velocities.
    Inputs:
        adpc = ADCPData object. projected to an xy regular grid projection
        depths = optional 1D array of depths that correspond the ensemble 
          dimension of velocity in adcp
        use_grid_only = True: use each grid cell to calc flows/mean velocities
          False: first find depth-average velocties, then use depths to find 
          flows/mean velocties
        xy_line = [[x0,y0],[x1,y1]] defining orientation of the transect
    Returns:
        scalar_mean_vel = mean veolocity of total flow shape [3] {m/s}
        depth_averaged_vel = depth averaged velocity, shape [n,3] {m/s}
        total_flow = total U,V, and W discharge [3] {m^3/s}
        total_survey_area = total area used for flow calculations {m^3}
    """    
    # check to see if adcp is child of ADCPTransectData ??
    if adcp.xy is None:
        raise ValueError("xy projection required")

    if adcp.rotation_angle is None:
        print('Warning - No alignment axis set: Calculating flows according to U=East and V=North')
    rfv = False
    if not "bt_depth" in dir(adcp):
        rfv = True
    elif adcp.bt_depth is None:
        rfv = True
    xd,yd,dd,xy_line = adcpy.util.find_projection_distances(adcp.xy,pline=xy_line)
    dxy = abs(dd[0]-dd[1])
    dz = abs(adcp.bin_center_elevation[0]-adcp.bin_center_elevation[1])
    (depths, velocity_mask) = adcp.get_velocity_mask(range_from_velocities=rfv,nan_mask=True)

    if use_grid_only:
        area_grid = velocity_mask*dxy*dz
        total_survey_area = np.nansum(np.nansum(area_grid))
        scalar_mean_vel = np.zeros(3)
        total_flow = np.zeros(3)
        depth_averaged_vel = np.zeros((adcp.n_ensembles,3))
        for i in range(3):
            total_flow[i] = np.nansum(np.nansum(adcp.velocity[:,:,i]*area_grid))
            masked_vel = adcp.velocity[:,:,i]*velocity_mask
            depth_averaged_vel[:,i] = util.nanmean(masked_vel,axis=1)
            scalar_mean_vel[i] = util.nanmean(masked_vel.ravel())
    else:      
        if rfv:
            print('Warning - No bottom depth set: Calculating flows according valid velocity bins only')
        total_survey_area = np.nansum(dxy*depths)
        depth_averaged_vel =  adcp.ensemble_mean_velocity(range_from_velocities=rfv)
        depth_integrated_flow = adcp.calc_ensemble_flow(range_from_velocities=rfv)
        scalar_mean_vel = util.nanmean(depth_averaged_vel,axis=0)
        total_flow = np.nansum(depth_integrated_flow,axis=0)
    
    return (scalar_mean_vel, depth_averaged_vel, total_flow, total_survey_area)


def find_centroid_distance_matrix(adcp_obs):
    """ 
    Calculates all possible distances between a list of ADCPData objects (twice...ineffcient)
    Inputs:
        adcp_obs = list ADCPData objects, shape [n]
    Returns:
        centers = list of centorids of ensemble locations of input ADCPData objects, shape [n]
        distances = xy distance between centers, shape [n-1]
    """    
    n_obs = len(adcp_obs)
    distances = np.empty((n_obs,n_obs),np.float64)
    centers = []
    for a in adcp_obs:
        if a.xy is not None:
            centers.append(adcpy.util.centroid(a.xy))
        else:
            centers.append(np.array([np.nan,np.nan]))
    centers = [adcpy.util.centroid(a.xy) for a in adcp_obs]
    for i in range(n_obs):
        for j in range(n_obs):
            distances[i,j] = adcpy.util.find_line_distance(centers[i],centers[j])
    return (centers,distances)


def transect_rotate(adcp_transect,rotation,xy_line=None):
    """ 
    Rotates ADCPTransectData U and V velocities.
    Inputs:
        adcp_transect = ADCPTransectData object
        rotation = one of:
          None - no rotation of averaged velocity profiles
         'normal' - rotation based upon the normal to the plotline (default rotation type)
         'pricipal flow' - uses the 1st principal component of variability in uv flow direction
         'Rozovski' - individual rotation of each vertical velocity to maximize U
         'no transverse flow' - rotation by the net flow vector is used to minimize V
        xy_line = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]], or None
    Returns
        adcp_transect = ADCPTransectData object with rotated uv velocities
    """
    if rotation == "normal":
        # find angle of line:
        if xy_line is None:
            if adcp_transect.xy is None:
                raise Exception("transect_rotate() error: ADCPData must be xy projected, or input xy_line must be supplied for normal rotation")
            xy_line = adcpy.util.map_xy_to_line(adcp_transect.xy)
        theta = adcpy.util.calc_normal_rotation(xy_line)
    elif rotation == "no transverse flow":
        flows = adcp_transect.calc_ensemble_flow(range_from_velocities=True)
        theta = adcpy.util.calc_net_flow_rotation(flows[:,0],flows[:,1])
    elif rotation == "Rozovski":
        flows = adcp_transect.calc_ensemble_flow(range_from_velocities=True)
        theta = adcpy.util.calc_Rozovski_rotation(flows[:,0],flows[:,1])
    elif rotation == "principal flow":
        flows = adcp_transect.calc_ensemble_flow(range_from_velocities=True)
        theta = adcpy.util.principal_axis(flows[:,0],flows[:,1],calc_type='EOF')
    elif type(rotation) is str:
        raise Exception("In transect_rotate(): input 'rotation' string not understood: %s"%rotation)
    else:
        theta = rotation
    
    adcp_transect.set_rotation(theta,'uv')

    return adcp_transect


def find_uv_dispersion(adcp):
    """
    Calculates dispersion coeffcients of velocties in adcp according to 
    Fischer et al. 1979
    Inputs:
        adcp = ADCPTransectData object
    Returns:
        ustbar = 
        Kx_3i = horizontal dispersion coefficients
        Ky_3i = lateral dispersion coefficients
    """
    # should check to see if it is regular grid - required for dispersion calc
    if adcp.xy is None:
        raise ValueError("adcp.xy (xy projection) must exist for dispersion calculation")
    if adcp.bt_depth in adcp.__dict__:
        depth = adcp.bt_depth
    else:
        (depth, velocity_mask) = adcp.get_velocity_mask(range_from_velocities=True,
                                                        nan_mask=True)
    xd,yd,dd,xy_line = adcpy.util.find_projection_distances(adcp.xy)
    return  adcpy.util.calcKxKy(adcp.velocity[:,:,0],
                                adcp.velocity[:,:,1],
                                dd,
                                adcp.bin_center_elevation,
                                depth)
