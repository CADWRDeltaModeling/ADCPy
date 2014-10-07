# -*- coding: utf-8 -*-
"""
ADCPy

Overview: ADCPy allows the user to read raw (unprocessed) data from ADCP 
instruments, perform a suite of processing functions and data transformations, 
and output summary data and related plots. By providing access to the raw ADCP 
velocities, ADCPy allows exacting user control over the diagnosis of water 
movement. Numerous built-in data transformation tools and associated utilities 
allow full scripting of ADCP processing, from raw data to final output plots 
and flow volumes. Raw ADCP data is stored as a python object class, which may 
be exported and imported to disk in the Network Common Data Format (Climate 
and Forecast Metadata Convention; NetCDF-CF).

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""

import os
import numpy as np
import scipy.stats.stats as sp
#import scipy.stats.morestats as ssm
import ADCPy_utilities as util
import ADCPy_plot as plot
import netCDF4

""" 
Make sure subclasses of ADPCData are recorded here, so they may be
loaded dynamically.
"""
adcpdata_subclass_names = [ 'ADCP_RdiWorkhorse_Data',  # RDI Workhorse / WinRiver II Raw File, subclass of ADCPTransectData
                            'ADCP_RdiRiverRay_Data',   # RDI River Ray / WinRiver II Raw File, subclass of ADCPTransectData
                            'ADCP_RdiChannelMaster_Data' ]  # RDI Channel Master Raw File, subclass of ADCPMooredData
                            #'ADCP_Transect_Data',      # General sub-class of ADCPData with transect-specific functionality
                            #'ADCP_Moored_Data',        # General sub-class of ADCPData with moored-specific functionality

def open_adcp(file_path,file_type=None,**kwargs):
    """ 
    Attempts to determine the type of a passed ADCP data file, and 
    will pass the file type to the appropriate subclass for reading, 
    and will return an populated AdcpData structure if possible.  Optionally
    the file type (and subclass reading) can be forced by assigning one
    of the known subclass names as a string.
    Inputs:
        file_path = path and filename of ADCPy-supported file to open [str]
        file_type = either 'raw_file' or 'nc_file' [str], optional to help in deciding how to open file
        ** additional keyword=argument pairs that will be passed to the appropriate open call
    Returns:
        ADCP_Data class object (or sub-class)
    """
    file_name,ext = os.path.splitext(file_path)
    
    # Native NetCDF file type - attempt to determine the correct
    # file_type/subclass module
    if ext == '.nc':      
        if not os.path.exists(file_path):
            raise IOError, "Cannot find %s"%file_path       
        rootgrp = netCDF4.Dataset(file_path, 'r', format='NETCDF4')
        if rootgrp.ADCPData_class_name:
            if rootgrp.ADCPData_class_name in adcpdata_subclass_names:
                file_type = rootgrp.ADCPData_class_name
            rootgrp.close()
            init_file_type = 'nc_file'
        else:
            print "Invalid ADCPy NetCDF datafile."
            raise
    else:   
        init_file_type = 'raw_file'

    # Import subclass module and init
    if file_type in adcpdata_subclass_names:
        try:
            #map(__import__,file_type)
            exec("import %s"%file_type)             
        except:
            print  "Import of module '" + file_type +"' failed"
            raise
        init_command = 'adata = %s.%s(%s=file_path,**kwargs)'%(file_type,file_type,init_file_type)
    # Init from local classes
    else:
        init_command = 'adata = %s(%s=file_path,**kwargs)'%(file_type,init_file_type)
    try:
        exec(init_command)                
        return adata      
    except:
        print  "Init of ADCPData file_type '" + file_type +"' failed"
        raise             

class ADCP_Data(object):
    """ 
    Encapsulates data from a single deployment of an ADCP
    Note that this class is intended to be general across multiple
    types of ADCPs and deployments - to the extent possible place 
    code specific to brand or type of deployment in subclasses.
    """
    # Attributes common to all adcps (though may be None if no such 
    # data is present
    base_data_names =  ('n_ensembles',  # time/horizontal dimension
                        'n_bins',       # vertical dimension
                        'velocity',     # [n_ensembles, n_bins, {0:u,1:v,2:w}] (m/s)
                        'mtime',        # [n_ensembles] - matplotlib date2num values
                        'bin_center_elevation', # [n_bins] - range from transducer head to bin center, negative downward (m)
                        'rotation_angle',     # V-direction alignment axis (=0 if u = East and v = North) (degrees from North)
                        'rotation_axes',     # V-direction alignment axis (=0 if u = East and v = North) (degrees from North)
                        'xy',           # [n_ensembles,{x,y}] - transformed positions
                        'xy_srs',       # name of projection of xy
                        'lonlat',       # [n_ensembles,{0:lon,1:lat}], or None if no nav data available.
                        'lonlat_srs',   # name of projection of lonlat
                        'title',        # A succinct description of what is in the dataset
                        'institution',  # affiliated institution
                        'source',       # description of data source and/or filename(s)
                        'references',   # important references for data/the model-version that produced it/publication that reported it
                        'comment',      # miscellaneous comments
                        'history')      # the history list contains processing evens descriptions/audit trail stored as a single
                                        # string delinieated by \n (newline) characters
    # Flags affecting behavior
    default_lonlat_srs = 'WGS84'
    # Which field is defined as the first dimension of the variables
    # in netcdf output.  'time' will generally cause software to 
    # consider the data a timeseries
    nc_ensemble_dim = 'ensemble' # 'time'

    def __init__(self,raw_file=None,nc_file=None,**kwargs):
        self.clean_base_data()
        self.messages = [] # list of messages related to reading/processing
        self.history = ""
        if nc_file is not None:
            self.read_nc(nc_file=nc_file)
        elif raw_file is not None:
            self.read_raw(raw_file=raw_file,**kwargs)    


    def clean_base_data(self):
        """ 
        Sets all base data variables to none.  Used in __init__, and 
        probably shouldn't be called anywhere else.
        """
        for var in self.base_data_names:
            exec("self.%s = None"%var)


    def copy_base_data(self):
        """ 
        Copies base ADCP data to a new ADCP_data calss - used to populate
        a new ADCP_Data class for the purposes of saving processed data
        or stripping out subclass information.
        Returns:
            adata = new ADCP_Data class object
        """
        adata = ADCP_Data()
        for var in self.base_data_names:
            exec("adata.%s = self.%s"%(var,var))
        self.history_append('copy_base_data/strip to base variables')
        return adata


    def copy_minimum_data(self):
        """ 
        Copies the minimum amount of data required to constitute ADCPy data.
        Returns:
            adata = new ADCP_Data class object        
        """
        min_data_names = ('n_ensembles',
                          'n_bins',
                          'velocity',
                          'bin_center_elevation')
        adata = ADCP_Data()
        for var in min_data_names:
            exec("adata.%s = self.%s"%(var,var))
        return adata


    def msg(self,s):
        """ 
        Collect information/debugging/warnings during reading and processing 
        of this data. 
        Inputs:
            s = python str to display
        """
        self.messages.append(s)
        print s
        
    def print_history(self):
        """ 
        Returns a string describing the filename and/or method
        used to get the data - analagous to CF conventions history.
        """
        if self.history is None or not self.history:
            print 'no history found'
        else:
            print self.history         

    def history_append(self,string):
        """ 
        Appends a string to the ADCP_Data class history
        Inputs:
            string = python str to append
        """    
        from datetime import datetime
        dtn = datetime.now()
        self.history = self.history + dtn.strftime('%Y-%m-%d %H:%M:%S ') + string + '; '

    def get_subclass_name(self):
        """ 
        Helper function thet returns the ultimate (sub)class name of self
        """
        return self.__class__.__name__
        
    def write_nc(self,filename,zlib=None):
        """ 
        Write the ADCP data to a CF-compliant netcdf file        
        Subclasses should generally leave this be, and implement
        extra attribute/variable writing via self.write_nc_extra(rootgrp)
        Inputs:
            filename = string with netcdf output filename and path
            zlib = if True, use variable compression
        """
        rootgrp = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        try:
            # NetCDF attribute containing (sub)class that wrote the files
            # This allows reading of the correct subclass netcdf variables
            # using only the file itself
            rootgrp.ADCPData_class_name = self.get_subclass_name()
            
            # dimensions:                
            time_dim = rootgrp.createDimension(self.nc_ensemble_dim,self.n_ensembles)
            bin_dim = rootgrp.createDimension('bin',self.n_bins)
            component3_dim = rootgrp.createDimension('component3',3)
            component2_dim = rootgrp.createDimension('component2',2)
            
            # Coordinate variables - depending on self.nc_ensemble_dim, time
            # may be a coordinate variable, or not...
            time_var = rootgrp.createVariable('time','f8',
                                              self.nc_ensemble_dim,
                                              fill_value=0.0,
                                              zlib=zlib)
            # these settings are supposed to match matplotlib date2num
            time_var.units='days since 0000-12-31 00:00:00.0'
            time_var.calendar='proleptic_gregorian'
            if self.mtime is not None:
                time_var[...] = self.mtime
            
            ens_var = rootgrp.createVariable('ensemble','i8',self.nc_ensemble_dim,
                                             zlib=zlib)
            ens_var.units = 'count'
            ens_var[...] = np.arange(self.n_ensembles)
            
            bin_var = rootgrp.createVariable('bin','f8','bin',zlib=zlib)
            bin_var.units = 'm'
            bin_var[:] = self.bin_center_elevation
            
            comp_var3 = rootgrp.createVariable('component3','S1','component3')
            comp_var3.units = 'none'
            comp_var3[:] = ['u','v','w']

            comp_var2 = rootgrp.createVariable('component2','S1','component2')
            comp_var2.units = 'none'
            comp_var2[:] = ['u','v']
            
            velocity_var = rootgrp.createVariable('velocity','f8',
                                                  (self.nc_ensemble_dim,'bin',
                                                   'component3'),zlib=zlib)
            velocity_var.units = 'm/s'
            velocity_var[...] = self.velocity
            
            if self.lonlat is not None:
                lat_var = rootgrp.createVariable('lat','f8',
                                                 self.nc_ensemble_dim,
                                                 zlib=zlib)
                lat_var.units = 'degrees_north'
                lat_var.standard_name = 'latitude'
                lat_var[...] = self.lonlat[...,1]
                
                lon_var = rootgrp.createVariable('lon','f8',self.nc_ensemble_dim,
                                                 zlib=zlib)
                lon_var.units = 'degrees_east'
                lon_var.standard_name = 'longitude'
                lon_var[...] = self.lonlat[...,0]
 
            if self.xy is not None:
                pro_var = rootgrp.createVariable('xy_projection','i4')
                pro_var.scale_factor_at_central_meridian = 0
                pro_var.grid_mapping_name = 'adcp_xy_projection'
                pro_var.epsg_code = self.xy_srs

                ens_x_var = rootgrp.createVariable('ens_x','f8',
                                                   self.nc_ensemble_dim,
                                                   zlib=zlib)
                ens_x_var.units = 'm'
                ens_x_var.standard_name = 'projection_x_coordinate'
                ens_x_var.long_name = 'Projected adcp ensemble x coordinate - hopefully in m'
                ens_x_var[...] = self.xy[...,0]
                
                ens_y_var = rootgrp.createVariable('ens_y','f8',self.nc_ensemble_dim,
                                                   zlib=zlib)
                ens_y_var.units = 'm'
                ens_y_var.standard_name = 'projection_y_coordinate'                
                ens_y_var.long_name = 'Projected adcp ensemble y coordinate - hopefully in m'
                ens_y_var[...] = self.xy[...,1]

            if self.rotation_angle is not None:
                rootgrp.rotation_angle = self.rotation_angle
                rootgrp.rotation_axes = self.rotation_axes
            
            # Some global attributes, recommended by CF
            rootgrp.Conventions="CF-1.0"
            if self.title is not None:
                rootgrp.title = self.title
            if self.institution is not None:
                rootgrp.institution = self.institution
            if self.source is not None:
                rootgrp.source = self.source
            if self.references is not None:
                rootgrp.references = self.references
            nc_history = "Wrote to NetCDF file %s"%(filename)
            self.history_append(nc_history)
            rootgrp.history = self.history
            
            
            self.write_nc_extra(rootgrp,zlib)
        finally:
            # Failing to close the file can cause it to fail on
            # subsequent calls
            rootgrp.close()
        
    def write_nc_extra(self,rootgrp,zlib):
        """ 
        Entry point for subclasses to add attributes and variables to the base
        ADCP_Data data.
        Inputs:
            rootgrp = Python NetCDF object
            zlib = if True, use variable compression
        """
        pass

    def read_nc(self,nc_file):
        """ 
        Read base/minimum ADCP_Data variables from a NetCDF format data file.        
        Inputs:
            nc_file = string with input netcdf filename and path
        """
        if not os.path.exists(nc_file):
            raise IOError, "Cannot find %s"%nc_file
        
        self.nc_file = nc_file
        
        rootgrp = netCDF4.Dataset(nc_file, 'r', format='NETCDF4')
    #try:

        # check subclass type of netcdf file to see if the read_nc_extras
        # matches/will work.
        if rootgrp.ADCPData_class_name != self.get_subclass_name():
            print 'WARNING - NetCDF file was not written by this ADCPData subclass.'
            print 'Variables besides those in the base class may be ignored.'
            print 'NetCDF ADCPData Class: ',rootgrp.ADCPData_class_name
            print 'Calling ADCPData Class: ',self.get_subclass_name()
 
        # read base variables
        # import pdb; pdb.set_trace()
        self.mtime = rootgrp.variables['time'][...]
        if self.mtime[0] == 0.0:
            self.mtime = None
        self.velocity = rootgrp.variables['velocity'][...]
        self.bin_center_elevation = rootgrp.variables['bin'][...]       
        self.n_ensembles, self.n_bins,n_vels = np.shape(self.velocity)
        
        # read attributes/scalars/strings
        attributes = ['rotation_angle', 'rotation_axes',
                      'title','institution','source',
                      'references','comment','history']                                                              
        for att in attributes:
            self.read_nc_att(rootgrp,att)

        if 'lat' in rootgrp.variables:
            self.lonlat = np.zeros((self.n_ensembles,2),np.float64)
            self.lonlat[:,1] = rootgrp.variables['lat'][...]
            self.lonlat[:,0] = rootgrp.variables['lon'][...]
        if 'ens_x' in rootgrp.variables:
            self.xy = np.zeros((self.n_ensembles,2),np.float64)
            self.xy[:,1] = rootgrp.variables['ens_y'][...]
            self.xy[:,0] = rootgrp.variables['ens_x'][...]
            self.xy_srs = rootgrp.variables['xy_projection'].epsg_code

            print 'Doing read_nc in ADCP_Data...'


        self.read_nc_extra(rootgrp)
        
    #except:
        
        #print 'NETCDF read of ADCP_Data failed for file: ',nc_file

    #finally:
        # Failing to close the file can cause it to fail on
        # subsequent calls
        rootgrp.close()
        
        self.history_append('Read in from netcdf file: %s'%nc_file)

    def read_nc_var(self,rg,var):
        """ 
        Helper method for reading netcdf variables from a netcdf file.
        Inputs:
            rg = Python NetCDF object
            var = string of variable name
        """
        if var in rg.variables:
            command = 'self.%s = rg.variables[\'%s\'][...]'%(var,var)
            exec(command)

    def read_nc_att(self,rg,att):
        """ 
        Helper method for reading netcdf attributes from a netcdf file.
        Inputs:
            rg = Python NetCDF object
            att = string of attribute name
        """
        if hasattr(rg,att):
            command = 'self.%s = rg.%s'%(att,att)
            exec(command)

    def read_nc_extra(self,rootgrp):
        """ 
        Entry point for subclasses to read extra attributes and variables.
        Inputs:
            rootgrp = Python NetCDF object
        """
        pass  

    def read_raw(self,raw_file,**kwargs):
        """ 
        Entry point for subclasses to read some sort of native format.
        Inputs:
            raw_file = path to raw file [str]
            ** additional keyword=argument pairs to be used by binary file
               specific read_raw definitions.           
        """
        pass  

    def lonlat_to_xy(self,xy_srs):
        """ 
        Project geographic coordinates (self.lonlat) to self.xy.  Specify
        a string ID for the intended projection (e.g. EPSG:26910).
        Inputs:
            xy_srs = EPSG code [str]
        """
        if self.lonlat is None:
            raise Exception,"lonlat_to_xy: Attempt to transform coordinates with no lat/lon"        
        if xy_srs is None:
            raise Exception,"lonlat_to_xy: Attempt to transform coordinates without supplying projection"        
        ll_srs = self.lonlat_srs
        if ll_srs is None:
            ll_srs = self.default_lonlat_srs            
        self.xy = util.coordinate_transform(self.lonlat,ll_srs,xy_srs,interp_nans=True)
        self.xy_srs = xy_srs
        self.history_append("lonlat_to_xy(xy_srs=%s)"%xy_srs)


    def xy_to_lonlat(self,lonlat_srs=None):
        """ 
        Transform projected coordinates (self.xy) to gerographic coordinates 
        (self.lonlat).  Specify a string ID for the intended projection 
        (e.g. EPSG:26910) or self.default_lonlat_srs is used.
        Inputs:
            lonlat_srs = EPSG code [str]
        """
        if self.xy is None:
            raise Exception,"xy_to_lonlat: Attempt to transform coordinates with no xy data"        
        if lonlat_srs is None:
            lonlat_srs = self.default_lonlat_srs            
        self.lonlat = util.coordinate_transform(self.xy,self.xy_srs,lonlat_srs,interp_nans=True)
        self.lonlat_srs = lonlat_srs
        self.history_append("xy_to_lonlat(lonlat_srs=%s)"%lonlat_srs)


    def xyzt(self):
        """
        Return the x,y,z, pojected positions, along with the time of every
        velocity in the velocity array.
        Returns:
            x = x-direction projected coordinates (ideally m)
            y = y-direction projected coordinates (ideally m)
            z = z posotion of velocity measure
            t = datetime of velocity measure
        """
        x = np.hstack([self.x[:,0] for i in range(self.n_bins)])
        y = np.hstack([self.x[:,0] for i in range(self.n_bins)])
        z = np.vstack([self.bin_center_elevation for i in range(self.n_ensembles)])
        if self.mtime is not None:
            t = np.hstack([self.mtime for i in range(self.n_bins)])
        else:
            t = None
        return (x,y,z,t)

    def self_copy(self):
        """
        Create an exact copy of an ADCP_Data class (or sub-class).
        Returns:
            new ADCP_Data class object  
        """
        import copy as cp
        return cp.deepcopy(self)

    def average_ensembles(self,ens_to_avg):
        """
        Averages adjacent ADCP profiles (ensembles) as a method of noise/data
        reduction.  Data is averaged in the time dimension; the resulting data
        retains its range (.n_bins) resolution.
        Inputs:
            ens_to_avg = integer number of adjacent ensembles to average
        """
        #create copy of adcp class
        a = self.self_copy()
     
        # find indices
        n2 = np.int(np.floor(a.n_ensembles/ens_to_avg))
        nn = range(n2*ens_to_avg)
    
        # take median on ensemble times
        a.mtime = np.median(a.mtime[nn].reshape(n2,ens_to_avg),1)
    
        # create averaged variables
        a.lonlat = util.average_array(self.lonlat[nn,:],(n2,ens_to_avg),axis=0)
        a.velocity = np.zeros((n2, a.n_bins, 3),np.float64)
        for i in range(3):
            a.velocity[:,:,i] = util.average_array(self.velocity[nn,:,i],(n2,ens_to_avg),axis=0)  
        if self.xy is not None:
            a.xy = util.average_array(self.xy[nn,:],(n2,ens_to_avg),axis=0)
        a.n_ensembles = n2
        a.history_append('average_ensembles(ens_to_avg=%i)'%ens_to_avg)        
        return a

    def remove_sidelobes(self,fsidelobe=0.10):
        """
        Throws out near-bottom cells b/c of side lobe problems 
        fSidelobe=0.10; used 0.15 in past, but Carr and Rehmann use 0.06...
        Inputs:
            fsidelobe = fraction of total elevation to drop 
        """
        side_lobes = util.find_sidelobes(fsidelobe,
                                       self.bt_depth,
                                       self.bin_center_elevation)
        for i in range(3):
          self.velocity[:,:,i][side_lobes] = np.nan
        self.history_append('remove_sidelobes(fsidelobe=%f)'%fsidelobe)


    def kernel_smooth(self,kernel_size):
        """
        Uses a box/uniform filter of size [kernel_size,kernel_size] to
        smooth velocity.
        Inputs:
            kernel_size = odd integer of size 3 or larger
        """
        for i in range(3):
            self.velocity[:,:,i] = util.kernel_smooth(kernel_size,self.velocity[:,:,i])
        self.history_append('kernel_smooth(kernel_size=%i)'%kernel_size)

        
    def sd_drop(self,sd=3.0,sd_axis='elevation',interp_holes=True,
                warning_fraction=0.05):
        """
        Throw out outliers and fill in gaps based upon standard deviation - 
        typical to use 3 standard deviations (sd=3).
        Inputs:
            sd = stadard defiation, scalar float
            sd_axis = either 'elevation' or 'ensemble'
            interp_holes = if True, interpolate to fill hole created by dropping values > sd
            warning_fraction = if fraction of cells dropped is greater than this,
              throw a warning
        """
        v_mag = np.sqrt(self.velocity[:,:,0]**2+self.velocity[:,:,1]**2)
        v_vert = self.velocity[:,:,2]
        if sd_axis=='elevation':
            elev = -1.0*self.bin_center_elevation
            axis = 1
        elif sd_axis=='ensemble':
            xd,yd,elev = util.find_projection_distances(self.xy)
            axis = 0
        else:
            print "Unkown axis '%s' passed to sd_drop; vaid options are 'elevation' and 'ensemble'"%sd_axis
            raise ValueError
        # drop U/V velocity values, using total UV magnitude
        drop = util.find_sd_greater(v_mag,elev,sd,axis=axis)          
        for i in range(3):
            if i == 2:
                # find separate drop values for vertical (W) velocities
                drop = util.find_sd_greater(v_vert,elev,sd,axis=axis) 
            self.velocity[:,:,i] = util.remove_values(self.velocity[:,:,i],
                                                    drop,
                                                    axis=axis,
                                                    elev=elev,
                                                    interp_holes=True,
                                                    warning_fraction=warning_fraction)   
        self.history_append('sd_drop(sd=%f,sd_axis=%s,interp_holes=%s,warning_fraction=%f)'%(sd,
                                                                                             sd_axis,
                                                                                             interp_holes,
                                                                                             warning_fraction))


    def rotate_UV_velocities(self,radian):
        """
        Re-orients U and V velocities to an arbitrary rotation, without self
        assignment.
        Inputs:
            radian = rotation in radians
        Returns:
            Python list of 2 velocity numpy arrays.
        """
        return util.rotate_velocity(radian,
                                  self.velocity[:,:,0],
                                  self.velocity[:,:,1])
                                  


    def set_rotation(self,radian,axes_string='UV'):
        """
        Re-orient designated velocities to an arbitrary rotation.
            Python list of 2 velocity numpy arrays.
        Inputs:
            radian = rotation in radians
            axes_string = 2-character string containing 'U','V', or 'W', 
              indicating which velocity axes to rotate, with the first
              being in the 0-degree direction and the second the 90-degree
              direction
        """
        
        ax = util.get_axis_num_from_str(axes_string)
        if len(ax) != 2:
            ValueError("ADCPy.rotate_velocities: axes_string '%s' not understood")
            raise

        # un-rotate previous rotation before setting new rotation
        if self.rotation_angle is not None:
            ax_old = util.get_axis_num_from_str(self.rotation_axes)
            (self.velocity[:,:,ax_old[0]], 
             self.velocity[:,:,ax_old[1]]) = self.rotate_velocities(-1.0*self.rotation_angle,
                                                                ax_old[0],
                                                                ax_old[1])
        if radian is None:
            self.rotation_angle = None
            self.rotation_axes = None
            self.history_append('rotate_velocities(None)')
        else:                          
            (self.velocity[:,:,ax[0]], 
             self.velocity[:,:,ax[1]]) = self.rotate_velocities(radian,
                                                                ax[0],
                                                                ax[1])
            self.rotation_angle = radian
            if np.size(radian) > 1:
                radian_str = 'multiple'
            else:
                radian_str = '%f'%radian
            self.rotation_axes = axes_string
            self.history_append('rotate_velocities(radian=%s,axes=%s)'%(radian_str,axes_string))        
        

    def get_unrotated_velocity(self):
        """
        If a rotation has been applied to two velocties, this methods un-rotates
        velocties back to raw rotations and returns a 3D velocity array.
        Returns:
            velocity = 3D numpy array of U,V,W velocities with zero rotation        
        """
        velocity = np.copy(self.velocity)
        if self.rotation_angle is not None:
            ax = util.get_axis_num_from_str(self.rotation_axes)
            (velocity[:,:,ax[0]], 
             velocity[:,:,ax[1]]) = util.rotate_velocity(-1.0*self.rotation_angle,
                                                         velocity[:,:,ax[0]],
                                                         velocity[:,:,ax[1]])
        return velocity

    def rotate_velocities(self,radian,ax1,ax2):
        """
        Re-orient velocities to an arbitrary rotation
        Inputs:
          radian = rotation in radians, either 1D numpy array or scalar
          ax1 = velocity component axis, scalar integer 1:3
          ax2 = velocity component axis, scalar integer 1:3
        Returns:
          list of 2 velocity array
        """
        return util.rotate_velocity(radian,
                                    self.velocity[:,:,ax1],
                                    self.velocity[:,:,ax2])

    def extrapolate_boundaries(self):
        """
        Extrapolates velocities to surface/bottom boundaries where ADCP measurements
        are typically not available or valid.
        """
        ex_elev,ex_depth,ex_bins = util.find_extrapolated_grid(self.n_ensembles,
                                                               -self.bin_center_elevation,
                                                               self.bt_depth,
                                                               self.adcp_depth)
        self.velocity = util.extrapolate_boundaries(self.velocity,
                                                    -self.bin_center_elevation,
                                                    ex_elev,ex_depth,ex_bins)
        self.bin_center_elevation = -ex_elev
        self.n_bins = np.size(self.bin_center_elevation)
        self.history_append('extrapolate_boundaries')


    def calc_ensemble_flow(self,elev_line=None,range_from_velocities=False):
        """ 
        Uses valid data and bin_center_elevation distances to construct
        flows in the ensemble axis.
        Inputs:
            elev_line = optional scalar or array elevation (distance from 
              transducer) neyond which velocity is invalid
            range_from_velocities = if True, calculates the elev_line
              from the range of valid (non-NaN) velocities in bins.
        Returns:
            2D numpy array, shape [self.n_ensembles,3], with U,V,W flows
        """
        (my_elev_line,mask) = self.get_velocity_mask(elev_line,range_from_velocities,nan_mask=True)
        flows = np.zeros((self.n_ensembles,3),np.float64)
        for i in range(3):
            flows[:,i] = np.nansum(self.velocity[:,:,i]*mask,axis=1)*my_elev_line
        return flows

    def ensemble_mean_velocity(self,elev_line=None,range_from_velocities=False):
        """ 
        Uses valid data and bin_center_elevation distances to construct
        mean flows in the ensemble axis.
        Inputs:
            elev_line = optional scalar or array elevation (distance from 
              transducer) neyond which velocity is invalid
            range_from_velocities = if True, calculates the elev_line
              from the range of valid (non-NaN) velocities in bins.
        Returns:
            2D numpy array, shape [self.n_ensembles,3], with mean U,V,W 
              velocities
        """
        (my_elev_line,mask) = self.get_velocity_mask(elev_line,range_from_velocities,nan_mask=True)
        mean_velocity = np.zeros((self.n_ensembles,3),np.float64)
        for i in range(3):
           mean_velocity[:,i] = sp.nanmean(self.velocity[:,:,i]*mask,axis=1)
        return mean_velocity


    def get_velocity_mask(self,elev_line=None,range_from_velocities=False,
                          mask_region='above',nan_mask=False):
        """ 
        Generates a either a boolean mask, or a 1/NaN mask, correspnding to
        valid velocties measurements. If elev_line is given values beyond
        this elevation are masked as invalid.
        Inputs:
            elev_line = optional scalar or array elevation (distance from 
              transducer) neyond which velocity is invalid
            range_from_velocities = if True, calculates the elev_line
              from the range of valid (non-NaN) velocities in bins.
            mask_region = 'above' 
        Returns:
            2D numpy array, shape [self.n_ensembles,3], with mean U,V,W 
              velocities
        """
       
        my_elev_line = np.copy(elev_line)
        if range_from_velocities:
            # find lowest non-nan data
            my_elev_line = util.find_max_elev_from_velocity(self.velocity[:,:,0],
                                                self.bin_center_elevation)        
        if my_elev_line is not None:
            # mask out velocities below depth
            mask = util.find_mask_from_vector(self.bin_center_elevation,
                                        my_elev_line,
                                        mask_area = mask_region)
        else:
            elev = abs(self.bin_center_elevation[-1]-self.bin_center_elevation[0])
            elev = elev + elev/self.n_bins   # add a single bin to get probable total height
            my_elev_line = np.ones(self.n_ensembles)*elev
            mask = np.ones(np.shape(self.velocity[:,:,0]))
        if nan_mask:
            mask = mask*1.0  # convert to float
            mask[mask==0] = np.nan  # set to nan for data elimination - important b/c zeros are valid data            
        return (my_elev_line,mask)

    def xy_regrid(self,dxy,dz,xy_srs=None,pline=None,sort=False,kind='bin average'):
        """ 
        Projects ensemble locations to a strain line in the xy-plane, and
        then regrids velocities onto a regular grid defined by dxy and dz.
        This process changes the the dimensions of almost every piece of data in
        the class. self.mtime is handled differently since it does not make sense
        to average or interpolate time based on location.  Returns intermediate
        calculations to facilitate regridding of additional data by subclasses.
        Inputs:
            dxy = new grid xy resolution in xy projection units
            dz = new grid z resolution in z units
            xy_srs = EPSG code [str]        
            pline = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]], or None
            sort = if True, pre-sort the data being regrided in terms of location
               on the projection line
             kind = one of ['bin_average', linear','cubic','nearest'], where 
               the later three are types of numpy interpolation
        Returns:
            xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
            xy_new = new grid x-y locations, 2D array of shape [ne2,2], where [:,0] is x and [:,1] is y
            z = z positions, 1D array of shape [nb]
            z_new = z positions of new grid, 1D array of shape [nb2]
            nn = sort order of xy locations on the projection line
            pre_calcs = python list of different intermediate things - see 
              ADCPy_utilities.py
        """        
        # switch to new projection if required
        if (xy_srs is not None and self.xy_srs != xy_srs) or self.xy is None:
            self.lonlat_to_xy(xy_srs)
        # find new retangular grid sides and xy locations
        (xy_range,xy_new_range,xy_new,z_new) = util.new_xy_grid(self.xy,
                                                              self.bin_center_elevation,
                                                              dxy,dz,
                                                              pline=pline)
        # pre calculate meshes for (faster) 2D interpolation, and in the interest
        # of speed when not reordering arrays...
        if sort:
            nn = np.argsort(xy_range)
            xy_range = xy_range[nn]
            xy = self.xy[nn,:]
            v_interp = self.velocity[nn,:,:]
        else:
            nn = None
            v_interp = self.velocity         
        zmesh_new, xymesh_new = np.meshgrid(z_new,xy_new_range)
        zmesh, xymesh = np.meshgrid(self.bin_center_elevation,xy_range)
        # package pre_calc variables
        pre_calcs = (xy_range,zmesh,xymesh,xy_new_range,zmesh_new, xymesh_new)
        # regrid and mask all xy-based vars
        self.velocity = util.xy_regrid_multiple(v_interp,self.xy,xy_new,
                                             self.bin_center_elevation,
                                             z_new,pre_calcs,kind)
        # mtime is special - we can't regrid if sorted 
        # removed loop times to get min transect cross time
        nn = util.find_xy_transect_loops(self.xy,xy_range=xy_range)
        pre_calcs_mtime = (xy_range[nn],None,None,xy_new_range,None,None,)
        self.mtime = util.xy_regrid(self.mtime[nn],self.xy[nn,...],xy_new,
                               pre_calcs=pre_calcs_mtime,kind=kind)
        self.mtime = util.interp_nans_1d(self.mtime)

        # update dependent variables
        xy = np.copy(self.xy)
        z = np.copy(self.bin_center_elevation)
        self.n_ensembles,self.n_bins = np.shape(self.velocity[:,:,0])
        self.bin_center_elevation = z_new
        self.xy = xy_new    
        self.xy_to_lonlat()
        self.history_append("xy_regrid(dxy=%f,dz=%f,xy_srs=%s,pline=%s)"%(dxy,dz,
                                                                          xy_srs,
                                                                          pline is not None))
        # return vars such that sub-classes with more xy dimension variables can regrid
        return (xy, xy_new, z, z_new, nn, pre_calcs)

    def split_by_ensemble(self,split_nums):
        """
        Given a list of ensemble indices, splits self into len(split_nums)+1
        ADCP_Data objects.
        Inputs:
            split_nums = Python list of ensemble numbers (indices)
        """
        n_splits = len(split_nums)
        sub_adcps = list()        
        for i in range(n_splits+1):
            # find bounding ensembles - extremely non-pythonic
            if i==0:
                l_bound = 0
            else:
                l_bound = max(0,split_nums[i-1]-1)
            if i==n_splits:
                u_bound = -1
            else:
                u_bound = min(self.n_ensembles-1,split_nums[i])
            # split data
            a = self.self_copy()
            a.velocity = a.velocity[l_bound:u_bound,...]
            if a.mtime is not None:
                a.mtime = a.mtime[l_bound:u_bound]
            if a.lonlat is not None:
                a.lonlat = a.lonlat[l_bound:u_bound,...]
            if a.xy is not None:
                a.xy = a.xy[l_bound:u_bound,...]
            a.n_ensembles = np.shape(a.velocity)[0]
            a.history_append('split_by_ensemble: %i:$i'%(l_bound,ubound))
            sub_adcps.append(a)
        return sub_adcps


class ADCP_Transect_Data(ADCP_Data):
    """ 
    Subclass of :py:class:ADCP_Data for transect-based ADCP surveys, 
    for when the ADCP intrument is moving.
    """

    adcp_depth = None # [n_bins] - depth of transducer under water 
    bt_velocity = None # [n_bins] - depth of transducer under water
    bt_depth = None    # [n_ensembles] - bottom depth/elevation from transducer face
 
    def write_nc_extra(self,grp,zlib=None):
        """ 
        Extra transect-specific data is written to a NetCDF output 
        file.
        Inputs:
            grp = Python NetCDF object
            zlib = if True, use variable compression
        """ 
        super(ADCP_Transect_Data,self).write_nc_extra(grp,zlib)
        if self.adcp_depth is not None:
            adcp_depth_var = grp.createVariable('adcp_depth','f8',
                                                   (self.nc_ensemble_dim),
                                                    zlib=zlib)
            adcp_depth_var.units = 'm'
            adcp_depth_var[...] = self.adcp_depth
        if self.bt_depth is not None:
            bt_depth_var = grp.createVariable('bt_depth','f8',
                                                   (self.nc_ensemble_dim),
                                                    zlib=zlib)
            bt_depth_var.units = 'm'
            bt_depth_var[...] = self.bt_depth

        if self.bt_velocity is not None:
            bt_velocity_var = grp.createVariable('bt_velocity','f8',
                                                   (self.nc_ensemble_dim,
                                                    'component2'),
                                                    zlib=zlib)
            bt_velocity_var.units = 'm/s'
            bt_velocity_var[...] = self.bt_velocity[:,:2]

    def read_nc_extra(self,grp):           
        """ 
        Extra transect-specific data is read from NetCDF output file.
        Inputs:
            grp = Python NetCDF object
        """ 
        super(ADCP_Transect_Data,self).read_nc_extra(grp)
        if 'adcp_depth' in grp.variables:
            self.adcp_depth = np.array(grp.variables['adcp_depth'][...])
        if 'bt_velocity' in grp.variables:
            self.bt_velocity = grp.variables['bt_velocity'][...]
        if 'bt_depth' in grp.variables:
            # need double array here so future transposes work
            self.bt_depth = np.array([grp.variables['bt_depth'][...]])
            
    def copy_headCorrect_vars(self,xy_srs=None):
        """
        Returns the raw data required to perform a headCorrection of a moving
        ADCP platform (i.e. a boat)
        Inputs:
            xy_srs = EPSG code [str] if projection to xy is desired, or None            
        """       
        if self.xy is None:
            self.lonlat_to_xy(xy_srs=xy_srs)    
        return (np.copy(self.mtime),                          # times
                np.copy(self.heading),                        # compass headings
                np.copy(self.bt_velocity),                    # bottom track velocities
                np.copy(self.xy))                             # xy positions
    
    def heading_correct(self,cf=None,u_min_bt=None,hdg_bin_size=None,
                        hdg_bin_min_samples=None,mag_dec=None):
        """
        Makes corrections to ADCP velocities based upon the heading of a moving
        ADCP platform (i.e. a boat)
        Inputs:
            cf = harmoic fit composed of the [scalar, sine, cosine] components, or None
            u_min_bt = minimum bottom track velocity - compass must be moving {m/s}, scalar
            hdg_bin_size = size of the correction bins in degrees, scalar
            hdg_bin_min_samples = minimum valid compass headings for a corection to bin
            mag_dec = magnetic declination, in degrees, or None
        """
        if self.heading is None:
            print "Error in heading_correct: self.heading must be assigned"
            print "Heading correcting not performed"            
            return 
        delta = util.find_headCorrect(self.heading,            # delta is in degrees  
                                    cf=cf,
                                    u_min_bt=u_min_bt,
                                    hdg_bin_size=hdg_bin_size,
                                    hdg_bin_min_samples=hdg_bin_min_samples,
                                    mag_dec=mag_dec,
                                    mtime_in=self.mtime,
                                    bt_vel_in=self.bt_velocity,
                                    xy_in=self.xy)
        (self.velocity[:,:,0], 
        self.velocity[:,:,1]) = self.rotate_velocities(delta*np.pi/180.0,0,1)
        if self.bt_velocity is not None:
            (self.bt_velocity[:,0], 
            self.bt_velocity[:,1]) = self.rotate_bt_velocities(delta*np.pi/180.0)
        self.heading = self.heading + delta
        head_str = "heading_correct(hdg_bin_size=%f,"%hdg_bin_size + \
                   "hdg_bin_min_samples=%i)"%hdg_bin_min_samples
        self.history_append(head_str)


    def rotate_bt_velocities(self,radian):
        """
        Re-orient bottom track velocities to an arbitrary rotation, without 
        self assignment.
        Inputs:
            radian = rotation in radians, either 1D numpy array or scalar
        Returns:
            
        """    
        if self.bt_velocity is not None:
            btU,btV = util.rotate_velocity(radian,
                                         -self.bt_velocity[:,0],
                                         -self.bt_velocity[:,1])
            return (-btU,-btV)
        else:
            return (0.0,0.0)        

    def set_rotation(self,radian,axes_string='UV'):
        """
        Re-orient designated velocities to an arbitrary rotation.
            Python list of 2 velocity numpy arrays.
        Inputs:
            radian = rotation in radians
            axes_string = 2-character string containing 'U','V', or 'W', 
              indicating which velocity axes to rotate, with the first
              being in the 0-degree direction and the second the 90-degree
              direction
        """
        old_rotation = self.rotation_angle
        super(ADCP_Transect_Data,self).set_totation(radian)
        if axes_string != 'UV':
            print "WARNING: bottom track velocity rotation not supported for axes: ",axes_string
        elif self.bt_velocity is not None:
            if old_rotation is not None:
                (self.bt_velocity[:,0], 
                 self.bt_velocity[:,1]) = self.rotate_bt_velocities(-1.0*old_rotation)           
            if self.rotation_angle is not None:
                (self.bt_velocity[:,0], 
                 self.bt_velocity[:,1]) = self.rotate_bt_velocities(self.rotation_angle)           
        

    def get_velocity_mask(self,elev_line=None,range_from_velocities=False,mask_region='above'):
        """ 
        Generates a either a boolean mask, or a 1/NaN mask, correspnding to
        valid velocties measurements. If elev_line is given values beyond
        this elevation are masked as invalid.
        Inputs:
            elev_line = optional scalar or array elevation (distance from 
              transducer) neyond which velocity is invalid
            range_from_velocities = if True, calculates the elev_line
              from the range of valid (non-NaN) velocities in bins.
            mask_region = 'above' 
        Returns:
            2D numpy array, shape [self.n_ensembles,3], with mean U,V,W 
              velocities
        """
        my_elev_line = np.copy(elev_line)
        if self.bt_depth is not None and elev_line is not None:
            my_elev_line = self.bt_depth
        return super(ADCP_Transect_Data,self).get_velocity_mask(my_elev_line,range_from_velocities)


    def calc_crossproduct_flow(self):
        """
        Calculates the discharge(flow) by finding the cross product of the water
        and bottom track velocities.  
        Returns:
            mean U and V velocity, U and V total flow, and survey area
        """
        vU = np.copy(self.velocity[:,:,0])
        vV = np.copy(self.velocity[:,:,1])
        btU =  np.copy(self.bt_velocity[:,0])
        btV =  np.copy(self.bt_velocity[:,1])
        return util.calc_crossproduct_flow(vU,vV,btU,btV,
                                         -self.bin_center_elevation,
                                         -self.bt_depth,
                                         self.mtime)


    def xy_regrid(self,dxy,dz,xy_srs=None,pline=None,sort=False,kind='bin average'):
        """ 
        Projects ensemble locations to a strain line in the xy-plane, and
        then regrids velocities onto a regular grid defined by dxy and dz.
        This process changes the the dimensions of almost every piece of data in
        the class. self.mtime is handled differently since it does not make sense
        to average or interpolate time based on location.  Returns intermediate
        calculations to facilitate regridding of additional data by subclasses.
        Inputs:
            dxy = new grid xy resolution in xy projection units
            dz = new grid z resolution in z units
            xy_srs = EPSG code [str]        
            pline = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]], or None
            sort = if True, pre-sort the data being regrided in terms of location
               on the projection line
             kind = one of ['bin_average', linear','cubic','nearest'], where 
               the later three are types of numpy interpolation
        Returns:
            xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
            xy_new = new grid x-y locations, 2D array of shape [ne2,2], where [:,0] is x and [:,1] is y
            z = z positions, 1D array of shape [nb]
            z_new = z positions of new grid, 1D array of shape [nb2]
            nn = sort order of xy locations on the projection line
            pre_calcs = python list of different intermediate things - see 
              ADCPy_utilities.py
        """        
        # call base method to start regridding of base data, and get 
        # new grid info    
        (xy, xy_new, z, z_new, nn, pre_calcs) = \
        super(ADCP_Transect_Data,self).xy_regrid(dxy,dz,xy_srs,pline,sort,kind)
    
        # pre-sort data if needed, for speed
        if sort:
            if np.size(self.adcp_depth) > 1:
                adcp_depth_interp = self.adcp_depth[nn]
            if self.bt_depth is not None:
                bt_depth_interp = np.squeeze(self.bt_depth)[nn]
            if self.heading is not None:
                heading_interp = self.heading[nn]
            if self.bt_velocity is not None:
                bt_velocity_interp = self.bt_velocity[nn,:]
        else:
            adcp_depth_interp = self.adcp_depth
            bt_depth_interp = np.squeeze(self.bt_depth)
            heading_interp = self.heading
            bt_velocity_interp = self.bt_velocity
            
        # regrid xy-based transect variables
        if np.size(self.adcp_depth) > 1:
            print 'adcp_depth',self.adcp_depth
            self.adcp_depth = util.xy_regrid(adcp_depth_interp,xy,xy_new,
                                           pre_calcs=pre_calcs,kind=kind)
        if self.bt_depth is not None:            
            self.bt_depth = util.xy_regrid(bt_depth_interp,xy,xy_new,
                                           pre_calcs=pre_calcs,kind=kind)

            self.bt_depth = np.array([self.bt_depth])
            mask = util.find_mask_from_vector(self.bin_center_elevation,
                                            self.bt_depth,
                                            mask_area = 'below')
            for i in range(3):
                self.velocity[:,:,i][mask] = np.nan
        if self.heading is not None:
            self.heading = util.xy_regrid(heading_interp,xy,xy_new,
                                           pre_calcs=pre_calcs,kind=kind)
        if self.bt_velocity is not None:
            self.bt_velocity = util.xy_regrid_multiple(bt_velocity_interp,xy,xy_new,
                                           pre_calcs=pre_calcs,kind=kind)
                           
 
class ADCP_Moored_Data(ADCP_Data):
    """ 
    Subclass of :py:class:ADCP_Data for moored ADCP surveys.
    """
   
    adcp_depth = None # [n_bins] - depth of transducer under water 
 
    def __init__(self,file_path,**kwargs):
        super(ADCP_Moored_Data,self).__init__(file_path=file_path,**kwargs)

    def write_nc_extra(self,grp,zlib=None):
        super(ADCP_Moored_Data,self).write_nc_extra(grp)       

    def read_nc_extra(self,grp):           
        super(ADCP_Moored_Data,self).read_nc_extra(grp)

