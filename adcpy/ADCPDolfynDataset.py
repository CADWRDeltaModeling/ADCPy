# -*- coding: utf-8 -*-
"""ADCPTransectData subclass for the Dolfyn datasets with transect support

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""

import numpy as np
import xarray as xr
import re,os
import io
from . import adcpy_utilities as au
#import scipy.stats.stats as sp
import scipy.stats.morestats as ssm
from . import adcpy
from datetime import datetime
from matplotlib.dates import date2num

from dolfyn.io.rdi import read_rdi #(filename, userdata=None, nens=None, debug_level=-1, vmdas_search=False, winriver=False, **kwargs)
from dolfyn.io.nortek2 import read_signature #(filename, userdata=True, nens=None, rebuild_index=False, debug=False, dual_profile=False, **kwargs)

class ADCPDolfynDataset(adcpy.ADCPTransectData):
    """
    Subclass of :py:class:AdcpTransectData for importing data from a Dolfyn-read xarray dataset
    """
    error_vel = None # [n_ensembles, n_bins]
    heading = None # [n_ensembles] or None if no nav data available (degrees from North)
    nens = None
    adcp_depth = 0.0
    DataSet = None

    #: Reference to Station object - necessary for some transformations where
    #  a cross-channel direction is needed
    station = None
    kwarg_options = ['nens',          # integer number of ensembles to read
                     'adcp_depth']    # tow depth of ADCP below surface [m]

    def read_raw(self,raw_file=None,**kwargs):
        """ 
        raw_file: path to a XXXXr.nnn raw file from an RDI ADCP
        nav_file: path to NMEA output which matches raw_file. If 'auto',
        then look for a nav file based on the raw file.  if None, no
        handling of n files.

        num_av: average this many samples together
        nens: if None, read all ensembles, 
              if [start,stop] read those ensembles (ending index is inclusive)
              if N, read the first N ensembles
        """        
        # set some defaults
        self.nav_file=None  # support external nav processing in future?

        for k, v in kwargs.items(): #kwarg_options:
            if k in self.kwarg_options:
                setattr(self,k,v)

        if not os.path.exists(raw_file):
            raise IOError("Cannot find %s"%raw_file)
    
        self.raw_file = raw_file
        if self.raw_file is not None:
            if self.raw_file.endswith('ad2cp'):
                self.Dataset = read_signature(self.raw_file, userdata=True, nens=self.nens, rebuild_index=False, debug=False, dual_profile=False)
            else:
                # assume it is an rdi file and try to read it
                self.Dataset = read_rdi(self.raw_file, userdata=None, nens=self.nens, debug_level=-1, vmdas_search=False, winriver=False)
        # test for return of empty ds??
        elif not isinstance(self.Dataset,xr.Dataset):
            raise ValueError('If no raw_file is specified, ')

        self.valid = self.read_from_ds(self.Dataset)


    def read_from_ds(self,ds):
        """ read into memory the ADCP data
        if no data is found, return false, otherwise return true
        """

        # Get the dimensions that were actually returned
        self.n_bins = ds.sizes['range']
        self.n_ensembles = ds.sizes['time']

        vel_transpose = ds.vel.values.transpose([2,1,0])
        self.velocity = vel_transpose[...,:3]
        self.error_vel = vel_transpose[...,3]

        # invert w velocity - is there a way to determine orientation from config?
        self.velocity[:,:,2] = self.velocity[:,:,2] * -1.0

        if 'vel_bt' in ds.variables:
            self.bt_velocity = ds.vel_bt.values.transpose((1,0))
            # problems w/ big spikes in bt -> not sure why
            # for j in range(0,2):
            #     bt_vel = self.bt_velocity[:,j]
            #     ii = np.greater(abs(bt_vel),5.0) # identify where velocity is > 5 m/s
            #     bt_vel[ii] = np.nan
            #     bt_vel = au.interp_nans_1d(bt_vel) # interpolate over nans
            #     self.bt_velocity[:,j] = bt_vel

            # we are not using the 4th bt_vel - maybe that is an error vel too we should use?
            self.bt_velocity = self.bt_velocity[...,:3]

        if 'dist_bt' in ds.variables:
            self.bt_depth = -1.0*np.array([np.mean(ds.dist_bt.values,1)])
            
        if 'depth' in ds.variables:
            self.adcp_depth = ds.depth.values
        elif self.adcp_depth is not None:
            self.adcp_depth = np.ones(self.n_ensembles)*self.adcp_depth

        self.mtime = date2num(ds['time'].values)
        

        # TODO
        #if self.raw_adcp.longitude is not None and self.raw_adcp.latitude is not None:
        #    self.lonlat = np.array( [self.raw_adcp.longitude,
        #                             self.raw_adcp.latitude] ).T[:Ne]

        if 'heading' in ds.variables:
            self.heading = ds.heading.values
                                     
        # Extract info about bins - but convert to a z=up, surface=0
        # coordinate system
        self.bin_center_elevation = -1*ds.range.values

        if self.bt_velocity is not None:

            # remove boat motion from water vel
            vbins = np.shape(self.bin_center_elevation)
            btE = np.copy(self.bt_velocity[:,0])
            btN = np.copy(self.bt_velocity[:,1])
            btW = np.copy(self.bt_velocity[:,2])
            vE = np.copy(self.velocity[:,:,0]) - np.ones(vbins)*np.array([btE]).T
            vN = np.copy(self.velocity[:,:,1]) - np.ones(vbins)*np.array([btN]).T
            vW = np.copy(self.velocity[:,:,2]) - np.ones(vbins)*np.array([btW]).T

            # rotate velocities from ship coordinates
            if ds.imag.coord_sys == 'ship':

                 # convert ship coord to enu
                delta = self.heading*np.pi/180
                delta2D = np.ones(vbins)*np.array([delta]).T # array of headings
                self.velocity[:,:,0] = np.cos(delta2D)*vE + np.sin(delta2D)*vN
                self.velocity[:,:,1] = -np.sin(delta2D)*vE + np.cos(delta2D)*vN
                self.bt_velocity[:,0] = np.cos(delta)*btE + np.sin(delta)*btN
                self.bt_velocity[:,1] = -np.sin(delta)*btE + np.cos(delta)*btN

            else:

                self.velocity[:,:,0] = vE
                self.velocity[:,:,1] = vN

            self.velocity[:,:,2] = vW

# -- previous method of rotation, before correction from Dave Ralton 4/26/2013
#            vbins = np.shape(self.bin_center_elevation)
#            delta = np.array([self.heading]).T # transpose to vertical
#            delta = np.ones(vbins)*delta*np.pi/180 # array of headings
#            vE = np.cos(delta)*self.velocity[:,:,0] + np.sin(delta)*self.velocity[:,:,1]
#            vN = -np.sin(delta)*self.velocity[:,:,0] + np.cos(delta)*self.velocity[:,:,1]
#            
#            delta = self.heading*np.pi/180
#            btE = np.cos(delta)*self.bt_velocity[:,0] + np.sin(delta)*self.bt_velocity[:,1]
#            btN = -np.sin(delta)*self.bt_velocity[:,0] + np.cos(delta)*self.bt_velocity[:,1]
#          
#            # remove boat motion from water vel
#            vN=vN-np.ones(vbins)*np.array([btN]).T
#            vE=vE-np.ones(vbins)*np.array([btE]).T
# 
#            # restore corrected velocities
#            self.velocity[:,:,0] = vE
#            self.velocity[:,:,1] = vN
#            self.bt_velocity[:,0] = btE
#            self.bt_velocity[:,1] = btN
      
        #lat=adcp.nav_latitude(nn);
        #lon=adcp.nav_longitude(nn);
        #% fix bad lon/lats
        #ii=abs(lon-mean(lonlat00(:,1)))>5;
        #lon(ii)=NaN;
        #ii=abs(lat-mean(lonlat00(:,2)))>5;
        #lat(ii)=NaN;
        #% if lat/lon not recorded every ping, fill in blanks
        #lat=interpnan(yd,lat)';
        #lon=interpnan(yd,lon)';
        
        read_raw_history = "Constructor DolfynDataset: Raw RDI file: %s ens=%s"%(self.raw_file,self.n_ensembles)

        self.history_append(read_raw_history)

        path,fname = os.path.split(self.raw_file)
        self.source = fname
        
        return True

    def split_by_ensemble(self,split_nums,extra_fields=[]):
        return super(ADCPDolfynDataset,self).split_by_ensemble(split_nums,
            extra_fields=extra_fields+['heading','error_vel'])

    def write_nc_extra(self,grp,zlib=None):
        super(ADCPDolfynDataset,self).write_nc_extra(grp,zlib)

        if self.error_vel is not None:
            (e_ens,e_bins) = np.shape(self.error_vel)
            grp.createDimension('error_bin',e_bins)
            grp.createDimension('error_ens',e_ens)
            error_vel_var = grp.createVariable('error_vel','f8',
                                               ('error_ens','error_bin'),
                                               zlib=zlib)
            error_vel_var.units = 'm/s'
            error_vel_var[...] = self.error_vel

        if self.heading is not None:
            heading_var = grp.createVariable('heading','f8',
                                                 self.nc_ensemble_dim,
                                                 zlib=zlib)
            heading_var.units = 'degrees'
            heading_var[...] = self.heading
        
        if 'raw_adcp' in self.__dict__:

            raw_adcp_grp = grp.createGroup('raw_adcp')
            config = raw_adcp_grp.createGroup('config')
            for k in self.raw_adcp.config.__dict__:
                v = self.raw_adcp.config.__dict__[k]
                try:
                    setattr(config,k,v)
                except Exception as ex:
                    print("Skipping config attribute %s"%k)
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)

            (raw_n_ens,raw_n_bins) = np.shape(self.raw_adcp.bin_data)
            raw_n_ens_dim = raw_adcp_grp.createDimension('raw_n_ensembles',raw_n_ens)
            raw_n_bins_dim = raw_adcp_grp.createDimension('raw_n_bins',raw_n_bins)
            
            ens_data_nc_dtype = raw_adcp_grp.createCompoundType(self.raw_adcp.ensemble_data.dtype,'ens_dtype')
            ens_data_var = raw_adcp_grp.createVariable('ensemble_data',
                                                       ens_data_nc_dtype,
                                                       'raw_n_ensembles',
                                                       zlib=zlib)
            ens_data_var[...] = self.raw_adcp.ensemble_data
            
            bin_data_nc_dtype = raw_adcp_grp.createCompoundType(self.raw_adcp.bin_data.dtype,'bin_dtype')
            bin_data_var = raw_adcp_grp.createVariable('bin_data',
                                                       bin_data_nc_dtype,
                                                       ('raw_n_ensembles','raw_n_bins'),
                                                       zlib=zlib)
            bin_data_var[...] = self.raw_adcp.bin_data
            

    def read_nc_extra(self,grp):           
        super(ADCPDolfynDataset,self).read_nc_extra(grp)

        print('Doing read_nc in ADCPDolfynDataset...')

        # read optional base variables
        if 'error_vel' in grp.variables:
            self.error_vel = grp.variables['error_vel'][...]
        if 'heading' in grp.variables:
            self.heading = grp.variables['heading'][...]
        
        if 'raw_adcp' in grp.groups:
            self.raw_adcp = rdradcp.Adcp()
            raw_grp = grp.groups['raw_adcp']
            if 'config' in raw_grp.groups:
                self.raw_adcp.config = rdradcp.Config()
                cfg = raw_grp.groups['config']
                for k in raw_grp.groups['config'].__dict__:
                    exec("self.raw_adcp.config.%s = cfg.%s"%(k,k))

            if 'ensemble_data' in raw_grp.variables:
                self.raw_adcp.ensemble_data = raw_grp.variables['ensemble_data'][...]
        
            if 'bin_data' in raw_grp.variables:
                self.raw_adcp.bin_data = raw_grp.variables['bin_data'][...]


    def append_ensembles_extra(self,a):
        super(ADCPDolfynDataset,self).append_ensembles_extra(a)
        if self.heading is not None:
            self.heading = au.concatenate_array_w_fill(self.heading,
                                                  (self.n_ensembles,),
                                                  a.heading,
                                                  (a.n_ensembles,))
        if self.error_vel is not None:
            self.error_vel = au.concatenate_array_w_fill(self.error_vel,
                                                  (self.n_ensembles,self.n_bins),
                                                  a.error_vel,
                                                  (a.n_ensembles,a.n_bins))

    def average_ensembles(self,ens_to_avg):
        """ Extra variables must be averaged for this subclass
        """
        a = super(ADCPDolfynDataset,self).average_ensembles(ens_to_avg)
        n2 = a.n_ensembles
        nn = range(n2*ens_to_avg)
        if a.heading is not None:
            head = a.heading[nn].reshape(n2,ens_to_avg)
            a.heading = np.zeros(n2,np.float64)
            for i in range(n2):
                a.heading[i] = ssm.circmean(head[i,:]*np.pi/180)*180/np.pi
        if a.bt_depth is not None:
            a.bt_depth = au.average_vector(self.bt_depth[0,nn],(n2,ens_to_avg))
            a.bt_depth = np.array([a.bt_depth]) # reformat into downward vector
        if a.adcp_depth is not None:
            a.adcp_depth = au.average_vector(self.adcp_depth[nn],(n2,ens_to_avg))
        if a.bt_velocity is not None:                
            a.bt_velocity = np.zeros((n2,2),np.float64)
            for i in range(2):
                a.bt_velocity[:,i] = au.average_array(self.bt_velocity[nn,i],(n2,ens_to_avg),axis=0)  
                a.bt_velocity = au.average_array(self.bt_velocity[nn,:],(n2,ens_to_avg),axis=0)
        if a.error_vel is not None:
            a.error_vel = au.average_array(self.error_vel[nn,:],(n2,ens_to_avg),axis=0)
    
        return a
         
    def xy_regrid(self,dxy,dz,xy_srs=None,pline=None,sort=False,kind='bin average',
                  sd_drop=0,mtime_regrid=False,sd_drop_alt=0,nonlinear=False):

        (xy, xy_new, z, z_new, nn, pre_calcs) = \
        super(ADCPDolfynDataset,self).xy_regrid(dxy, dz, xy_srs = xy_srs, pline = pline, sort = sort,
                    kind = kind, sd_drop = sd_drop, mtime_regrid = mtime_regrid, nonlinear = nonlinear)

        # seems like RDI-specifc vars like bottom track stuff should go here and not in transect class?

        if self.error_vel is not None:
            error_vel_interp = self.error_vel
            self.error_vel = au.xy_regrid(error_vel_interp,xy,xy_new,
                                           pre_calcs=pre_calcs,kind=kind, sd_drop=sd_drop)
        return (xy, xy_new, z, z_new, nn, pre_calcs)


    def t_regrid(self,dt,dz,sd_drop=0,sd_drop_alt=0):

        (t, t_new, z, z_new, dummy, pre_calcs) = super(ADCPDolfynDataset,self).t_regrid(dt,dz,sd_drop,sd_drop_alt)

        if self.error_vel is not None:
            error_vel_interp = self.error_vel
            self.error_vel = au.xy_regrid(error_vel_interp, t ,t_new, z, z_new,pre_calcs,
                                            kind='bin average', sd_drop=sd_drop)

            # return vars such that sub-classes with more xy dimension variables can regrid
            return (t, t_new, z, z_new, None, pre_calcs)


    def split_by_ensemble(self, split_nums, extra_fields=[]):
        sub_adcps = super(ADCPDolfynDataset, self).split_by_ensemble(split_nums, extra_fields=extra_fields + ['error_vel', 'heading'])
        return sub_adcps

    def crop(self, l_bound, u_bound, extra_fields=[], axis='ensemble'):
        a = super(ADCPDolfynDataset, self).crop(l_bound, u_bound,
                                               extra_fields=extra_fields + ['error_vel', 'heading'],
                                               axis=axis)
        return a


