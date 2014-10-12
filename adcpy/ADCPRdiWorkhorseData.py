# -*- coding: utf-8 -*-
"""ADCPTransectData subclass for the RDI Workhorse ADCP in transect mode

Reads RDI raw files from the RDI Workhorse ADCP. Uses rdradcp.

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""

import numpy as np
import re,os
#import netCDF4
import rdradcp
reload(rdradcp)
import pynmea.streamer
import cStringIO
import adcpy_utilities as au
#import scipy.stats.stats as sp
import scipy.stats.morestats as ssm
import adcpy


class ADCPRdiWorkhorseData(adcpy.ADCPTransectData):
    """
    Subclass of :py:class:AdcpData for reading raw .rNNN RDI ADCP data files,
    and optionally accompanying navigational data in .nNNN files.
    """
    error_vel = None # [n_ensembles, n_bins]
    heading = None # [n_ensembles] or None if no nav data available (degrees from North)
    
    #: Reference to Station object - necessary for some transformations where
    #  a cross-channel direction is needed
    station = None

    # parameters passed to rdradcp
    baseyear = 2000
    despike = 'no'
    quiet = True
    
    kwarg_options = ['nav_file',      # file path of optional NMEA navigational file
                     'num_av',        # integer - may be used to average ensembles during reading
                     'nens',          # number or range of ensembles to read 
                     'adcp_depth']    # tow depth of ADCP below surface [m]

    def read_raw(self,raw_file,**kwargs):
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
        nav_file=None
        num_av=1
        nens=None
        adcp_depth=None
        for kwarg in self.kwarg_options:
            if kwarg in kwargs:
                exec("%s = kwargs[kwarg]"%kwarg)
    
        # set parameters passed to rdradcp
        self.rdradcp_num_av = num_av
        self.rdradcp_nens = nens
        self.rdradcp_adcp_depth = adcp_depth

        if not os.path.exists(raw_file):
            raise IOError, "Cannot find %s"%raw_file
    
        self.raw_file = raw_file

        if nav_file == 'auto':
            nav_file = None
            # NOTE: this will only find lowercase 'n' files:
            possible_nav_file = re.sub(r'r(\.\d+)$',r'n\1',raw_file)
            if possible_nav_file != raw_file and os.path.exists(possible_nav_file):
                nav_file = possible_nav_file
                if not self.quiet:
                    self.msg("found nav file %s"%nav_file)
            else:
                nav_file = None
    
        if nav_file and not os.path.exists(nav_file):
            raise IOError,"Cannot find %s"%nav_file

        self.nav_file = nav_file

        if self.nav_file:
            self.read_nav()
        self.valid = self.read_raw_data()

    def read_nav(self):
        """ Reads NMEA stream data, prepared for transforn to latlon and
        mtime data structures
        """
        fp = open(self.nav_file)
        
        nmea = pynmea.streamer.NMEAStream(fp)
        
        self.ensemble_gps_indexes = [] # [ensemble #, index into gps_data]
        self.gps_data = [] # [ day_fraction, lat, lon] 

        while 1:
            next_data = nmea.get_objects()
            if not next_data:
                break

            for sentence in next_data:
                try:
                    if sentence.sen_type == 'GPGGA': # a fix
                        if sentence.gps_qual < 1:
                            continue # not a valid fix.
                        lat_s = sentence.latitude
                        lon_s = sentence.longitude
                        try:
                            lat = int(lat_s[:2]) + float(lat_s[2:])/60.
                            lon = int(lon_s[:3]) + float(lon_s[3:])/60.
                        except ValueError:
                            # every once in a while the strings are corrupted
                            continue
                        if sentence.lat_direction == 'S':
                            lat *= -1
                        if sentence.lon_direction == 'W':
                            lon *= -1
    
                        hours=int(sentence.timestamp[:2])
                        minutes= int(sentence.timestamp[2:4])
                        seconds= float(sentence.timestamp[4:])
                        day_fraction = (hours + (minutes + (seconds/60.))/60.)/24.0
    
                        self.gps_data.append( [day_fraction,lat,lon] )
    
                    elif sentence.sen_type == 'RDENS':
                        # assume that this marker goes with the *next* NMEA location
                        # output.
                        self.ensemble_gps_indexes.append( [int(sentence.ensemble),
                                                           len(self.gps_data)] )
                except AttributeError,exc:
                    print "While parsing NMEA: "
                    print exc
                    print "Ignoring this NMEA sentence"
                    continue
                    
        self.ensemble_gps_indexes = np.array(self.ensemble_gps_indexes)
        self.gps_data = np.array(self.gps_data)

    def read_raw_data(self):
        """ read into memory the ADCP data
        if no data is found, return false, otherwise return true
        """
        if self.rdradcp_nens is None:
            nens = -1 # translate to the matlab-ish calling convention
        else:
            nens = self.rdradcp_nens

        if self.quiet: 
            log_fp = cStringIO.StringIO()
        else:
            log_fp = None
            
        self.raw_adcp = rdradcp.rdradcp(self.raw_file,num_av = self.rdradcp_num_av,nens=nens,
                                        baseyear=self.baseyear,despike=self.despike,
                                        log_fp = log_fp)
        if self.quiet: 
            self.rdradcp_log = log_fp.getvalue()
            log_fp.close()
            
        if self.raw_adcp is None:
            return False
            
        # Rusty need to handle case when there are no valid ensembles

        
        # Get the dimensions that were actually returned
        self.n_bins = self.raw_adcp.east_vel.shape[1]
        self.raw_adcp.n_bins = self.n_bins
        self.n_ensembles = self.raw_adcp.east_vel.shape[0]

        bin_fields = self.raw_adcp.bin_data.dtype.names
        ens_fields = self.raw_adcp.ensemble_data.dtype.names
        
        # Due to a bug inherited from rdradcp, the last ensemble is not
        # valid.  In the long run, this should be fixed in rdradcp.py
        self.n_ensembles -= 1
        Ne = self.n_ensembles
        
        # velocity: [Ntimes,Nbins,{u,v,w}]
        self.velocity = np.array( [self.raw_adcp.east_vel,
                                   self.raw_adcp.north_vel,
                                   self.raw_adcp.vert_vel] ).transpose([1,2,0])[:Ne]
        # invert w velocity - is there a way to determine orientation from config?
        self.velocity[:,:,2] = self.velocity[:,:,2] * -1.0

        if 'bt_vel' in ens_fields:
            self.bt_velocity = self.raw_adcp.bt_vel[:Ne]*1.0e-3 # mm/s -> m/s
            # problems w/ big spikes in bt -> not sure why
            for j in range(0,1):
                bt_vel = self.bt_velocity[:,j]
                ii = np.greater(bt_vel,5.0) # identify where depth is > 5 m/s
                bt_vel[ii] = np.nan
                bt_vel = au.interp_nans_1d(bt_vel) # interpolate over nans
                self.bt_velocity[:,j] = bt_vel   
           
        if 'bt_range' in ens_fields:
            self.bt_depth = -1.0*np.array([np.mean(self.raw_adcp.bt_range[:Ne],1)])
            
        if 'depth' in ens_fields:
            self.adcp_depth = self.raw_adcp.depth[:Ne]
        if self.rdradcp_adcp_depth is not None:
            self.adcp_depth = np.ones(Ne)*self.rdradcp_adcp_depth
        
        # [Ntimes,Nbins]
        self.error_vel = self.raw_adcp.error_vel[:Ne]
        self.mtime = self.raw_adcp.mtime[:Ne]
        
        if self.raw_adcp.longitude is not None and self.raw_adcp.latitude is not None:
            self.lonlat = np.array( [self.raw_adcp.longitude,
                                     self.raw_adcp.latitude] ).T[:Ne]

        if 'heading' in ens_fields:
            self.heading = self.raw_adcp.heading[:Ne]
                                     
        # Extract info about bins - but convert to a z=up, surface=0
        # coordinate system
        self.bin_center_elevation = -1*self.raw_adcp.config.ranges  
        
        if self.n_ensembles < 1:
            self.msg("Dropping empty last ensemble lead to empty ADCP")
            return False
            
        # ben adding processing specific to adcpRaw_rdr.m

        # remove boat motion from water vel
        vbins = np.shape(self.bin_center_elevation)
        btE = np.copy(self.bt_velocity[:,0])
        btN = np.copy(self.bt_velocity[:,1])
        btW = np.copy(self.bt_velocity[:,2])
        vE = np.copy(self.velocity[:,:,0]) - np.ones(vbins)*np.array([btE]).T
        vN = np.copy(self.velocity[:,:,1]) - np.ones(vbins)*np.array([btN]).T
        vW = np.copy(self.velocity[:,:,2]) - np.ones(vbins)*np.array([btW]).T
        
        # rotate velocities from ship coordibates
        if self.raw_adcp.config.coord_sys is 'ship':
                                    
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
        
        read_raw_history = "Constructor RdiWorkhorseAdcpData: Raw RDI file: %s nnum_av=%s ens=%s"%(self.raw_file,
                                                     self.rdradcp_num_av,
                                                     self.n_ensembles)

        self.history_append(read_raw_history)

        path,fname = os.path.split(self.raw_file)
        self.source = fname
        
        return True

    def write_nc_extra(self,grp,zlib=None):
        super(ADCPRdiWorkhorseData,self).write_nc_extra(grp,zlib)

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
                except Exception,exc:
                    print exc
                    print "Skipping config attribute %s"%k
            
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
        super(ADCPRdiWorkhorseData,self).read_nc_extra(grp)

        print 'Doing read_nc in ADCPRdiWorkhorseData...'

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


    def average_ensembles(self,ens_to_avg):
        """ Extra variables must be averaged for this subclass
        """
        a = super(ADCPRdiWorkhorseData,self).average_ensembles(ens_to_avg)
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
         
         
 


