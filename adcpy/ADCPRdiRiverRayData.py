import os
import numpy as np
import rdi_pd0

import adcpy_utilities as au
import adcpy

class ADCPRdiRiverRayData(adcpy.ADCPTransectData):
    """
    Subclass of :py:class:AdcpData for reading PD0 RiverRay files.
    """
    error_vel = None # [n_ensembles, n_bins]
    heading = None # [n_ensembles] or None if no nav data available (degrees from North)
    
    #: Reference to Station object - necessary for some transformations where
    #  a cross-channel direction is needed
    station = None

    def read_raw(self,raw_file):
        """ 
        raw_file: path to a XXXX.PD0 raw file from an RDI ADCP, assumed to be 
        a Riverray.
        """        
        if not os.path.exists(raw_file):
            raise IOError("Cannot find %s"%raw_file)
    
        self.raw_file = raw_file

        self.valid = self.read_raw_data()

    def read_nav(self):
        # TODO: copy parsed data back into this form
        self.ensemble_gps_indexes = [] # [ [ensemble #, index into gps_data], ... ]
        self.gps_data = [] # [ [ day_fraction, lat, lon], ... ]

    def read_raw_data(self):
        """ read into memory the ADCP data
        if no data is found, return false, otherwise return true
        """
        self.raw_adcp = rdi_pd0.parse(self.raw_file,cls=rdi_pd0.RiverrayFrame)
        if self.raw_adcp is None:
            return False
        ens=self.raw_adcp['data'] # ensemble data array
        
        # Get the dimensions that were actually returned
        self.n_bins = self.raw_adcp['Ncells']
        Ne = self.n_ensembles = self.raw_adcp['Nensembles']

        # velocity: [Ntimes,Nbins,{u,v,w}]
        self.velocity = ens['velocity'][:,:,:3] # omit error vel.

        # This is in the workhorse code - need to check more closely to see 
        # if it's necessary for RiverRay.
        # BEN?
        # invert w velocity - is there a way to determine orientation from config?
        # self.velocity[:,:,2] *= -1

        fields=ens.dtype.names

        if 'bt_vel' in fields:
            self.bt_velocity = ens['bt_vel']
            # skipping the spike removal until deemed necessary

        if 'vertical_range' in fields:
            depth=-ens['vertical_range']
            # sometimes it's all nan
            if np.any( np.isfinite(depth) ):
                self.vertical_depth = depth
        if 'bt_range' in fields:
            self.bt_depth = np.array([-ens['bt_range'].mean(axis=1)])            
            
        if 'transducer_depth' in fields:
            self.adcp_depth = ens['transducer_depth']
        else:
            self.adcp_depth = np.zeros(Ne)
        
        # [Ntimes,Nbins]
        self.error_vel = ens['velocity'][:,:,3]
        self.mtime = ens['dn'] # python datenum
        
        if 'gps_index' in fields:
            lat = self.raw_adcp['gps_data'][ ens['gps_index'] ]['lat']
            lon = self.raw_adcp['gps_data'][ ens['gps_index'] ]['lon']
            self.lonlat = np.array( [lon,lat] ).T

        if 'heading' in fields:
            self.heading = ens['heading'][:Ne]
                                     
        # Extract info about bins - but convert to a z=up, surface=0
        # coordinate system
        self.bin_center_elevation = -self.raw_adcp['bin_distances']
        
        if self.n_ensembles < 1:
            self.msg("Dropping empty last ensemble lead to empty ADCP")
            return False
            
        # ben adding processing specific to adcpRaw_rdr.m

        # HERE: seems like this should be done conditionally based on
        # the coordinate system 
        # currently in one of the test files coord_transform is BEAM.
        
        # remove boat motion from water vel
        vbins = np.shape(self.bin_center_elevation)
        btE = np.copy(self.bt_velocity[:,0])
        btN = np.copy(self.bt_velocity[:,1])
        btW = np.copy(self.bt_velocity[:,2])
        vE = np.copy(self.velocity[:,:,0]) - np.ones(vbins)*np.array([btE]).T
        vN = np.copy(self.velocity[:,:,1]) - np.ones(vbins)*np.array([btN]).T
        vW = np.copy(self.velocity[:,:,2]) - np.ones(vbins)*np.array([btW]).T
        
        # rotate velocities from ship coordibates
        coords=self.raw_adcp['coord_sys']
        if coords is 'ship':
            # convert ship coord to enu
            delta = self.heading*np.pi/180
            delta2D = np.ones(vbins)*np.array([delta]).T # array of headings
            self.velocity[:,:,0] = np.cos(delta2D)*vE + np.sin(delta2D)*vN
            self.velocity[:,:,1] = -np.sin(delta2D)*vE + np.cos(delta2D)*vN
            self.velocity[:,:,2] = vW

            self.bt_velocity[:,0] = np.cos(delta)*btE + np.sin(delta)*btN
            self.bt_velocity[:,1] = -np.sin(delta)*btE + np.cos(delta)*btN
        elif coords is 'beam':
            print "WARNING: looks like beam coordinates, not ready for that!"
            print "  will punt and pretend it's east north up. "
            self.velocity[:,:,0] = vE
            self.velocity[:,:,1] = vN
            self.velocity[:,:,2] = vW
        else:
            self.velocity[:,:,0] = vE
            self.velocity[:,:,1] = vN
            self.velocity[:,:,2] = vW

        read_raw_history = "Constructor ADCPRdiRiverRayData: Raw RDI file: %s"%(self.raw_file)

        self.history_append(read_raw_history)

        path,fname = os.path.split(self.raw_file)
        self.source = fname
        
        return True

    def write_nc_extra(self,grp,zlib=None):
        super(ADCPRdiRiverRayData,self).write_nc_extra(grp,zlib)

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
        
#        if 'raw_adcp' in self.__dict__:
#
#            raw_adcp_grp = grp.createGroup('raw_adcp')
#            config = raw_adcp_grp.createGroup('config')
#            for k in self.raw_adcp.config.__dict__:
#                v = self.raw_adcp.config.__dict__[k]
#                try:
#                    setattr(config,k,v)
#                except Exception,exc:
#                    print exc
#                    print "Skipping config attribute %s"%k
#            
#            (raw_n_ens,raw_n_bins) = np.shape(self.raw_adcp.bin_data)
#            raw_n_ens_dim = raw_adcp_grp.createDimension('raw_n_ensembles',raw_n_ens)
#            raw_n_bins_dim = raw_adcp_grp.createDimension('raw_n_bins',raw_n_bins)
#            
#            ens_data_nc_dtype = raw_adcp_grp.createCompoundType(self.raw_adcp.ensemble_data.dtype,'ens_dtype')
#            ens_data_var = raw_adcp_grp.createVariable('ensemble_data',
#                                                       ens_data_nc_dtype,
#                                                       'raw_n_ensembles',
#                                                       zlib=zlib)
#            ens_data_var[...] = self.raw_adcp.ensemble_data
#            
#            bin_data_nc_dtype = raw_adcp_grp.createCompoundType(self.raw_adcp.bin_data.dtype,'bin_dtype')
#            bin_data_var = raw_adcp_grp.createVariable('bin_data',
#                                                       bin_data_nc_dtype,
#                                                       ('raw_n_ensembles','raw_n_bins'),
#                                                       zlib=zlib)
#            bin_data_var[...] = self.raw_adcp.bin_data
            

    def read_nc_extra(self,grp):           
        super(ADCP_RdiWorkhorse_Data,self).read_nc_extra(grp)

        print 'Doing read_nc in ADCP_RdiWorkhorse_Data...'

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
                    print 'config var: ',k
                    exec("self.raw_adcp.config.%s = cfg.%s"%(k,k))

            if 'ensemble_data' in raw_grp.variables:
                self.raw_adcp.ensemble_data = raw_grp.variables['ensemble_data'][...]
        
            if 'bin_data' in raw_grp.variables:
                self.raw_adcp.bin_data = raw_grp.variables['bin_data'][...]


    def average_ensembles(self,ens_to_avg):
        """ Extra variables must be averaged for this subclass
        """
        a = super(ADCP_RdiWorkhorse_Data,self).average_ensembles(ens_to_avg)
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
         
         
 


