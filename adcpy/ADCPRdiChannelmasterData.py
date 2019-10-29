"""
Still under heavy development.  Not ready for use."
"""
import numpy as np
import re,os
from . import adcpy_utilities as au
from . import adcpy
from . import rdi_pd0


class ADCPRdiChannelmasterData(adcpy.ADCPMooredData):
    """
    Subclass of :py:class:AdcpData for reading raw .pd0 RDI ADCP data files,
    and optionally accompanying navigational data in .nNNN files.
    """
    error_vel = None # [n_ensembles, n_bins]
    
    kwarg_options = ['nav_file',      # file path of optional NMEA navigational file
                     'num_av',        # integer - may be used to average ensembles during reading
                     'nens',          # number or range of ensembles to read 
                     'adcp_depth']    # tow depth of ADCP below surface [m]

    def read_raw(self,raw_file,**kwargs):
        """ 
        raw_file: path to a XXXX.nnn raw file from an RDI Channel Master ADCP
        """ 
        self.raw_file = raw_file
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
                except AttributeError as exc:
                    print("While parsing NMEA: ")
                    print(exc)
                    print("Ignoring this NMEA sentence")
                    continue
                    
        self.ensemble_gps_indexes = np.array(self.ensemble_gps_indexes)
        self.gps_data = np.array(self.gps_data)

    def read_raw_data(self):
        """ read into memory the ADCP data
        if no data is found, return false, otherwise return true
        """
        self.raw_adcp = rdi_pd0.parse(self.raw_file,cls=rdi_pd0.ChannelmasterFrame)
        if self.raw_adcp is None:
            return False
        ens=self.raw_adcp['data'] # ensemble data array

        # Get the dimensions that were actually returned
        self.n_bins = self.raw_adcp['Ncells']
        Ne = self.n_ensembles = self.raw_adcp['Nensembles']

        # velocity: [Ntimes,Nbins,{u,v,w}]
        self.velocity = ens['velocity'][:,:,:2] # omit error vel.

        fields=ens.dtype.names

        # HERE 
        if 'depth' in fields:
            self.adcp_depth = self.raw_adcp.depth[:Ne]
        elif 'transducer_depth' in fields:
            self.adcp_depth = ens['transducer_depth']
        else:
            self.adcp_depth = np.zeros(Ne)
        
        # [Ntimes,Nbins]
        #self.error_vel = ens['velocity'][:,:,2]
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

        vbins = np.shape(self.bin_center_elevation)
        vE = np.copy(self.velocity[:,:,0])
        vN = np.copy(self.velocity[:,:,1])
        
        # rotate velocities from ship coordibates
        coords=self.raw_adcp['coord_sys']
        if coords is 'ship':
            # convert ship coord to enu
            delta = self.heading*np.pi/180
            delta2D = np.ones(vbins)*np.array([delta]).T # array of headings
            self.velocity[:,:,0] = np.cos(delta2D)*vE + np.sin(delta2D)*vN
            self.velocity[:,:,1] = -np.sin(delta2D)*vE + np.cos(delta2D)*vN
        elif coords is 'beam':
            print("WARNING: looks like beam coordinates, not ready for that!")
            print("  will punt and pretend it's east north up. ")
            self.velocity[:,:,0] = vE
            self.velocity[:,:,1] = vN
        else:
            self.velocity[:,:,0] = vE
            self.velocity[:,:,1] = vN
                    
        read_raw_history = "Constructor ADCPRdiChannelmasterData: Raw RDI file: %s"%self.raw_file

        self.history_append(read_raw_history)

        path,fname = os.path.split(self.raw_file)
        self.source = fname
        
        return True

    def write_nc_extra(self,grp,zlib=None):
        super(ADCPRdiChannelmasterData,self).write_nc_extra(grp,zlib)

        if self.error_vel is not None:
            error_vel_var = grp.createVariable('error_vel','f8',
                                                   (self.nc_ensemble_dim,
                                                    'bin'),zlib=zlib)
            error_vel_var.units = 'm/s'
            error_vel_var[...] = self.error_vel

        if self.heading is not None:
            heading_var = grp.createVariable('heading','f8',
                                                 self.nc_ensemble_dim,
                                                 zlib=zlib)
            heading_var.units = 'degrees'
            heading_var[...] = self.heading
                        
        config = grp.createGroup('config')
        
        for k in self.raw_adcp.config.__dict__:
            v = self.raw_adcp.config.__dict__[k]
            try:
                setattr(config,k,v)
            except Exception as exc:
                print(exc)
                print("Skipping config attribute %s"%k)

        # And some ensemble data: 
        beam_dim = grp.createDimension('beam',self.raw_adcp.config.numbeams)
        raw_bin_dim = grp.createDimension('raw_bin',self.raw_adcp.n_bins)
        beam_var = grp.createVariable('beam','i8','beam')
        # number them like RDI does - 1-based
        beam_var[:] = 1 + np.arange(self.raw_adcp.config.numbeams)
        
        for k in self.raw_adcp.ensemble_data.dtype.names:
            # skip ones that are generic and are written by the superclass
            if k in ['nav_latitude','nav_mtime','nav_longitude',
                     'mtime','heading']:
                 continue
            # pull one element out to see what the dtype is
            data = self.raw_adcp.ensemble_data[k]
            element = data[0]
            dims = [self.nc_ensemble_dim]
            if element.shape: 
                if element.shape[0] == len(beam_var):
                    # assume that the dimension is beam
                    dims.append('beam')
            k_var = grp.createVariable(k,element.dtype,dims,zlib=zlib)
            # because of the rdradcp bug of an extra ensemble, explicitly
            # ask for the right length here
            k_var[:] = data[:self.n_ensembles]
        # bin data
        for k in self.raw_adcp.bin_data.dtype.names:
            # skip ones that are generic and are written by the superclass
            if k in ['east_vel','north_vel','vert_vel','error_vel']:
                 continue
            # pull one element out to see what the dtype is
            data = self.raw_adcp.bin_data[k]
            element = data[0,0]
            dims = [self.nc_ensemble_dim,'raw_bin']
            if element.shape: 
                if element.shape[0] == len(beam_var):
                    # assume that the dimension is beam
                    dims.append('beam')
            k_var = grp.createVariable(k,element.dtype,dims,zlib=zlib)
            # because of the rdradcp bug of an extra ensemble, explicitly
            # ask for the right length here
            k_var[:] = data[:self.n_ensembles]

    def read_nc_extra(self,grp):           
        super(ADCPRdiChannelmasterData,self).read_nc_extra(grp)

        # read optional base variables
        if 'error_vel' in grp.variables:
            self.error_vel = grp.variables['error_vel'][...]
        if 'heading' in grp.variables:
            self.heading = grp.variables['heading'][...]

    def average_ensembles(self,ens_to_avg):
        """ Extra variables must be averaged for this subclass
        """
        a = super(ADCPMooredData,self).average_ensembles(ens_to_avg)
        n2 = a.n_ensembles
        nn = list(range(n2*ens_to_avg))
        if a.heading is not None:
            head = a.heading[nn].reshape(n2,ens_to_avg)
            a.heading = np.zeros(n2,np.float64)
            for i in range(n2):
                a.heading[i] = ssm.circmean(head[i,:]*np.pi/180)*180/np.pi
        if a.adcp_depth is not None:
            a.adcp_depth = au.average_vector(self.adcp_depth[nn],(n2,ens_to_avg))
        if a.error_vel is not None:
            a.error_vel = au.average_array(self.error_vel[nn,:],(n2,ens_to_avg),axis=0)
    
        return a
         
         
 


