import numpy as np
import pdb
import os
import sys
import datetime
from matplotlib.dates import date2num
from collections import defaultdict

warnings=defaultdict(lambda : defaultdict(lambda :0)) # warning message => {filename=>count}

def warn(msg,fn):
    warnings[msg][fn]+=1
    if warnings[msg][fn]==1:
        print msg


ll_sign=dict(N=1,S=-1,E=1,W=-1)

def center_to_edge(c,dx_single=None):
    """
    take 'cell' center locations c, and infer boundary locations.
    first/last cells get width of the first/last inter-cell spacing.
    if there is only one sample and dx_single is specified, use that
    for width.  otherwise error.
    """
    d=np.ones(len(c)+1)
    d[1:-1] = 0.5*(c[1:] + c[:-1])
    if len(c)>1:
        d[0]=c[0]-0.5*(c[1]-c[0])
        d[-1]=c[-1]+0.5*(c[-1]-c[-2])
    elif dx_single:
        d[0]=c[0]-0.5*dx_single
        d[1]=c[0]+0.5*dx_single
    else:
        raise Exception("only a single data point given to center to edge with no dx_single")
    return d

def nearest(A,x):
    """ like searchsorted, but return the index of the nearest value,
    not just the first value greater than x
    """
    N=len(A)

    xi_right=np.searchsorted(A,x).clip(0,N-1) # the index of the value to the right of x
    xi_left=(xi_right-1).clip(0,N-1)
    dx_right=np.abs(x-A[xi_right])
    dx_left=np.abs(x-A[xi_left])
    
    xi=xi_right
    sel_left=dx_left < dx_right
    if xi.ndim:
        xi[sel_left] = xi_left[sel_left]
    else:
        if sel_left:
            xi=xi_left
    return xi


class Pd0Exception(Exception):
    """
    All exceptions raised in this code are subclasses of
    this exception type
    """
    pass

class Pd0Eof(Pd0Exception):
    """ end-of-file was reached while trying to read
    a frame.  This is raised even when end-of-file
    is exactly at the end of a frame - i.e. this does
    not necessarily imply any errors.
    """
    pass

class Pd0ChecksumError(Pd0Exception):
    """ The computed checksum and the given checksum 
    in a frame did not match
    """
    pass

class Pd0VariableFrametypeError(Pd0Exception):
    pass

class Pd0Unsupported(Pd0Exception):
    """ signifies that a subclass implementation of a method 
    is missing.
    """
    pass

pd0_header_id=0x7F
src_id_workhorse=0x7F # appears to hold for channel master, too
src_id_waves=0x79 # unsure - inferred from NCCOOS pd0.py

header_dtype=np.dtype([
    ('hdr_id','u1'),
    ('src_id','u1'),
    ('nbytes','<u2'),
    ('spare','u1'),
    ('ntypes','u1')] )
offset_dtype=np.dtype('<u2')

# this prefix of the fixed leader should be enough (along with
# src_id) to choose the right Pd0Frame subclass for parsing the
# rest of the frame
fixed_prefix_leader_dtype=np.dtype([('cpu_firmware','u1'),
                                    ('cpu_revision','u1'),
                                    ('system_configuration','<u2')])

# It's possible to have interleaved types of frames, say one current frame
# then a wave frame.
# In the case of the Riverray, there is probably an additional issue that some
# parameters which are "fixed" in the fixed leader actually vary.
# so the building block of parsing is to parse a single block, then
# have separate logic which strings them together afterwards

def decode_system_configuration(config_bytes):
    """ 
    Apply any model-specific tweaks to the information in 
    fill in some informational fields in self.system_configuration
    based on the fixed leader
    """
    conf={}

        
    systems=['75-kHz SYSTEM',
             '150-kHz SYSTEM',
             '300-kHz SYSTEM',
             '600-kHz SYSTEM',
             '1200-kHz SYSTEM',
             '2400-kHz SYSTEM',
             '38-kHz SYSTEM',
             'unknown system']

    conf['system']=systems[ config_bytes&7 ]

    conf['pattern']=['concave','convex'][ (config_bytes&8)>>3 ]

    conf['sensor_config']=1 + ((config_bytes>>4)&3)
    conf['head_attached']= (config_bytes>>6) & 1

    conf['facing']=['down','up'][ (config_bytes>>7)&1 ]

    conf['angle']=[15,20,30,25][ (config_bytes>>8) & 3 ]

    config_id=config_bytes>>12

    if config_id==2:
        conf['description']='2-beam + vert. stage'
    elif config_id==4:
        conf['description']='4-beam Janus'
    elif config_id==5:
        conf['description']='5-beam Janus demod'
    elif config_id==15:
        conf['description']='5-beam Janus 2 demod'
    else:
        conf['description']='N/A'
 
    return conf

class Pd0Frame(object):
    """ Represents a single frame of data from a PD0 file.
    Generally (at least for an ADCP), this represents one ensemble

    A separate step is required to combined Frames
    into a time series, and provide reasonably user-facing interfaces
    to the data
    """

    def __init__(self,raw_frame,header,offsets,**kwargs):
        """
        raw_frame: the entire ensemble
        header: an already parsed header block
        offsets: the offsets from the end of the header
        kwargs: other variables which will be set as attributes on the 
           instance (e.g. initial_file_pos, source filename, etc.)
        """
        self.raw_frame=raw_frame
        self.header=header
        self.offsets=offsets

        self.fixed=None
        self.variable=None
        self.velocity=None
        self.correlation=None
        self.echo=None
        self.pct_good=None
        self.status=None
        self.stage=None
        self.text=None
        self.messages=[]
        # 
        self.unparsed=[] # tuples of block id and the raw data

        self.firmware_info=None

        self.filename='n/a'
        self.__dict__.update(kwargs)

    @property
    def coord_transform(self):
        """ return a string describing the coordinate transform
        does not get into the details of tilt/no tilt, just 
        beam, instrument, ship, or earth
        (note that ChannelMaster cannot be SHIP or EARTH)
        """
        bits=(self.fixed['coord_transform'] & 0b00011000) >> 3
        transforms=['beam','instrument','ship','earth']
        return transforms[bits]

    def blocks(self):
        """ return list [ (block_idx,block_id,raw), ... ]
        where block_idx just counts from 0
        block_id is the <u2 valued block id word
        and raw is the remaining bytes of the block
        """
        blks=[]
        for block_idx,start_byte in enumerate(self.offsets):
            if block_idx+1<len(self.offsets):
                end_byte=self.offsets[block_idx+1]
            else:
                # 4 bytes: 2 for the checksum, 2 reserved
                end_byte=self.header['nbytes']-4

            block_id=np.fromstring(self.raw_frame[start_byte:start_byte+2],'<u2')[0]
            raw=self.raw_frame[start_byte+2:end_byte]
            blks.append( (block_idx,block_id,raw) )
        return blks
        
    def parse(self):
        for block_idx,block_id,raw in self.blocks():
            self.parse_block(block_idx,block_id,raw)

    def to_dtype(self,raw,dtype,desc='n/a'):
        expect=dtype.itemsize

        if len(raw)!=expect:
            warn("%s: expected %d bytes but got %d"%(desc,expect,len(raw)),
                 self.filename)
            if len(raw)<expect:
                raw = raw + "\0"*(expect-len(raw))
            else:
                raw=raw[:expect]
        return np.fromstring(raw,dtype)[0]

    def block_summary(self):
        """ print metadata for this frame, specifically block ids and sizes.
        """
        for block_idx,block_id,raw in self.blocks():
            print "[%02d] 0x%04x  %3d bytes %s"%(block_idx,block_id,len(raw),
                                                 self.block_id_to_name(block_id))

    def block_id_to_name(self,blk_id):
        """ just for debugging/info purposes.  returns a string describing
        the type of block based on the 2-byte '<u2' block id.
        """
        block_types={0x0000:'fixed leader',
                     0x0080:'variable leader',
                     0x0100:'velocity',
                     0x0200:'correlation',
                     0x0300:'echo intensity',
                     0x0400:'percent good',
                     0x0500:'status',
                     0x0002:'firmware info'}
        return block_types.get(blk_id,'n/a')
                     
    def parse_block(self,block_idx,block_id,raw):
        if block_idx==0:
            assert(block_id==0x0000)
            self.fixed=self.to_dtype(raw,self.fixed_leader_dtype,'fixed leader')
        elif block_idx==1:
            assert(block_id==0x0080)
            self.variable=self.to_dtype(raw,self.variable_leader_dtype,'variable leader')
        elif block_id==0x0100:
            # slight differences in handling velocity
            self.parse_velocity(raw)
        elif block_id==0x0200:
            # ncells, 1 byte per sample, 4 'beams' regardless of hardware
            self.correlation=np.fromstring(raw,
                                           'u1').reshape([-1,4])
        elif block_id==0x0300:
            self.echo=np.fromstring(raw,
                                    'u1').reshape([-1,4])
        elif block_id==0x0400:
            self.pct_good=np.fromstring(raw, 
                                        'u1').reshape([-1,4])
        elif block_id==0x0500:
            self.status=np.fromstring(raw,
                                      'u1').reshape([-1,4])
        # this is probably common to most ADCPs
        elif block_id==0x2:
            # ASCII encoded firmware versions
            self.firmware_info=raw
        else:
            msg="Header id 0x%x not understood"%(block_id)
            warn(msg,self.filename)
            self.unparsed.append( (block_id,raw) )

    @classmethod
    def check_frame_types(cls,frames):
        """ return True if all frames are instances of the same subclass
        """
        return all( [f.__class__ == frames[0].__class__ for f in frames] )

    @classmethod
    def datenum_from_var_leader(cls,var_leaders,dest=None):
        """ Given a struct array with rtc_* fields, 
        return the dates parsed to matplotlib datenums.
        """
        if dest is not None:
            dns=dest
        else:
            dns=np.zeros( len(var_leaders),'f8')

        years=var_leaders['rtc_year'] + 1900
        years[ years<1985 ] += 100

        N=len(var_leaders)
        for fi in xrange(N):
            dns[fi]=date2num(datetime.datetime(years[fi],
                                               var_leaders['rtc_month'][fi],
                                               var_leaders['rtc_day'][fi],
                                               var_leaders['rtc_hour'][fi],
                                               var_leaders['rtc_minute'][fi],
                                               var_leaders['rtc_second'][fi],
                                               var_leaders['rtc_hundredth'][fi].astype('i4')*10000 ))
        return dns

    @classmethod
    def metadata(cls,frames):
        """ Return a dict with non-varying information about the dataset
        This generic version just copies data from the fixed leader of
        the first frame.  As needed, coded fields can be decoded here or
        in subclasses.
        """
        md={}
        f=frames[0]
        for n in f.fixed.dtype.names:
            md[n]=f.fixed[n]

        # human readable version of the coordinate system
        md['coord_sys']=f.coord_transform

        md['Nensembles']=len(frames)

        return md

    @classmethod 
    def concatenate_frames(cls,frames):
        """ returns a dictionary collecting all the parsed data.
        typically has a 'data' referring to the time series data,
        possibly a gps_data, gps_ensemble_idx if NMEA data was processed
        """
        raise Pd0Unsupported("Subclasses must implement concatenate_frames()")

    def frame_dtype(self):
        raise Pd0Unsupported("Subclasses must implement frame_dtype()")

        
class ChannelmasterFrame(Pd0Frame):
    fixed_leader_dtype=np.dtype([('cpu_firmware','u1'),
                                 ('cpu_revision','u1'),
                                 ('system_configuration','<u2'), # bitmask
                                 ('resvd00','u1'), 
                                 ('resvd01','u1'), 
                                 ('nbeams','u1'),
                                 ('Ncells','u1'),
                                 ('pings_per_ensemble','<u2'),
                                 ('cell_length','<u2'),
                                 ('blank_after_xmit','<u2'),
                                 ('resvd02','u1'),
                                 ('low_corr_thresh','u1'),
                                 ('ncode_reps','u1'),
                                 ('resvd03','u1'),
                                 ('error_vel_max','<u2'),
                                 ('tpp_minutes','u1'),
                                 ('tpp_seconds','u1'),
                                 ('tpp_hundredths','u1'),
                                 ('coord_transform','u1'),
                                 ('resvd04','<i2'), 
                                 ('resvd05','<i2'), 
                                 ('sensor_src','u1'),
                                 ('sensors_avail','u1'),
                                 ('bin1_distance_cm','<u2'),
                                 ('resvd06','<u2'),
                                 ('resvd07','u1'),
                                 ('resvd08','u1'),
                                 ('false_target_thresh','u1'),
                                 ('resvd09','u1'),
                                 ('xmit_lag_distance','<u2'),
                                 ('cpu_board_serial','<u8'),
                                 ('sys_bandwidth','u1'), # one byte for CM!
                                 ('resvd10','u1'),
                                 ('resvd11','<u2'),
                                 ('instrument_serial','<u4')])


    surface_commands_dtype=np.dtype( [('blank','<u2'),
                                      ('pings','<u2'),
                                      ('bw','u1'),
                                      ('detectmode','u1'),
                                      ('press_screen','u1'),
                                      ('range_screen','u1'),
                                      ('edge_detect_thresh','u1'),
                                      ('edge_detect_delta','<u2'),
                                      ('rx_gain','u1'),
                                      ('offset_tenths_mm','<u2'),
                                      ('scale_ppm','<u2'),
                                      ('max_range','<u2'),
                                      ('subpings','<u2'),
                                      ('tx_length','<u2'),
                                      ('w_thresh','u1'),
                                      ('w_width','<u2'),
                                      ('tx_power','u1')])

    surface_status_dtype=np.dtype( [('depth_corrected','<u4'),
                                    ('depth_raw','<u4'),
                                    ('eval_amplitude','u1'),
                                    # ('depth_sig','u1'), # M.S. code
                                    ('amplitude_at_surface','u1'),
                                    ('pct_good_surface','u1'), # byte #13
                                    ('std_surface','<u4'),
                                    ('min_surface','<u4'),
                                    ('max_surface','<u4'),
                                    ('pressure_correction','<u4'),
                                    ('depth_pressure','<u4'),
                                    ('pct_good_pressure','u1'),
                                    ('std_depth_pressure','<u4'),
                                    ('min_depth_pressure','<u4'),
                                    ('max_depth_pressure','<u4')] )

    surface_amplitude_dtype=np.dtype( [('pings_in_burst','u1'),
                                       ('surface_bin','<u2'), # these are all means
                                       ('filter_eval_amp','u1'),
                                       ('filter_surface_amp','u1'),
                                       ('wfilter_bin','<u2'),
                                       ('wfilter_eval_amp','u1'),
                                       ('wfilter_surface_amp','u1'),
                                       ('leading_edge_bin','<u2'),
                                       ('leading_edge_eval_amp','u1'),
                                       ('leading_edge_surface_amp','u1')] )


    variable_leader_dtype=np.dtype([('ensemble_number','<u2'),
                                    ('rtc_year','u1'),
                                    ('rtc_month','u1'),
                                    ('rtc_day','u1'),
                                    ('rtc_hour','u1'),
                                    ('rtc_minute','u1'),
                                    ('rtc_second','u1'),
                                    ('rtc_hundredth','u1'),
                                    ('ens_msb','u1'),
                                    ('bit_result','<u2'),
                                    ('c_sound','<u2'),
                                    ('transducer_depth','<u2'),
                                    ('resvd00','<u2'),
                                    ('pitch','<i2'),
                                    ('roll','<i2'),
                                    ('salinity','<u2'),
                                    ('temperature','<i2'),
                                    ('mpt_minute','u1'),
                                    ('mpt_second','u1'),
                                    ('mpt_hundredth','u1'),
                                    ('resvd01','u1'),
                                    ('pitch_std','u1'),
                                    ('roll_std','u1'),
                                    ('resvd02','u1'), # 
                                    ('voltage','u1'),
                                    ('resvd03','u1',12),
                                    ('pressure','<i4'),
                                    ('resvd04','u1',8)])
    def __init__(self,*args,**kwargs):
        super(ChannelmasterFrame,self).__init__(*args,**kwargs)
        # CM specific:
        self.surface_status=None
        self.surface_commands=None
        self.surface_amplitude=None
        
    def parse_velocity(self,raw):
        # ncells, 2 bytes per sample, apparently 4 'beams' regardless of
        # hardware
        # CM manual shows the latter 2 'beams' as reserved.
        velocity=np.fromstring(raw, 
                               '<i2').reshape([-1,4])
        self.velocity=velocity[:,:2] # latter 2 'beams' are reserved in CM


    def parse_block(self,block_idx,block_id,raw):
        # These are specific to Channel Master:
        if block_id==0x4000: 
            # VMSTageID, from Mike Simpson's code
            # only makes sense for channel master
            self.surface_status=np.fromstring(raw, 
                                              self.surface_status_dtype)[0]
        elif block_id==0x4001:
            self.surface_commands=np.fromstring(raw, 
                                                self.surface_commands_dtype)[0]
        elif block_id==0x4002:
            self.surface_amplitude_means=np.fromstring(raw[:13],
                                                       self.surface_amplitude_dtype)[0]
            Nping=self.surface_amplitude_means['pings_in_burst']
            pings=np.zeros( Nping,
                            dtype=[('w_bin','<u2'),
                                   ('w_eval_amp','u1'),
                                   ('w_amp','u1'),
                                   ('status','u1')])

            # The manual gives these as 2,3,1,1 bytes - 
            # but there are only 5 bytes per ping in the data file
            pings['w_bin']     =np.fromstring(raw[13:13+2*Nping],'<u2') 
            # manual gives these as 3-byte values - file looks like they
            # are actually 1 byte values
            pings['w_eval_amp']=np.fromstring(raw[13+2*Nping:13+3*Nping],'u1')
            pings['w_amp']     =np.fromstring(raw[13+3*Nping:13+4*Nping],'u1')
            pings['status']    =np.fromstring(raw[13+4*Nping:13+5*Nping],'u1')
            self.surface_amplitude_pings=pings
        else:
            super(ChannelmasterFrame,self).parse_block(block_idx,block_id,raw)


    def frame_dtype(self):
        """ return list of fields suitable for storing the time-varying
        portion of the data associated with this frame.
        """
        # For space-savings, only some of the data are included
        # here -
        Ncells=self.fixed['Ncells']

        fields=[ # from variable leader:
            ('ensemble_number','i4'), 
            ('dn','f8'), # matplotlib datenum 
            ('transducer_depth','f8'), # meters
            ('pitch','f8'), # degrees
            ('roll','f8'), # degrees
            ('salinity','f8'), # ppt
            ('temperature','f8'), # degrees C
            ('pressure_kPa','f8'), # kPa
            # from surface_status
            ('depth_corrected','f8'),
            # and per-bin values
            ('velocity','f8',(Ncells,2)),# m/s
            ('correlation','u1',(Ncells,2)), # 0-255 linear 
            ('echo','u1',(Ncells,2)),  # scaled to 0.45dB
            ('status','u1',(Ncells,2))
            ]

        if self.coord_transform=='BEAM':
            fields.append( ('pct_good','u1',(Ncells,2)) )
        elif self.coord_transform=='INSTRUMENT':
            # note that for channel master, partial solution is
            # the best possible, since there aren't enough beams
            # to get an error velocity (i.e. a full solution).
            # so 
            fields += [ ('pct_partial','u1',Ncells),
                        ('pct_rejected','u1',Ncells),
                        ('pct_nosolution','u1',Ncells) 
                        # ('pct_full','u1',Ncells)  # omit b/c always 0 for CM
                    ]
        return fields

    def bin_distances(self):
        """ the bin distances for just *this* frame, in m
        """
        f=self.fixed
        return 0.01*( f['bin1_distance_cm'] + np.arange(f['Ncells'])*f['cell_length'] )

    def bin_edges(self):
        """ like bin_distances, but returns N+1 distances corresponding 
        to boundaries between bins 
        """
        f=self.fixed
        dx=f['cell_length']
        return 0.01*( f['bin1_distance_cm'] + (np.arange(f['Ncells']+1)-0.5)*dx )


    @classmethod 
    def concatenate_frames(cls,frames):
        """ given a list of frames all of the same type, concatenate
        the time-varying fields to get all timeseries data in a single
        struct array (with dtype given by frames[0].frame_dtype()

        returns (metadata,packed) 
          packed: array of the ensembles
          metadata: dictionary describing the whole dataset
        """
        if not cls.check_frame_types(frames):
            raise Pd0VariableFrametypeError("Multiple frame types are not supported")

        md=cls.metadata(frames)

        # Allocate a single array for the whole dataset
        N=len(frames)
        fdata=np.zeros(N,dtype=frames[0].frame_dtype() )

        # 

        # velocity
        all_v=np.concatenate( [f.velocity[None,...] for f in frames], axis=0)
        # convert to m/s
        fdata['velocity']=all_v/1000.0
        fdata['velocity'][ all_v==-32768 ] = np.nan

        # ens. number:
        var_leaders=np.array( [f.variable for f in frames] )
        fdata['ensemble_number']=var_leaders['ensemble_number'] + var_leaders['ens_msb']*2**16

        # parse timestamp to a python datenum
        cls.datenum_from_var_leader(var_leaders,dest=fdata['dn'])

        # 
        # scalar fields with minimal processing
        fdata['transducer_depth']=var_leaders['transducer_depth']*0.1 # meters, with decimeter resolution
        fdata['transducer_depth'][ var_leaders['transducer_depth']==-1 ] = np.nan

        fdata['salinity']=var_leaders['salinity'] # in ppt
        fdata['salinity'][ var_leaders['salinity']==-1]=np.nan

        # 
        # pretty sure these are all scaled to 0.01 degree,
        # but the channel master manual is unclear for pitch/roll.
        # workhorse manual shows them all as 0.01 degree.
        for k in ['temperature','pitch','roll']:
            fdata[k]=var_leaders[k]*0.01
            fdata[k][ var_leaders[k]==-32768 ]=np.nan
        # 

        # pressure original in decapascals - convert to kPa
        # relative to one atmosphere
        fdata['pressure_kPa']=var_leaders['pressure']/100.0
        fdata['pressure_kPa'][ var_leaders['pressure']==-2147483648 ] = np.nan

        surf_stats=np.array( [f.surface_status for f in frames] )
        fdata['depth_corrected']=surf_stats['depth_corrected']*0.0001

        fdata['correlation']=np.array( [f.correlation for f in frames] )[:,:,:2]
        fdata['echo']=np.array( [f.echo for f in frames] )[:,:,:2]
        fdata['status']=np.array( [f.status for f in frames] )[:,:,:2]

        # interpretation of pct_good depends on coordinate transform
        pct_good=np.array( [f.pct_good for f in frames] )
        if frames[0].coord_transform=='BEAM':
            fdata['pct_good']=pct_good[:,:,:2]
        elif frames[0].coord_transform=='INSTRUMENT':
            fdata['pct_partial']=pct_good[:,:,0]
            fdata['pct_rejected']=pct_good[:,:,1]
            fdata['pct_nosolution']=pct_good[:,:,2]
            # fdata['pct_full']=pct_good[:,:,3]

        md['data']=fdata
        return md

    @classmethod
    def metadata(cls,frames):
        """ Interpret some of the config values
        """
        
        # bin distance should be constant across file
        md=super(ChannelmasterFrame,cls).metadata(frames)
        md['bin_distances'] = frames[0].bin_distances()
        return md

        
class WorkhorseFrame(Pd0Frame):
    fixed_leader_dtype=np.dtype([('cpu_firmware','u1'),
                                 ('cpu_revision','u1'),
                                 ('system_configuration','<u2'), # bitmask
                                 ('is_sim','u1'), 
                                 ('lag_length','u1'),
                                 ('nbeams','u1'),
                                 ('Ncells','u1'),
                                 ('pings_per_ensemble','<u2'),
                                 ('cell_length','<u2'),
                                 ('blank_after_xmit','<u2'),
                                 ('profiling_mode','u1'),
                                 ('low_corr_thresh','u1'),
                                 ('ncode_reps','u1'),
                                 ('pct_good_min','u1'),
                                 ('error_vel_max','<u2'),
                                 ('tpp_minutes','u1'),
                                 ('tpp_seconds','u1'),
                                 ('tpp_hundredths','u1'),
                                 ('coord_transform','u1'),
                                 ('heading_alignment','<i2'), # 0.01 deg
                                 ('heading_bias','<i2'),      # 0.01 deg
                                 ('sensor_src','u1'),
                                 ('sensors_avail','u1'),
                                 ('bin1_distance_cm','<u2'),
                                 ('xmit_pulse_length_cm','<u2'),
                                 ('wp_ref_layer_start','u1'),   
                                 ('wp_ref_layer_end','u1'),     
                                 ('false_target_thresh','u1'),
                                 ('cx','u1'), # also listed as spare
                                 ('xmit_lag_distance','<u2'),
                                 ('cpu_board_serial','<u8'),
                                 ('sys_bandwidth','<u2'), # one byte for CM!
                                 ('sys_power','u1'),
                                 ('spare','u1'),
                                 ('instrument_serial','<u4'),
                                 ('beam_angle','u1')])
    variable_leader_dtype=np.dtype([('ensemble_number','<u2'),
                                    ('rtc_year','u1'),
                                    ('rtc_month','u1'),
                                    ('rtc_day','u1'),
                                    ('rtc_hour','u1'),
                                    ('rtc_minute','u1'),
                                    ('rtc_second','u1'),
                                    ('rtc_hundredth','u1'),
                                    ('ens_msb','u1'),
                                    ('bit_result','<u2'),
                                    ('c_sound','<u2'),
                                    ('transducer_depth','<u2'),
                                    ('heading','<u2'),# resvd on CM
                                    ('pitch','<i2'),
                                    ('roll','<i2'),
                                    ('salinity','<u2'),
                                    ('temperature','<i2'),
                                    ('mpt_minute','u1'),
                                    ('mpt_second','u1'),
                                    ('mpt_hundredth','u1'),
                                    ('heading_std','u1'), # resvd on CM
                                    ('pitch_std','u1'),
                                    ('roll_std','u1'),
                                    ('adc','u1',8), # 
                                    ('error_status','u4'),
                                    ('spare0','<u2'),
                                    ('pressure','<i4'),
                                    ('pressure_variance','<i4'), # manual gives it as signed...
                                    ('spare1','u1'),
                                    ('rtc_y2k_century','u1'), # y2k moniker taken from USF-COT code
                                    ('rtc_y2k_year','u1'),
                                    ('rtc_y2k_month','u1'),
                                    ('rtc_y2k_day','u1'),
                                    ('rtc_y2k_hour','u1'),
                                    ('rtc_y2k_minute','u1'),
                                    ('rtc_y2k_second','u1'),
                                    ('rtc_y2k_hundredth','u1')])


    bottom_track_dtype=np.dtype([('pings_per_ens','<u2'),
                                 ('delay_before_reaq','<u2'),
                                 ('min_corr_mag','u1'),
                                 ('min_eval_amp','u1'),
                                 ('min_pct_good','u1'),
                                 ('mode','u1'),
                                 ('max_err_vel','<u2'),
                                 ('reserved','<u4'),
                                 ('range','<u2',4),
                                 ('vel','<u2',4),
                                 ('corr','<u2',4),
                                 ('eval_amp','<u2',4),
                                 ('pct_good','<u2',4),
                                 ('ref_layer_min','<u2'),
                                 ('ref_layer_near','<u2'),
                                 ('ref_layer_far','<u2'),
                                 ('ref_layer_vel','<u2',4),
                                 ('ref_corr','u1',4),
                                 ('ref_int','u1',4),
                                 ('ref_pct_good','u1',4),
                                 ('max_depth','<u2'),
                                 ('rssi_amp','u1',4),
                                 ('gain','u1'),
                                 ('range_msb','u1',4),
                                 ('reserved1','<u4')])
    def __init__(self,*args,**kwargs):
        super(WorkhorseFrame,self).__init__(*args,**kwargs)
        self.bottom_track=None

    def parse_velocity(self,raw):
        # ncells, 2 bytes per sample, apparently 4 'beams' regardless of
        # hardware
        # mm/s, bad values are -32768
        self.velocity=np.fromstring(raw, 
                                    '<i2').reshape([-1,4])


    def block_id_to_name(self,blk_id):
        """ just for debugging/info purposes.  returns a string describing
        the type of block based on the 2-byte '<u2' block id.
        """
        if blk_id==0x0600:
            return 'bottom track'
        else:
            return super(WorkhorseFrame,self).block_id_to_name(blk_id)

    def parse_block(self,block_idx,block_id,raw):
        """ workhorse specific blocks 
        """
        if block_id==0x0600:
            self.bottom_track=np.fromstring(raw, 
                                            self.bottom_track_dtype)[0]
            # note that range actually has an msb field separate from the
            # low two bytes - needs some postprocessing
        else:
            super(WorkhorseFrame,self).parse_block(block_idx,block_id,raw)


class RiverrayFrame(WorkhorseFrame):
    # fixed_leader is almost identical to Workhorse 
    fixed_leader_dtype=np.dtype([('cpu_firmware','u1'),
                                 ('cpu_revision','u1'),
                                 ('system_configuration','<u2'), # bitmask
                                 ('is_sim','u1'), 
                                 ('lag_length','u1'), 
                                 ('nbeams','u1'),
                                 ('Ncells','u1'),
                                 ('pings_per_ensemble','<u2'),
                                 ('cell_length','<u2'),
                                 ('blank_after_xmit','<u2'),
                                 ('profiling_mode','u1'),
                                 ('low_corr_thresh','u1'),
                                 ('ncode_reps','u1'),
                                 ('pct_good_min','u1'),
                                 ('error_vel_max','<u2'),
                                 ('tpp_minutes','u1'),
                                 ('tpp_seconds','u1'),
                                 ('tpp_hundredths','u1'),
                                 ('coord_transform','u1'),
                                 ('heading_alignment','<i2'), 
                                 ('heading_bias','<i2'), 
                                 ('sensor_src','u1'),
                                 ('sensors_avail','u1'),
                                 ('bin1_distance_cm','<u2'),
                                 ('xmit_pulse_length','<u2'),
                                 ('resvd00','u1'),
                                 ('resvd01','u1'),
                                 ('false_target_thresh','u1'),
                                 ('resvd02','u1'),
                                 ('xmit_lag_distance','<u2'),
                                 ('cpu_board_serial','<u8'),
                                 ('sys_bandwidth','<u2'),
                                 ('sys_power','u1'),
                                 ('resvd03','u1'),
                                 ('instrument_serial','<u4'),
                                 ('beam_angle','u1') ])

    # again, very similar to WorkHorse
    # DBG: this is 64 bytes, WorkHorse above is 63 bytes, and
    # the sample data file is 58 bytes.
    # RR docs give it as 66 bytes, less two bytes for ID is 64.
    # those are all correct.
    # the sample file is from WinRiver - is WinRiver screwing with the
    # data?
    variable_leader_dtype=np.dtype([('ensemble_number','<u2'),
                                    ('rtc_year','u1'),
                                    ('rtc_month','u1'),
                                    ('rtc_day','u1'),
                                    ('rtc_hour','u1'),
                                    ('rtc_minute','u1'),
                                    ('rtc_second','u1'),
                                    ('rtc_hundredth','u1'),
                                    ('ens_msb','u1'),
                                    ('bit_fault','u1'),
                                    ('bit_count','u1'),
                                    ('c_sound','<u2'),
                                    ('transducer_depth','<u2'),
                                    ('heading','<u2'),
                                    ('pitch','<i2'),
                                    ('roll','<i2'),
                                    ('salinity','<u2'),
                                    ('temperature','<i2'),
                                    ('mpt_minute','u1'),
                                    ('mpt_second','u1'),
                                    ('mpt_hundredth','u1'),
                                    ('heading_std','u1'),
                                    ('pitch_std','u1'),
                                    ('roll_std','u1'),
                                    ('adc0','u1'), # 
                                    ('voltage','u1'),
                                    ('adc2to7','u1',6),
                                    ('resvd00','u1',15),
                                    ('rtc_y2k_century','u1'), 
                                    ('rtc_y2k_year','u1'),
                                    ('rtc_y2k_month','u1'),
                                    ('rtc_y2k_day','u1'),
                                    ('rtc_y2k_hour','u1'),
                                    ('rtc_y2k_minute','u1'),
                                    ('rtc_y2k_second','u1'),
                                    ('rtc_y2k_hundredth','u1'),
                                    ('lag_near_bottom','u1')])

    # some minor differences
    bottom_track_dtype=np.dtype([('pings_per_ens','<u2'),
                                 ('delay_before_reaq','<u2'),
                                 ('min_corr_mag','u1'),
                                 ('min_eval_amp','u1'),
                                 ('min_pct_good','u1'),
                                 ('mode','u1'),
                                 ('max_err_vel','<u2'),
                                 ('reserved','<u4'),
                                 ('range','<u2',4),
                                 ('vel','<i2',4), # details unspecified
                                 ('corr','u1',4),
                                 ('eval_amp','u1',4),
                                 ('pct_good','u1',4),
                                 ('resvd00','u1',26),# present on WH
                                 ('max_depth','<u2'),
                                 ('rssi_amp','u1',4),
                                 ('gain','u1'),
                                 ('range_msb','u1',4),
                                 ('resvd01','<u4'),
                                 # unsure why lsb range is duplicated
                                 ('range_lsb','u1',4)])

    vertical_dtype=np.dtype( [('eval_amp','u1'),
                              ('rssi_amp','u1'),
                              ('range_to_bottom','<u4'),
                              ('status','u1')] )

    surface_leader_dtype=np.dtype( [('cell_count','u1'),
                                    ('cell_size','<u2'),
                                    ('cell1_distance','<u2')] )

    automatic_beam_dtype=np.dtype( [('mode','u1'),
                                    ('depth','<u2'),
                                    ('ping_count','u1'),
                                    ('ping_type','u1'),
                                    ('cell_count','<u2'),
                                    ('cell_size','<u2'),
                                    ('bin1_mid','<u2'),
                                    ('code_reps','u1'),
                                    ('xmit_length','<u2'),
                                    ('lag_length','<u2'),
                                    ('tx_bandwidth','u1'),
                                    ('rx_bandwidth','u1'),
                                    ('min_ping_interval','<u2')] )

    fw_status_dtype=np.dtype( [('alpha','S1'),
                               ('branch','S1',14),
                               ('test_data','<u2'),
                               ('test_switches','<u2'),
                               ('resvd00','u1') ] )

    nmea_dtype=np.dtype( [('msg_id','<u2'), # 104 GG, 105 VTG
                          ('msg_size','<u2'),
                          ('delta_time','f8')] )

    # winriver mangles GPGGA sentences, replacing most of the
    # data with binary values.
    wr_gga_dtype=np.dtype( [ ('nmea_type','S7'),
                             ('time_string','S10'), # HHMMSS.SS
                             ('lat','<f8'),
                             ('lat_sign','S1'),
                             ('lon','<f8'),
                             ('lon_sign','S1'),
                             ('quality','u1'),
                             ('satellites','u1'),
                             ('hdop','<f4'),
                             ('alt','<f4'),
                             ('alt_units','S1'),
                             ('geoid_height','<f4'),
                             ('geoid_height_units','S1'),
                             ('time_since_dgps','<f4'),
                             ('dgps_station','<u2') ] )
    # size of actual message depends on msg_id


    # ADCP_RdiWorkhorse_Data constructs
    # (a) list ensemble_gps_indexes [ (ens #, index into gps_data), ... ]
    # (b) list gps_data [ (day_fraction,lat,lon), ... ]
    # It doesn't look like there is any code that really uses these, but
    # it would be easy enough to ultimately repackage the data in that way.
    # here we assemble basically all the data, and let 
    # ADCP_RdiRiverray_Data.py massage into those two lists.

    nmea_data_dtype=np.dtype( [('lat','f8'),
                               ('lon','f8'),
                               ('day_fraction','f8'), 
                               ('altitude','f8'),
                               ('geoid_height','f8'),
                               ('quality','u1'),
                               ('hdop','f8'),
                               ('satellites','u1')] )

    def __init__(self,*args,**kwargs):
        super(RiverrayFrame,self).__init__(*args,**kwargs)
        # RR specific:
        self.vertical=None
        self.surface_leader=None
        self.surface_velocity=None
        self.surface_correlation=None
        self.surface_echo=None
        self.surface_pct_good=None
        self.surface_status=None
        self.automatic=None
        self.fw_status=None

        # several different types of data here:

        # the result of parsing 0x2022 blocks, which have a bit of 
        # metadata, and either the raw or winriver-mangled NMEA sentence.
        self.nmea_msgs=[] # [ (nmea_metadata,nmea_msg), ... ]

        # the relevant data extracted from nmea_msgs, including
        # lat,lon,timestamp,altitude,geoid_height,quality,satellites
        # packed into a arrays
        # This can only be done when it's a GGA sentence in the WinRiver
        # mangled format
        self.nmea_data=[]
        
        # looks like the one most recent NMEA sentence, verbatim as
        # text.  Even though it appears that there is at most one of 
        # these per ensemble, this is a list
        self.nmea_raw_sentences=[] # [ "$GPGGA,...", ... ]

    def block_id_to_name(self,blk_id):
        """ just for debugging/info purposes.  returns a string describing
        the type of block based on the 2-byte '<u2' block id.
        """
        block_types={0x4100:'vertical range',
                     0x0010:'surface leader',
                     0x0110:'surface velocity',
                     0x0210:'surface correlation',
                     0x0310:'surface echo intensity',
                     0x0410:'surface percent good',
                     0x0510:'surface status',
                     0x4401:'automatic mode setup',
                     0x4400:'riverray firmware status',
                     0x2022:'timestamped nmea sentence',
                     0x2101:'raw nmea sentence'}
        if blk_id in block_types:
            return block_types[blk_id]
        else:
            return super(RiverrayFrame,self).block_id_to_name(blk_id)


    def parse_block(self,block_idx,block_id,raw):
        # These are specific to RiverRay:
        if block_id==0x4100: 
            # this is failing with 5 bytes of input, but vertical_dtype has 7 bytes
            # with zero padding, it returns 47, 120, 2945, [0], but the msb of the
            # vertical range is padded to zero then - no good.
            # which would be 2.945m deep
            # the exact bytes are array([ 47, 120, 129,  11,   0], dtype=uint8)
            # would assume that vertical range is still 4 bytes - but with 5 bytes,
            # only two options, and they result in values of either 754m or 193km
            # max legal value is 100m.
            # so far, most reasonable solution is equivalent to padding the frame
            # with 0s - same as dropping the status byte and the msb for the range
            if len(raw)==5:
                raw=raw+"\0\0" 
            self.vertical=np.fromstring(raw, 
                                        self.vertical_dtype)[0]
        elif block_id==0x0010:
            self.surface_leader=np.fromstring(raw,
                                              self.surface_leader_dtype)[0]
        elif block_id==0x0110:
            # 2-5 cells - should match Ncells from surface leader
            self.surface_velocity=np.fromstring(raw,
                                                '<i2').reshape([-1,4])
            assert(self.surface_velocity.shape[0]==self.surface_leader['cell_count'])
        elif block_id==0x0210:
            self.surface_correlation=np.fromstring(raw,'u1').reshape([-1,4])
            assert(self.surface_correlation.shape[0]==self.surface_leader['cell_count'])
        elif block_id==0x0310:
            self.surface_echo=np.fromstring(raw,'u1').reshape([-1,4])
            assert(self.surface_echo.shape[0]==self.surface_leader['cell_count'])
        elif block_id==0x0410:
            self.surface_pct_good=np.fromstring(raw,'u1').reshape([-1,4])
            assert(self.surface_pct_good.shape[0]==self.surface_leader['cell_count'])
        elif block_id==0x0510:
            self.surface_status=np.fromstring(raw,'u1').reshape([-1,4])
            assert(self.surface_status.shape[0]==self.surface_leader['cell_count'])
        elif block_id==0x4401: # this had been 0x0144, but I think that's byteswapped
            beam_count=np.fromstring(raw[:1],'u1')
            self.automatic=np.fromstring(raw[1:-1],self.automatic_beam_dtype)
            assert(len(self.automatic)==beam_count)
        elif block_id==0x4400: # this had been byteswapped, too.
            # actual format unclear, doesn't match the RiverRay manual
            # leave it as raw bytes.
            self.fw_status=raw
            # self.fw_status=np.fromstring(raw,self.fw_status_dtype)[0]
        elif block_id==0x2022:
            meta_size=self.nmea_dtype.itemsize
            nmea_metadata=np.fromstring(raw[:meta_size],
                                        self.nmea_dtype)[0]
            nmea_msg=raw[meta_size:]
            self.nmea_msgs.append( (nmea_metadata,nmea_msg) )

            if len(nmea_msg)==self.wr_gga_dtype.itemsize:
                # good - can parse it
                gga=np.fromstring(nmea_msg,self.wr_gga_dtype)[0]
                nmea_datum=np.zeros((),self.nmea_data_dtype)
                
                nmea_datum['lat']=ll_sign[gga['lat_sign']] * gga['lat']
                nmea_datum['lon']=ll_sign[gga['lon_sign']] * gga['lon']
                hh=int(gga['time_string'][:2])
                mm=int(gga['time_string'][2:4])
                ss=float(gga['time_string'][4:]) # ss.ss
                nmea_datum['day_fraction']=((ss/60. + mm)/60 + hh)/24.0
                nmea_datum['altitude']=gga['alt']
                nmea_datum['geoid_height']=gga['geoid_height']
                nmea_datum['quality']=gga['quality']
                nmea_datum['hdop']=gga['hdop']
                nmea_datum['satellites']=gga['satellites']
                self.nmea_data.append(nmea_datum)
            elif len(nmea_msg)==28:
                warn("Likely GPVTG, not parsed.",self.filename)
            else:
                warn("Unknown timestamped NMEA sentence",self.filename)

        elif block_id==0x2101 or block_id==0x2102:
            # 2101: GGA
            # 2102: VTG
            # e.g. ",\x00$GPVTG,213.610,T,0.000,M,0.230,N,0.430,K,D"
            self.nmea_raw_sentences.append(raw[2:])
        else:
            super(RiverrayFrame,self).parse_block(block_idx,block_id,raw)

    def bin_distances(self):
        """ the bin distances for just *this* frame, in m
        """
        f=self.fixed
        return 0.01*( f['bin1_distance_cm'] + np.arange(f['Ncells'])*f['cell_length'] )

    def bin_edges(self):
        """ like bin_distances, but returns N+1 distances corresponding 
        to boundaries between bins 
        """
        f=self.fixed
        dx=f['cell_length']
        return 0.01*( f['bin1_distance_cm'] + (np.arange(f['Ncells']+1)-0.5)*dx )

            
    @classmethod
    def frame_dtype(cls,data):
        """ return list of fields suitable for storing the time-varying
        portion of the data associated with this frame.
        """
        # this changes between ensembles
        # Ncells=self.fixed['Ncells']
        Ncells=data['velocity'].shape[1]

        fields=[ # from variable leader:
            ('ensemble_number','i4'), 
            ('dn','f8'), # matplotlib datenum 
            ('transducer_depth','f8'), # meters
            ('heading','f8'), # degrees
            ('pitch','f8'), # degrees
            ('roll','f8'), # degrees
            ('salinity','f8'), # ppt
            ('temperature','f8'), # degrees C
            # from bottom track
            ('bt_range','f8',4),
            ('bt_vel','f8',4),
            ('bt_corr','u1',4),
            ('bt_pct_good','u1',4),
            # from vertical
            ('vertical_range','f8'),
            # from surface related frames
            ('surf_Ncells','u1'),
            ('surf_cellsize','f8'),
            ('surf_cell1_distance','f8'),
            ('surf_vel','f8',(5,4)), # contains 2-5 values
            ('surf_corr','u1',(5,4)),#  2-5 values
            ('surf_pct_good','u1',(5,4)), # 2-5 values
            ('surf_echo','f8',(5,4)), # 2-5 values
            # automatic: skip
            # nmea:
            ('nmea_text','S41'),
            ('nmea_delta','f8'),
            # and per-bin values
            ('velocity','f8',(Ncells,4)),# m/s
            ('correlation','u1',(Ncells,4)), # 0-255 linear 
            ('echo','u1',(Ncells,4)),  # scaled to 0.45dB
            ('status','u1',(Ncells,4)),
            ('pct_good','u1',(Ncells,4))
            ]

        if 'gps_index' in data:
            fields.append( ('gps_index','i4') )

        return fields

    @classmethod 
    def concatenate_frames(cls,frames):
        """ given a list of frames all of the same type, concatenate
        the time-varying fields to get all timeseries data in a single
        struct array (with dtype given by frames[0].frame_dtype()

        returns (metadata,packed) 
          packed: array of the ensembles
          metadata: dictionary describing the whole dataset
        """
        if not cls.check_frame_types(frames):
            raise Pd0VariableFrametypeError("Multiple frame types are not supported")

        md=cls.metadata(frames)

        # Can't allocate ahead of time b/c sizes are not known for RiverRay.
        N=len(frames)
        fdata={} 

        #### 

        # ens. number:
        var_leaders=np.array( [f.variable for f in frames] )
        fdata['ensemble_number']=var_leaders['ensemble_number'] + var_leaders['ens_msb']*2**16

        # parse timestamp to a python datenum
        fdata['dn']=cls.datenum_from_var_leader(var_leaders)

        # the RiverRay manual does not make explicit the error values
        # here - these values are taken from the WorkHorse manual.

        # scalar fields with minimal processing
        fdata['transducer_depth']=var_leaders['transducer_depth']*0.1 # meters, with decimeter resolution
        fdata['transducer_depth'][ var_leaders['transducer_depth']==-1 ] = np.nan 

        fdata['salinity']=var_leaders['salinity'].astype('f8') # in ppt
        fdata['salinity'][ var_leaders['salinity']==-1]=np.nan

        for k in ['heading','pitch','roll','temperature']:
            fdata[k]=var_leaders[k]*0.01
            fdata[k][ var_leaders[k]==-32768 ]=np.nan
        # 

        bts=np.array( [f.bottom_track for f in frames] )

        # meters
        fdata['bt_range']=0.01*(bts['range'] + bts['range_msb']*(2**16))
        fdata['bt_range'][bts['range']==0] = np.nan

        # m/s
        fdata['bt_vel']=bts['vel']*0.001 # assuming same scaling as water vel.
        fdata['bt_corr']=bts['corr'] # same as water col.
        fdata['bt_pct_good']=bts['pct_good']

        # from vertical
        missing=np.zeros((),dtype=cls.vertical_dtype)
        missing['range_to_bottom']=0
        verts=np.array( [f.vertical or missing for f in frames] )
        fdata['vertical_range']= verts['range_to_bottom']*0.001 # m
        # this would work according to the manual, but the files so far have
        # a short vertical frame, and status is left unknown.
        # invalid= (verts['status']&0x3) == 0
        invalid=( verts['range_to_bottom']==0 )
        fdata['vertical_range'][invalid]=np.nan

        # from surface related frames
        # These are a pain because they aren't all the same size
        fdata['surf_vel']=np.nan*np.ones((N,5,4),'f8')
        fdata['surf_corr']=np.ones((N,5,4),'u1')
        fdata['surf_pct_good']=np.ones_like(fdata['surf_corr'])
        fdata['surf_echo']=np.nan*np.ones_like(fdata['surf_vel'])
        fdata['surf_Ncells']=np.zeros(N,'u1')
        fdata['surf_cellsize']=np.nan*np.ones(N,'f8')
        fdata['surf_cell1_distance']=np.nan*np.ones(N,'f8')
        fdata['nmea_text']=np.zeros(N,'S41')
        fdata['nmea_delta']=np.zeros(N,'f8')

        for i in range(len(frames)):
            f=frames[i]

            if f.surface_leader is not None:
                nsurf=fdata['surf_Ncells'][i]=f.surface_leader['cell_count']
                fdata['surf_cellsize'][i]=f.surface_leader['cell_size']*0.01 # m
                fdata['surf_cell1_distance'][i]=f.surface_leader['cell1_distance']*0.01 # m

            if f.surface_velocity is not None:
                # manual doesn't give scaling, assume same as other velocity cells
                fdata['surf_vel'][i][:nsurf,:] = f.surface_velocity*0.001 
            if f.surface_correlation is not None:
                fdata['surf_corr'][i][:nsurf,:] = f.surface_correlation
            if f.surface_pct_good is not None:
                fdata['surf_pct_good'][i][:nsurf,:]=f.surface_pct_good
            if f.surface_echo is not None:
                fdata['surf_echo'][i][:nsurf,:]=f.surface_echo

            # automatic: skip

        # and per-bin values
        fdata['velocity']=np.zeros( (N,md['Ncells'],4), 'f8' )
        fdata['velocity'][:,:,:]=np.nan
        for fld in ['correlation','echo','status','pct_good']:
            fdata[fld]=np.zeros( (N,md['Ncells'],4), 'u1')

        for i,f in enumerate(frames):
            # at least in the first test file, when it switches to
            # coarser bins, they are exactly offset so that one coarse
            # bin maps to two fine bins.
            vels=f.velocity/1000.0
            vels[ f.velocity==-32768 ] = np.nan
            
            # infer the boundaries between bins in this frame
            edges=f.bin_edges()
            # and map the output bin_centers to indices relative
            # to those boundaries
            srcs=np.searchsorted(edges,
                                 md['bin_distances']) - 1
            # limit those to bin centers which map into the interior
            # of src bins
            valid=(srcs>=0) & (srcs<len(vels))
            srcs=srcs[valid]

            fdata['velocity'][i,valid,:] = vels[srcs,:]

            for fld in ['correlation','echo','status','pct_good']:
                per_frame=getattr(f,fld)
                if per_frame is not None:
                    fdata[fld][i,valid]=per_frame[srcs]

        if 1: # handle GPS data
            count=0
            per_ensemble=[]
            gps_index=np.zeros(len(frames),'i4')

            for fi,f in enumerate(frames):
                gps_index[fi]=count
                if f.nmea_data:
                    this_ensemble=np.array(f.nmea_data)
                    per_ensemble.append(this_ensemble)
                    count+=len(this_ensemble)
                elif f.nmea_raw_sentences:
                    warn("Ensemble had no parsed NMEA, but has raw NMEA",
                         f.filename)
            if count:
                md['gps_data']=np.concatenate( per_ensemble )
                fdata['gps_index']=gps_index

        # Transfer from dictionary to a numpy struct array

        # nan out float point fields 
        packed=np.zeros(N,dtype=cls.frame_dtype(fdata))
        for f in packed.dtype.names:
            try:
                packed[f][...]=np.nan
            except TypeError:
                # N.B. this error is NOT robust - it's okay
                # here, but at least some numpy will silently
                # let this go, even if it's an integer field.
                pass
        for f in fdata.keys():
            packed[f]=fdata[f]


        md['data']=packed

        return md

    @classmethod
    def metadata(cls,frames):
        """ Interpret some of the config values
        """
        md=super(RiverrayFrame,cls).metadata(frames)

        # RiverRay takes some more involved processing since some of these
        # fields vary between ensembles
        cell_length_cm   = np.array( [f.fixed['cell_length'] for f in frames] )
        ncells           = np.array( [f.fixed['Ncells'] for f in frames] )
        bin1_distance_cm = np.array( [f.fixed['bin1_distance_cm'] for f in frames] )

        # replace values (which came from the first frame), with the intended
        # values for the full dataset (i.e. after sampling to a common vertical axis)
        del md['cell_length']
        del md['bin1_distance_cm']
        md['cell_length_m'] = 0.01*cell_length_cm.min()
        md['bin1_distance_m'] = 0.01*bin1_distance_cm.min()

        # center of the last possible bin
        binN_distance_cm=( bin1_distance_cm + (ncells-1)*cell_length_cm ).max()

        # offchance of getting an extra cell here due to f.p. operations, but no significant
        # downside to that.
        effective_max_ncells=1 + int(np.ceil( (binN_distance_cm-bin1_distance_cm.min()) / float(cell_length_cm.min())))
        md['Ncells']=effective_max_ncells

        md['bin_distances'] = md['bin1_distance_m'] + np.arange(md['Ncells'])*md['cell_length_m']
        return md


class EofFile(file):
    def read(self,nbytes):
        # wrapper which will throw an exception on EOF
        data=file.read(self,nbytes)
        if len(data) < nbytes:
            raise Pd0Eof()
        return data


class Pd0Reader(object):
    """ These methods are bundled as a class just for organization.
    The entry point is parse, and is assigned to the method pd0_parse(fn)
    below
    """
    def __init__(self):
        pass

    def parse(self,fn,cls=None):
        frames=self.parse_all_frames(fn,cls=cls)
        if frames:
            return frames[0].concatenate_frames(frames)
        else:
            print "%s had no frames!"%fn
            return None,None

    def parse_all_frames(self,fn,cls=None):
        """
        fn: filename
        returns: a list of parsed frames
        """
        fp=EofFile(fn,'rb')

        # get size of file:
        fp.seek(0,os.SEEK_END)
        eof=fp.tell()
        fp.seek(0)

        frames=[]

        while fp.tell() < eof:
            try:
                frames.append( self.parse_one_frame(fp,filename=fn,cls=cls) )
            except Pd0Eof:
                break
        return frames

    def frame_class(self,header,system_configuration):
        if header['src_id']==0x7f:
            config_id=system_configuration>>12

            if config_id==2:
                return ChannelmasterFrame
            else:
                # HERE: need to determine this more robustly
                #return WorkhorseFrame
                return RiverrayFrame
        else:
            print "Unknown src_id %x, will try Workhorse as a last resort"%header['src_id']
            
    def scan_for_ensemble(self,fp):
        # Scan for 0x7f, 0x7f
        hdr_id=ord(fp.read(1))
        src_id=ord(fp.read(1))

        # FUTURE: could support other src_ids
        while hdr_id!=pd0_header_id or src_id!=src_id_workhorse:
            hdr_id,src_id=src_id,ord(fp.read(1))
            sys.stdout.write('.') ; sys.stdout.flush()

        fp.seek(fp.tell()-2) # reposition just befor the marks

    def read_ensemble(self,fp):
        """ Assumes that fp is positioned just *before* the hdr_id,src_id bytes.
        returns the header, complete raw_frame, and offsets.
        (header and offsets parsed, raw_frame just bytes)
        """
        pos=fp.tell() # save position at very beginning of ensemble (before hdr_id)
        header=np.fromstring(fp.read(header_dtype.itemsize),header_dtype)[0]

        # Read the whole frame into a buffer:
        fp.seek(pos) # or maybe we do get the hdr/src id?? step back to start of the frame, but not including hdr_id/src_id
        nbytes=header['nbytes'] # doesn't include 2 byte checksum
        raw_frame=fp.read(nbytes+2) # 2 bytes for the checksum

        # Checksum - This is failing.  header looks okay.
        body=np.fromstring(raw_frame[:-2],'u1') # drop the checksum here
        calc_checksum=body.astype('u4').sum() & np.uint32(0xFFFF)
        given_checksum=np.fromstring(raw_frame[-2:],'<u2')[0]
        if calc_checksum != given_checksum:
            raise Pd0ChecksumError()

        offsets_start=header_dtype.itemsize # maybe??
        offsets=np.fromstring(raw_frame[offsets_start:offsets_start+header['ntypes']*offset_dtype.itemsize],
                              offset_dtype)

        # parse the beginning of the first block - this is the fixed leader
        # (while the order of blocks is in general not guaranteed, the manual
        # does say that channelmaster firmware puts the fixed leader first)

        return header,raw_frame,offsets

    def parse_one_frame(self,fp,filename='n/a',cls=None):
        """ fp: open file-like object
        scans from the start of a frame (0x7f, {0x7f,0x79}).

        assuming everything checks out, returns a Pd0Frame subclass instance,
        and leaves fp read pointer at the end of this frame, presumably
        ready for the next frame
        """
        # initial_file_pos=fp.tell() # record where we started scanning the file

        self.scan_for_ensemble(fp)

        # The real start of the frame - 
        # file_pos=fp.tell()-2 # i.e. just before the two byte ID

        header,raw_frame,offsets=self.read_ensemble(fp)

        # This seems to be missing the block ids!
        raw=raw_frame[offsets[0]:offsets[0]+fixed_prefix_leader_dtype.itemsize]
        fixed_prefix=np.fromstring(raw,fixed_prefix_leader_dtype)

        cls=cls or self.frame_class(header,
                                    fixed_prefix['system_configuration'])
        F=cls(raw_frame,
              header=header,
              offsets=offsets,
              # initial_pos=initial_file_pos,
              # file_pos=file_pos,
              filename=filename)
        F.parse()
        
        return F

reader=Pd0Reader()
parse=reader.parse

#   Rough and Ready Island (RRI)/2013/RRI020613/RRI020613-2_0_000.PD0 had a bad size for the fixed
#     leader - 50 bytes instead of 57.  maybe it's a different device?  checksum was okay, data isn't
#     truncated.  Could be a workhorse?
#   Workhorse should be 56.  

#          fixed leader   variable leader
#  streampro: 57+2
#  riverray:  57+2
#  workhorese: 56+2
#   bad files: 50+2

# The bad files are reported as 20 deg, 4-beam Janus, down-facing, convex, 600-kHz.
# with sensor_config=1
# just padding the fixed info - 
#  ncells=30 - good
#  cell_length is 50 - very reasonable.
#  bin1_distance_cm is 96 - a bit odd


# WHOA: at least one of the test files is in BEAM coordinates.
# at least RiverRay (theoretically) doesn't need a device-specific
# transformation matrix, but still.



