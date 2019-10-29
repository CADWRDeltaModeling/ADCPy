
import numpy as np

from . import adcpy
from . import adcpy_utilities as au

class ADCPVMTData(adcpy.ADCPData):
    """
    Read from a directory containing csv output from Velocity Mapping Toolbox (VMT)

    A single velocity dataset will have muliple csv files containing arrays of data, where the a string common to the
    dataset forms the beginning of each filename. The different dataset files end with standard descriptors:

    <dataset_string>_Depth.csv
    <dataset_string>_Distance.csv
    <dataset_string>_Easting.csv
    <dataset_string>_Northing.csv
    <dataset_string>_Timerange.csv
    <dataset_string>_U.csv
    <dataset_string>_Ux.csv
    <dataset_string>_Uy.csv
    <dataset_string>_V.csv
    <dataset_string>_W.csv
    """

    def __init__(self,VMT_base_name,epsg_code,**kwargs):
        super(ADCPVMTData,self).__init__(raw_file=None,nc_file=None,**kwargs)

        self.VMT_base_name=VMT_base_name
        self.epsg_code=epsg_code
        self.convert_from_VMT()

    def convert_from_VMT(self):
        """
        Need to populate several arrays for ADCPData
        base_data_names =  ('n_ensembles',  # time/horizontal dimension
                        'n_bins',       # along-beam (typ. vertical) dimension
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
                        'history',      # the history list contains processing evens descriptions/audit trail stored as a single
                                        # string delinieated by \n (newline) characters
                        'xy_line',      # [[x0,y0],[x1,y1]] for projection onto slice
                        )
        """
        self.northing = np.genfromtxt(self.VMT_base_name + 'Northing.csv',delimiter=',')
        self.easting = np.genfromtxt(self.VMT_base_name + 'Easting.csv',delimiter=',')
        (self.n_bins,self.n_ensembles) = np.shape(self.northing)
        self.xy = np.full((self.n_ensembles,2),np.nan)
        self.xy[:,0] = self.easting[0,:]
        self.xy[:,1] = self.northing[0,:]
        self.xy_srs = self.epsg_code
        self.lonlat_srs = self.default_lonlat_srs
        self.xy_to_lonlat()

        self.depth = np.genfromtxt(self.VMT_base_name + 'Depth.csv',delimiter=',')
        self.bin_center_elevation = -1.0*self.depth[:,0]

        self.velocity = np.full((self.n_ensembles,self.n_bins,3),np.nan)
        self.velocity[:,:,0] = np.rot90(np.loadtxt(self.VMT_base_name + 'Ux.csv',delimiter=','))
        self.velocity[:,:,1] = np.rot90(np.loadtxt(self.VMT_base_name + 'Uy.csv',delimiter=','))
        self.velocity[:,:,2] = np.rot90(np.loadtxt(self.VMT_base_name + 'W.csv',delimiter=','))

        self.references="VMT proocessed data"

