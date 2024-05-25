"""
transect_preprocessor.py Input File
Set options in this file in order to use process adcp transects using the 
For further information on setting these options, the ADCPy Documentation 

IMPORTANT:  This options file is uses Python 2.X code convenctions, meaning:
 1) there may not be any tab characters in the file;
 2) option lines may have no leadng spaces;
 3) strings (text in quotes) should be preceded by an 'r' for maximum compatibility.
 
This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel

"""
file_list = None
file_ext = r'r.000'
file_type = r'ADCPRdiWorkhorseData'



# ADCP Data File location(s)
# -----------------------------------------------------------------------------
working_directory = r'Y:\temp\adcp_anaylsis_stations\RIO20100309_avg_test'        # or None for current directory

# Processing Options
# -----------------------------------------------------------------------------
xy_projection = r'EPSG:26910' # The text-based EPSG code describing the map projection (in Northern CA, UTM Zone 10N = 'EPSG:26910')
do_head_correct = False         # Switch for using/not using heading correction due to magnetic compass declination and errors. {True or False}
head_correct_spanning = False   # perform heading correction on all data files binned together {True or False}
mag_declination = 15          # magnetic compass declination - this value will be used to correct compass heading if head_correcting is not used {degrees E of true North, or None}
u_min_bt=0.3                  # minimum bottom track velocity for head_correct {typically 0-0.3 [m/s] or None}
hdg_bin_size=5                # bin size of heading correction {typically 5,10 [degrees]}
hdg_bin_min_samples=10        # minimum number of sample headings in a heading bin for consideration in heading correction {typically 10-50, more is safer}
sidelobe_drop=0.1             # fraction of vertical profile to drop due to sidelobe/bottom interaction {typically 0.5-0.15 [fraction]}
std_drop=3.0                  # standard deviation of velocity, above which samples are dropped from analysis {0.0=no dropping, 2.0-3.0 typically [number of standard deviations]}
std_interp=True               # perform interpolation of holes in velocity profiles left by high standard deviation removal {typically True with std_drop > 0.0}
smooth_kernel=0               # smooth velocity data using a square kernel box-filter, with side dimension =
extrap_boundaries=False        # extrapolate velocity profiles upward toward surface, and downward to the sounder-detected bottom  {True or False}
average_ens = 1            # average adjacent (in time) velocity profiles {typically 0-15 [number of adjacent velocity profiles(ensembles)]}
regrid_horiz_m = None          # horizontal grid resolution used when regridding results {resonable fraction of transect width, or None for default(2m) [m]}
regrid_vert_m = None          # vertical grid resolution used when regridding results {resonable fraction of transect depth, or None for default(0.1) [m]} 
adcp_depth = 0.244            # depth of the adcp face under the surface {[m] or None}
p1lat = 38.1619               # latitude of origin of optional transect plot line [degrees E] or None
p1lon = -121.6843             # longitude of origin of optional transect plot line [degrees N] or None
p2lat = 38.1578               # latitude of end of optional transect plot line [degrees E] or None
p2lon = -121.6781             # longitude of end of optional transect plot line [degrees N] or None

# Data Output Options
# -----------------------------------------------------------------------------
save_raw_data_to_netcdf = True          # Switch to output raw data to netCDF-CF format.  {True or False}
save_preprocessed_data_to_netcdf = True    # Switch to output results to netCDF-CF format.  {True or False}
use_netcdf_data_compression = True      # Switch to use NetCDF 4 data compression to save disk space in data and results files.  {True or False}

# Debug options
debug_stop_after_n_transects = False       # False, or number to limit return to
