# PROCESS ADCP INPUT FILE
# Set options in this file in order to use process adcp transects using the 
# For further information on setting these options, the ADCP Python Documentation 
#
# IMPORTANT:  This options file is uses Python 2.X code convenctions, meaning:
# 1) there may not be any tab characters in the file;
# 2) option lines may have no leadng spaces;
# 3) strings (test in quotes) should be preceded by an 'r' for maximum compatibility.


# ADCP Data File location(s)
# -----------------------------------------------------------------------------

#working_directory = r'Y:\temp\ADCP_2008\NDSOS_DLADCP.VelocityData\5thRelease\GEO5thRelease\GEO20090116'

working_directory = r'C:\Delta\ADCPy\GEO20090116'
        # or None for current directory
#working_directory = r'/Volumes/Aux/temp/adcp_anaylsis_stations/RIO20100309'        # or None for current directory

#working_directory = r'Z:\temp\adcp_anaylsis_stations\WGB20090721'
#working_directory = r'Z:\temp\adcp_anaylsis_stations\WGA20090722'        # or None for current directory
#working_directory = r'Z:\temp\adcp_anaylsis_stations\TMS20090513'
#working_directory = r'Z:\temp\adcp_anaylsis_stations\JPT20080618'
#working_directory = r'Z:\temp\adcp_anaylsis_stations\MRU022510' # can't headcorrect, not enough bins
#working_directory = r'Z:\temp\adcp_anaylsis_stations\MRU012810'
#working_directory = r'Z:\temp\adcp_anaylsis_stations\MRU060408' # no nav
#working_directory = r'Z:\temp\adcp_anaylsis_stations\MRU112707' # no nav
#working_directory = r'Z:\temp\adcp_anaylsis_stations\WCI042011' # done
#working_directory = r'Z:\temp\adcp_anaylsis_stations\WCI102009' # done
#working_directory = r'Z:\temp\adcp_anaylsis_stations\WCI050608'

# Processing Options
# -----------------------------------------------------------------------------
xy_projection = r'EPSG:26910' # The text-based EPSG code describing the map projection (in Northern CA, UTM Zone 10N = 'EPSG:26910')
do_headCorrect = False         # Switch for using/not using heading correction due to magnetic compass declination and errors. {True or False}
headCorrect_spanning = False   # perform heading correction on all data files binned together {True or False}
mag_declination = 14.7          # magnetic compass declination - this value will be used to correct compass heading if head_correcting is not used {degrees E of true North, or None}
u_min_bt=0.3                  # minimum bottom track velocity for headCorrect {typically 0-0.3 [m/s] or None}
hdg_bin_size=5                # bin size of heading correction {typically 5,10 [degrees]}
hdg_bin_min_samples=10        # minimum number of sample headings in a heading bin for consideration in heading correction {typically 10-50, more is safer}
sidelobe_drop=0.1             # fraction of vertical profile to drop due to sidelobe/bottom interaction {typically 0.5-0.15 [fraction]}
std_drop=3.0                  # standard deviation of velocity, above which samples are dropped from analysis {0.0=no dropping, 2.0-3.0 typically [number of standard deviations]}
std_interp=True               # perform interpolation of holes in velocity profiles left by high standard deviation removal {typically True with std_drop > 0.0}
smooth_kernel=0               # smooth velocity data using a square kernel box-filter, with side dimension =
extrap_boundaries=True        # extrapolate velocity profiles upward toward surface, and downward to the sounder-detected bottom  {True or False}
average_ens = 1            # average adjacent (in time) velocity profiles {typically 0-15 [number of adjacent velocity profiles(ensembles)]}
regrid_horiz_m = None          # horizontal grid resolution used when regridding results {resonable fraction of transect width, or None for default(2m) [m]}
regrid_vert_m = None          # vertical grid resolution used when regridding results {resonable fraction of transect depth, or None for default(0.1) [m]} 
adcp_depth = 0.100579092       # depth of the adcp face under the surface {[m] or None}
#p1lat = 38.0527               # latitude of origin of optional transect plot line [degrees E] or None
#p1lon = -121.6943             # longitude of origin of optional transect plot line [degrees N] or None
#p2lat = 38.0505               # latitude of end of optional transect plot line [degrees E] or None
#p2lon = -121.6900             # longitude of end of optional transect plot line [degrees N] or None
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
debug_stop_after_n_transects = 16       # False, or number to limit return to