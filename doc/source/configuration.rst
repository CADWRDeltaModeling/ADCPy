
Options and Configuration
=========================

Options for processing transects are specified using a python file full of key=value statements.
The options file is usually specified in the command line (with a default location), then at the beginning of a script it is dynamically loaded and parsed. The default name of the options file is 
transect_preprocessor_input.py

Preprocessing options
---------------------

Here are the transect_preprocessor_input.py options  

============================     ==============
Name                             Description
============================     ==============
working_directory                Path to the directory containing data files (WinRiver raw or ADCP Python NetCDF) for processing.   Note required ‘r’ character before first text quote.
xy_projection                    The text-based EPSG code describing the map projection (in Northern CA, UTM Zone 10N = ‘EPSG:26910’) to use for projecting ADCP profile locations onto an regular grid. 
do_head_correct                  Switch for heading correction due to magnetic compass declination and errors. {True or False}.
Headhead_correct_spanning        Switch to perform heading correction on all data files binned together (True), or on each file individually (False).  This option should be set to true whenever possible; errors in processing can occur if the sample size of headings used for correction is small. {True or False}.
mag_declination                  Magnetic compass declination; this value will be used to correct compass heading if do_head_correct is False {degrees E of true North, or None}.
u_min_bt                         Minimum bottom track velocity for use in heading correcting.  If the survey platform (boat) is moving too slowly, the GPS-based navigation heading may be invalid.  {typically 0-0.3 [m/s] or None}.
hdg_bin_size                     The size of the heading bin size of heading correction.  A value of 5 means headings will be grouped and averaged in over a range of 5 degrees.  Experimentation with this value may be required to produce a valid heading correction. {typically 5,10 [degrees]}.
hdg_bin_min_samples              Minimum number of sample headings in a heading bin for consideration in heading correction.  It is wise to use a larger number here if there is a large amount of data/number of data files, however using too large a number may exclude important data used for fitting with less data.  Experimentation with this value may be required. {typically 10-50 }.
sidelobe_drop                    The fraction of vertical profile to drop due from analysis due to sidelobe/bottom interaction. {typically 0.05-0.15 [fraction]}.
std_drop                         The calculated standard deviation from the mean velocity, above which velocity samples are dropped from analysis. {0.0=no dropping, 2.0-3.0 typically [number of standard deviations]}.
std_interp                       Switch to perform interpolation of holes in velocity profiles left by high standard deviation removal {typically True with std_drop > 0.0}.
smooth_kernel                    Remove noise from ADCP velocities through smoothing data using a square-kernel boxcar-filter. The square filter average neighboring velocities in a square pattern (kernel), with the sidelength of the square = smooth_kernel (i.e. smooth_kernel=3 specifies a 3x3 square, effectively averaging the 9 neighboring velocities). {0 for no smoothing, or odd integer between 3-9 }.
extrap_boundaries                Switch to extrapolate velocity profiles upward toward surface, and downward to the sounder-detected bottom.  {True or False}
average_ens	Specifies how many adjacent (in time) velocity profiles should be averaged together to reduce noise. {typically 0-15 [number of adjacent velocity profiles(ensembles)]}.
regrid_horiz_m                   Horizontal resolution of averaging bins {m, or None for no regridding}.
regrid_vert_m                    Vertical resolution of averaging bins {m, or None for no regridding}.
adcp_depth                       Scalar value indicating at the depth of the ADCP face underwater (positive downward from zero at surface) {m}.
p1lat, p1lon, p2lat, p2lon       Latitude/Longitude coordinates of points p1 and p2 which designate a plotline  for projection and regridding.
save_raw_data_to_netcdf          Switch to output raw data to netCDF-CF format.  {True or False}.
save_preprocessed_data_to_netcdfSwitch to output results to netCDF-CF format. {True or False}.
use_netcdf_data_compression	Switch to use NetCDF 4 data compression to save disk space in data and results files.  {True or False}.
debug_stop_after_n_transects     Limits the number of ADCP_Data objects returned to this scalar integer value.  {True or False}.
============================     ==============
