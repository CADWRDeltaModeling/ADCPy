.. ADCPy documentation master file, created by
   sphinx-quickstart on Tue Oct 07 11:54:34 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*****
ADCPy
*****  
   
Introduction
=================

ADCPy is a package much like a scripting version of the Velocity Mapping Toolkit in Matlab.
The tools aid in the analysis of data from Acoustic Doppler Current Profilers. 

ADCPy allows the user to read raw (unprocessed) data from ADCP instruments, perform a suite of processing functions and data transformations, and output summary data and related plots. By providing access to the raw ADCP velocities, ADCPy allows exacting user control over the diagnosis of water movement. Numerous built-in data transformation tools and associated utilities allow full scripting of ADCP processing, from raw data to final output plots and flow volumes. Raw ADCP data is stored as a python object class, which may be exported and imported to disk in the Network Common Data Format (Climate and Forecast Metadata Convention; NetCDF-CF). 
An abbreviated list of ADCPy functions:
 * Read native ADCP instrument binary file formats
 * Georeference/re-project ADCP measurements onto fitted or user-defined grids
 * ADCP Data processing methods:
   
   * Correct measured current velocities to account for platform/vessel speed
   * Correct measured velocities for instrument compass errors
   * Drop outliers, remove side lobe contamination, kernel smoothing
   * Combine/average ADCP ensembles from different sources
   * Extrapolation of boundaries (i.e. towards channel/sea bed, and towards ADCP face)

 * Calculate dispersion
 * Archive and read to NetCDF-CF files
 * Export velocities to comma-separated values (CSV file) for easy porting
 * Generate various surface and arrow (quiver) plots showing 2D velocity profiles, mean flow vectors, and survey geometries
 * A processing history is automatically generated and updated for ADCPy data classes, allowing tracking of processing methods



The California Department of Water Resources (DWR) commissioned the development of a tool to provide methods for ADCP transect analysis that are more customizable to different tasks and processing parameters than currently-available closed-source solutions.  These Python-based tools are designed to facilitate quality control and projection/extrapolation of ADCP surveys, conversion of ADCP transects to streamwise coordinates, re-gridding of ADCP transect profiles from vessel tracks onto a uniform grid, estimation and output of streamwise flows and velocities (NetCDF-CF format, see http://cfconventions.org/), and calculation of lateral and longitudinal dispersion coefficients (Fischer et al., 1979). 

Contents
========

.. toctree::
   :maxdepth: 2
   
   installation
   scripting
   terms
   api/modules Package Documentation
   
Key modules
===========

.. currentmodule:: adcpy

.. autosummary::
    adcpy
    adcpy_plot
    adcpy_utilities
    adcpy_recipes
    transect_average
    transect_preprocessor
    ADCPRdiWorkhorseData
    rdradcp
    
   
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

