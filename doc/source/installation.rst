
Installation
================

Python Prerequisites
--------------------

External dependencies for running the ADCPy include:
  1. A working Python 2.5-2.7 distribution (Python version 3 is not compatible)
  2. Specific Python packages that might not be included with the base Python distribution:
    a. Numpy version 1.7
    b. Scipy version 0.11.0
    c. Matplotlib version 1.2.0
    d. Pynmea (currently prepackaged inside ADCPy)
    e. Gdal
    f. Netcdf4

The ADCP Python code is developed and tested on Windows XP (32-bit) using the Python(x,y) python distribution (http://code.google.com/p/pythonxy/), and on Windows 7 (64-bit) using 64-bit Python 2.7 and associated libraries compiled and available at the time of writing at: http://www.lfd.uci.edu/~gohlke/pythonlibs/. It has also been tested to a lesser extent with the Anaconda 32 and 64 bit Python distributinos. Running the Spyder Python development environment is very helpful, but not necessary. 

IMPORTANT: Python version 3 is not backwards-compatible with versions 2.5-2.7.  ADCP Python tools have not been tested under Python 3 and likely don’t work.

ADCP Python does not use any platform-dependent code, and will likely run on any other system capable of supporting the above Python distribution and associated packages.  This has not been tested however.


Installation
------------


