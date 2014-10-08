""" Calculations used by the adcpy module such as smoothing, principal flow direction and averaging
This module is independent of adcpy, but is imported by it and is available as adcpy.util.  
This tools were abstracted out of other classes, either because of potential for reuse in
recipes, automated scripting or with data from outside adcpy.  They allows potentially complicated
data processing using the adcpy module to remain readable, hopefully.

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""
import numpy as np
import scipy.stats.stats as sp
import scipy.stats.morestats as ssm
import scipy.interpolate
import warnings
from osgeo import osr

try:
    import fftw3
    has_fftw = True

    def fftwn(array, nthreads=1):
        array = array.astype('complex').copy()
        outarray = array.copy()
        fft_forward = fftw3.Plan(array, outarray, direction='forward',
                flags=['estimate'], nthreads=nthreads)
        fft_forward.execute()
        return outarray

    def ifftwn(array, nthreads=1):
        array = array.astype('complex').copy()
        outarray = array.copy()
        fft_backward = fftw3.Plan(array, outarray, direction='backward',
                flags=['estimate'], nthreads=nthreads)
        fft_backward.execute()
        return outarray / np.size(array)
except ImportError:
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    has_fftw = False
# I performed some fft speed tests and found that scipy is slower than numpy
# http://code.google.com/p/agpy/source/browse/trunk/tests/test_ffts.py However,
# the speed varied on machines - YMMV.  If someone finds that scipy's fft is
# faster, we should add that as an option here... not sure how exactly

class new_object(object):
    def __init__(self,_d={},**kwargs):
        kwargs.update(_d)
        self.__dict__=kwargs


def convolvend(array, kernel, boundary='fill', fill_value=0,
        crop=True, return_fft=False, fftshift=True, fft_pad=True,
        psf_pad=False, interpolate_nan=False, quiet=False,
        ignore_edge_zeros=False, min_wt=0.0, normalize_kernel=False,
        use_numpy_fft=not has_fftw, nthreads=1):
    """
    Source:
    http://agpy.googlecode.com/svn/trunk/AG_fft_tools/convolve_nd.py
    On: 1/31/2013

    Convolve an ndarray with an nd-kernel.  Returns a convolved image with shape =
    array.shape.  Assumes image & kernel are centered.

    Parameters
    ----------
    array: `numpy.ndarray`
          Array to be convolved with *kernel*
    kernel: `numpy.ndarray`
          Will be normalized if *normalize_kernel* is set.  Assumed to be
          centered (i.e., shifts may result if your kernel is asymmetric)

    Options
    -------
    boundary: str, optional
        A flag indicating how to handle boundaries:
            * 'fill' : set values outside the array boundary to fill_value
                       (default)
            * 'wrap' : periodic boundary
    interpolate_nan: bool
        attempts to re-weight assuming NAN values are meant to be ignored, not
        treated as zero.  If this is off, all NaN values will be treated as
        zero.
    ignore_edge_zeros: bool
        Ignore the zero-pad-created zeros.  This will effectively decrease
        the kernel area on the edges but will not re-normalize the kernel.
        This parameter may result in 'edge-brightening' effects if you're using
        a normalized kernel
    min_wt: float
        If ignoring NANs/zeros, force all grid points with a weight less than
        this value to NAN (the weight of a grid point with *no* ignored
        neighbors is 1.0).  
        If `min_wt` == 0.0, then all zero-weight points will be set to zero
        instead of NAN (which they would be otherwise, because 1/0 = nan).
        See the examples below
    normalize_kernel: function or boolean
        if specified, function to divide kernel by to normalize it.  e.g.,
        normalize_kernel=np.sum means that kernel will be modified to be:
        kernel = kernel / np.sum(kernel).  If True, defaults to
        normalize_kernel = np.sum

    Advanced options
    ----------------
    fft_pad: bool
        Default on.  Zero-pad image to the nearest 2^n
    psf_pad: bool
        Default off.  Zero-pad image to be at least the sum of the image sizes
        (in order to avoid edge-wrapping when smoothing)
    crop: bool
        Default on.  Return an image of the size of the largest input image.
        If the images are asymmetric in opposite directions, will return the
        largest image in both directions.
        For example, if an input image has shape [100,3] but a kernel with shape
      [6,6] is used, the output will be [100,6].
    return_fft: bool
        Return the fft(image)*fft(kernel) instead of the convolution (which is
        ifft(fft(image)*fft(kernel))).  Useful for making PSDs.
    fftshift: bool
        If return_fft on, will shift & crop image to appropriate dimensions
    nthreads: int
        if fftw3 is installed, can specify the number of threads to allow FFTs
        to use.  Probably only helpful for large arrays
    use_numpy_fft: bool
        Force the code to use the numpy FFTs instead of FFTW even if FFTW is
        installed

    Returns
    -------
    default: `array` convolved with `kernel`
    if return_fft: fft(`array`) * fft(`kernel`)
      * if fftshift: Determines whether the fft will be shifted before
        returning
    if not(`crop`) : Returns the image, but with the fft-padded size
        instead of the input size

    Examples
    --------
    >>> convolvend([1,0,3],[1,1,1])
    array([ 1.,  4.,  3.])

    >>> convolvend([1,np.nan,3],[1,1,1],quiet=True)
    array([ 1.,  4.,  3.])

    >>> convolvend([1,0,3],[0,1,0])
    array([ 1.,  0.,  3.])

    >>> convolvend([1,2,3],[1])
    array([ 1.,  2.,  3.])

    >>> convolvend([1,np.nan,3],[0,1,0], interpolate_nan=True)
    array([ 1.,  0.,  3.])

    >>> convolvend([1,np.nan,3],[0,1,0], interpolate_nan=True, min_wt=1e-8)
    array([  1.,  nan,   3.])

    >>> convolvend([1,np.nan,3],[1,1,1], interpolate_nan=True)
    array([ 1.,  4.,  3.])

    >>> convolvend([1,np.nan,3],[1,1,1], interpolate_nan=True, normalize_kernel=True, ignore_edge_zeros=True)
    array([ 1.,  2.,  3.])

    """


    # Checking copied from convolve.py - however, since FFTs have real &
    # complex components, we change the types.  Only the real part will be
    # returned!
    # Check that the arguments are lists or Numpy arrays
    array = np.asarray(array, dtype=np.complex)
    kernel = np.asarray(kernel, dtype=np.complex)

    # Check that the number of dimensions is compatible
    if array.ndim != kernel.ndim:
        raise Exception('array and kernel have differing number of'
                        'dimensions')

    # store the dtype for conversion back later
    array_dtype = array.dtype
    # turn the arrays into 'complex' arrays
    if array.dtype.kind != 'c':
        array = array.astype(np.complex)
    if kernel.dtype.kind != 'c':
        kernel = kernel.astype(np.complex)

    # mask catching - masks must be turned into NaNs for use later
    if np.ma.is_masked(array):
        mask = array.mask
        array = np.array(array)
        array[mask] = np.nan
    if np.ma.is_masked(kernel):
        mask = kernel.mask
        kernel = np.array(kernel)
        kernel[mask] = np.nan

    # replace fftn if has_fftw so that nthreads can be passed
    global fftn, ifftn
    if has_fftw and not use_numpy_fft:
        def fftn(*args, **kwargs):
            return fftwn(*args, nthreads=nthreads, **kwargs)

        def ifftn(*args, **kwargs):
            return ifftwn(*args, nthreads=nthreads, **kwargs)
    elif use_numpy_fft:
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn


    # NAN catching
    nanmaskarray = (array != array)
    array[nanmaskarray] = 0
    nanmaskkernel = (kernel != kernel)
    kernel[nanmaskkernel] = 0
    if ((nanmaskarray.sum() > 0 or nanmaskkernel.sum() > 0) and not interpolate_nan
            and not quiet):
        warnings.warn("NOT ignoring nan values even though they are present" +
                " (they are treated as 0)")

    if normalize_kernel is True:
        kernel = kernel / kernel.sum()
        kernel_is_normalized = True
    elif normalize_kernel:
        # try this.  If a function is not passed, the code will just crash... I
        # think type checking would be better but PEPs say otherwise...
        kernel = kernel / normalize_kernel(kernel)
        kernel_is_normalized = True
    else:
        if np.abs(kernel.sum() - 1) < 1e-8:
            kernel_is_normalized = True
        else:
            kernel_is_normalized = False


    if boundary is None:
        WARNING = ("The convolvend version of boundary=None is equivalent" +
                " to the convolve boundary='fill'.  There is no FFT " +
                " equivalent to convolve's zero-if-kernel-leaves-boundary" )
        warnings.warn(WARNING)
        psf_pad = True
    elif boundary == 'fill':
        # create a boundary region at least as large as the kernel
        psf_pad = True
    elif boundary == 'wrap':
        psf_pad = False
        fft_pad = False
        fill_value = 0 # force zero; it should not be used
    elif boundary == 'extend':
        raise NotImplementedError("The 'extend' option is not implemented " +
                "for fft-based convolution")

    arrayshape = array.shape
    kernshape = kernel.shape
    ndim = len(array.shape)
    if ndim != len(kernshape):
        raise ValueError("Image and kernel must " +
            "have same number of dimensions")
    # find ideal size (power of 2) for fft.
    # Can add shapes because they are tuples
    if fft_pad:
        if psf_pad:
            # add the dimensions and then take the max (bigger)
            fsize = 2**np.ceil(np.log2(
                np.max(np.array(arrayshape) + np.array(kernshape))))
        else:
            # add the shape lists (max of a list of length 4) (smaller)
            # also makes the shapes square
            fsize = 2**np.ceil(np.log2(np.max(arrayshape+kernshape)))
        newshape = np.array([fsize for ii in range(ndim)])
    else:
        if psf_pad:
            # just add the biggest dimensions
            newshape = np.array(arrayshape)+np.array(kernshape)
        else:
            newshape = np.array([np.max([imsh, kernsh])
                for imsh, kernsh in zip(arrayshape, kernshape)])


    # separate each dimension by the padding size...  this is to determine the
    # appropriate slice size to get back to the input dimensions
    arrayslices = []
    kernslices = []
    for ii, (newdimsize, arraydimsize, kerndimsize) in enumerate(zip(newshape, arrayshape, kernshape)):
        center = newdimsize - (newdimsize+1)//2
        arrayslices += [slice(center - arraydimsize//2,
            center + (arraydimsize+1)//2)]
        kernslices += [slice(center - kerndimsize//2,
            center + (kerndimsize+1)//2)]

    bigarray = np.ones(newshape, dtype=np.complex128) * fill_value
    bigkernel = np.zeros(newshape, dtype=np.complex128)
    bigarray[arrayslices] = array
    bigkernel[kernslices] = kernel
    arrayfft = fftn(bigarray)
    # need to shift the kernel so that, e.g., [0,0,1,0] -> [1,0,0,0] = unity
    kernfft = fftn(np.fft.ifftshift(bigkernel))
    fftmult = arrayfft*kernfft
    if (interpolate_nan or ignore_edge_zeros) and kernel_is_normalized:
        if ignore_edge_zeros:
            bigimwt = np.zeros(newshape, dtype=np.complex128)
        else:
            bigimwt = np.ones(newshape, dtype=np.complex128)
        bigimwt[arrayslices] = 1.0-nanmaskarray*interpolate_nan
        wtfft = fftn(bigimwt)
        # I think this one HAS to be normalized (i.e., the weights can't be
        # computed with a non-normalized kernel)
        wtfftmult = wtfft*kernfft/kernel.sum()
        wtsm = ifftn(wtfftmult)
        # need to re-zero weights outside of the image (if it is padded, we
        # still don't weight those regions)
        bigimwt[arrayslices] = wtsm.real[arrayslices]
        # curiously, at the floating-point limit, can get slightly negative numbers
        # they break the min_wt=0 "flag" and must therefore be removed
        bigimwt[bigimwt<0] = 0
    else:
        bigimwt = 1


    if np.isnan(fftmult).any():
        # this check should be unnecessary; call it an insanity check
        raise ValueError("Encountered NaNs in convolve.  This is disallowed.")

    # restore nans in original image (they were modified inplace earlier)
    # We don't have to worry about masked arrays - if input was masked, it was
    # copied
    array[nanmaskarray] = np.nan
    kernel[nanmaskkernel] = np.nan

    if return_fft:
        if fftshift: # default on
            if crop:
                return np.fft.fftshift(fftmult)[arrayslices]
            else:
                return np.fft.fftshift(fftmult)
        else:
            return fftmult

    if interpolate_nan or ignore_edge_zeros:
        rifft = (ifftn(fftmult)) / bigimwt
        if not np.isscalar(bigimwt):
            rifft[bigimwt < min_wt] = np.nan
            if min_wt == 0.0:
                rifft[bigimwt == 0.0] = 0.0
    else:
        rifft = (ifftn(fftmult))

    if crop:
        result = rifft[arrayslices].real
        return result
    else:
        return rifft.real

#import pytest
#import itertools
#params = list(itertools.product((True,False),(True,False),(True,False)))
#@pytest.mark.parametrize(('psf_pad','use_numpy_fft','force_ignore_zeros_off'),params)
#def test_3d(psf_pad, use_numpy_fft, force_ignore_zeros_off, debug=False, tolerance=1e-17):
#    array = np.zeros([32,32,32])
#    array[15,15,15]=1
#    array[15,0,15]=1
#    kern = np.zeros([32,32,32])
#    kern[14:19,14:19,14:19] = 1
#
#    conv1 = convolvend(array, kern, psf_pad=psf_pad, force_ignore_zeros_off=force_ignore_zeros_off, debug=debug)
#
#    print "psf_pad=%s use_numpy=%s force_ignore_zeros_off=%s" % (psf_pad, use_numpy_fft, force_ignore_zeros_off)
#    print "side,center: %g,%g" % (conv1[15,0,15],conv1[15,15,15])
#    if force_ignore_zeros_off or not psf_pad:
#        assert(np.abs(conv1[15,0,15] - 1./125.) < tolerance)
#        assert(np.abs(conv1[15,1,15] - 1./125.) < tolerance)
#        assert(np.abs(conv1[15,15,15] - 1./125.) < tolerance)
#    else:
#        assert(np.abs(conv1[15,0,15] - 1./75.) < tolerance)
#        assert(np.abs(conv1[15,1,15] - 1./100.) < tolerance)
#        assert(np.abs(conv1[15,15,15] - 1./125.) < tolerance) 
 

def get_axis_num_from_str(axes_string):
    """
    u,v,w correspond to 0,1,2 in the trailing axis of adcpy velocity arrays.
    This method returns a list of 0,1, and 2s corresponding to an input 
    string composed u,v, and ws.
    Inputs:
        axes_string = string composed of U V or W only [str]
    Returns:
        ax_list = python list containing the integers 0,1, or 2 
    """    
    if type(axes_string) is not str:
        ValueError("axes_string argument must be a string")
        raise
    ax_list = []
    for char in axes_string:
        if char in 'UVW': char = char.lower()
        if char not in 'uvw':
            ValueError("axes_string letters must be u,v, or w only")
            raise
        if char == 'u':
            ax_list.append(0)
        elif char == 'v':
            ax_list.append(1)
        elif char == 'w':
            ax_list.append(2)
    return ax_list
            
 
def fit_headerror(headin,errin):
    """
    Least-squares harmonic fit of heading error
    Inputs:
        headin   = headings (binned) --> assumes units of degrees
        errin    = heading errors at each headin (binned)
    Returns:
        coeff = harmonic fit coefficients [y0 a b] where y0 is the offset, 
          a is the coefficient for the cosine and b is the coefficient for the
          sine.
        errfit = fitted error by heading
    """
    # from Dave - this code used to do arbitrary numbers of
    # periods, but for ADCP stuff just fit one period.
    per=np.array([360.0]) # the set of periods to fit
    valid = ~np.isnan(headin+errin)
    
    yy = errin[valid]
    tt = headin[valid]
    nper = len(per) 

    # the angular frequencies to fit - for us, just fit the first fourier mode
    si = 2*np.pi/per  # 

    M  = np.zeros( (1+2*nper,1+2*nper), np.float64) # 1+2*np is 1 DC component, and np sin/cos pairs
    x  = np.zeros(1+2*nper,np.float64) 
   
    for ic in range(1,2*nper+2):
        if ic == 1: # DC component
            x[ic-1] = sum(yy)
            for ir in range(1,2+2*nper):
                if ir == 1:
                    M[ic-1,ir-1] = len(tt)
                elif ir%2 == 1:
                    sr = si[(ir-1)/2-1] # HERE - need to figure out what si really is.
                    M[ic-1,ir-1] = sum(np.sin(sr*tt)) 
                elif ir%2 == 0:
                    sr = si[ir/2-1] # HERE - same
                    M[ic-1,ir-1] = sum(np.cos(sr*tt)) 
        elif ic % 2 == 1:
            sc = si[ (ic-1)/2 -1] # HERE
            x[ic-1] = sum(yy*np.sin(sc*tt)) # 
            for ir in range(1,2+2*nper):
                if ir == 1:
                    M[ic-1,ir-1] = sum(np.sin(sc*tt)) 
                elif ir%2 == 1:
                    sr = si[(ir-1)/2-1]
                    M[ic-1,ir-1] = sum(np.sin(sc*tt) * np.sin(sr*tt)) 
                elif ir%2 == 0:
                    sr = si[ir/2-1]
                    M[ic-1,ir-1] = sum(np.sin(sc*tt) * np.cos(sr*tt)) 
        elif ic%2 == 0:
            sc = si[ic/2-1] # I think ic={1,2} should map to si[1]
            x[ic-1] = sum(yy*np.cos(sc*tt))
            for ir in range(1,2+2*nper):
                if ir == 1:
                    M[ic-1,ir-1] = sum(np.cos(sc*tt)) 
                elif ir%2 == 1:
                    sr = si[(ir-1)/2-1]
                    M[ic-1,ir-1] = sum(np.cos(sc*tt) * np.sin(sr*tt))
                elif ir%2 == 0:
                    sr = si[ir/2-1]
                    M[ic-1,ir-1] = sum(np.cos(sc*tt) * np.cos(sr*tt)) 

    coeff = np.linalg.solve(M,x)
    errfit = np.zeros(len(errin),np.float64) 
    errfit = errfit + coeff[0]
    for ic in range(2,2*nper+2):
        if ic%2 == 1: 
            sc = si[ (ic-1)/2 -1]
            errfit = errfit + coeff[ic-1]*np.sin(sc*headin) 
        elif ic%2==0:
            sc = si[ ic/2 -1]
            errfit = errfit + coeff[ic-1]*np.cos(sc*headin)

    return (coeff,errfit)


def createLine(v1,v2):
    """
    CREATELINE create a line with various inputs - adapted from MATLAB.

    Line is represented in a parametric form : [x0 y0 dx dy]
    x = x0 + t*dx
    y = y0 + t*dy;

    l = CREATELINE(p1, p2) return the line going through the two given
    points.

    TODO :
    use a 5th parameter, to represent orientation of line. It can be used
    for rays to make difference between rays ]-Inf 0] and [0 Inf[.
    Also add support for cartesian line creation (solve ambiguity for
    direction).

    ---------
    author : David Legland 
    INRA - TPV URPOI - BIA IMASTE
    created the 31/10/2003.

    HISTORY :
    18/02/2004 : add more possibilities to create lines (4 parameters,
    all param in a single tab, and point + dx + dy.
    Also add support for creation of arrays of lines.

    NOTE : A line can also be represented with a 1*5 array : 
    [x0 y0 dx dy t].
    whith 't' being one of the following : 
    - t=0 : line is a singleton (x0,y0)
    - t=1 : line is an edge segment, between points (x0,y0) and (x0+dx,
    y0+dy).
    - t=Inf : line is a Ray, originated from (x0,y0) and going to infinity
    in the direction(dx,dy).
    - t=-Inf : line is a Ray, originated from (x0,y0) and going to infinity
    in the direction(-dx,-dy).
    - t=NaN : line is a real straight line, and contains all points
    verifying the above equation.
    This seems us a convenient way to represent uniformly all kind of lines
    (including edges, rays, and even point).

    NOTE2 : Any line object can be represented using a 1x6 array :
    [x0 y0 dx dy t0 t1]
    the first 4 parameters define the supporting line,
    t0 represent the position of the first point on the line, 
    and t1 the position of the last point.
    * for edges : t0 = 0, and t1=1
    * for straight lines : t0 = -inf, t1=inf
    * for rays : t0=0, t1=inf (or t0=-inf,t1=0 for inverted ray).
    I propose to call these objects 'lineArc'
    """
    
    if len(v1)==2 and len(v2)==2:
        #first input parameter is first point, and second input is the
        #second point.
        line = (v1[0], v1[1], v2[0]-v1[0], v2[1]-v1[1])   
    else:
        # error
        print 'createLine argument error: Please enter a pair of x-y points(as lists)'

    return line


def linePosition(point, line):
    """
    LINEPOSITION return position of a point on a line
 
    L = LINEPOSITION(POINT, LINE)
    compute position of point POINT on the line LINE, relative to origin
    point and direction vector of the line.
    LINE has the form [x0 y0 dx dy],
    POINT has the form [x y], and is assumed to belong to line.
 
    L = LINEPOSITION(POINT, LINES)
    if LINES is an array of NL lines, return NL positions, corresponding to
    each line.
 
    L = LINEPOSITION(POINTS, LINE)
    if POINTS is an array of NP points, return NP positions, corresponding
    to each point.
 
    L = LINEPOSITION(POINTS, LINES)
    if POINTS is an array of NP points and LINES is an array of NL lines,
    return an array of [NP NL] position, corresponding to each couple
    point-line.
 
    see createLine for more details on line representation.
 
    ---------
 
    author : David Legland 
    INRA - TPV URPOI - BIA IMASTE
    created the 25/05/2004. 

    HISTORY :
    07/07/2005 : manage multiple input 
    """

    Nl = len(line.shape)
    if Nl is not 1:
        Nl, cl = line.shape  #Nl = size(line, 1);
    Np = len(point.shape)
    if Np is not 1:
        Np, cp = point.shape  #Np = size(point, 1);
    
    line_local = np.copy(line)
    point_local = np.copy(point)
        
    if Nl is 1 and Np > 1:

        line_local = np.tile(line,(Np,1))

    elif Np is 1 and Nl > 1:
        
        point_local = np.tile(point,(Nl,1))       

    try:        

        dxl = line_local[...,2] # dxl = line(:, 3);
        dyl = line_local[...,3] # dyl = line(:, 4);
        dxp = point_local[...,0] - line_local[...,0] # dxp = point(:, 1) - line(:, 1);
        dyp = point_local[...,1] - line_local[...,1] # dyp = point(:, 2) - line(:, 2);
        
    except:
              
       print 'linePosition: line and point must be equal or singular - this is probably the error.'
       raise
    
    #print 'dxl,dyl',dxl,dyl
    #print 'dxp,dyp',dxp,dyp
    #d = (dxp.*dxl + dyp.*dyl)./(dxl.*dxl+dyl.*dyl);
    return (dxp*dxl + dyp*dyl)/(dxl*dxl+dyl*dyl)


def meanangle(inangle,dim=0,sens=1e-12):
    """
    MEANANGLE will calculate the mean of a set of angles (in degrees) based
    on polar considerations.
    
    Usage: [out] = meanangle(in,dim)
    
    in is a vector or matrix of angles (in degrees)
    out is the mean of these angles along the dimension dim
    
    If dim is not specified, the first non-singleton dimension is used.
    
    A sensitivity factor is used to determine oppositeness, and is how close
    the mean of the complex representations of the angles can be to zero
    before being called zero.  For nearly all cases, this parameter is fine
    at its default (1e-12), but it can be readjusted as a third parameter if
    necessary:
    
    [out] = meanangle(in,dim,sensitivity)
    
    Written by J.A. Dunne, 10-20-05
    """  
    ind = sum(np.shape(inangle))
    if ind == 1 or np.shape(inangle) :
        #This is a scalar
        print 'Scalar input encountered, aborting'
        out = inangle
        return out
    if dim > ind:
        print 'Dimension requested is greater than dimension of input angles, aborting.'
        out = inangle
        return out

    in1 = inangle * np.pi/180

    in1 = np.exp(1j*in1)
    mid = np.mean(in1,dim)
    out = np.arctan2(np.imag(mid),np.real(mid))*180/np.pi    
    
    #ii = abs(mid)<sens
    #out[ii] = np.nan
    return out


def princax(w):
    """
    PRINCAX Principal axis, rotation angle, principal ellipse
    
    [theta,maj,min,wr]=princax(w)
    
    Input: w = complex vector time series (u+i*v)
    
    Output: theta = angle of maximum variance, math notation (east == 0, north=90)
    maj = major axis of principal ellipse
    min = minor axis of principal ellipse
    wr = rotated time series, where real(wr) is aligned with
    the major axis.
    
    For derivation, see Emery and Thompson, "Data Analysis Methods
    in Oceanography", 1998, Pergamon, pages 325-327. ISBN 0 08 0314341
    ###################################################################
    Version 1.0 (12/4/1996) Rich Signell (rsignell@usgs.gov)
    Version 1.1 (4/21/1999) Rich Signell (rsignell@usgs.gov)
    fixed bug that sometimes caused the imaginary part
    of the rotated time series to be aligned with major axis.
    Also simplified the code.
    Version 1.2 (3/1/2000) Rich Signell (rsignell@usgs.gov)
    Simplified maj and min axis computations and added reference
    to Emery and Thompson book
    conveted to python - B Saenz 2/1/2013 
    ###################################################################
    """
    # use only the good (finite) points
    ind = np.isfinite(w) #ind=find(isfinite(w));
    wr = w  #wr=w;
    w_work = w[ind] #w=w(ind);
    
    # find covariance matrix    
    cv_data = np.array([np.real(w_work), np.imag(w_work)]) # arrange data
    cv=np.cov(cv_data) #cv=cov([real(w(:)) imag(w(:))]);
    
    # find direction of maximum variance
    theta = 0.5*np.arctan2(2.0*cv[1,0],cv[0,0]-cv[1,1]) #theta=0.5*atan2(2.*cv(2,1),(cv(1,1)-cv(2,2)) );
    
    # find major and minor axis amplitudes
    
    term1 = cv[0,0]+cv[1,1] #term1=(cv(1,1)+cv(2,2);
    term2 = np.sqrt((cv[0,0]-cv[1,1])**2 + 4.0*cv[1,0]**2) #term2=sqrt((cv(1,1)-cv(2,2)).^2 + 4.*cv(2,1).^2);
    maj1 = np.sqrt(0.5*(term1+term2))  #maj=sqrt(.5*(term1+term2));
    min1 = np.sqrt(0.5*(term1-term2))  #min=sqrt(.5*(term1-term2));
    
    # rotate into principal ellipse orientation
    wr[ind] = w_work*np.exp(-1j*theta)  #wr(ind)=w.*exp(-i*theta);
    #theta=theta*180./np.pi;
    
    #return (theta,maj1,min1,wr)
    return theta


def get_eof(x):
    """
    Finds the empirical orthogonal function of the 2D numpy array (matrix) x.
    Inputs:
        x = 2D numpy array
    Returns:

    """

    rows, cols = np.shape(x)
    #cov = np.dot(np.mat(x) , np.mat(x).T) / cols
    cov = x.T.dot(x) / rows
    U, S, V = np.linalg.svd(cov) #, full_matrices=True)
    B = U.T
    al = x.dot(B.T)
    
    return (B, S, al)


def fillProParab(u,z1,depth1,bbc=0):
    """
    Extrapolates the velocity profile to the bed and to the top value of zin 
    using a parabolic fit..
    Inputs:
        u = velocity profile that extends form the surface to the bed [1D or 2D numpy array]
        z1 = -H:0, cell center depths, corresponding to rightmost axis of u
        depth1 = the depth of the flow, corresponding to the left-most axis of u, or singular
           if u is 1D.
        bbc = bottom boundary condition, where
          0 ==> U(-H)=0
          1 ==> dSdz(-H)=0
    Returns:
        unew = velocity profile with nan's at top and bottom replaced with 
            extrapolated values

    """

    # depth is negative ?
    depth = depth1
    if sp.nanmean(depth) < 0:
        depth=-depth
    
    # find dimensions
    if len(np.shape(depth)) > 1:
        nt = len(depth)
        aa = z1.shape
        nz = aa[np.logical_and(aa!=nt,aa>1)]
    elif (len(z1.shape)) == 1:
        nz = len(z1)
        aa = u.shape
        nt = aa[aa!=nz]
    else:        
        nt,nz = u.shape

    # flush out z if not equal in shape to u
    z=z1
    if len(z.shape) == 1:
        try:
            z = z.T
        except:
            z = np.array([z]).T
        z = np.tile(z,(nt,1))
    
    
    unew=np.nan*u
    
    for kk in range(nt):
        d=depth[kk]
        uin=u[kk,:]
        zin=z[kk,:]
        vf=uin;
        a=np.nonzero(~np.isnan(uin))[0];
        b=np.nonzero(np.isnan(uin))[0];
        
        if len(a) < 3:
    
            print 'fillProParab: insufficient data in profile %i'%kk
    
        else:

            # find internal nans and intepolate linearly
            if len(np.nonzero(uin[a[0]:a[-1]])[0]) > 0:
                interp_nans_1d(uin[a[0]:a[-1]])

            # near surface --> dudz=0 at z=0
            jj = np.nonzero(np.greater(zin, np.max(zin[a])))
            if len(jj[0]) > 0:
                
                if np.greater( zin[a[-1]], zin[a[0]] ):           
                    u0 = np.mean(uin[a[-3]:a[-1]])
                    z0 = np.mean(zin[a[-3]:a[-1]])
                    u0z = (uin[a[-3]] - uin[a[-1]]) / (zin[a[-3]]-zin[a[-1]])
                else: 
                    u0 = np.mean(uin[a[0]:a[2]])
                    z0 = np.mean(zin[a[0]:a[2]])
                    u0z = (uin[a[0]] - uin[a[2]]) / (zin[a[0]] - zin[a[2]])
    
                aa = u0z / (2.0*z0)
                cc = u0 - aa*(z0**2)
                vf[jj]=aa*zin[jj]**2 + cc
            
            # near bot --> u=0 at z=-h
            jj = np.nonzero(np.logical_and( np.less(zin, np.min(zin[a])) ,
                                            np.greater(zin, -1.0*d) ))      
            if len(jj[0]) > 0:
    
                if np.greater( zin[a[-1]], zin[a[0]] ):           
                    u0 = np.mean(uin[a[0]:a[2]])
                    z0 = np.mean(zin[a[0]:a[2]])
                    u0z = (uin[a[0]] - uin[a[2]]) / (zin[a[0]] - zin[a[2]])
                else:
                    u0 = np.mean(uin[a[-3]:a[-1]])
                    z0 = np.mean(zin[a[-3]:a[-1]])
                    u0z = (uin[a[-3]] - uin[a[-1]]) / (zin[a[-3]]-zin[a[-1]])
    
                if bbc==0:                
                    aa = -u0/(z0+d)**2 + u0z/(z0+d)
                    bb = u0z-2.0*aa*z0
                    cc = bb*d-aa*d**2
                elif bbc==1:
                    aa = u0z/(2.0*(z0+d))
                    bb = 2.0*aa*d
                    cc = u0 - 2.0*aa*z0**2 - bb*z0
    
                vf[jj]=aa*(zin[jj]**2) + bb*zin[jj] + cc 
                
        unew[kk,:] = vf
    
    return unew

def calcKxKy(vU,vV,dd,z,depth):
    """
    Calculates dispersion coeffcients according to Fischer et al. 1979
    Inputs:
        vU(profiles,depths) = U direction velocities
        vV(profiles,depths) = V direction velocities
        dd(profiles = distance between profiles
        z(depths) = velocty bin depths
        depth(profiles) = profile bottom depth
    Returns:
        ustbar = 
        Kx_3i = horizontal dispersion coefficients
        Ky_3i = lateral dispersion coefficients
        
    """
    ############ calc Ky --> transverse mixing #################
    Ubar = sp.nanmean(np.reshape(vU,(np.size(vU),1)))
    Vbar = sp.nanmean(np.reshape(vV,(np.size(vV),1)))
    
    nx = np.size(depth)
    nz = np.size(z)
    #Ubar = sp.nanmean(vU,1)        # Ubar(n) = nanmean(A(n).uuex(:));
    #Vbar = sp.nanmean(vV,1)        # Vbar(n) = nanmean(A(n).vvex(:));
    
    bb = np.max(dd) - np.min(dd)      # bb(n) = max(A(n).dd)-min(A(n).dd);
    # cross-sect avg depth (as in Deng) --> thalweg instead?
    #Hbar = sp.nanmean(self.bt_depth) # 
    
    ### calc ustar
    # pick a vel from a constant ht above bed --> choose 2 m  
#        zz = ( -np.ones((self.n_bins))*self.bt_depth - 
#               self.bin_center_elevation*np.ones((self.n_ensembles)) )
    d1 = np.squeeze(depth)
    depths = np.array([d1]).T * np.ones((1,nz))
    bins = np.array([z])*(np.ones((nx,1)))
    zz = -depths + bins
   
    #zz = ( -np.ones((self.n_grid_z))*self.depth - 
    #       self.grid_z*(np.ones((self.n_grid_xy,1)).T) )

    
    zztmp = np.copy(zz)    
    zztmp[np.greater(zztmp, 2)] = np.nan
    #jnk,ii = np.max(np.nonzero(~np.isnan(zztmp)))
    ii = np.argmax(~np.isnan(zztmp),axis=1)
    U2m = np.zeros(len(dd))
    for i in np.arange(np.min(ii),np.max(ii)):
        nn = np.nonzero(np.equal(ii,i))
        U2m[nn] = vU[nn,i]
    
    # calc ustar: ustar^2 =  Cd*U^2
    Cd = 0.003
    ustar = np.sqrt(Cd*U2m**2)
    ustbar = sp.nanmean(ustar)
    U2mbar = sp.nanmean(U2m)
    
    # Ky - just 1 lateral sections
    vpr = sp.nanmean(vV)
    vpr = vpr-sp.nanmean(vpr)
    kwet = np.nonzero(~np.isnan(vpr))
    vpr = vpr[kwet]
    nzgw = np.size(kwet)
    #dzg = np.abs(self.bin_center_elevation[1]-self.bin_center_elevation[0])
    dzg = np.abs(z[1]-z[0])
    zsec = dzg*np.arange(0,nzgw)
    hsec = np.max(zsec)+dzg/2           
    ustsec = ustbar
                    
    # vvvvv--------- choose Kz here: assume parabolic profile
    kap=0.4
    Kzg = kap*ustsec*hsec*(zsec/hsec)*(1-zsec/hsec)
        
    # ^^^^^--------- choose Kz here
    Kyt = 0.15*ustsec*hsec # lateral turbulent diffusivity

    # Ky: fischer's (1967) triple integral, also Eqn 5.16 in Fischer et al 1979:
    c1 = np.zeros(np.size(kwet)) 
    c2 = c1
    c3 = c1 #  terms   
    for j in range(1,nzgw):
        c1[j] = vpr[j]*(zsec[j]-zsec[j-1])

    c1 = np.cumsum(c1)
    for j in range(1,nzgw):
        c2[j] = c1[j]/Kzg[j]*(zsec[j]-zsec[j-1])

    c2 = np.cumsum(c2)
    for j in range(1,nzgw):
        c3[j] = vpr[j]*c2[j]*(zsec[j]-zsec[j-1])

    Ky_3i = -1.0*(np.sum(c3)/hsec)+Kyt            
    
    # %%%%%%% calc Kx
    zwet = -3                                  # depth to deep enough to include in xsect
    iwet = np.nonzero(np.less(d1,zwet)) # wet (and moderately deep) columns    
    nygw = np.size(iwet)                        # no. wet cells    
    upr = sp.nanmean(vU,1)-Ubar                # depth avg - xsect mean
    #bg = dd[iwet[nygw-1]] - dd[iwet[0]]           # wet width
    dyg = abs(dd[1] - dd[0])  
    #alph = dyg/bg*np.ones(len(iwet))           # fractional width (const in this case)
    uprwet = upr[iwet]
    uprwet[np.isnan(uprwet)] = 0.0
    hwet = -d1[iwet] 
    hwet[np.isnan(hwet)] = 0.0
    ygwet = dd[iwet]
    
    # vvvvv--------- choose Ky here -----------
    Kyg = Ky_3i*np.ones(np.size(iwet))   
    # ^^^^^--------- choose Ky here -----------

    # Kx: fischer's triple integral: -1/A int(0toB) u'(y)h(y)dy int(0toy)1/(D_yh(y'))dy' int(0toy'')u'(y'')h(y'')dy'' 
    # where u'(y)=u(y)-Ubar, u(y) is depth avg vel, Ubar=xsect avg
    
    dAg = dyg*dzg
    Ag = dAg*np.sum(np.sum(~np.isnan(vU)))
    c1 = np.zeros(np.size(iwet))
    c2 = c1
    c3 = c1  # terms   
    for j in range(1,nygw):
        c1[j] = uprwet[j]*hwet[j]*np.abs(ygwet[j]-ygwet[j-1])

    c1 = np.cumsum(c1)
    for j in range(1,nygw):
        c2[j] = c1[j]/(Kyg[j]*hwet[j])*np.abs(ygwet[j]-ygwet[j-1])

    c2 = np.cumsum(c2)
    for j in range(1,nygw):
        c3[j] = uprwet[j]*hwet[j]*c2[j]*np.abs(ygwet[j]-ygwet[j-1])

    Kx_3i = -1.0*(np.sum(c3)/Ag)    

    return (ustbar,Kx_3i,Ky_3i)


def interp_nans_1d(data):
    """
    Linearly interpolates interior NaN values in a numpy array
    Inputs:
        data = 1D numpy array witn NaN values
    Returns:
        data = same dimension numpy array with NaN replaced by interpolated values
        
    """   
    # Create a boolean array indicating where the nans are
    bad_indexes = np.isnan(data)
    # Create a boolean array indicating where the good values area
    good_indexes = np.logical_not(bad_indexes)
    # A restricted version of the original data excluding the nans
    good_data = data[good_indexes]
    # Run all the bad indexes through interpolation
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    # Replace the original data with the interpolated values.
    data[bad_indexes] = interpolated
    
    return data


def points_to_xy(ll_points,xy_srs,ll_srs='WGS84'):
    """
    Project geographic coordinates ll_points (with projection ll_srs) to 
    projection xy_srs.
    Inputs:
        ll_points = 2D numpy array shape [n,2], where [:,0] are lon and [:,1] are lat
        xy_srs = new projection, as an EPSG string
        ll_srs = projection of ll_points, as an EPSG string
    Returns:
        xy_points = ll_point locations in xy_srs projection, 2D array of shape 
          [n,2], where [:,0] is x and [:,1] is y        
    """   
    
    from_srs = osr.SpatialReference()
    from_srs.SetFromUserInput(ll_srs)
    to_srs = osr.SpatialReference()
    to_srs.SetFromUserInput(xy_srs)
    
    xform = osr.CoordinateTransformation(from_srs,to_srs)

    xy_points = np.zeros( np.shape(ll_points), np.float64)
    
    npoints, two = np.shape(ll_points)
    for i in range(npoints):
        x,y,z = xform.TransformPoint(ll_points[i,0],ll_points[i,1],0 )
        xy_points[i] = [x,y]

    return xy_points


def principal_axis(Uflow,Vflow,calc_type='EOF'):
    """
    Returns the principal axis of varibility of U/V velocity profiles
    Inputs:
        Uflow = 2D numpy array shape of U direction velocities
        Vflow = 2D numpy array shape of V direction velocities
        calc_type = string ['EOF' = eigenvector PCA calculation,
                            'princax' = princax PCA calculation]
    Returns:
        principal flow variability axis, in radians      
    """   
    if calc_type == 'princeax':
        # This method seems to fail with diverse vecities, sometimes getting it 180 degrees off
        return principal_axis_from_princax(Uflow,Vflow)
    elif calc_type == 'EOF':
        return principal_axis_from_get_eof(Uflow,Vflow)


def principal_axis_from_princax(Uflow,Vflow):
    """
    Helper method for calculating the principal axis using princeax
    """
    nn = ~np.isnan(Uflow+Vflow)
    return princax(Uflow[nn]+1j*Vflow[nn])      


def principal_axis_from_get_eof(Uflow,Vflow):
    """
    Helper method for calculating the principal axis using eigenvectors
    """    
    nn = ~np.isnan(Uflow+Vflow)
    vEh = Uflow[nn]; vNh = Vflow[nn]
    
    #eof_input = zeros((2,self.n_ensembles))
    B,S,al = get_eof(np.column_stack((vEh, vNh)))
    return -np.arcsin(B[0,1])

def find_max_elev_from_velocity(vE,elev,assume_regular_grid=True):
    """
    Returns the elevation of the largest non-nan element in vE
    Inputs:
        vE = 2D numpy array, shape [ne,nb]
        elev = elevation of cells, array shape [nb]
        assume_regular_grid = True, assumes that grid cells are the same
          length in the elev direction, and adds half a grid cell to the
          bottom non-nan cell to arrive at max_evel.  False, reports the
          elev value of the deepest non-nan cell.
    Returns:
        max_elev = 1D numpy array, shape [ne], of max elevation of vE      
    """    
    n_ens,n_bins = np.shape(vE)
    max_elev = np.zeros(n_ens)
    # don't know grid, but assume regular grid so that we can add 1/2 to
    # the last center elevation to get total depth
    if assume_regular_grid:
        half = elev[1]-elev[0]/2.0
    else:
        half = 0.0
    
    for i in range(n_ens):
        idx = ~np.isnan(vE[i,:])
        if idx.any():
            idx = np.where(idx)[0]
            max_elev[i] = elev[idx[-1]] + half
    return max_elev
    

def calc_normal_rotation(xy_line):
    """
    Returns the angle normal to xy_line in radians
    Inputs:
        xy_line = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]]
    Output:
        The normal angle to xy_line in radians
    """
    return np.pi/2 - np.arctan2(xy_line[1,1]-xy_line[0,1],xy_line[1,0]-xy_line[0,0])    


def calc_Rozovski_rotation(Uflow,Vflow):
    """
    Returns the streamwise angle for all elevation-summed flows in U,V
    Inputs:
        Uflow = 1d array of flow [volume] values in U direction, shape [ne]
        Vflow = 1d array of flow [volume] values in V direction, shape [ne]
    Output:
        Streamwise angles, shape [ne]
    """
    return np.arctan2(Vflow,Uflow)


def calc_net_flow_rotation(Uflow,Vflow):
    """
    Returns the streamwise angle for summed Uflow, Vflow
    Inputs:
        Uflow = 1d array of flow [volume] values in U direction, shape [ne]
        Vflow = 1d array of flow [volume] values in V direction, shape [ne]
    Output:
        Streamwise angle, scalar, in radians 
    """
    return np.arctan2(np.nansum(Vflow),np.nansum(Uflow))
    

def average_vector(npvector,avg_shape):
    """
    Takes the average of sequential values of a numpy vector
    (reducing the resolution).
    Inputs:
        npvector = 1d numpy array, evenly divisible by avg_shape
        avg_shape = new shape [x,y] for array where x*y=np.size(npvector) 
    Output:
        sequentially averaged npvector data 
    """    
    return sp.nanmean(npvector.reshape(avg_shape),1)


def average_vector_clip(npvector,n_avg):
    """
    Takes the average of n_avg sequential values of a numpy vector
    (reducing the resolution).  Drops trailing npvector values that remain 
    after dividing np.size(npvector) by n_avg.
    Inputs:
        npvector = 1d numpy array, evenly divisible by avg_shape
        n_avg = scalar number of sequential values to average 
    Output:
        sequentially averaged npvector data 
    """    
    n = np.size(npvector)
    nnew = np.int(np.floor(n/n_avg))
    nn = range(nnew*n_avg)
    return average_vector(npvector[nn],(nnew,n_avg))

def average_array(nparray,avg_shape,axis):
    """
    Takes the average of equential values in the zero index of a numpy array
    (reducing the resolution).
     Inputs:
        nparray = 2d numpy array, with leftmost axis evenly divisible by avg_shape
        avg_shape = new shape [x,y] for array where x*y=np.size(nparray[n,:]) 
    Output:
        sequentially averaged nparray data   
    """
    # There may be a faster way of doing this - apparently this is not vectorized
    return np.apply_along_axis(average_vector,axis,nparray,avg_shape)


def average_array_clip(nparray,n_avg,axis):
    """
    Takes the average of n_avg values in the zero index of a numpy array
    (reducing the resolution). Drops trailing nparray values in the leftmost 
    dimension that remain after dividing np.size(nparray[n,:]) by n_avg.
    Inputs:
        nparray = 2d numpy array, with leftmost axis evenly divisible by avg_shape
        n_avg = scalar number of sequential values to average 
    Output:
        sequentially averaged nparray data   
    """    
    ne,nbins = np.size(nparray)
    if axis == 0:
        nnew = np.int(np.floor(ne/n_avg))
        nn = range(nnew*n_avg)
        return average_array(nparray[nn,:],(nnew,n_avg),axis)
    else:
        nnew = np.int(np.floor(nbins/n_avg))
        nn = range(nnew*n_avg)
        return average_array(nparray[:,nn],(nnew,n_avg),axis)


def centroid(xy):
    """
    Determines the centroids x-y postion from a point cloud.
    Inputs:
        xy_in = 2D numpy array, projected x/y positions of headings{m}, shape [n,2]
    Output:
        numpy array of shape [1,2] contain x-y centroid position   
    """    
    npoints,temp = np.shape(xy)    
    return np.array([[np.sum(xy[:,0])/npoints,np.sum(xy[:,1])/npoints]])

def distance_betweeen_point_clouds(xy1,xy2):
    """
    Determines the distance between the centroids of 2 point clouds.
    Inputs:
        xy1 = 2D numpy array, projected x/y positions of headings{m}, shape [n,2]
        xy2 = 2D numpy array, projected x/y positions of headings{m}, shape [n,2]
    Output:
        scalar distance in the same units as xy1,xy2   
    """    
    return find_line_distance(centroid(xy1),centroid(xy2))


def fit_head_correct(mtime_in,hdg_in,bt_vel_in,xy_in,u_min_bt=None,
                hdg_bin_size=None,hdg_bin_min_samples=None):      
    """
    Using raw data, generates a heading correction for a moving
    ADCP platform (i.e. a boat).  This circular correction should account for
    magnetic irregularities in compass headings due to metal near the compass,
    as well as magnetic declination.  It requires many compass headings
    distributed around 0-360 degrees in order to properly come up with a fit;
    otherwise compass headings may be worse than before fitting.
    Inputs:
        mtime_in = 1D numpy array, containing matplotlib date nums, shape [n]
        hdg_in = 1D numpy array, containing compass headings, in degrees, shape [n]
        bt_vel_in = 2D numpy array, containing bottom track u/v velocities {m/s}, shape [n,2]
        xy_in = 2D numpy array, projected x/y positions of headings{m}, shape [n,2]
        u_min_bt = minimum bottom track velocity - compass must be moving {m/s}, scalar
        hdg_bin_size = size of the correction bins in degrees, scalar
        hdg_bin_min_samples = minimum valid compass headings for a corection to bin
    Output:
        harmoic fit composed of the [scalar, sine, cosine] components
    """    
    #import mean_angle

    # parameters for the heading correct process:
    if u_min_bt is None:
        u_min_bt = 0.33 # np.nan #  min velocity cutoff for bottom track (nan for none)
    if hdg_bin_size is None:
        hdg_bin_size=10 # bin size for heading correction
    if hdg_bin_min_samples is None:
        hdg_bin_min_samples=10 # min number of samples per bin for head correction
    
    ## bt_vel should probably get promoted to an optional field of AdcpData
    #  likewise for heading
    #  Some of this may also have to change depending on info from Dave
    #  about how bottom track info is handled inside WinRiver (apparently
    #  the .r files we have do not have the transformed bottom track info,
    #  but processing them with winriver to get a .p file performs this
    #  transformation - I think Dave will supply code to do this in matlab/python)
    
    # These vars are modified during processing; we therefore need copies
    mtime = np.copy(mtime_in)
    hdg = np.copy(hdg_in)
    bt_vel = np.copy( -bt_vel_in[:,:2])  # note modification of standard bt_data
    xy = np.copy(xy_in)

    # find velocity from UTM positions:
    # Backward differences - first velocity assumed 0
    uv_nav = np.concatenate( [ [ [0.0,0.0] ],
                            np.diff(xy,axis=0) / np.diff(86400*mtime)[:,np.newaxis] ],
                            axis=0 )
    
    
    # The comments suggest that this is based on water-column current speed,
    # but it appears to be throwing out data where neither GPS nor bottom track
    # give a speed greater than u_min_bt.  In sample input files u_min_bt = nan,
    # so this doesn't run anyway.
    if u_min_bt>0:
        # UNTESTED - so far u_min_bt is always nan
        speed_nav = np.sqrt( np.sum(uv_nav**2,axis=1) )
        speed_bt = np.sqrt( np.sum(bt_vel**2,axis=1) )
        valid = np.nonzero( (speed_nav>=u_min_bt) & (speed_bt>=u_min_bt))[0]
        uv_nav=uv_nav[valid]
        hdg=hdg[valid]
        bt_vel = bt_vel[valid]
        mtime = mtime[valid]        
    
    ## heading from compass
    hdg = hdg%360
    #heading based on bottom tracking
    hdg_bt = (180/np.pi)*np.angle( bt_vel[:,1] + 1j*bt_vel[:,0]) % 360
    
    #heading from nav data
    hdg_nav = (180/np.pi)*np.angle(uv_nav[:,1] + 1j*uv_nav[:,0]) % 360
    
    # identify data that need shifting by 2pi --> depends on each data set
    # print "2 pi shift depends on the location/data!"
    
    # remove nans from pool of headings
    bad = np.isnan(hdg) | np.isnan(hdg_bt) | np.isnan(hdg_nav)
    good = np.nonzero(~bad)
    hdg = hdg[good]
    hdg_bt = hdg_bt[good]
    hdg_nav = hdg_nav[good]    
    
    
    hdg_bt[hdg_bt-hdg_nav>180] -= 360
    
    #print hdg_bt
    
    #print hdg_nav
    
    # toss data that looks like noise [commented out in head_correct.m]
    # nn=find(~(heada>266 & hdn>266 & hdbt<245));
    # heada = heada(nn) ; hdbt = hdbt(nn) ; hdn = hdn(nn) ; na=na(nn);
    
    # bin nav and bottom track data to get deviation.
    #hdg_nav_sorted = np.sort(hdg_nav)
    
    # transition points 
    #bin_centers = np.arange( hdg_bin_size/2.0,
    bin_centers = np.arange( hdg_nav.min() + hdg_bin_size/2.0,
                            hdg_nav.max() - hdg_bin_size/2.0,
                            hdg_bin_size)
    bin_breaks = bin_centers + hdg_bin_size/2.0
    hdg_to_bins = np.searchsorted(bin_breaks,hdg_nav)

    #print 'bin_breaks:',bin_breaks
    #print 'bin_centers:',bin_centers
    #print 'hdg_to_bins:',hdg_to_bins
    #print 'hdg_nav:',hdg_nav
  
    Nbins = len(bin_breaks)
    hdg_bt_bin_mean =   np.zeros(Nbins,np.float64)

    hdg_bt_bin_stddev = np.zeros(Nbins,np.float64)
    hdg_bt_bin_count  = np.zeros(Nbins,np.int32)

    for bin_idx in range(len(bin_breaks)):

        in_bin = hdg_to_bins==bin_idx
        hdg_bt_in_bin = hdg_bt[in_bin]

        hdg_bt_bin_count[bin_idx] = sum(in_bin)
        if hdg_bt_bin_count[bin_idx] >= hdg_bin_min_samples:
            #hdg_bt_bin_stddev[bin_idx] = sp.nanstd(hdg_bt_in_bin)
            #hdg_bt_bin_mean[bin_idx]   = sp.nanmean(hdg_bt_in_bin)
            #print 'mean1: ',hdg_bt_bin_mean[bin_idx]
            #hdg_bt_bin_stddev[bin_idx] = ssm.circstd(hdg_bt_in_bin,high=360,low=0)
            #hdg_bt_bin_mean[bin_idx] = ssm.circmean(hdg_bt_in_bin,high=360,low=0)

            #hdg_bt_bin_stddev[bin_idx] = ssm.circstd(hdg_bt_in_bin*np.pi/180)*180/np.pi

            hdg_bt_bin_mean[bin_idx] = ssm.circmean(hdg_bt_in_bin*np.pi/180)*180/np.pi
            #print 'hdg_bt_in_bin',bin_idx,hdg_bt_in_bin                       
            hdg_bt_bin_stddev[bin_idx] = ssm.circstd(hdg_bt_in_bin*np.pi/180)*180/np.pi

            #if hdg_bt_bin_mean[bin_idx] < 0:
            #    hdg_bt_bin_mean[bin_idx] += 360
            #print 'mean2: ',hdg_bt_bin_mean[bin_idx]
        else:
            #don't keep if small sample size
            hdg_bt_bin_stddev[bin_idx] = np.nan
            hdg_bt_bin_mean[bin_idx]   = np.nan  
    
    # pull headings to below 360 degrees
    hdg_bt_bin_mean = hdg_bt_bin_mean%360

    # rectify headings close to 360/zero so that fitting doesn't blow up, for instance when bins have averages of 355 and 4
    nn=np.nonzero(~np.isnan(hdg_bt_bin_mean))
    hdmi2=hdg_bt_bin_mean[nn]
    for n in range(2,len(hdmi2)):
        if abs(hdmi2[n]-hdmi2[n-1]) > abs(hdmi2[n]-hdmi2[n-1]+360):
            hdmi2[n]=hdmi2[n]+360
    hdg_bt_bin_mean[nn]=hdmi2
    
    if sum(~np.isnan(hdg_bt_bin_mean)) < 3:
        print "Not enough valid heading bins for head_correct."
        print "Try reducing hdg_bin_size and/or hdg_bin_min_samples"
        exit()
    
    # hdg_bt_bin_mean[len(hdg_bt_bin_mean)-1]=360  # hack-in test
    #print 'hdg_bt_bin_count:',hdg_bt_bin_count
    #print 'bin_centers:',bin_centers
    #print 'hdg_bt_bin_mean:',hdg_bt_bin_mean
    delta_hdg = bin_centers-hdg_bt_bin_mean  
    
    #import fit_hdg_error
    #reload(fit_hdg_error)
    
    #print 'hdg_bt_bin_mean:',hdg_bt_bin_mean,'delta_hdg',delta_hdg                        

    
    if np.sum(~np.isnan(hdg_bt_bin_mean)) < 5:
        # perform linear fit if data is sparse
        
        cf = (-sp.nanmean(delta_hdg),None,None)
    else:
        # perform harmonic fit for data that spans a large number of headings            
        (cf,yf) = fit_headerror(hdg_bt_bin_mean,delta_hdg)
    
    print 'cf:',cf
    
    return cf        
    

def find_head_correct(hdg_in,                
                    cf=None,
                    u_min_bt=None,
                    hdg_bin_size=None,
                    hdg_bin_min_samples=None,
                    mag_dec=None,
                    mtime_in=None,
                    bt_vel_in=None,
                    xy_in=None):
    """
    Makes harmonic heading corrections to input headings, either from supplied
    fit (cf) or my generating a new fit using (mtime_in,hgd_in,bt_vel_in, and
    xy_in). It requires many compass headings distributed around 0-360 degrees 
    in order to properly come up with a fit; otherwise compass headings may be worse than before fitting.
    Inputs:
        cf = harmoic fit composed of the [scalar, sine, cosine] components, or None
        mtime_in = 1D numpy array, containing matplotlib date nums, shape [n]
        hdg_in = 1D numpy array, containing compass headings, in degrees, shape [n]
        bt_vel_in = 2D numpy array, containing bottom track u/v velocities {m/s}, shape [n,2]
        xy_in = 2D numpy array, projected x/y positions of headings{m}, shape [n,2]
        u_min_bt = minimum bottom track velocity - compass must be moving {m/s}, scalar
        hdg_bin_size = size of the correction bins in degrees, scalar
        hdg_bin_min_samples = minimum valid compass headings for a corection to bin
        mag_dec = magnetic declination, in degrees, or None
    Output:
        Fitted heading difference from hdg_in, shape [n]
    """    
    # find correction factor if none it supplied. This may be inaccurate for
    # a single ADCP transect; normally it requires a large amount of data.
    if cf is None:
        
        cf = np.zeros(3,dtype=np.float64)
        
        if mag_dec is not None:
            print 'No fitted heading correction found - performing single magnetic declination correction'            
            cf[0] = mag_dec
            
        # if no 'cf' fit data is supplied, generate fit from self
        else: 
            print 'Warning: attemping to fit heading correcton based on single file.'
            try:
                cf = fit_head_correct(mtime_in,hdg_in,bt_vel_in,xy_in,
                                     u_min_bt=u_min_bt,
                                     hdg_bin_size=hdg_bin_size,
                                     hdg_bin_min_samples=hdg_bin_min_samples)
            except:
                print 'head_correct fitting failure - heading correction not performed!'
                return 0.0
        
    return cf[0] + cf[1]*np.cos((np.pi/180)*hdg_in) + cf[2]*np.sin((np.pi/180)*hdg_in)
    

def coordinate_transform(xy_in,in_srs,xy_srs,interp_nans=False):
    """
    Tranforms (re-projects) coordinates xy_in (with EPSG projection in_srs) to 
    new projection xy_srs, with optional linear interpolation of missing values.
    Inputs:
        xy_in = 2D numpy array, projected positions, shape [n,2]
        in_srs = EPSG code of xy_in positions [str]
        xy_srs = output EPSG code [str]
        interp_nans = True: interpolate nans in output positions, False: do nothing
    Output:
        xy = 2D numpy array, re-projected positions, shape [n,2]
    """
    from_srs = osr.SpatialReference() ; from_srs.SetFromUserInput(in_srs)
    to_srs = osr.SpatialReference() ; to_srs.SetFromUserInput(xy_srs)
    
    xform = osr.CoordinateTransformation(from_srs,to_srs)
    n_points,temp = np.shape(xy_in)
    xy = np.zeros((n_points,2), np.float64)
    
    for i in range(n_points):
        x,y,z = xform.TransformPoint(xy_in[i,0],xy_in[i,1],0)
        xy[i] = [x,y]

    # interpolate nans if needed
    if interp_nans:
        if (np.sum(np.sum(np.isnan(xy_in))) > 0):
            try:
                xy[:,0] = interp_nans_1d(xy[:,0])
                xy[:,1] = interp_nans_1d(xy[:,1])
            except:
                print 'lonlat_to_xy: Not enough valid navigation locations to fill NaNs'
                raise
    
    return xy


def rotate_velocity(delta,vE_in,vN_in):
    """
    Rotates vectors based by delta (radians).  If delta is a scalar,
    it is expanded to all velocities (zero index dimension).  If velocities are 2D,
    and delta is a vector, delta is applied to all bins (one index dimension).
    Inputs:
        delta = 1D numpy array, or scalar rotation angle(s), in radians
        vE_in = 1D or 2D numpy array of East or U velocities
        vN_in = 1D or 2D numpy array of North or V velocities
        interp_nans = True: interpolate nans in output positions, False: do nothing
    Output:
        vE, vN = velocities rotationed by delta, in same shape as input vE_in, vN_in
    """
    dims = np.shape(vE_in)
    error_dims = 0
    delta1 = np.copy(delta)
    # 2D velocity arrays
    if len(dims) == 2:
        ne = dims[0]
        nbins = dims[1]
        if np.size(delta) != ne and np.size(delta) != 1:
            error_dims = 1
        elif np.size(delta) == 1:
            print 'ne',ne
            print 'delta',delta
            delta1 = np.ones([ne,1],np.float64)*delta  # create vertical array
        else:
            if len(np.shape(delta)) == 1:
                delta1 = np.array([delta]).T  # transpose to vertical
            delta1 = np.ones(nbins,np.float64)*delta1 # generate 2D delta array        
    # 1D velocity vectors
    elif len(dims) == 1:      
        ne = dims[0]
        if np.size(delta) != ne and np.size(delta) != 1:
            error_dims = 1
        elif np.size(delta) == 1:
            delta1 = np.ones(ne,np.float64)*delta  # create array            
    # scalar velocities
    elif len(dims) == 0:
        if np.size(delta) > 1:
           error_dims = 1

    if error_dims == 1:
        print "Error in rotate_velocity: delta is not mappable to velocities."
        print "Check sizes of input delta and velocity."
        raise ValueError
    
    # need to coorect for the fact that sometimes there are fewer headers than velocities!
    vE = np.cos(delta1)*vE_in + np.sin(delta1)*vN_in
    vN = -np.sin(delta1)*vE_in + np.cos(delta1)*vN_in    
    return (vE, vN)


def find_sidelobes(fsidelobe,bt_depth,elev):
    """
    Finds near-bottom cells that may have side lobe problems 
    fSidelobe=0.10; used 15% in past, but Carr and Rehmann use 6%
    Inputs:
        fsidelobe = fraction of cells closer than bottom/valid range cosidered bad data
        bt_depth = bottom distance, 1D numpy array
        elev = bin center elveation from transducer, 1D numpy array
    Output:
        numpy boolean array where True is bad cells
    """
    nens = np.size(bt_depth)
    nbins = np.size(elev)
    ranges_loc = -1.0*elev
    zz = ranges_loc * np.ones([nens,1])  # generate array of ranges
    depth = -1.0*bt_depth.T  # need rank-2 array so we can transpose it
    return np.greater(zz,(1-fsidelobe)*np.ones([nbins]) * depth) # identify where depth is too great
    


def find_sd_greater(nparray,elev,sd=3,axis=1):
    """
    Find outliers in nparray > sd, with sd generated along nparray(axis),
    Inputs:
        nparray = 2D numpy array
        elev = bin center elveation from transducer, 1D numpy array
        sd = threshold standard deviation
        axis=0 is 1st dimension, axis=1 is 2nd dimenstion (default)
    Output:
        numpy boolean array where True is cells > axis standard deviation
    """
    nens,vbins = np.shape(nparray)
    if axis == 1:
        vsig = np.array([sp.nanstd(nparray,1)]).T # transpose to vertical
        test = sd*np.ones([vbins])*vsig + np.ones([vbins])*np.array([sp.nanmean(nparray,1)]).T
    else:
        vsig = np.array([sp.nanstd(nparray)])
        test = sd*np.ones((nens,1))*vsig + np.ones((nens,1))*np.array([sp.nanmean(nparray)])    
    return np.greater(nparray,test) 


def remove_values(nparray,rm,axis=None,elev=None,interp_holes=False,warning_fraction=0.05):
    """
    Throw out outliers and fill in gaps based upon standard deviation - 
    typical to use 3 standard deviations (sd=3)
    Inputs:
        nparray = 2D numpy array
        elev = bin center elveation from transducer, 1D numpy array
        rm = numpy boolean array where True is cells to be dropped, same shape as nparry
        axis = determines in which direction interpolation is performed
          0 is 1st dimension, axis=1 is 2nd dimenstion (default)
        interp_holes = True means interpolate values just removed, False = don't
        warning_fractoin = scalar threshold above which the fraction of removed cells throws a warning
    Output:
        new_array = numpy 2D array with rm values removed, and optionally interpolated
    """
    
    # generate warning if neccessary
    good_vels = np.sum(np.sum(~np.isnan(nparray)))
    fraction_dropped = np.sum(rm) / good_vels
    if fraction_dropped > warning_fraction:
       print 'Warning: greater than %3.2f%% of velocities will be removed.'%warning_fraction        
    
    # drop values    
    new_array = np.copy(nparray)
    new_array[rm] = np.nan

    # interpolate holes if desired
    if interp_holes:
        if elev is None:
            print "Error in remove_values: setting 'elev' is required to interpolate holes"
            return 0.0
        if axis is None:
            print "Error in remove_values: setting 'axis' is required to interpolate holes"
            return 0.0
        nens,vbins = np.shape(new_array)
        i, j = np.nonzero(rm)
        # generate array to interpolate into, so interpolted values are
        # not used in further interpolation
        new_interp = np.copy(new_array)  # generate array to interpolate into, so interpolted values
        if axis == 1:
            if np.size(elev) != vbins:
                print "Error in remove_values:  nparray size, elev size, and axis do not agree"
                return 0.0
            # interpolate in 2nd dimension of new_array
            for m in range(len(i)):
                nn=np.nonzero(~np.isnan(new_array[i[m],:]))
                new_interp[i[m],j[m]] = np.interp(elev[j[m]],elev[nn],np.squeeze(new_array[i[m],nn]))
        else:
            if np.size(elev) != nens:
                print "Error in remove_values:  nparray size, elev size, and axis do not agree"
                return 0.0
            # interpolate in 1st dimension of new_array
            for m in range(len(i)):
                nn=np.nonzero(~np.isnan(new_array[:,j[m]]))
                new_interp[i[m],j[m]] = np.interp(elev[i[m]],elev[nn],np.squeeze(new_array[nn,j[m]]))
        
        new_array = new_interp
        
    return new_array


def kernel_smooth(kernel_size,nparray):
    """
    Uses a nan-safe boxcar/uniform filter to smooth the data.  
    Smooth_kernel must be an odd integer >= 3
    Inputs:
        kernel_size = odd integer >= 3
        nparray = 2D numpy array
    Output:
        nparray_out = smoothed nparray
    """
    
    #from scipy.ndimage.filters import uniform_filter as boxcar
    #import convolve_nd
    
    if kernel_size < 3 or kernel_size > min(np.shape(nparray)):
        print 'Error: kernel_size must be between 3 and the smallest array dimension'
        return 0.0    
    kernel = np.ones([kernel_size,kernel_size])        
    nparray_out = np.copy(nparray)    
    nn = np.isnan(nparray)
    nparray_out = convolvend(nparray_out,kernel,
                             interpolate_nan=True,
                             normalize_kernel=True,
                             ignore_edge_zeros=True)
    nparray_out[nn] = np.nan
    return nparray_out
    

def find_xy_transect_loops(xy,xy_range=None,pline=None):
    """
    Uses x-y postion/projection to a line to determine where a sequence of 
    positions folds back on itself.
    Inputs:
        xy = 2D numpy array, projected x/y positions of headings{m}, shape [n,2]
        xy_range = projected distance between points
        pline = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]]
    Output:
        boolean 1D array of xy positions that are fold back compared to the 
        mean path
    """
    if (xy_range is None):
        xx,yy,xy_range = find_projection_distances(xy,pline=pline)
    #determine if mostly increasing or decreasing
    xy_diff = np.diff(xy_range)
    xy_sign = np.sign(np.sum(np.sign(xy_diff)))
    if xy_sign > 0:
        return np.nonzero(xy_diff > 0)
    else:
        return np.nonzero(xy_diff < 0)


def find_extrapolated_grid(n_ens,elev,bot_depth=None,adcp_depth=None):
    """
    Generates a new grid in the elevation/range dimension from the transducer,
    filling in missing bins up to surface and down to bot_depth.
    Inputs:
        n_ens = integer desribing the number of ensembles bot_depth and grid will
          be expanded to
        elev = bin center evelation of current grid
        bot_depth = scalar or 1D nparray of shape [n_ens] descibing bot_depth, or None
        adcp_depth = scalar descibing the depth of the adcp face underwater - useful
          for downward and upward looking deployments. 
    Output:
        zex = new elev, descibing bin center elevations of new grid 
        depth = max valid range elevation from transducer
        new_bins = new bin elevations nearest transducer
    """

    z = elev   # expecting positive values
    if adcp_depth is not None:
        z = z + sp.nanmean(adcp_depth)  # not currently extrapolating individual profiles...
    dz = z[1] - z[0]
    new_bins = np.array([np.arange(z[0]-dz,0,-dz)]).T
    zex = np.append(np.sort(new_bins,0),z)

    if bot_depth is None:
        depth = np.ones(n_ens)*(max(zex)+1)   # all bins will be valid if no bottom depth
    elif len(np.shape(bot_depth)) == 0:
        depth = np.ones(n_ens)*bot_depth      # extrapolate single bot_depth to all
    else:
        bt_shape = np.shape(bot_depth)
        if len(bt_shape) > 1 and bt_shape[0] == 1:
            depth = bot_depth.T
        else:
            depth = np.array([bot_depth]).T        
    return (zex, depth, new_bins)




def extrapolate_boundaries(velocity,elev,ex_evel,depth,new_bins):
    """
    Extrapolates velocities to surface/bottom boundaries where ADCP measurements
    are typically not available or valid.  Uses a parabolic extrapolation towards
    the a free surface, and towartds zero at the bottom.  This version is appropriate
    for downward-looking ADCPs.
    Inputs:
        velocity = 2D numpy array of velocities, shape [ne,nb]
        elev = bin center evelation of current grid, shape [nb]
        ex_evel = bin center evelation of new exrapolated grid, 1D array > [nb2]
        depth = scalar or 1D nparray of shape [ne] descibing bot_depth, or None
          for downward and upward looking deployments.
        new_bins = new bin elevations nearest transducer
    Output:
        ex_velocity = new extrapolated 2D velocity array, of shape [ne,nb2]
    """
    # find new grid elevation with extended bins
    vel_shape = np.shape(velocity)
    n_ens = vel_shape[0]
    n_bins = vel_shape[1]
    if len(vel_shape) == 2:
        n_vels = 1
        vel_in = velocity.reshape((n_ens,n_bins,1))
    else:
        n_vels = vel_shape[2]
        vel_in = velocity
    new_vel_shape = (n_ens,np.size(new_bins)+n_bins,n_vels)
    ex_velocity = np.zeros(new_vel_shape,np.float64)
    zztemp = -ex_evel*np.ones((n_ens,1))
    for i in range(n_vels):
        new_vel_shape1 = np.column_stack((np.nan*np.ones((n_ens,np.size(new_bins))),
                                       vel_in[:,:,i]))                
        ex_velocity1 = fillProParab(new_vel_shape1,zztemp,depth,0)  # don't need to flip arrays b/c they are already flipped relative to matlab
        ex_velocity1[zztemp < np.ones(len(ex_evel))*depth] = np.nan  # depths here are negative
        ex_velocity[:,:,i] = ex_velocity1
    if n_vels == 1:
        ex_velocity = np.squeeze(ex_velocity,axis=2)    
    return ex_velocity


def create_depth_mask(elev,depths):
    """
    Returns a boolean matrix with positive as valid velocity bins above depth
    Inputs:
        elev = bin center evelation of current grid, shape [nb]
        depths = 1D nparray of shape [ne] descibing the max valid elevation
    Output:
        depth_mask = boolean numpy array of shape [ne,nb] with True above input
          depths
    """
    nens = np.size(depths)
    vbins = np.size(elev)
    bt_local = np.copy(depths)
    
    d1 = np.zeros(vbins,np.float64)
    half = abs(elev[1]-elev[0])/2
    d1[0] = min(abs(elev[0]),half) + half
    for i in range(1,vbins-1):
        half_old = half
        half = abs(elev[i+1]-elev[i])/2              
        d1[i] = half + half_old
    d1[vbins-1] = half*2            
    depth_mask = np.tile(d1,(nens,1))
    zz = elev * np.ones([nens,1])  # generate array of ranges
    if (np.shape(bt_local) < 2):
        bt_local = np.array([bt_local])
    d2 = bt_local.T  # need rank-2 array so we can transpose it
    ii = np.greater(zz,np.ones([vbins]) * d2) # identify where depth is too great
    depth_mask[ii] = 0
    return depth_mask       
   

def calc_crossproduct_flow(vU,vV,btU_in,btV_in,elev,bt_depth,mtime):        
    """
    Calculates the discharge(flow) by finding the cross product of the water
    and bottom track velocities. **elev and bt_depth are positive**
    Inputs:
        vU = U velocity, 2D numpy array, shape [ne,nb] {m/s}
        vV = V velocity, 2D numpy array, shape [ne,nb] {m/s}
        btU_in = Bottom track (i.e. transducer speed) U velocity, 1D numpy array, shape [ne] {m/s}
        btV_in = Bottom track (i.e. transducer speed) V velocity, 1D numpy array, shape [ne] {m/s}
        elev = bin center evelation velocities, shape [nb] - must be positive and increasing
        depths = 1D nparray of shape [ne] descibing the max valid elevation - must be positive and increasing
        mtime = 1D numpy array, shape [ne], with matplotlib datenums of ensemble measurement times
    Output:
        Ums = Ensemble flow, 1D numpy array, shape [ne] {m^3/s}
        U = Total cross-sectional flow {m^3/s}
        total_survey_area = ensemble-to-ensemble 2D survey area {m^2}
        total_cross_sectional_area = total valid survey area in U-direction {m^2}
    """      
    # spread bottom track velocities across all bins
    nens,vbins = np.shape(vU)
    btU =  np.ones(vbins)*np.array([btU_in]).T
    btV =  np.ones(vbins)*np.array([btV_in]).T 
    cp_velocity = vU*btV - vV*btU # cross product velocities
    
     # construct depth matrix
    depths = create_depth_mask(elev,bt_depth)  

    # construct time matrix
    time = np.zeros(nens,np.float64)
    #depth = np.zeros((nens,vbins))
    time[0] = (abs(mtime[0]-mtime[1]))/2.0
    time[nens-1] = (abs(mtime[nens-2]-mtime[nens-1]))/2.0
    for i in range(1,nens-1):
        time[i] = (abs(mtime[i-1] - mtime[i+1]))/2.0                                                
    times = np.tile(time,(vbins,1))            
    times = times.T*3600*24 # rotate, convert from days to seconds  
  
    # integrate flow, reverse if heading is backwards compared to alignment
    # axis
    flow = np.abs(cp_velocity*(times*depths))      
    bt_mag = np.sqrt(btU**2+btV**2)
    survey_area = times*depths*bt_mag
    nn = np.logical_or(np.isnan(vU),np.isnan(vV))
    survey_area[nn] = np.nan
    total_survey_area = np.nansum(np.nansum(survey_area))
    cross_sectional_area = survey_area*(btV/bt_mag) # fraction of survey in U direction            
    total_cross_sectional_area = \
    np.abs(np.nansum(np.nansum(cross_sectional_area)))
    
    U = np.nansum(np.nansum(flow))
    Ums = U/total_cross_sectional_area

    return (Ums, U, total_survey_area, total_cross_sectional_area)


def map_xy_to_line(xy):
    """
    Finds a best linear fit to a point cloud, using numpy polyfit.
    Inputs:
        xy = x-y locations , 2D array of shape [n,2], where [:,0] is x and [:,1] is y
    Output:
        numpy array of line defined by 2 points: [[x1,y1],[x2,y2]]
    """      
    x0 = np.min(xy[:,0])
    y0 = np.min(xy[:,1])
    xd = xy[:,0] - x0
    yd = xy[:,1] - y0
    coefs = np.polyfit(xd,yd, 1)
    yd_fit = np.polyval(coefs,xd)+y0
    y0_fit = np.min(yd_fit)
    yd1 = yd_fit - y0_fit
    y00_fit = np.max(yd1)
    # find line ends
    mag_fit = np.sqrt(y00_fit*y00_fit + xd*xd)
    mini = np.argmin(mag_fit)
    maxi = np.argmax(mag_fit)
    return np.array([[xy[mini,0],yd_fit[mini]],[xy[maxi,0],yd_fit[maxi]]])    
    

def find_line_distance(xy1,xy2):
    """
    Finds distance between two points on a regular grid.  Can accept arrays of
    points.
    Inputs:
        xy1 = x-y locations , 2D array of shape [n,2], where [:,0] is x and [:,1] is y
        xy2 = x-y locations , 2D array of shape [n,2], where [:,0] is x and [:,1] is y
    Output:
        numpy array of line defined by 2 points: [[x1,y1],[x2,y2]]
    """      
    return np.sqrt((xy2[:,0]-xy1[:,0])**2 + \
                   (xy2[:,1]-xy1[:,1])**2)
    

def find_projection_distances_new(xy,pline=None):
    """
    Finds the distances between profiles(dd) along either a linear fit of 
    transect positions, or along a supplied line given by the pline.
    Also returns the x (xd) and y (yd) distances of xy points along this axis
    Inputs:
        xy = x-y locations , 2D array of shape [n,2], where [:,0] is x and [:,1] is y
        pline = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]], or None
    Output:
        xd = x normal (minimum distance) projection of xy points onto fitted line
        yd = y normal (minimum distance) projection of xy points onto fitted line
        dd = distance along fitted line, in the direction of the fit line
    """
    if pline is None:
        print 'No plot line specified - performing linear fit of projected data.'
        xy_line = map_xy_to_line(xy)
    else:
        xy_line = pline
    x0 = xy_line[0,0]                
    y0 = xy_line[0,1]                
    x1 = xy_line[1,0]                
    y1 = xy_line[1,1]            
    x00 = x1 - x0
    y00 = y1 - y0
    xd = xy[:,0] - x0
    yd = xy[:,1] - y0
    
    #print 'x0 x1 xy[0,0] xy[-1,0] x00 xd[0] xd[-1]'
    #print x0, x1, xy[0,0], xy[-1,0], x00, xd[0], xd[-1]
    #print 'y0 y1 xy[0,1] xy[-1,1] y00 yd[0] yd[-1]'
    #print y0, y1, xy[0,1], xy[-1,1], y00, yd[0], yd[-1]
    
    # below sourced from: http://stackoverflow.com/questions/3120357/get-closest-point-to-a-line
    xy_line_d_sq = x00*x00 + y00*y00
    dot_product = xd*x00 + yd*y00
    normal_vector = dot_product / xy_line_d_sq
    closest_points = np.zeros((len(xd),2),np.float64)
    closest_points[:,0] = x0 + x00*normal_vector
    closest_points[:,1] = y0 + y00*normal_vector
    #fig=plt.figure()
    #plt.scatter(closest_points[:,0],closest_points[:,1])
    #plt.show()
    weird_line_format = np.array(createLine(xy_line[0,:],xy_line[1,:]))
    
    dd = linePosition(closest_points,weird_line_format)*np.sqrt(xy_line_d_sq)
    return closest_points[:,0],closest_points[:,1],dd
    

def find_projection_distances(xy,pline=None):
    """
    Finds the distances between profiles(dd) along either a linear fit of 
    transect positions, or along a supplied line given by the pline.
    Also returns the x (xd) and y (yd) distances of xy points along this axis
    Inputs:
        xy = x-y locations , 2D array of shape [n,2], where [:,0] is x and [:,1] is y
        pline = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]], or None
    Output:
        xd = x normal (minimum distance) projection of xy points onto fitted line
        yd = y normal (minimum distance) projection of xy points onto fitted line
        dd = distance along fitted line, in the direction of the fit line
    """
    if (pline is None):
        #print 'No plot line specified - performing linear fit of projected data.'
        xy_line_point = map_xy_to_line(xy)
    else:
        xy_line_point = pline
    x0 = xy_line_point[0,0]                
    y0 = xy_line_point[0,1]                
    x00 = xy_line_point[1,0] - x0          
    y00 = xy_line_point[1,1] - y0           
    xd = xy[:,0] - x0
    yd = xy[:,1] - y0
    test_flip = False
    
    plot_line = np.array(createLine((0.0,0.0),(x00,y00)))
    d_plot_line = np.sqrt(plot_line[2]**2 + plot_line[3]**2)

    # map profiles to line                             
    points = np.zeros((len(xd),2),np.float64)
    points[:,0] =  xd
    points[:,1] =  yd
    dd = linePosition(points,plot_line)*d_plot_line
    #dd = linePosition(points,plot_line)   
    if test_flip:
        if yd1[maxi] < yd1[mini]:
            #flip it
            dd = np.max(dd) - dd
    
    #fig=plt.figure()
    #plt.scatter(xd,yd)
    #plt.show()
    return xd,yd,dd


def new_xy_grid(xy,z,dx,dz,pline=None,fit_to_xy=True):
    """
    Generates a regular grid (staight in the xy plane) with spacing
    set by dx and dy, using the same distance units as the current projection. 
    Generates a linear fit, or fits to input pline.
    Inputs:
        xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
        z = z position vector of current grid, shape [nb]
        dxy = new grid xy resolution in xy projection units
        dz = new grid z resolution in z units
        pline = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]], or None
        fit_to_xy = if True, returns new grid that encompasses xy data only, if
          False and pline is given, returns a regular grid exactly the length of pline
    Returns:
        xy_new = xy positions of new grid shape, 2D numpy array
        z_new = z positions of new grid, 1D numpy array
    """
    # reverse dz if necessary
    z_is_negative = np.less(sp.nanmean(z),0)
    if z_is_negative == (dz < 0):
        my_dz = dz
    else:
        my_dz = -dz
    xd,yd,dd = find_projection_distances(xy,pline=pline)
    # reverse dx if necessary
    dd_start = np.min(dd)
    dd_end = np.max(dd)
    dd_increasing = np.less(yd[np.argmin(dd)],yd[np.argmax(dd)])
    if dd_increasing == (dx > 0):
        my_dx = dx
    else:
        my_dx = -dx
        tmp = dd_end
        dd_end = dd_start
        dd_start = tmp     
    # find gridding dimensions
    if pline is not None and not fit_to_xy:
        x0 = pline[0,0]
        y0 = pline[0,1]
        x00 = pline[1,0]-pline[0,0]
        y00 = pline[1,1]-pline[0,1]
        pline_distance = np.sqrt(x00**2 + y00**2)
        #if dx < 0:
        #    xy_new_range = np.arrange(0,pline_distance,abs(dx))
        xy_new_range = np.arange(0,pline_distance,np.abs(dx))
        grid_angle = np.arctan2(y00,x00)
        xy_new = np.zeros((np.size(xy_new_range),2),dtype=np.float64)
        xy_new[:,0] = xy_new_range*np.cos(grid_angle) + x0 # back to projection x - might be offset by up to dx
        xy_new[:,1] = xy_new_range*np.sin(grid_angle) + y0 # back to projection y - might be offset by up to dy
        poo1, poo2, xy_new_range = find_projection_distances(xy_new,pline=pline)
    else:
        xy_new_range = np.arange(dd_start,dd_end,my_dx)
        # find x/y ppojected locations of new grid
        min_dd = np.argmin(dd)
        max_dd = np.argmax(dd)
        x0 = xy[min_dd,0]
        y0 = xy[min_dd,1]
        x00 = xy[max_dd,0] - x0
        y00 = xy[max_dd,1] - y0
        grid_angle = np.arctan2(y00,x00)
        xy_new = np.zeros((np.size(xy_new_range),2),dtype=np.float64)
        xy_new[:,0] = xy_new_range*np.cos(grid_angle) + x0 # back to projection x - might be offset by up to dx
        xy_new[:,1] = xy_new_range*np.sin(grid_angle) + y0 # back to projection y - might be offset by up to dy
    z_new = np.arange(z[0],z[-1],my_dz)  # find z1
    
    return (dd,xy_new_range,xy_new,z_new)


def newer_new_xy_grid(xy,z,dx,dz,pline=None):
    """
    Generates a regular grid (staight in the xy plane) with spacing
    set by dx and dy, using the same distance units as the current projection. 
    Generates a linear fit, or fits to input pline.
    Inputs:
        xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
        z = z position vector of current grid, shape [nb]
        dxy = new grid xy resolution in xy projection units
        dz = new grid z resolution in z units
        pline = numpy array of line defined by 2 points: [[x1,y1],[x2,y2]], or None
    Returns:
        xy_new = xy positions of new grid shape, 2D numpy array
        z_new = z positions of new grid, 1D numpy array
    """
    # reverse dz if necessary
    z_is_negative = np.less(sp.nanmean(z),0)
    if z_is_negative == (dz < 0):
        my_dz = dz
    else:
        my_dz = -dz
    xd,yd,dd = find_projection_distances(xy,pline=pline)
    # reverse dx if necessary
    dd_start = np.min(dd)
    dd_end = np.max(dd)
    dd_increasing = np.less(yd[np.argmin(dd)],yd[np.argmax(dd)])
    if dd_increasing == (dx > 0):
        my_dx = dx
    else:
        my_dx = -dx
        tmp = dd_end
        dd_end = dd_start
        dd_start = tmp     
    # find gridding dimensions
    xy_new_range = np.arange(dd_start,dd_end,my_dx)
    z_new = np.arange(z[0],z[-1],my_dz)  # find z1
    # find x/y ppojected locations of new grid
    min_dd = np.argmin(dd)
    max_dd = np.argmax(dd)
    x0 = xy[min_dd,0]
    y0 = xy[min_dd,1]
    x00 = xy[max_dd,0] - x0
    y00 = xy[max_dd,1] - y0
    grid_angle = np.arctan2(y00,x00)
    xy_new = np.zeros((np.size(xy_new_range),2),dtype=np.float64)
    xy_new[:,0] = xy_new_range*np.cos(grid_angle) + x0 # back to projection x - might be offset by up to dx
    xy_new[:,1] = xy_new_range*np.sin(grid_angle) + y0 # back to projection y - might be offset by up to dy
    
    return (dd,xy_new_range,xy_new,z_new)


def find_regular_bin_edges_from_centers(centers):
    """
    Finds bin (grid cell) edges from center positions.  Assumes a regular grid.
    Inputs:
        centers = bin/grid center position vector of current grid, shape [nb]
    Returns:
        edges = edge positions of bins (grid), shape [nb+1]
    """
    edges = np.zeros(np.size(centers)+1,np.float64)
    offset = (centers[1]-centers[0])*0.5  # should be regular grid
    edges[1:] = centers + offset
    edges[0] = centers[0] - offset
    edges.sort()
    return edges


def xy_regrid(nparray,xy,xy_new,z=None,z_new=None,pre_calcs=None,kind='bin average'):
    """
    Re-grids the values in the 1D or 2D nparray onto a new grid defined by xy_new
    (and z_new, if 2D).  Accepts a bunch of stuff as pre_calcs so to save computation
    if multiple xy_regrids are being called with the same input and new grids.
    Inputs:
        nparray = 1D or 2D numpy array
        xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
        xy_new = new grid x-y locations, 2D array of shape [ne2,2], where [:,0] is x and [:,1] is y
        z = z positions, 1D array of shape [nb]
        z_new = z positions of new grid, 1D array of shape [nb2]
        pre_calcs = python list of different intermediate things, specifically the
          returns of prep_xy_regrid()
        kind = string, either 'bin average' or one ofthe kinds of interpolatation
          known by scipy.interpolate
    Returns:
        nparray values regridded to shape [ne2],1D or [ne2,nb2],2D
    """   
    if kind == 'bin average':
        # this fuction optionally returns a tuple, we only want 1st element which is means
        return  xy_bin_average(nparray,xy,xy_new,z,z_new,pre_calcs,return_stats=False)[0]
    else:
        return xy_interpolate(nparray,xy,xy_new,z,z_new,pre_calcs,kind)


def xy_regrid_multiple(nparray,xy,xy_new,z=None,z_new=None,pre_calcs=None,kind='bin average'):
    """
    Iterates of 3D arrays, calling xy_regrid() on each 2D slice [:,:,n]
    Inputs:
        nparray = 1D or 2D numpy array
        xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
        xy_new = new grid x-y locations, 2D array of shape [ne2,2], where [:,0] is x and [:,1] is y
        z = z positions, 1D array of shape [nb]
        z_new = z positions of new grid, 1D array of shape [nb2]
        pre_calcs = python list of different intermediate things, specifically the
          returns of prep_xy_regrid()
        kind = string, either 'bin average' or 'interpolate' to determine how
          the regridding is accomplished
    Returns:
        nparray values regridded to shape [ne2],1D or [ne2,nb2],2D
    """
    
    dims = np.shape(nparray)
    if len(dims) == 1:
        print 'xy_regrid_multiple: nparray must be 2D or 3D'
        raise ValueError
    if len(dims) == 2:
        new_dims = (np.size(xy_new[:,0]),dims[-1])
    else:
        new_dims = (np.size(xy_new[:,0]),np.size(z_new),dims[-1])
    gridded_array = np.zeros(new_dims,np.float64)
    for i in range(dims[-1]):
        gridded_array[...,i] = xy_regrid(nparray[...,i],xy,xy_new,
                                         z,z_new,pre_calcs,kind)
    return gridded_array


def xy_z_linearize_array(xy_range,z,nparray):
    """
    Helper function to linearize the values in nparray, along with their
    xy and z positions, while removing nans.
    Inputs:
        nparray = 1D or 2D numpy array, shape [ne,nb]
        xy = xy grid positions, shape [ne]
        z = z locations, 1D array of shape [nb]
    Returns:
        xy1 = xy grid positions, shape [ne*nb - nans]
        z1 = z grid positions, shape [ne*nb - nans]
        v1 = nparray values, shape [ne*nb - nans]
    """
    v1 = nparray.reshape(np.size(nparray))
    nnan = ~np.isnan(v1)
    v1 = v1[nnan]
    xy1 = np.repeat(xy_range,np.size(z))[nnan]
    z1 = np.tile(z,np.size(xy_range))[nnan]
    return (xy1, z1, v1)


def un_flip_bin_average(xy_range,z,avg):
    """
    Depending in the orientation of the bin edges input into bin_average(), the
    bin edges must be sorted ascending.  This function reverses output bin average
    arrays if sorting occured, so that the bin_average results are oriented in
    the same fashion as input bin edges.
    Inputs:
        xy = xy grid edge positions, shape [xyb]
        z = z locations, 1D array of shape [nb]
        z_bins = z grid edge positions, shape [zb]
        avg = list or arrays to be conditionally flipped 
    Returns:
        list of input arrays avg,conditinally flipped   
    """
    flipped = []
    ud = (xy_range[-1]-xy_range[0]) < 0
    lr = False
    if z is not None:
        lr = (z[-1]-z[0]) < 0
    for a in avg:
        if lr:
            a = np.fliplr(a)
        if ud:
            a = np.flipud(a)
        flipped.append(a)
    return flipped

def bin_average(xy,xy_bins,values,z=None,z_bins=None,return_stats=False):
    """
    Bins  input values in the 1D or 2D nparray 'values' into the bins with
    edges defined by xy_bins (and z_bins if 2D).  Optionally returns the number
    of values in each bin, and the standard deviation of values in each bin.
    Inputs:
        xy = xy locations, 1D array of shape [ne]
        xy_bins = xy grid edge positions, shape [xyb]
        values = values to bin average, 1D or 2D numpy array, shape [ne] or [ne,nb]
        z = z locations, 1D array of shape [nb]
        z_bins = z grid edge positions, shape [zb]
        return_stats = optionally returns number of nparray values per bin, and
          standard devation per bin
    Returns:
        bin_mean = values bin-averaged to shape [xyb] or [xyb,zb]
        if return_stats = True, returns (bin_mean, bin_n, bin_sd),
        all of shape [xyb] or [xyb,zb]
    """        
    z_not_none = False
    if z is not None:
        z_not_none = True

    if z_not_none:
        # 2D bin average 
        bin_n, e1, e2 = np.histogram2d(xy,z,bins = (xy_bins,z_bins))
        bin_sum, e1, e2 = np.histogram2d(xy,z,bins = (xy_bins,z_bins),
                                        weights = values)            
    else:
        # 1D bin average
        bin_sum, e1 = np.histogram(xy,bins = xy_bins,weights = values)
        bin_n, e1 = np.histogram(xy,bins = xy_bins)

    bin_mean = bin_sum/bin_n
            
    if return_stats:
        xy_bin_num = np.digitize(xy,xy_bins)
        xy_bin_num[xy_bin_num>=np.size(xy_bins)] = 0
        sq_sums = np.zeros(np.shape(bin_mean),np.float64)
        if z_not_none:
            z_bin_num = np.digitize(z,z_bins)
            z_bin_num[z_bin_num>=np.size(z_bins)] = 0
        for n in range(np.size(xy)):
            i = xy_bin_num[n]-1
            if z_not_none:
                j = z_bin_num[n]-1
                if i > 0 and j > 0:
                    sq_sums[i,j] += (values[n] - bin_mean[i,j])**2
            elif i > 0:
                sq_sums[i] += (values[n] - bin_mean[i])**2
        bin_sd = np.sqrt(sq_sums/bin_n)
        return (bin_mean, bin_n, bin_sd)
    else:
        return (bin_mean,)
        

def prep_xy_regrid(nparray,xy,xy_new,z=None,z_new=None,pre_calcs=None):
    """
    Computes new grids, grid boundaries and meshgrids in preparation for
    calls to xy-regrid and related methods.  If pre_calcs is not None, it
    simply returns those values, skipping re-computation.
    Inputs:
        nparray = 1D or 2D numpy array
        xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
        xy_new = new grid x-y locations, 2D array of shape [ne2,2], where [:,0] is x and [:,1] is y
        z = z positions, 1D array of shape [nb]
        z_new = z positions of new grid, 1D array of shape [nb2]
        pre_calcs = python list of different intermediate things, specifically the
          returns of prep_xy_regrid()
    Returns:
        is_array = True if nparray is 2D, False if nparray is 1D
        xy_range = distance along fitted line, in the direction of the fit line, current grid
        zmesh = mesh of z-direction values resulting from meshgrid(), current grid
        xymesh = mesh of fitted-xy-direction values resulting from meshgrid()
        xy_new_range = distance along fitted line, in the direction of the fit line, new grid
        zmesh_new = mesh of z-direction values resulting from meshgrid(), new grid
        xymesh_new = mesh of fitted-xy-direction values resulting from meshgrid(), new grid
    """   
    if len(np.shape(nparray)) == 1:
        is_array = False
    elif z is None or z_new is None:
        print 'Error - to regrid a 2D array, arguments z and z_new are required'
        raise ValueError
    else:
        is_array = True       
    if pre_calcs is None:
        # generate projected distances between xy points
        pline = np.array([[xy_new[0,0],xy_new[0,1]],[xy_new[-1,0],xy_new[-1,1]]])
        xtemp,ytemp,xy_range = find_projection_distances(xy,pline=pline)
        xtemp,ytemp,xy_new_range = find_projection_distances(xy_new)
        if is_array:
            zmesh_new, xymesh_new = np.meshgrid(z_new,xy_new_range)
            zmesh, xymesh = np.meshgrid(z,xy_range)
        else:
            zmesh_new, xymesh_new, zmesh, xymesh = (None,None,None,None)
    else:
        xy_range,zmesh,xymesh,xy_new_range,zmesh_new,xymesh_new = pre_calcs
    
    return (is_array,xy_range,zmesh,xymesh,xy_new_range,zmesh_new,xymesh_new)


def xy_bin_average(nparray,xy,xy_new,z=None,z_new=None,pre_calcs=None,return_stats=False):
    """
    Bin-averages the values in the 1D or 2D nparray onto a new grid defined by xy_new
    (and z_new, if 2D).  Accepts a bunch of stuff as pre_calcs so to save computation
    if multiple xy_regrids are being called with the same input and new grids.
    Inputs:
        nparray = 1D or 2D numpy array
        xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
        xy_new = new grid x-y locations, 2D array of shape [ne2,2], where [:,0] is x and [:,1] is y
        z = z positions, 1D array of shape [nb]
        z_new = z positions of new grid, 1D array of shape [nb2]
        pre_calcs = python list of different intermediate things, specifically the
          returns of prep_xy_regrid()
        return_stats = optionally returns number of nparray values per bin, and
          standard devation per bin
    Returns:
        avg = nparray values regridded to shape [ne2],1D or [ne2,nb2],2D
        if return_stats = True, returns (re_grid_nparray, bin_n_nparray, bin_sd_nparray),
        all of shape [ne2] [ne2,nb2]
    """      
    (is_array,xy_range,zmesh,xymesh,xy_new_range,zmesh_new,xymesh_new) = \
        prep_xy_regrid(nparray,xy,xy_new,z,z_new,pre_calcs)

    xy_bins = find_regular_bin_edges_from_centers(xy_new_range)
    if is_array:
        z_bins = find_regular_bin_edges_from_centers(z_new)
        xy_tiled, z_tiled, valid_data = \
            xy_z_linearize_array(xy_range,z,nparray)
    else:
        z_bins = None
        z_tiled = None
        nnan = ~np.isnan(nparray)
        xy_tiled = xy_range[nnan]
        valid_data = nparray[nnan]

    avg = bin_average(xy_tiled,xy_bins,valid_data,z_tiled,z_bins,return_stats=False)    
    return un_flip_bin_average(xy_new_range,z_new,avg)


def xy_interpolate(nparray,xy,xy_new,z=None,z_new=None,pre_calcs=None,kind='cubic'):
    """
    Interpolates the values in the 1D or 2D nparray onto a new grid defined by xy_new
    (and z_new, if 2D).  Accepts a bunch of stuff as pre_calcs so to save computation
    if multiple xy_regrids are being called with the same input and new grids.
    Inputs:
        nparray = 1D or 2D numpy array
        xy = x-y locations , 2D array of shape [ne,2], where [:,0] is x and [:,1] is y
        xy_new = new grid x-y locations, 2D array of shape [ne2,2], where [:,0] is x and [:,1] is y
        z = z positions, 1D array of shape [nb]
        z_new = z positions of new grid, 1D array of shape [nb2]
        pre_calcs = python list of different intermediate things, specifically the
          returns of prep_xy_regrid()
        kind = one of the string options for scipy.interpolate: ['nearest','linear','cubic']
    Returns:
        nparray values regridded to shape [ne2],1D or [ne2,nb2],2D
    """   
    griddata_kinds = ['nearest','linear','cubic']

    (is_array,xy_range,zmesh,xymesh,xy_new_range,zmesh_new,xymesh_new) = \
        prep_xy_regrid(nparray,xy,xy_new,z,z_new,pre_calcs)

    if kind not in griddata_kinds:
        raise Exception,"Unknown regrid kind in xy_interpolate()"
    
    if is_array:
        valid = np.nonzero(~np.isnan(nparray))
        return scipy.interpolate.griddata(zip(xymesh[valid],zmesh[valid]),
                                          nparray[valid],
                                          (xymesh_new, zmesh_new),
                                          method=kind)
    else:
        valid = np.nonzero(~np.isnan(nparray))
        nn = np.argsort(xy_range)
        f = scipy.interpolate.interp1d(xy_range[nn][valid],nparray[nn][valid],
                                       kind=kind,bounds_error=False)
        return f(xy_new_range)
        #return np.interp(xy_new_range,xy_range[valid],nparray[valid])


def find_mask_from_vector(z,z_values,mask_area):
    """
    Masked values either above or below the z_values in the grid space defined
    by z.
    Inputs:
        z = z positions, 1D array of shape [nb]
        z_values = elevation within the values of z, shape [ne]
        mask_area = "above" or "below", describing desired mask location
    Returns:
        numpy boolean array, shape [ne,nb]
    """      
    z_values_T = -1.0 * np.array([np.squeeze(z_values)]).T  # need rank-2 array so we can transpose it
    z_values_array = z_values_T * np.ones((1,len(z)))
    z_array = -z * np.ones((len(z_values),1))
    if mask_area == 'below':
        return np.greater(z_array,z_values_array)
    elif mask_area == 'above':
        return np.less(z_array,z_values_array)
    else:
        print "Input mask_area must be set to either 'above' or 'below'"
        raise ValueError
        
            
    