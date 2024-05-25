from __future__ import print_function

import xarray as xr
import numpy as np

from  . import adcpy

class ADCPXArrayData(adcpy.ADCPData):
    """
    Read from a pre-existing xarray Dataset.

    the dataset should have fields like Ve,Vn,Vu
    """
    def __init__(self,ds,**kwargs):
        super(ADCPXArrayData,self).__init__(**kwargs)

        self.name=ds.attrs.get('name',str(self))
        self.filename=ds.attrs.get('filename',self.name)

        self.ds=ds 
        self.convert_from_ds()

    def convert_from_ds(self):
        # Set ADCPData members from self.ds

        # use round(...,4) to drop some FP roundoff trash
        # model data comes in with absolute z coordinate, but convert to
        # depth below surface:
        _,z_2d = xr.broadcast(self.ds.Ve, self.ds.z_surf-self.ds.location)
        z_in=z_2d.values # self.ds.location.values
        valid=np.isfinite(self.ds.Ve.values)

        min_z=round(np.nanmin(z_in[valid]),4)
        min_dz=np.round(np.nanmin(np.diff(z_in,axis=1)),4)
        max_z=round(np.nanmax(z_in[valid]),4)

        dz_sgn=np.sign(min_dz)
        min_dz=np.abs(min_dz)

        nbins=1+int(round( (max_z-min_z)/min_dz))
        new_z=np.linspace(min_z,max_z,nbins)

        def resamp(orig,axis=-1):
            """ orig: [samples,cells].
            interpolate each sample to from z_in[sample,:] to new_z
            """
            n_samples=orig.shape[0]
            new_A=np.zeros( (n_samples,len(new_z)),np.float64)
            for sample in range(n_samples):
                new_A[sample,:]= np.interp(dz_sgn*new_z,
                                           dz_sgn*z_in[sample,:],orig[sample,:],
                                           left=np.nan,right=np.nan)
            return new_A

        self.n_ensembles=len(self.ds.sample)
        self.velocity=np.array( (resamp(self.ds.Ve.values),
                                 resamp(self.ds.Vn.values),
                                 resamp(self.ds.Vu.values)) ).transpose(1,2,0)
        self.bin_center_elevation=-new_z # make it negative downward
        self.n_bins=len(new_z)
        if 'time' in self.ds:
            self.mtime=utils.to_dnum(self.ds.time.values)
        else:
            mtime=utils.to_dnum(np.datetime64("2000-01-01"))
            self.mtime=mtime * np.ones(len(self.ds.sample))

        #self.rotation_angle=0 -- should be default=None
        #self.rotation_axes=0 -- should be default=None - zero is not a valid axes rotation - it's a string ike 'uv'
        self.lonlat=np.c_[self.ds.lon.values,self.ds.lat.values]
        self.source=self.ds.attrs['source'] # self.filename
        self.name=self.name
        self.references="UnTRIM"
