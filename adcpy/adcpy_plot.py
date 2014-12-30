# -*- coding: utf-8 -*-
"""Tools for visualizing ADCP data that is read and processed by the adcpy module
This module is imported under the main adcpy, and should be available as 
adcpy.plot. Some methods can be used to visualize flat arrays, independent of
adcpy, and the plots may be created quickly using the IPanel and QPanel 
classes.

This code is open source, and defined by the included MIT Copyright License 

Designed for Python 2.7; NumPy 1.7; SciPy 0.11.0; Matplotlib 1.2.0
2014-09 - First Release; blsaenz, esatel
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import num2date#,date2num,
import scipy.stats.stats as sp

import adcpy
from adcpy_recipes import calc_transect_flows_from_uniform_velocity_grid

U_str = 'u'
V_str = 'v'
W_str = 'w'
vel_strs = (U_str,V_str,W_str)


# Common formatting for datenums:
def fmt_dnum(dn):
    return num2date(dn).strftime('%c')


class IPanel(object):
    """
    This object stores and plots a 2D velocity map as an image.  Any of the data
    fields  (kwarg_options) may be specificed as kwargs during initialization. 
    At minimum IPanel requires 'velocity' to be set.
    """
    kwarg_options = ['use_pcolormesh',
                     'minv',      
                     'maxv',       
                     'velocity',          
                     'title',
                     'units',
                     'xlabel',
                     'ylabel',
                     'x',
                     'y',
                     'chop_off_nans',
                     'x_is_mtime',
                     'arrow_color',
                     'xy_is_lonlat',
                     'interpolation',
                     'shading',
                     'my_axes']

    def __init__(self,**kwargs):
        
        # init everything to None
        for kwarg in self.kwarg_options:
                exec("self.%s = None"%kwarg)        
        # set defaults
        self.minv = -0.25
        self.maxv = 0.25
        self.x_is_mtime = False
        self.interpolation = 'nearest'
        self.use_pcolormesh = False
        self.shading = 'flat'
        self.xy_is_lonlat = False
        self.chop_off_nans = False
        
        # read/save arguments
        for kwarg in self.kwarg_options:
            if kwarg in kwargs:
                exec("self.%s = kwargs[kwarg]"%kwarg)        
        
    def plot(self,ax=None):
        """
        Plots the data in IPanel onto the axis ax, or if ax is None,
        onto self.my_axes.
        Inputs:
            ax = matplotlib axes object, or None
        Returns:
            Nothing
        """      
        # set desired axes
        if ax is not None:
            plt.sca(ax)
        elif self.my_axes is not None:
            ax = plt.sca(self.my_axes)
        else:
            ax = plt.gca()                
        if self.minv is not None:
            mnv = ",vmin=self.minv"
        else:
            mnv = ""
        if self.minv is not None:
            mxv = ",vmax=self.maxv"
        else:
            mxv = ""
        if self.use_pcolormesh:
            vel_masked = np.ma.array(self.velocity,mask=np.isnan(self.velocity))
            if self.x is not None and self.y is not None:
                xy = "self.x,self.y,"
            else:            
                xy = ""
            plot_cmd = "pc=plt.pcolormesh(%svel_masked.T,shading=self.shading%s%s)"%(xy,mnv,mxv)
            exec(plot_cmd)
        else:
            if self.x is not None and self.y is not None:
                xy = ",extent=[self.x[0],self.x[-1],self.y[-1],self.y[0]]"
            else:            
                xy = ""
            plot_cmd = "pc=plt.imshow(self.velocity.T%s,interpolation=self.interpolation%s%s)"%(xy,mnv,mxv)

        exec(plot_cmd)
        if self.title is not None:
            plt.title(self.title)
        plt.axis('tight')
        if self.chop_off_nans:
            x_test = np.nansum(self.velocity,axis=1)
            x_test = ~np.isnan(x_test)*np.arange(np.size(x_test))
            if self.x is None:
                plt.xlim([np.nanmin(x_test),np.nanmax(x_test)])
            else:
                plt.xlim([self.x[np.nanmin(x_test)],self.x[np.nanmax(x_test)]])
                if self.x[-1] < self.x[0]:
                    plt.xlim(plt.xlim()[::-1])
            y_test = np.nansum(self.velocity,axis=0)
            y_test = ~np.isnan(y_test)*np.arange(np.size(y_test))
            plt.ylim([np.nanmin(y_test),np.nanmax(y_test)])
            if self.y is None:
                plt.ylim([np.nanmin(y_test),np.nanmax(y_test)])
            else:
                plt.ylim([self.y[np.nanmin(y_test)],self.y[np.nanmax(y_test)]])
                if self.y[-1] < self.y[0]:
                    plt.ylim(plt.ylim()[::-1])
        if self.x_is_mtime:
            ax.xaxis_date()
            plt.gcf().autofmt_xdate()
        elif self.xy_is_lonlat:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%7.4f'))       
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%7.4f'))
            plt.ylabel('Latitude [degrees N]')
            plt.xlabel('Longitude [degrees E]')            
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        plt.colorbar(pc, use_gridspec=True)
        

class QPanel(object):
    """
    This object stores and plots a 1D or 2D velocity map as a quiver plot.  Any 
    of the data fields (kwarg_options) may be specificed as kwargs during 
    initialization.  At minimum QPanel requires 'velocity' to be set.
    """
    kwarg_options = ['u_vecs',
                     'v_vecs',
                     'velocity',          
                     'title',
                     'units',
                     'xlabel',
                     'ylabel',
                     'x',
                     'y',
                     'v_scale',  # make arrow bigger or smaller, relatively speaking, defaults to 1
                     'xpand',    # fractional buffer around xy extent, to capture arrow ends
                     'x_is_mtime',
                     'arrow_color',
                     'xy_is_lonlat',
                     'equal_axes',
                     'my_axes',
                     'plot_calcs']

    def __init__(self,**kwargs):
        
        # init everything to None
        for kwarg in self.kwarg_options:
                exec("self.%s = None"%kwarg)        
        # set defaults
        self.u_vecs = 50
        self.v_vecs = 50
        self.x_is_mtime = False
        self.xy_is_lonlat = False
        self.arrow_color = 'k'
        self.v_scale = 1.0
        self.xpand = 0.33
        self.equal_axes = False
         
        # read/save arguments
        for kwarg in self.kwarg_options:
            if kwarg in kwargs:
                exec("self.%s = kwargs[kwarg]"%kwarg)

    def get_plot_calcs(self):
        """ Returns varous parameters generated from self that are required
        for plotting.
        Returns:
            u_indices = plotting indices to velocity data in u-dimension
            v_indices = plotting indices to velocity data in v-dimension
            vScale = internal scaling number for quiver arrows
            qk_value = scale number used to generate quiver legend
            x1,x2,y1,y2 = plot axes limits
        """
        dims = np.shape(self.velocity)
        u_reduction = max(1,int(dims[0]/self.u_vecs))
        u_indices = np.arange(0,dims[0],u_reduction)
        v_mag = np.sqrt(self.velocity[...,0]**2 + self.velocity[...,1]**2)
        if len(dims) == 2:
            vScale = np.nanmax(v_mag[u_indices])
            v_indices = None
        elif len(dims) == 3:
            v_reduction = max(1,int(dims[1]/self.v_vecs))
            v_indices = np.arange(0,dims[1],v_reduction)
            v_mag = v_mag[u_indices,:]
            v_mag = v_mag[:,v_indices]
            vScale = np.nanmax(np.nanmax(v_mag))
        vScale = max(vScale,0.126)
        qk_value = np.round(vScale*4)/4

        if self.xpand is not None:
            xpand = self.xpand
            xspan = np.max(self.x) - np.min(self.x)
            yspan = np.max(self.y) - np.min(self.y)
            xspan = max(xspan,yspan)
            yspan = xspan                        
            x1 = np.min(self.x) - xpand*xspan
            x2 = np.max(self.x) + xpand*xspan          
            y1 = np.min(self.y) - xpand*yspan
            y2 = np.max(self.y) + xpand*yspan
        else:
            x1=x2=y1=y2 = None

        return (u_indices,v_indices,vScale,qk_value,x1,x2,y1,y2)            
        
       

    def plot(self,ax=None,use_plot_calcs=False):
        """
        Plots the data in QPanel onto the axis ax, or if ax is None,
        onto self.my_axes.
        Inputs:
            ax = matplotlib axes object, or None
        Returns:
            Nothing
        """       
        # set desired axes
        if ax is not None:
            plt.sca(ax)
        elif self.my_axes is not None:
            ax = plt.sca(self.my_axes)
        else:
            ax = plt.gca()
            
        if not use_plot_calcs or self.plot_calcs is None:
            self.plot_calcs = self.get_plot_calcs()
        (u_indices,v_indices,vScale,qk_value,x1,x2,y1,y2) = self.plot_calcs
        dims = np.shape(self.velocity)
        if len(dims) == 2:
            local_vel = self.velocity[u_indices,...]
            local_u = local_vel[:,0]
            local_v = local_vel[:,1]
            local_x = self.x[u_indices]
            local_y = self.y[u_indices]
        elif len(dims) == 3:
            local_vel = self.velocity[u_indices,:,:]
            local_vel = local_vel[:,v_indices,:]
            local_u = local_vel[:,:,0].T
            local_v = local_vel[:,:,1].T
            local_x,local_y = np.meshgrid(self.x[u_indices],self.y[v_indices])
        
        Q = plt.quiver(local_x,local_y,
                   local_u,local_v,
                   width=0.0015*self.v_scale,
                   headlength=10.0,
                   headwidth=7.0,
                   scale = 10.0*vScale/self.v_scale,   #scale = 0.005,
                   color = self.arrow_color,
                   scale_units = 'width')
        if self.equal_axes:
            ax.set_aspect('equal')
        if self.xpand is not None:
            if x1 is not None and x2 is not None:
                plt.xlim([x1, x2])
            if y1 is not None and y2 is not None:
                plt.ylim([y1, y2])
        qk = plt.quiverkey(Q, 0.5, 0.08, qk_value, 
                       r'%3.2f '%qk_value + r'$ \frac{m}{s}$', labelpos='W',)
        if self.title is not None:
            plt.title(self.title,y=1.06)
        if self.x_is_mtime:
            ax.xaxis_date()
            plt.gcf().autofmt_xdate()
        elif self.xy_is_lonlat:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%7.4f'))       
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%7.4f'))
            plt.ylabel('Latitude [degrees N]')
            plt.xlabel('Longitude [degrees E]')
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        #plt.autoscale(True)
    
    
def get_fig(fig):
    """
    Returns a new figure if figure is None, otherwise passes returns fig.
    Inputs:
        fig = matplotlib figure object, or None
    Returns:
        fig = either passes, or new matplotlib figure object
    """
    if fig is None:
        return plt.figure()
    else:
        return fig

def plot_vertical_panels(vpanels,fig=None,title=None):
    """
    Plots a list of panels in a vertical arangement in in figure window.
    Inputs:
        fig = matplotlib figure object in which to plot, or None for a new figure
    Returns:
        nothing
    """

    fig_handle = get_fig(fig)
    plt.clf()
    n_panels = len(vpanels)
    sp_base = 100*n_panels+10
    for i in range(n_panels):
        plt.subplot(sp_base+i+1)
        vpanels[i].plot()
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    return fig_handle

def show():
    """
    Shortcut to matplotlib.pyplot.show()
    """
    plt.show()

def find_array_value_bounds(nparray,resolution):
    """
    Find the bounds of the array values, adding up + resolution to make the bounds
    a round out of a multiple of resolution.
    Inputs:
        nparray = array of numbers for which bounds are needed
        resoution = number of which the bounds will be rounded up toward
    Returns:
        minv = minimum bound value of nparray
        maxv = maximum bound value of nparray
    """
    if resolution is None:
        my_res = 0.1
    else:
        my_res = resolution
    inv = 1.0/my_res
    mtest = np.floor(nparray*inv)
    minv = np.nanmin(np.nanmin(mtest))*my_res
    mtest = np.ceil(nparray*inv)    
    maxv = np.nanmax(np.nanmax(mtest))*my_res
    return (minv,maxv)
    

def find_plot_min_max_from_velocity(velocity_2d,res=None,equal_res_about_zero=True):
    """
    Finds bounds as in find_array_value_bounds(), then optinoally 
    equates then +/- from zero.  If res is None, returns None
    Inputs:
        nparray = array of numbers for which bounds are needed [2D numpy array]
        res = number of which the bounds will be rounded up toward [number]
        equal_res_about_zero = toggle to switch [True/False]
    Returns:
        minv =- minimum bound value of nparray, or None
        maxv = maximum bound value of nparray, or None
    """    
    minv, maxv = find_array_value_bounds(velocity_2d,res)
    if equal_res_about_zero:
        maxv = np.max(np.abs(minv),np.abs(maxv))
        minv = -1.0*maxv
    return (minv,maxv)

def get_basic_velocity_panel(velocity_2d,res=None,equal_res_about_zero=True):
    """
    Returns an IPanel with from a 2D velocity array.
    Inputs:
        nparray = array of numbers for which bounds are needed
        res = number of which the bounds will be rounded up toward
        equal_res_about_zero = toggle to switch [True/False]
    Returns:
        IPanel onject 
    """
    minv, maxv = find_plot_min_max_from_velocity(velocity_2d,res,
                                                 equal_res_about_zero)
    return IPanel(velocity = velocity_2d,
                   x = None,
                   y = None,
                   minv = minv,
                   maxv = maxv,
                   units = 'm/s')
    

def plot_uvw_velocity_array(velocity,fig=None,title=None,ures=None,vres=None,wres=None,
                            equal_res_about_zero=True):
    """
    Generates a figure with three panels showing U,V,W velocity from a single 3D
    velocity array
    Inputs:
        velocity = [x,y,3] shape numpy array of 2D velocities
        fig = input figure number [integer or None]
        ures,vres,wres = numbers by which the velocity bounds will be rounded up toward [number or None]
        equal_res_about_zero = toggle to switch [True/False]
    Returns:
        fig = matplotlib figure object
    """    
    panels = []
    res = [ures, vres, wres]
    for i in range(3):
        if i == 0 and title is not None:
            title_str = title + " - "
        else:
            title_str = ""
        panels.append(get_basic_velocity_panel(velocity[:,:,i],res=res[i],equal_res_about_zero=False))
        panels[-1].title = "%s%s Velocity [m/s]"%(title_str,vel_strs[i])
        panels[-1].use_pcolormesh = False
    fig = plot_vertical_panels(panels)
    plt.tight_layout()
    return fig


def plot_secondary_circulation(adcp,u_vecs,v_vecs,fig=None,title=None):
    """
    Generates a with a single panel, plotting U velocity as an IPanel, overlain by
    VW vectors from a QPanel.
    Inputs:
        adcp = ADCPData object
        u_vecs,v_vecs = desired number of horizontal/vertical vectors [integers]
        fig = input figure number [integer or None]
        title = figure title text [string or None]
    Returns:
        fig = matplotlib figure object
    """      
    if fig is None:
        fig = plt.figure(fig,figsize=(10,4))
    else:
        plt.clf()
    xd,yd,dd,xy_line = adcpy.util.find_projection_distances(adcp.xy)
    stream_wise = get_basic_velocity_panel(adcp.velocity[:,:,1],res=0.01)
    stream_wise.x = dd
    stream_wise.y = adcp.bin_center_elevation
    stream_wise.chop_off_nans = True
    secondary = QPanel(velocity = adcp.velocity[:,:,1:],
                      x = dd,
                      y = adcp.bin_center_elevation,
                      xpand = None,
                      v_scale = 1.5,
                      u_vecs = u_vecs,
                      v_vecs = v_vecs,
                      arrow_color = 'k',
                      units = 'm/s')
    stream_wise.plot()
    secondary.plot()
    if title is not None:
        plt.title(title)
    return fig
    
def plot_secondary_circulation_over_streamwise(adcp,u_vecs,v_vecs,fig=None,title=None):
    """
    Generates a with a single panel, plotting U velocity as an IPanel, overlain by
    VW vectors from a QPanel.
    Inputs:
        adcp = ADCPData object
        u_vecs,v_vecs = desired number of horizontal/vertical vectors [integers]
        fig = input figure number [integer or None]
        title = figure title text [string or None]
    Returns:
        fig = matplotlib figure object
    """      
    if fig is None:
        fig = plt.figure(fig,figsize=(10,4))
    else:
        plt.clf()
    xd,yd,dd,xy_line = adcpy.util.find_projection_distances(adcp.xy)
    stream_wise = get_basic_velocity_panel(adcp.velocity[:,:,0],res=0.01)
    stream_wise.x = dd
    stream_wise.y = adcp.bin_center_elevation
    stream_wise.chop_off_nans = True
    secondary = QPanel(velocity = adcp.velocity[:,:,1:],
                      x = dd,
                      y = adcp.bin_center_elevation,
                      xpand = None,
                      v_scale = 1.5,
                      u_vecs = u_vecs,
                      v_vecs = v_vecs,
                      arrow_color = 'k',
                      units = 'm/s')
    stream_wise.plot()
    secondary.plot()
    if title is not None:
        plt.title(title)
    return fig

def plot_ensemble_mean_vectors(adcp,fig=None,title=None,n_vectors=50,return_panel=False):
    """
    Generates a QPanel, plotting mean uv velocity vectors in the x-y plane.
    Inputs:
        adcp = ADCPData object
        fig = input figure number [integer or None]
        title = figure title text [string or None]
        n_vectors = desired number of vectors [integer]
        return_panel = optinally return the QPanel instead of the figure
    Returns:
        fig = matplotlib figure object, or
        vectors = QPanel object
    """
    dude = np.zeros((adcp.n_ensembles,2),np.float64)
    velocity = adcp.get_unrotated_velocity()
    # this doesn't factor in depth, may integrate bad values if the have not been filtered into NaNs somehow
    dude[:,0] = sp.nanmean(velocity[:,:,0],axis=1)
    dude[:,1] = sp.nanmean(velocity[:,:,1],axis=1)
    vectors = QPanel(velocity = dude,
                      u_vecs = n_vectors,
                      arrow_color = 'k',
                      title = title,
                      units = 'm/s')
    if adcp.xy is not None:        
        vectors.x = adcp.xy[:,0]
        vectors.y = adcp.xy[:,1]
        vectors.xlabel = 'm'
        vectors.ylabel = 'm'
        vectors.equal_axes = True
    elif adcp.lonlat is not None:
        vectors.x = adcp.lonlat[:,0]
        vectors.y = adcp.lonlat[:,1]
        vectors.xy_is_lonlat = True
    else:
        vectors.x = adcp.mtime
        vectors.y = np.zeros(np.size(vectors.x))
        vectors.x_is_mtime = True
    if return_panel:
        return vectors
    else:                  
        fig = get_fig(fig)
        vectors.plot()
        plt.tight_layout()
        return fig


def plot_ensemble_uv(adcp,ens_num,fig=None,title=None,n_vectors=50,return_panel=False):
    """
    Generates a QPanel, plotting mean uv velocity vectors against 
    elevation (self.bin_center_elevation).
    Inputs:
        adcp = ADCPData object
        ens_num = ensemble index number to plot
        fig = input figure number [integer or None]
        title = figure title text [string or None]
        n_vectors = desired number of vectors [integer]
        return_panel = optinally return the QPanel instead of the figure
    Returns:
        fig = matplotlib figure object, or
        vectors = QPanel object
    """
    dude = np.zeros((adcp.n_bins,2),np.float64)
    velocity = adcp.get_unrotated_velocity()
    # this doesn't factor in depth, may integrate bad values if the have not been filtered into NaNs somehow
    dude[:,0] = velocity[ens_num,:,0]
    dude[:,1] = velocity[ens_num,:,1]
    vectors = QPanel(velocity = dude,
                      u_vecs = n_vectors,
                      arrow_color = 'k',
                      title = title,
                      units = 'm/s')
    vectors.y = -1.0 * adcp.bin_center_elevation
    vectors.x = np.zeros(np.size(adcp.bin_center_elevation))
    vectors.xlabel = 'm'
    vectors.equal_axes = True
    if return_panel:
        return vectors
    else:                  
        fig = get_fig(fig)
        vectors.plot()
        plt.tight_layout()
        return fig


def plot_obs_group_xy_lines(adcp_obs,fig=None,title=None):
    """
    Produces a quick plot of the adcp ensemble x-y locations, from
    a list of ADCPData objects.  x-y tracks lines are colored differently
    for each ADCPData object.
    Inputs:
        adcp_obs = list ADCPData objects
        fig = input figure number [integer or None]
        title = figure title text [string or None]
    Returns:
        fig = matplotlib figure object
    """        
   
    fig = get_fig(fig)
    plt.hold(True)
    legends = []
    for a in adcp_obs:
        if a.mtime is not None:
            label = a.source+"; "+fmt_dnum(a.mtime[0])
        else:
            label = a.source
        plot_xy_line(a,fig,label=label,use_stars_at_xy_locations=False)
    plt.legend(prop={'size':10})
    if title is not None:
        plt.title(title,y=1.06)
    return fig

def plot_xy_line(adcp,fig=None,title=None,label=None,use_stars_at_xy_locations=True):
    """
    Produces a quick plot of the adcp ensemble x-y locations, from an ADCPData
    object.
    Inputs:
        adcp_obs = list ADCPData objects
        fig = input figure number [integer or None]
        title = figure title text [string or None]
        use_stars_at_xy_locations = plots * at actual ensemble locations [True/False]
    Returns:
        fig = matplotlib figure object
    """
    fig = get_fig(fig)
    if adcp.xy is not None:
        x = adcp.xy[:,0]
        y = adcp.xy[:,1]
    elif adcp.lonlat is not None:
        x = adcp.lonlat[:,0]
        y = adcp.lonlat[:,1]
    else:
        raise Exception,"plot_xy_line(): no position data in ADCPData object"
    if use_stars_at_xy_locations:
        plt.plot(x,y,marker='*',label=label)
    else: 
        plt.plot(x,y,label=label)
    if title is not None:
        plt.title(title,y=1.06)
    formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    return fig
    

def plot_uvw_velocity(adcp,uvw='uvw',fig=None,title=None,ures=None,vres=None,wres=None,
                            equal_res_about_zero=True,return_panels=False,match_scales=True):
    """
    Produces a quick plot of the adcp ensemble x-y locations, from an ADCPData
    object.
    Inputs:
        adcp_obs = list ADCPData objects
        fig = input figure number [integer or None]
        title = figure title text [string or None]
        use_stars_at_xy_locations = plots * at actual ensemble locations [True/False]
    Returns:
        fig = matplotlib figure object
    """        
    panels = []
    dx = None
    dt = None
    res = [ures, vres, wres]
    if match_scales:
        res = min(res)
        res = [res, res, res]
        minv,maxv = (0.0,0.0)        
    
    if adcp.xy is not None:
        if np.size(adcp.xy[:,0]) == adcp.n_ensembles:
            xd,yd,dx,xy_line = adcpy.util.find_projection_distances(adcp.xy)
    if adcp.mtime is not None:
        if np.size(adcp.mtime) == adcp.n_ensembles:
            dt = adcp.mtime
    ax = adcpy.util.get_axis_num_from_str(uvw)
    
    for i in ax:
        if i == ax[0] and title is not None:
            title_str = title + " - "
        else:
            title_str = ""
        panels.append(get_basic_velocity_panel(adcp.velocity[:,:,i],res=res[i],
                                               equal_res_about_zero=equal_res_about_zero))
        if match_scales:
            minv = min(minv,panels[-1].minv)
            maxv = max(maxv,panels[-1].maxv)
        panels[-1].title = "%s%s Velocity [m/s]"%(title_str,vel_strs[i])
        if dx is not None:
            # plotting velocity projected along a line
            panels[-1].x = dx
            panels[-1].xlabel = 'm'                
            panels[-1].ylabel = 'm'                
            panels[-1].y = adcp.bin_center_elevation
        elif dt is not None:
            # plotting velocity ensembles vs time
            panels[-1].x = dt
            panels[-1].x_is_mtime = True
            panels[-1].y =  adcp.bin_center_elevation
            panels[-1].ylabel = 'm'                               
            #panels[-1].use_pcolormesh = False
        else:
            # super basic plot
            panels[-1].use_pcolormesh = False
            
    if match_scales:
        for p in panels:
            p.minv = minv
            p.maxv = maxv
    
    if return_panels:
        return panels
    else:
        fig = plot_vertical_panels(panels)
        return fig


def plot_flow_summmary(adcp,title=None,fig=None,ures=None,vres=None,use_grid_flows=False):
    """
    Plots projected mean flow vectors, U and V velocity profiles, and 
    associated text data on a single plot.
    Inputs:
        adcp_obs = list ADCPData objects
        fig = input figure number [integer or None]
        title = figure title text [string or None]
        ures,vres = numbers by which the velocity bounds will be rounded up toward [number or None]
        use_grid_flows = calculates flows using crossproduct flow (if available) 
          [True] or by weighted summing of grid cells [False]   
    Returns: 
        fig = matplotlib figure object
    """
    
    if adcp.xy is None:            
        ValueError('Cannot plot summary without projected data.')
        raise
    if fig is None:
        fig = plt.figure(fig,figsize=(8,10.5))
    else:
        plt.clf()
        
    vectors = plot_ensemble_mean_vectors(adcp,n_vectors=30,return_panel=True)
    vectors.x = vectors.x - np.min(vectors.x)
    vectors.y = vectors.y - np.min(vectors.y)
    u_panel,v_panel = plot_uvw_velocity(adcp,uvw='uv',fig=fig,ures=ures,
                                        vres=vres,return_panels=True)
    
    u_panel.chop_off_nans = True
    u_panel.xlabel = None
    v_panel.chop_off_nans = True

    xd,yd,dd,xy_line = adcpy.util.find_projection_distances(adcp.xy)           

    plt.subplot(221)
    vectors.plot()
    plt.subplot(413)
    u_panel.plot()
    plt.subplot(414)
    v_panel.plot()  
    plt.tight_layout()

    if title is not None:
        plt.text(0.55,0.933,title,
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)
   
    if adcp.mtime is not None:
        plt.text(0.55,0.9,'Start of Data: %s'%( num2date(adcp.mtime[0]).strftime('%c')),
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)

    if adcp.rotation_angle is not None:
        if np.size(adcp.rotation_angle) > 1:
            rot_str = 'Rozovski'
        else:
            rot_str = '%5.2f degrees'%(adcp.rotation_angle*180.0/np.pi)
    else:
        rot_str = 'None'
    plt.text(0.55,0.866,'Streawise Rotation: %s'%rot_str,
    horizontalalignment='left',
    verticalalignment='center',
    fontsize=10,
    transform=fig.transFigure)
   
   
    x1 = min(adcp.xy[:,0][np.nonzero(~np.isnan(adcp.xy[:,0]))])
    y1 = min(adcp.xy[:,1][np.nonzero(~np.isnan(adcp.xy[:,1]))])
    
    loc_string = 'Plot origin (%s) = (%i,%i)'%(adcp.xy_srs,
                                               int(x1),
                                               int(y1))
    
    plt.text(0.55,0.833,loc_string,
    horizontalalignment='left',
    verticalalignment='center',
    fontsize=10,
    transform = fig.transFigure)
    
    if not use_grid_flows and 'calc_crossproduct_flow' in dir(adcp):    
    
        wrums,wru,tsa,tcsa = adcp.calc_crossproduct_flow()
        
        plt.text(0.55,0.8,'Mean cross-product velocity [m/s]: %3.2f'%wrums,
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)
        
        plt.text(0.55,0.766,'Mean cross-product flow [m^3/s]: %12.2f'%wru,
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)

    else:
        
        (scalar_mean_vel, depth_averaged_vel, total_flow, total_survey_area) = \
            calc_transect_flows_from_uniform_velocity_grid(adcp,use_grid_only=True)        
        
        plt.text(0.55,0.8,'Mean U velocity [m/s]: %3.2f'%scalar_mean_vel[0],
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)
        
        plt.text(0.55,0.766,'Mean V velocity [m/s]: %3.2f'%scalar_mean_vel[1],
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)
        
        plt.text(0.55,0.733,'Mean U flow [m^3/s]: %12.2f'%total_flow[0],
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)

        plt.text(0.55,0.7,'Mean V flow [m^3/s]: %12.2f'%total_flow[1],
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)

    if adcp.source is not None:
        plt.text(0.55,0.633,'Sources:\n%s'%adcp.source,
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10,
        transform = fig.transFigure)
       
    return fig



def animate_plot_ensemble_uv(a,frames,interval=1000,span=None,fig=None,title=None,n_vectors=50):
    """
    Generates a matplotlib animation object, where frames are quiver plots of
    uv velocities vs. elevation.
    Inputs:
        a = ADCPData object
        frames = number of frames in animation
        internal = time between frames when showing (ms)
        span = number of ensembles to span between frames (integer),  If None,
        span is calulated by dividing the number ensembles by the number of 
        frames to arrive at even spacing of frames across data.
        fig = input figure number [integer or None]
        title = figure title text [string or None]
        n_vectors = desired number of vectors [integer]
    Returns:
        ani = matplotlib.animation.FuncAnimation object 
    """

    if span is None:
        span = min(1,int(a.n_ensembles/frames))
    stop = span*frames    
    ens_nums = range(0,stop,span)
    print a.n_ensembles, frames, span
    print 'ens_nums: ',ens_nums
    
    vec_frame = plot_ensemble_uv(a,ens_nums[0],fig=fig,title=title,n_vectors=n_vectors,return_panel=True)
    vec_frame.v_scale = 2.0
    velocity = a.get_unrotated_velocity()

    def update_plot(i,ens_nums,velocity,vec_frame):
        print 'frame #',i
        plt.clf()
        vec_frame.velocity[:,0] = velocity[ens_nums[i],:,0]
        vec_frame.velocity[:,1] = velocity[ens_nums[i],:,1]
        vec_frame.plot(use_plot_calcs=True)
    
    fig_handle = get_fig(fig)
    ani = animation.FuncAnimation(fig_handle,update_plot,frames,fargs=(ens_nums,velocity,vec_frame),interval=interval)
    return ani

#    
        
#
#
#def ani_frame():
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.set_aspect('equal')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    im = ax.imshow(rand(300,300),cmap='gray',interpolation='nearest')
#    im.set_clim([0,1])
#    fig.set_size_inches([5,5])
#
#
#    tight_layout()
#
#
#    def update_img(n):
#        tmp = rand(300,300)
#        im.set_data(tmp)
#        return im
#
#    #legend(loc=0)
#    ani = animation.FuncAnimation(fig,update_img,300,interval=30)
#    writer = animation.writers['ffmpeg'](fps=30)
#
#    ani.save('demo.mp4',writer=writer,dpi=dpi)
#    return ani


