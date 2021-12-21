

#!/usr/bin/python
#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Module to perform ray tracing to get path lengths for attenuation tomography.

# Input variables:

# Output variables:

# Created by Tom Hudson, 4th August 2020

# Notes:
# Depends upon ttcrpy - An external python module for computing travel times and ray tracing (see Nasr et al 2020)
# ttcrpy depends on vtk

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
# from mayavi import mlab
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import pickle
import time
import gc
from scipy.sparse.linalg import lsqr
from scipy.interpolate import griddata
import copy
import NonLinLocPy
import ttcrpy.rgrid as ttcrpy_rgrid
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

#----------------------------------------------- Define constants and parameters -----------------------------------------------
# Define any constants/parameters:

#----------------------------------------------- End: Define constants and parameters -----------------------------------------------


#----------------------------------------------- Define main functions -----------------------------------------------
class rays:
    """Class to obtain rays and path lengths for tomography, given event locations, station locations, 
    and grid nodes. Creates a class describing rays for a particular seismic phase.
    Note: All units are SI units, apart from km rather than m scales."""

    def __init__(self, x_node_labels, y_node_labels, z_node_labels, vel_grid, n_threads=1):
        """Function to initialise rays class. 
        Inputs:
        x_labels - Array of x labels for vel_grid nodes, in km. (1D np array)
        x_labels - Array of x labels for vel_grid nodes, in km. (1D np array)
        x_labels - Array of x labels for vel_grid nodes, in km. (1D np array)
        vel_grid - A 3D grid describing the velocity, in km/s, for a particular seismic phase. (np array, of shape x,y,z)
        Optional:
        n_threads - Number of threads to try and use.
        """
        # Assign the grids:
        self.x_node_labels = x_node_labels
        self.y_node_labels = y_node_labels
        self.z_node_labels = z_node_labels
        self.n_threads = n_threads
        self.grid = ttcrpy_rgrid.Grid3d(x_node_labels, y_node_labels, z_node_labels, cell_slowness=False, n_threads=self.n_threads)
        self.vel_grid = vel_grid
        # Initialise other key variables:
        self.rays_coords = [] # Will store ray coords of each (xs,ys,zs) as list of arrays
        self.rays_cell_path_lengths_grids = [] # Will store ray cell path lengths for each ray
        #self.thread_no = 1 # Number of threads to use for calculations

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx


    def add_event_rays(self, station_coords, event_coords):
        """Function to add rays for event.
        Required inputs:
        station_coords - An array of n x station coords for the event. (np array of shape n x 3)
        event_coords - A list of event coords (list, of len 3)"""
        # Calculate rays:
        tt, curr_event_rays = self.grid.raytrace(event_coords, station_coords, 1./self.vel_grid, return_rays=True)#, thread_no=self.thread_no)
        # Append to data stores:
        for i in range(len(curr_event_rays)):
            self.rays_coords.append(curr_event_rays[i])


    def calc_rays_cells_path_lengths(self):
        """Function to calculate rays cell path lengths, from curr_event_rays. Returns array of cell path lengths.
        Appends rays to  self.rays_cell_path_lengths_grids, flattened/ravelled for use in inversions."""
        # Do some initial clearing up, if required:
        try:
            self.unsampled_cell_idxs
            del self.unsampled_cell_idxs
            del self.sampled_cell_idxs
            gc.collect()
        except AttributeError:
            print('')
        # Loop over all rays, assigning path lengths to grid:
        for j in range(len(self.rays_coords)):
            if (j+1) % 1000 == 0:
                print('Processing rays for ray ',j+1,'/',len(self.rays_coords))
            ray = self.rays_coords[j]

            # Create new grid and fill:
            new_cell_path_lengths_grid = np.zeros(self.vel_grid.shape)

            # Loop over individual ray sections, calculating path length for a particular cell:
            # Try to find if multiple ray coords:
            try:
                ray.shape[1]
                multi_rays_exist = True
            except IndexError:
                multi_rays_exist = False
                #print('Skipping ray, as only one set of coordinates.')
            # And calculate cell path lengths, if exists:
            if multi_rays_exist:
                for i in range(ray.shape[0] - 1):
                    # Get cell indices for current ray section:
                    val, idx_x = self.find_nearest(self.x_node_labels, ray[i, 0])
                    val, idx_y = self.find_nearest(self.y_node_labels, ray[i, 1])
                    val, idx_z = self.find_nearest(self.z_node_labels, ray[i, 2])
                    # And calculate current path length, and append to grid:
                    path_len_curr = np.sqrt(np.sum((ray[i+1,:] - ray[i,:])**2))
                    new_cell_path_lengths_grid[idx_x, idx_y, idx_z] = path_len_curr

            # Append new grid to data store:
            self.rays_cell_path_lengths_grids.append(new_cell_path_lengths_grid.ravel())
        
        # And convert from list to path lengths second order tensor:
        self.rays_cell_path_lengths_grids = np.array(self.rays_cell_path_lengths_grids)

    def find_number_of_ray_passes_through_cells(self):
        """Function to find number of ray passes through cells.
        Creates object self.ray_sampling_grid."""
        # Create grid of ray sampling:
        self.ray_sampling_grid = np.zeros(self.vel_grid.shape)

        # Loop over all rays, assigning path lengths to grid:
        for j in range(len(self.rays_coords)):
            if (j+1) % 1000 == 0:
                print('Processing rays for ray ',j+1,'/',len(self.rays_coords))
            ray = self.rays_coords[j]

            # Loop over individual ray sections, calculating path length for a particular cell:
            # Try to find if multiple ray coords:
            try:
                ray.shape[1]
                multi_rays_exist = True
            except IndexError:
                multi_rays_exist = False
                #print('Skipping ray, as only one set of coordinates.')
            # And calculate cell path lengths, if exists:
            if multi_rays_exist:
                for i in range(ray.shape[0] - 1):
                    # Get cell indices for current ray section:
                    val, idx_x = self.find_nearest(self.x_node_labels, ray[i, 0])
                    val, idx_y = self.find_nearest(self.y_node_labels, ray[i, 1])
                    val, idx_z = self.find_nearest(self.z_node_labels, ray[i, 2])
                    # And calculate current path length, and append to grid:
                    self.ray_sampling_grid[idx_x,idx_y,idx_z] += 1
    

    def consolidate_rays_cell_path_lengths_grids(self):
        """Function to consolidate rays_cell_path_lengths_grids to 
        only hold non-zero values."""
        # Protect from running this function twice, as if do, would 
        # lose non-sampled array information:
        try:
            self.unsampled_cell_idxs
            run_func = False
            print('Not consolidated again, as already undetaken.')
        except AttributeError:
            run_func = True
        if run_func:
            # Find ray passes, if possible:
            try:
                self.ray_sampling_grid
            except AttributeError:
                self.find_number_of_ray_passes_through_cells()
            # Find indices of non-sampled cells:
            ray_sampling_grid_ravelled = self.ray_sampling_grid.ravel()
            self.unsampled_cell_idxs = np.argwhere(ray_sampling_grid_ravelled == 0)[:,0]
            self.sampled_cell_idxs = np.argwhere(ray_sampling_grid_ravelled != 0)[:,0] # (And find sampled cells, for easy reconstruction later)
            # And remove non-sampled cells from rays_cell_path_lengths_grids:
            self.rays_cell_path_lengths_grids = np.delete(self.rays_cell_path_lengths_grids, self.unsampled_cell_idxs, axis=1)
            # And find consolidated vel_grid:
            self.vel_grid_ravelled = self.vel_grid.ravel()
            self.vel_grid_ravelled = np.delete(self.vel_grid_ravelled, self.unsampled_cell_idxs, axis=0)
            print("Consolidated arrays by removing non-sampled cells. \n The info on these removed cells is held in: self.unsampled_cell_idxs.")
            # And tidy:
            gc.collect()



    def plot_vel_model_slice(self, slice_idx=0, slice_axis=0):
        """Function to plot velocity model slices.
        Optional inputs:
        slice_idx - The slice index to slice the model for (int)
        slice_axis - The axis to slice for (int)"""
        plt.figure()
        if slice_axis == 0:
            plt.imshow(self.vel_grid[slice_idx,:,:].transpose())
            plt.xlabel('y-direction (indices)')
            plt.ylabel('z-direction (indices)')
        elif slice_axis == 1:
            plt.imshow(self.vel_grid[:,slice_idx,:].transpose())
            plt.xlabel('x-direction (indices)')
            plt.ylabel('z-direction (indices)')
        elif slice_axis == 2:
            plt.imshow(self.vel_grid[:,:,slice_idx].transpose())
            plt.xlabel('x-direction (indices)')
            plt.ylabel('y-direction (indices)')
        plt.colorbar(label='Velocity ($m$ $s^{-1}$)')
        plt.show()

    def plot_all_ray_paths(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for r in self.rays_coords:
            ax.plot(r[:,0], r[:,1], r[:,2],'-k', alpha=0.1)
        ax.invert_zaxis()
        ax.set_title("All ray paths")
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        fig.tight_layout()
        plt.show()

    def save_ray_info(self, out_fname='ray_info.pkl'):
        """Saves ray class object to file.
        Optional inputs:
        out_fname - Path/filename to save data to (str)
        """
        f = open(out_fname,'wb')
        pickle.dump(self, f)
        print('Saved class object to file: ',out_fname)
        print('(Note: To load, load using pickle as rb)')


    def load_ray_info(self, in_fname):
        """Loads ray class object from file.
        Required inputs:
        in_fname - Path/filename to load data from (str)
        """      
        f = open('ray_info_S.pkl', 'rb')
        self.rays = pickle.load(f)
        return(self.rays)


class inversion:
    """Class to perform attenuation tomography inversion."""

    def __init__(self, rays):
        """Function to initialise inversion class. 
        Inputs:
        rays - A rays class containing ray tracing, as in the class described in this module.

        """
        # Assign input arguments to the class object:
        self.rays = rays
        # Assign other paramters:
        self.G = []
        self.t_stars = []
        self.seismic_phase_to_use = 'P' # Can be P or S
        self.Q_stdev_filt = 400. # Standard deviation in Q filter to use
        self.inv_info_fname = "inv_info.pkl"


    def prep_rays_for_inversion(self):
        """Function to prep the ray-tracing data for the inversion."""
        # Find ray path lengths, and the tomography tensor:
        self.rays.rays_cell_path_lengths_grids = []
        gc.collect()
        self.rays.calc_rays_cells_path_lengths()
        # Find and plot ray sampling through grid, and consolidate/remove cells with no samples:
        print("Shape before consolidation:", self.rays.rays_cell_path_lengths_grids.shape)
        self.rays.consolidate_rays_cell_path_lengths_grids()
        print("Shape after consolidation:", self.rays.rays_cell_path_lengths_grids.shape)
        # Get tomography tensor:
        self.G = self.rays.rays_cell_path_lengths_grids.copy() / self.rays.vel_grid_ravelled
        del self.rays.rays_cell_path_lengths_grids
        gc.collect()
        print('Finished data preparation for inversion')


    def prep_t_stars_from_SeisSrcMoment_for_inversion(self, moment_mags_dict_fname, seismic_phase_to_use, Q_stdev_filt=400.):
        """Function to prep t-star values found using SeisSrcMoment for inversion."""
        # Initialise parameters input into function:
        self.moment_mags_dict_fname = moment_mags_dict_fname
        self.seismic_phase_to_use = seismic_phase_to_use
        self.Q_stdev_filt = Q_stdev_filt

        # Load in magnitudes analysis data:
        # (Note: Found using SeisSrcMoment)
        mag_dict = pickle.load(open(self.moment_mags_dict_fname, 'rb'))

        # Get all t* values, in order for event:
        # (Note: order is really important, as must correspond to ray path lengths below)
        event_fnames = list(mag_dict.keys())
        t_stars = []
        for event_fname in event_fnames:
            nonlinloc_hyp_data = NonLinLocPy.read_nonlinloc.read_hyp_file(event_fname)
            stations = list(nonlinloc_hyp_data.phase_data.keys()) #list(mag_dict[event_fname].keys())
            for station in stations:
                # Check if current station t_star exists and is positive:
                try:
                    t_star_curr = mag_dict[event_fname][station]['t_star']
                except KeyError:
                    continue
                Q_stdev_curr = mag_dict[event_fname][station]['Q_stdev']
                if t_star_curr > 0.:
                    if Q_stdev_curr < Q_stdev_filt:
                        # And check whether any current station for the current t* for the current seismic phase:
                        try:
                            nonlinloc_hyp_data.phase_data[station][seismic_phase_to_use]['StaLoc']['x']
                        except KeyError:
                            continue
                        # And append t* if criteria met:
                        t_stars.append(t_star_curr)
        self.t_stars = np.array(t_stars)
        print('Number of t_star observations to use:', len(self.t_stars))


    def reconstruct_full_threeD_grid_soln(self, m):
        """Function to reconstruct full 3D grid solution.
        Note: This is neccessary since the inversion was undertaken after consolidating/removing zero 
        values from G matrix prior to the inversion.
        
        Returns:
        Q_tomo_array - An array of 1/Q values output from the inversion."""
        # Add unsampled cells back in then reshape solution back to 3D grid:
        m_all = np.zeros(len(self.rays.vel_grid.ravel()))
        m_all[self.rays.sampled_cell_idxs] = m
        self.Q_tomo_array = np.reshape(m_all, self.rays.vel_grid.shape)
        return(self.Q_tomo_array)


    def perform_inversion(self, lamb=1., Q_init=250., result_out_fname=''):
        """Function to perform the inversion, using lsqr method.
        Inputs:
        Optional:
        lamb - The damping coefficient/regularisation coefficient to use. Default is 1. (float)
        Q_init - Initial guess of Q value. Can be a single value or a 1D array of flattened
                values describing Q for each point in the consolidated 3D grid. (float, or 1D 
                array of floats)
        result_out_fname - The path/filename to save the inversion output to. If unspecified by 
                            user, will not save to file (str)
        """
        # Initialise function input parameters:
        self.lamb = lamb # Damping
        self.Q_init = Q_init # Initial guess at Q
        self.result_out_fname = result_out_fname
        # perform lsqr inversion:
        x0 = np.ones(self.G.shape[1]) / self.Q_init # Initial guess
        result = lsqr(self.G, self.t_stars, damp=self.lamb, show=True, x0=x0)
        self.m = result[0]
        # And save result, if specified:
        if len(result_out_fname) > 0:
            pickle.dump(result, open(self.result_out_fname, 'wb'))
        # And get reconstructed inversion result:
        # (Adding back in zero values)
        self.Q_tomo_array = self.reconstruct_full_threeD_grid_soln(self.m)
        return(self.Q_tomo_array)


    def perform_multi_lambda_reg_inversion(self, lambs=[1., 0.1, 1e-2, 1e-3, 1e-4], Q_init=250., results_out_fname_prefix='result_lsqr_lamb_'):
        """Function to perform inversion for mulitple damping coefficients, to find the
        optimal regualarised solution.
        Inputs:
        Optional:
        lambs - The damping coefficient/regularisation coefficients to use. (float)
        Q_init - Initial guess of Q value. Can be a single value or a 1D array of flattened
                values describing Q for each point in the consolidated 3D grid. (float, or 1D 
                array of floats)
        results_out_fname_prefix - The path/filename prefix to save the inversion output to. 
                                    If unspecified by user, will not save to file (str)
        """
        # Initialise function input parameters:
        self.lambs = lambs # List of damping/reg. coefficients
        self.Q_init = Q_init # Initial guess at Q
        self.results_out_fname_prefix = results_out_fname_prefix
        # Loop over damping coefficients, performing inversion:
        for i in range(len(self.lambs)):
            fname_out = self.results_out_fname_prefix+str(self.lambs[i])+'_'+self.seismic_phase_to_use+'.pkl'
            # Use lsqr method:
            x0 = np.ones(self.G.shape[1]) / self.Q_init # Initial guess
            result = lsqr(self.G, self.t_stars, damp=self.lambs[i], show=True, x0=x0)
            # Save result:
            pickle.dump(result, open(fname_out, 'wb'))


    def save_inversion_obj(self, out_fname='inv_info.pkl'):
        """Loads inversion class object from file.
        Required inputs:
        in_fname - Path/filename to load data from (str)
        """      
        self.inv_info_fname = out_fname
        try:
            f = open(out_fname, 'wb')
            pickle.dump(self, f)
        except OverflowError:
            f = open(out_fname, 'wb')
            inv_out = copy.deepcopy(self)
            inv_out.G = []
            gc.collect()
            pickle.dump(inv_out, f)
            del inv_out
            gc.collect()
            print('Note: Failed to save G, as >4gb')
        print('Saved class object to file: ',out_fname)
        print('(Note: To load, load using pickle as rb)')


    def load_inversion_obj(self):
        """Loads inversion class object from file.
        Required inputs:
        in_fname - Path/filename to load data from (str)
        """      
        f = open('ray_info_S.pkl', 'rb')
        self.rays = pickle.load(f)
        return(self.rays)



class plot:
    """Class to plot attenuation tomography inversion results."""

    def __init__(self, rays, inv):
        """Function to initialise plot class. 
        Inputs:
        rays - Ray tracing class containing info on ray paths. (class object)
        inv - Inversion class containing info on the inversion (class object)
        """
        # Assign input arguments to the class object:
        self.rays = rays
        self.inv = inv
        # Assign other paramters:
        self.G = []
        self.t_stars = []
        self.seismic_phase_to_use = self.inv.seismic_phase_to_use # Can be P or S
        self.Q_stdev_filt = self.inv.Q_stdev_filt # Standard deviation in Q filter to use

    
    def plot_L_curve(self):
        """Function to plot L-curve analysis for choice of damping/
        regularisation parameter.
        """
        # Calculate 2-norms to find L-curve:
        soln_norms = np.zeros(len(self.inv.lambs))
        res_norms = np.zeros(len(self.inv.lambs))
        for i in range(len(self.inv.lambs)):
            fname_in = self.inv.results_out_fname_prefix+str(self.inv.lambs[i])+'_'+self.inv.seismic_phase_to_use+'.pkl'
            result = pickle.load(open(fname_in, 'rb'))
            m = result[0]
            soln_norms[i] = np.sqrt(np.sum(m**2))
            res_norms[i] = np.sqrt(np.sum((np.matmul(self.inv.G,m) - self.inv.t_stars)**2))

        # And plot results:
        plt.figure()
        plt.plot(res_norms, soln_norms)
        for i in range(len(self.inv.lambs)):
            plt.annotate(str(self.inv.lambs[i]), (res_norms[i], soln_norms[i]))    
        plt.xlabel('Residual norms $||A x - b||_2$')
        plt.ylabel('Solution norms $|| x ||_2$')
        plt.show()


    def psuedo_threeD_interpolation(self):
        """Function to perform psuedo-3D interpolation of results.
        (Note: Currently interpolates in X-Y plane)
        (Note: Currently only interpolates for real, physical Q values (i.e. > 0))
        """
        # Setup requried data:
        X, Y = np.meshgrid(self.rays.x_node_labels, self.rays.y_node_labels)
        self.opt_Q_tomo_array_interp = np.zeros(self.rays.vel_grid.shape)

        # Loop over 2D planes in Z:
        for i in range(len(self.rays.z_node_labels)):
            # And select non-zeros values only:
            non_zero_idxs = np.argwhere(self.opt_Q_tomo_array[:,:,i] > 0.)
            # And check that there are some non-zero values:
            if non_zero_idxs.shape[0] > 0.:
                x_idxs = non_zero_idxs[:,0]
                y_idxs = non_zero_idxs[:,1]
                points = np.zeros((len(x_idxs),2))
                points[:,0] = X[x_idxs,y_idxs].ravel()
                points[:,1] = Y[x_idxs,y_idxs].ravel()
                values = self.opt_Q_tomo_array[x_idxs,y_idxs,i].ravel()
                gridded_data = griddata(points, values, (X, Y), method='linear') #, method='nearest') #, method='linear')
                self.opt_Q_tomo_array_interp[:,:,i] = gridded_data
        # And save interpolated result:
        fname_out = 'opt_Q_tomo_array_interp_'+self.inv.seismic_phase_to_use
        np.save(fname_out, self.opt_Q_tomo_array_interp)
        print('Saved interpolated data to: ', fname_out)
        return(self.opt_Q_tomo_array_interp)

    
    def load_opt_Q_tomo_result_interpolated_smoothed(self, inv_fname, spatial_smooth_sigma_km=0.0):
        """Function to load optimal Q tomography result from file
        and interpolate data.
        Inputs:
        inv_fname - The inversion data fname to plot data for.
        Optional:
        spatial_smooth_sigma - The spatial smoothing to apply, in km. 
                                Applies Gaussian filtering. Default is 0.0, 
                                which applies no filtering (float)

        Returns:
        opt_Q_tomo_array_interp - Optimal tomography array, 
                                interpolated. (3D np array)
        """
        # Load optimal data:
        opt_result = pickle.load(open(inv_fname, 'rb'))
        opt_m = opt_result[0]

        # Reconstruct full model 3D grid result from data:
        # (Add unsampled cells back in then reshape solution back to 3D grid)
        self.opt_Q_tomo_array = self.inv.reconstruct_full_threeD_grid_soln(opt_m)

        # Interpolate results:
        self.opt_Q_tomo_array_interp = self.psuedo_threeD_interpolation()

        # Apply spatial filtering, if specified:
        if spatial_smooth_sigma_km > 0.:
            grid_spacing_km = self.rays.x_node_labels[1] - self.rays.x_node_labels[0] # (Note: Assumes uniform grid spacing in x,y,z)
            gauss_filt_sigma = spatial_smooth_sigma_km / grid_spacing_km
            self.opt_Q_tomo_array_interp_smooth = gaussian_filter(self.opt_Q_tomo_array_interp, sigma=gauss_filt_sigma)
        else:
            self.opt_Q_tomo_array_interp_smooth = self.opt_Q_tomo_array_interp

        return(self.opt_Q_tomo_array_interp_smooth)


    def plot_inversion_result(self, inv_fname, plane='xz', plane_idx=0, spatial_smooth_sigma_km=0.0, cmap='viridis',
                                 fig_out_fname='', vmin=10., vmax=1000., xlims=[], ylims=[], checkerboard_inv=None,
                                 earthquakes_nonlinloc_fnames=None):
        """Plot inversion result for optimal damping parameter.
        Inputs:
        inv_fname - The inversion data fname to plot data for.
        Optional:
        plane - The plane to plot. Can be xy, xz, or yz. (str)
        plane_idx - The index of the plane to plot (int)
        spatial_smooth_sigma - The spatial smoothing to apply, in km. 
                                Applies Gaussian filtering. Default is 0.0, 
                                which applies no filtering (float)
        cmap - The matplotlib colormap to use. Default is viridis (str)
        fig_out_fname - The name of the file to save the file to, if 
                        specified. Default is not to save to file. (str)
        xlims, ylims - The x and y minimum and maximum extents to plot 
                        for the specified plane, in km. In format 
                        [xmin, xmax] , [ymin, ymax]. Default is [], 
                        which means it will use the full extent. 
                        (list of two floats each)
        checkerboard_inv - Checkerboard object. If provided, will plot the
                            locations of synthetic spikes (their widths).
                            Default = None, so will not plot. (checkerboard 
                            object)
        """
        # Load optimal data:
        opt_result = pickle.load(open(inv_fname, 'rb'))
        opt_m = opt_result[0]

        # Reconstruct full model 3D grid result from data:
        # (Add unsampled cells back in then reshape solution back to 3D grid)
        self.opt_Q_tomo_array = self.inv.reconstruct_full_threeD_grid_soln(opt_m)

        # Interpolate results:
        self.opt_Q_tomo_array_interp = self.psuedo_threeD_interpolation()

        # Apply spatial filtering, if specified:
        if spatial_smooth_sigma_km > 0.:
            grid_spacing_km = self.rays.x_node_labels[1] - self.rays.x_node_labels[0] # (Note: Assumes uniform grid spacing in x,y,z)
            gauss_filt_sigma = spatial_smooth_sigma_km / grid_spacing_km
            self.opt_Q_tomo_array_interp_smooth = gaussian_filter(self.opt_Q_tomo_array_interp, sigma=gauss_filt_sigma)
        else:
            self.opt_Q_tomo_array_interp_smooth = self.opt_Q_tomo_array_interp

        # Plot result:
        if len(xlims) > 0 and len(ylims) > 0:
            max_lim_tmp = np.max(np.abs(np.array((xlims, ylims))))
            xlims = np.array(xlims)
            ylims = np.array(ylims)
            fig, ax = plt.subplots(figsize=(3*((xlims[1]-xlims[0])/max_lim_tmp),(3*((ylims[1]-ylims[0])/max_lim_tmp))))
        else:
            fig, ax = plt.subplots(figsize=(8,4))
        # Specify plot limits:
        if len(xlims) > 0 and len(ylims) > 0:
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
        if plane == 'xy':
            # Plot data:
            Y, X = np.meshgrid(self.rays.y_node_labels, self.rays.x_node_labels)
            im = ax.pcolormesh(X, Y, 1./self.opt_Q_tomo_array_interp_smooth[:,:,plane_idx], vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm(), cmap=cmap)
            # Add text:
            ax.set_title(' '.join(("XY-plane, z =",str(self.rays.z_node_labels[plane_idx]),"km")))
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            # And plot checkerboard synthetic spike locations, if specified:
            if checkerboard_inv:
                for i in range(len(checkerboard_inv.spike_x_idxs)):
                    for j in range(len(checkerboard_inv.spike_y_idxs)):
                        x_tmp = checkerboard_inv.rays.x_node_labels[checkerboard_inv.spike_x_idxs[i]] + ((checkerboard_inv.rays.x_node_labels[1] - checkerboard_inv.rays.x_node_labels[0]) / 2)
                        y_tmp = checkerboard_inv.rays.y_node_labels[checkerboard_inv.spike_y_idxs[j]] + ((checkerboard_inv.rays.y_node_labels[1] - checkerboard_inv.rays.y_node_labels[0]) / 2)
                        circle_tmp = matplotlib.patches.Circle((x_tmp,y_tmp), radius=checkerboard_inv.spike_width_km/2., fill=False, edgecolor='white', linestyle='--')
                        ax.add_artist(circle_tmp)
            # And plot seismicity, if specified:
            if earthquakes_nonlinloc_fnames:
                for i in range(len(earthquakes_nonlinloc_fnames)):
                    nonlinloc_hyp_data = NonLinLocPy.read_nonlinloc.read_hyp_file(earthquakes_nonlinloc_fnames[i])
                    ax.scatter(nonlinloc_hyp_data.max_prob_hypocenter['x'], nonlinloc_hyp_data.max_prob_hypocenter['y'], s=2.5, c='k', alpha=0.5)
        elif plane == 'xz':         
            # Plot data:
            Z, X = np.meshgrid(self.rays.z_node_labels, self.rays.x_node_labels)
            im = ax.pcolormesh(X, Z, 1./self.opt_Q_tomo_array_interp_smooth[:,plane_idx,:], vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm(), cmap=cmap)
            ax.invert_yaxis()
            # Add text:
            ax.set_title(' '.join(("XZ-plane, y =",str(self.rays.y_node_labels[plane_idx]),"km")))
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Z (km)')
            # And plot checkerboard synthetic spike locations, if specified:
            if checkerboard_inv:
                for i in range(len(checkerboard_inv.spike_x_idxs)):
                    for j in range(len(checkerboard_inv.spike_z_idxs)):
                        x_tmp = checkerboard_inv.rays.x_node_labels[checkerboard_inv.spike_x_idxs[i]] + ((checkerboard_inv.rays.x_node_labels[1] - checkerboard_inv.rays.x_node_labels[0]) / 2)
                        z_tmp = checkerboard_inv.rays.z_node_labels[checkerboard_inv.spike_z_idxs[j]] + ((checkerboard_inv.rays.z_node_labels[1] - checkerboard_inv.rays.z_node_labels[0]) / 2)
                        circle_tmp = matplotlib.patches.Circle((x_tmp,z_tmp), radius=checkerboard_inv.spike_width_km/2., fill=False, edgecolor='white', linestyle='--')
                        ax.add_artist(circle_tmp)
            # And plot seismicity, if specified:
            if earthquakes_nonlinloc_fnames:
                for i in range(len(earthquakes_nonlinloc_fnames)):
                    nonlinloc_hyp_data = NonLinLocPy.read_nonlinloc.read_hyp_file(earthquakes_nonlinloc_fnames[i])
                    ax.scatter(nonlinloc_hyp_data.max_prob_hypocenter['x'], nonlinloc_hyp_data.max_prob_hypocenter['z'], s=2.5, c='k', alpha=0.5)
        elif plane == 'yz':
            # Plot data:
            Z, Y = np.meshgrid(self.rays.z_node_labels, self.rays.y_node_labels)
            im = ax.pcolormesh(Y, Z, 1./self.opt_Q_tomo_array_interp_smooth[plane_idx,:,:], vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm(), cmap=cmap)
            ax.invert_yaxis()
            # Add text:
            ax.set_title(' '.join(("YZ-plane, x =",str(self.rays.x_node_labels[plane_idx]),"km")))
            ax.set_xlabel('Y (km)')
            ax.set_ylabel('Z (km)')
            # And plot checkerboard synthetic spike locations, if specified:
            if checkerboard_inv:
                for i in range(len(checkerboard_inv.spike_y_idxs)):
                    for j in range(len(checkerboard_inv.spike_z_idxs)):
                        y_tmp = checkerboard_inv.rays.y_node_labels[checkerboard_inv.spike_y_idxs[i]] + ((checkerboard_inv.rays.y_node_labels[1] - checkerboard_inv.rays.y_node_labels[0]) / 2)
                        z_tmp = checkerboard_inv.rays.z_node_labels[checkerboard_inv.spike_z_idxs[j]] + ((checkerboard_inv.rays.z_node_labels[1] - checkerboard_inv.rays.z_node_labels[0]) / 2)
                        circle_tmp = matplotlib.patches.Circle((y_tmp,z_tmp), radius=checkerboard_inv.spike_width_km/2., fill=False, edgecolor='white', linestyle='--')
                        ax.add_artist(circle_tmp)
            # And plot seismicity, if specified:
            if earthquakes_nonlinloc_fnames:
                for i in range(len(earthquakes_nonlinloc_fnames)):
                    nonlinloc_hyp_data = NonLinLocPy.read_nonlinloc.read_hyp_file(earthquakes_nonlinloc_fnames[i])
                    ax.scatter(nonlinloc_hyp_data.max_prob_hypocenter['y'], nonlinloc_hyp_data.max_prob_hypocenter['z'], s=2.5, c='k', alpha=0.5)
        else:
            print('Error: Plane option', plane, 'does not exist. Exiting.')
            sys.exit()
        # plt.colorbar(label='$Q_'+self.inv.seismic_phase_to_use+'$')
        fig.colorbar(im, label='$Q_'+self.inv.seismic_phase_to_use+'$')
        # Save figure, if specified:
        if len(fig_out_fname) > 0:
            plt.savefig(fig_out_fname, dpi=300)
            print('Saved figure to:',fig_out_fname)
        # And show figure:
        plt.show()


    def plot_inversion_result_3D_slices(self, opt_lamb, plane='xz', spatial_smooth_sigma_km=0.0, cmap='viridis', fig_out_fname='', vmin=10, vmax=1000):
        """Plot inversion result for optimal damping parameter as a 3D plot with a 
        number of 2D surfaces.
        Inputs:
        opt_lamb - The optimal damping/regularisation parameter (decided 
        upon based on L-curve analysis).
        Optional:
        plane - The plane to plot. Can be xy, xz, or yz. (str)
        spatial_smooth_sigma - The spatial smoothing to apply, in km. 
                                Applies Gaussian filtering. Default is 0.0, 
                                which applies no filtering (float)
        cmap - The matplotlib colormap to use. Default is viridis (str)
        fig_out_fname - The name of the file to save the file to, if 
                        specified. Default is not to save to file. (str)
        """
        # Load optimal data:
        opt_result = pickle.load(open(self.inv.results_out_fname_prefix+str(opt_lamb)+'_'+self.seismic_phase_to_use+'.pkl', 'rb'))
        opt_m = opt_result[0]

        # Reconstruct full model 3D grid result from data:
        # (Add unsampled cells back in then reshape solution back to 3D grid)
        self.opt_Q_tomo_array = self.inv.reconstruct_full_threeD_grid_soln(opt_m)

        # Interpolate results:
        self.opt_Q_tomo_array_interp = self.psuedo_threeD_interpolation()

        # Apply spatial filtering, if specified:
        if spatial_smooth_sigma_km > 0.:
            grid_spacing_km = self.rays.x_node_labels[1] - self.rays.x_node_labels[0] # (Note: Assumes uniform grid spacing in x,y,z)
            gauss_filt_sigma = spatial_smooth_sigma_km / grid_spacing_km
            print(gauss_filt_sigma)
            self.opt_Q_tomo_array_interp_smooth = gaussian_filter(self.opt_Q_tomo_array_interp, sigma=gauss_filt_sigma)
        else:
            self.opt_Q_tomo_array_interp_smooth = self.opt_Q_tomo_array_interp

        # # # Plot result (plotly):
        # # Setup figure:
        # fig = make_subplots(rows=1, cols=1,
        #     specs=[[{'is_3d': True}]],
        #     subplot_titles=['Color corresponds to z'],)
        # # fig = make_subplots(rows=1, cols=2,
        # #             specs=[[{'is_3d': True}, {'is_3d': True}]],
        # #             subplot_titles=['Color corresponds to z', 'Color corresponds to distance to origin'],)
        # # --------- Plot xy plane: ---------
        # plane_idx = int(len(self.rays.z_node_labels) / 2. )
        # Y, X = np.meshgrid(self.rays.y_node_labels, self.rays.x_node_labels)
        # Z = self.rays.z_node_labels[plane_idx] * np.ones(X.shape)
        # # Create forth dimension to colour surfaces:
        # colour_dimension = 1./self.opt_Q_tomo_array_interp_smooth[:,:,plane_idx]
        # # And plot:
        # fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.log10(colour_dimension), cmin=np.log10(vmin), cmax=np.log10(vmax), opacity=0.6, colorscale=cmap), 1, 1)
        # # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, vmin=vmin, vmax=vmax, shade=False)
        # # --------- Plot xz plane: ---------
        # plane_idx = int(len(self.rays.y_node_labels) / 2. )
        # Z, X = np.meshgrid(self.rays.z_node_labels, self.rays.x_node_labels)
        # Y = self.rays.y_node_labels[plane_idx] * np.ones(X.shape)
        # # Create forth dimension to colour surfaces:
        # colour_dimension = 1./self.opt_Q_tomo_array_interp_smooth[:,plane_idx,:]
        # # And plot:
        # fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.log10(colour_dimension), cmin=np.log10(vmin), cmax=np.log10(vmax), opacity=0.6, colorscale=cmap), 1, 1)
        # # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, vmin=vmin, vmax=vmax, shade=False)
        # # --------- Plot xz plane: ---------
        # plane_idx = int(len(self.rays.y_node_labels) / 2. )
        # Z, X = np.meshgrid(self.rays.z_node_labels, self.rays.x_node_labels)
        # Y = self.rays.y_node_labels[plane_idx] * np.ones(X.shape)
        # # Create forth dimension to colour surfaces:
        # colour_dimension = 1./self.opt_Q_tomo_array_interp_smooth[:,plane_idx,:]
        # # And plot:
        # fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.log10(colour_dimension), cmin=np.log10(vmin), cmax=np.log10(vmax), opacity=0.6, colorscale=cmap), 1, 1)
        # # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, vmin=vmin, vmax=vmax, shade=False)
        # # --------- Finish plotting: ---------
        # fig.update_layout(coloraxis_colorbar=dict(
        #     title="QP",
        #     tickvals=[1,2,3],
        #     ticktext=["10", "100", "1000"],
        # ))
        # print(fig['layout']) #['zaxis']['autorange'] = "reversed"
        # fig.show()

        # # Plot result (mayavi):
        # # Setup figure:
        # fig = mlab.figure()
        # # --------- Plot xy plane: ---------
        # plane_idx = int(len(self.rays.z_node_labels) / 2. )
        # Y, X = np.meshgrid(self.rays.y_node_labels, self.rays.x_node_labels)
        # Z = self.rays.z_node_labels[plane_idx] * np.ones(X.shape)
        # # Create forth dimension to colour surfaces:
        # colour_dimension = 1./self.opt_Q_tomo_array_interp_smooth[:,:,plane_idx]
        # # And plot:
        # surf_xy = mlab.volume_slice(X, Y, Z, colour_dimension, vmin=vmin, vmax=vmax, plane_opacity=0.5, plane_orientation='x_axes', slice_index=plane_idx) #colormap='inferno_r', 
        # # # --------- Plot xz plane: ---------
        # # plane_idx = int(len(self.rays.y_node_labels) / 2. )
        # # Z, X = np.meshgrid(self.rays.z_node_labels, self.rays.x_node_labels)
        # # Y = self.rays.y_node_labels[plane_idx] * np.ones(X.shape)
        # # # Create forth dimension to colour surfaces:
        # # colour_dimension = 1./self.opt_Q_tomo_array_interp_smooth[:,plane_idx,:]
        # # fcolors = cm.to_rgba(colour_dimension)
        # # # And plot:
        # # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, vmin=vmin, vmax=vmax, shade=False, zorder=1)
        # fig.show()

        # Plot result:
        # Setup figure:
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca(projection='3d')
        # Setup colour dimension info:
        norm = matplotlib.colors.LogNorm(vmin, vmax) #matplotlib.colors.Normalize(vmin, vmax)
        cm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cm.set_array([])
        # --------- Plot xy plane: ---------
        plane_idx = int(len(self.rays.z_node_labels) / 2. )
        Y, X = np.meshgrid(self.rays.y_node_labels, self.rays.x_node_labels)
        Z = self.rays.z_node_labels[plane_idx] * np.ones(X.shape)
        # Create forth dimension to colour surfaces:
        colour_dimension = 1./self.opt_Q_tomo_array_interp_smooth[:,:,plane_idx]
        fcolors = cm.to_rgba(colour_dimension, alpha=0.2)
        # And plot:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, vmin=vmin, vmax=vmax, shade=False, zorder=2)
        # --------- Plot xz plane: ---------
        plane_idx = int(len(self.rays.y_node_labels) / 2. )
        Z, X = np.meshgrid(self.rays.z_node_labels, self.rays.x_node_labels)
        Y = self.rays.y_node_labels[plane_idx] * np.ones(X.shape)
        # Create forth dimension to colour surfaces:
        colour_dimension = 1./self.opt_Q_tomo_array_interp_smooth[:,plane_idx,:]
        fcolors = cm.to_rgba(colour_dimension)
        # And plot:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, vmin=vmin, vmax=vmax, shade=False, zorder=1)
        plt.show()
            

class checkerboard:
    """Class to perform checkerboard testing."""

    def __init__(self, rays):
        """Function to initialise checkerboard class. 
        Inputs:
        rays - A rays class containing ray tracing, as in the class described in this module.

        """
        # Assign input arguments to the class object:
        self.rays = rays

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def create_checkerboard_spikes_grid(self, spike_spacing_km, spike_width_km, Q_background=250., spike_rel_amp=0.2, plot_out_fname='', cmap='viridis'):
        """Function to create checkerboard spikes grid, from specified 
        spikes size and spacing.
        Inputs:
        spike_spacing_km - Spike spacing, in km. (float)
        spike_width_km - Spike half width, in km. (float)
        Optional:
        Q_background - The background Q value. Default is 250. (float)
        spike_rel_amp - The relative amplitude of the spikes above the 
                        background level, Q_background. (float)
        """
        # Specify Q grid from velocity grid:
        Q_grid = np.ones(self.rays.vel_grid.shape) * Q_background
        
        # Find spike x indices:
        i = 0
        spike_x_idxs = []
        val, idx = self.find_nearest(self.rays.x_node_labels, self.rays.x_node_labels[i] + spike_spacing_km)
        spike_spacing_lim = idx.copy() + 1
        while i < len(self.rays.x_node_labels) - spike_spacing_lim:
            if i==0:
                val, idx = self.find_nearest(self.rays.x_node_labels, self.rays.x_node_labels[i] + spike_spacing_km/2)
            else:
                val, idx = self.find_nearest(self.rays.x_node_labels, self.rays.x_node_labels[i] + spike_spacing_km)
            spike_x_idxs.append(idx)
            i = idx
        # Find spike y indices:
        i = 0
        spike_y_idxs = []
        val, idx = self.find_nearest(self.rays.y_node_labels, self.rays.y_node_labels[i] + spike_spacing_km)
        spike_spacing_lim = idx.copy() + 1
        while i < len(self.rays.y_node_labels) - spike_spacing_lim:
            if i==0:
                val, idx = self.find_nearest(self.rays.y_node_labels, self.rays.y_node_labels[i] + spike_spacing_km/2)
            else:
                val, idx = self.find_nearest(self.rays.y_node_labels, self.rays.y_node_labels[i] + spike_spacing_km)
            spike_y_idxs.append(idx)
            i = idx
        # Find spike z indices:
        i = 0
        spike_z_idxs = []
        val, idx = self.find_nearest(self.rays.z_node_labels, self.rays.z_node_labels[i] + spike_spacing_km)
        spike_spacing_lim = idx.copy() + 1
        while i < len(self.rays.z_node_labels) - spike_spacing_lim:
            if i==0:
                val, idx = self.find_nearest(self.rays.z_node_labels, self.rays.z_node_labels[i] + spike_spacing_km/2)
            else:
                val, idx = self.find_nearest(self.rays.z_node_labels, self.rays.z_node_labels[i] + spike_spacing_km)
            spike_z_idxs.append(idx)
            i = idx

        # Add multivariate Gaussian spikes into Q grid:
        X, Y, Z = np.meshgrid(self.rays.x_node_labels, self.rays.y_node_labels, self.rays.z_node_labels)
        spike_amp = Q_background * spike_rel_amp
        # Set coords for multivar gaussian:
        multivar_gauss_pos = np.zeros((X.shape[0],X.shape[1],X.shape[2],3))
        multivar_gauss_pos[:,:,:,0] = X
        multivar_gauss_pos[:,:,:,1] = Y
        multivar_gauss_pos[:,:,:,2] = Z
        # Loop over spike indices, adding to field:
        for i in range(len(spike_x_idxs)):
            mu_x = self.rays.x_node_labels[spike_x_idxs[i]]
            print(100*(i+1)/len(spike_x_idxs),'% complete')
            for j in range(len(spike_y_idxs)):
                mu_y = self.rays.y_node_labels[spike_y_idxs[j]]
                for k in range(len(spike_z_idxs)):
                    mu_z = self.rays.z_node_labels[spike_z_idxs[k]]
                    # Add a multivariate gaussian spike:
                    rv = multivariate_normal(mean=[mu_x, mu_y, mu_z], cov=(spike_width_km**2), allow_singular=True)
                    curr_gauss_spike_vals = rv.pdf(multivar_gauss_pos)
                    Q_grid = Q_grid + ( spike_amp * curr_gauss_spike_vals / np.max(curr_gauss_spike_vals) )
        del X,Y,Z,curr_gauss_spike_vals
        gc.collect()

        # Plot Q grid:
        print('Plotting Q grid')
        fig, ax = plt.subplots(figsize=(8,4))
        Z, X = np.meshgrid(self.rays.z_node_labels, self.rays.x_node_labels)
        im = ax.pcolormesh(X, Z, Q_grid[:,spike_y_idxs[int(len(spike_y_idxs)/2)],:], norm=matplotlib.colors.LogNorm(), cmap=cmap)
        ax.invert_yaxis()
        fig.colorbar(im, label='Q')
        ax.set_title('Q synth in')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        # Plot spike locations (for comparison with other plots):
        for i in range(len(spike_x_idxs)):
            for j in range(len(spike_z_idxs)):
                x_tmp = self.rays.x_node_labels[spike_x_idxs[i]] + ((self.rays.x_node_labels[1] - self.rays.x_node_labels[0]) / 2)
                z_tmp = self.rays.z_node_labels[spike_z_idxs[j]] + ((self.rays.z_node_labels[1] - self.rays.z_node_labels[0]) / 2)
                circle_tmp = matplotlib.patches.Circle((x_tmp,z_tmp), radius=spike_width_km/2., fill=False, edgecolor='white', linestyle='--')
                ax.add_artist(circle_tmp)
        if len(plot_out_fname) > 0:
            plt.savefig(plot_out_fname, dpi=300)
        plt.show()

        # And create inv Q grid and tidy up:
        self.inv_Q_grid = 1. / Q_grid
        self.spike_x_idxs = spike_x_idxs
        self.spike_y_idxs = spike_y_idxs
        self.spike_z_idxs = spike_z_idxs
        self.spike_spacing_km = spike_spacing_km
        self.spike_width_km = spike_width_km
        del Q_grid
        gc.collect()


    def create_synth_t_stars(self):
        """Creates synthetic t stars using the path lengths and velocity model
        from the rays object, and 1/Q from the cehckerboard spikes input 
        (created using checkerboard.create_checkerboard_spikes_grid()).
        Creates self.inv.t_stars output.
        """
        # Calculate G from path lengths, if haven't already:
        try:
            self.inv.G
            print('self.inv.G already exists, so continuing without recalculation.')
        except AttributeError:
            self.inv = inversion(self.rays)
            self.inv.prep_rays_for_inversion()

        # Consolidate Q values, to only use those that have ray paths going through them:
        inv_Q_grid_consolidated_ravelled = np.delete(self.inv_Q_grid.ravel(), self.inv.rays.unsampled_cell_idxs, axis=0)
            
        # And calculate synth t stars from the path lengths, vel grid and the Q grid:
        self.inv.t_stars = np.matmul(self.inv.G, inv_Q_grid_consolidated_ravelled)
        print('Number of synth t_star observations to use:', len(self.inv.t_stars))


    def perform_synth_inversion(self, lamb, Q_init=250., synth_result_out_fname=''):
        """
        Function to perform the inversion on synthetic input data.
        Inputs:
        Required:
        lamb - The damping/regularisation parameter. This should
                be the same value as used in the real data inversion.
                (float)
        Optional:
        Q_init - The initial Q value to use in the initial lsqr 
                inversion conditions. This value should be equal
                to Q_background used in the synthetics for normal
                use. Default is 250. (float)
        synth_result_out_fname - The filename to save data out to.
                                Default is no output. (str)

        Returns:
        synth_Q_tomo_array - An array containing the synthetic Q 
                            tomography result.
        """
        # Perform the inversion:
        if len(synth_result_out_fname) == 0:
            self.synth_Q_tomo_array = self.inv.perform_inversion(lamb=lamb, Q_init=Q_init)
        else:
            self.synth_Q_tomo_array = self.inv.perform_inversion(lamb=lamb, Q_init=Q_init, result_out_fname=synth_result_out_fname)
        return(self.synth_Q_tomo_array)
        


#----------------------------------------------- End: Define main functions -----------------------------------------------


#----------------------------------------------- Run script -----------------------------------------------
if __name__ == "__main__":
    # Add main script run commands here...

    print("Finished")
