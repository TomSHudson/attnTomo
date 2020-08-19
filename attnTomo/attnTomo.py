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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import time
import gc
from scipy.sparse.linalg import lsqr
from scipy.interpolate import griddata
import copy
import NonLinLocPy
import ttcrpy.rgrid as ttcrpy_rgrid
from scipy.ndimage import gaussian_filter

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
        self.lamb = 1. # Damping
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


    def plot_inversion_result(self, opt_lamb, plane='xz', plane_idx=0, spatial_smooth_sigma_km=0.0, cmap='viridis', fig_out_fname=''):
        """Plot inversion result for optimal damping parameter.
        Inputs:
        opt_lamb - The optimal damping/regularisation parameter (decided 
        upon based on L-curve analysis).
        Optional:
        plane - The plane to plot. Can be xy, xz, or yz. (str)
        plane_idx - The index of the plane to plot (int)
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

        # Plot result:
        plt.figure(figsize=(8,4))
        if plane == 'xy':
            # Plot data:
            Y, X = np.meshgrid(self.rays.y_node_labels, self.rays.x_node_labels)
            plt.pcolormesh(X, Y, 1./self.opt_Q_tomo_array_interp_smooth[:,:,plane_idx], vmin=10., vmax=1000., norm=matplotlib.colors.LogNorm(), cmap=cmap)
            plt.gca().invert_yaxis()
            # Add text:
            plt.title(' '.join(("XY-plane, z =",str(self.rays.z_node_labels[plane_idx]),"km")))
        elif plane == 'xz':         
            # Plot data:
            Z, X = np.meshgrid(self.rays.z_node_labels, self.rays.x_node_labels)
            plt.pcolormesh(X, Z, 1./self.opt_Q_tomo_array_interp_smooth[:,plane_idx,:], vmin=10., vmax=1000., norm=matplotlib.colors.LogNorm(), cmap=cmap)
            plt.gca().invert_yaxis()
            # Add text:
            plt.title(' '.join(("XZ-plane, y =",str(self.rays.y_node_labels[plane_idx]),"km")))
        elif plane == 'yz':
            # Plot data:
            Z, Y = np.meshgrid(self.rays.z_node_labels, self.rays.y_node_labels)
            plt.pcolormesh(Y, Z, 1./self.opt_Q_tomo_array_interp_smooth[plane_idx,:,:], vmin=10., vmax=1000., norm=matplotlib.colors.LogNorm(), cmap=cmap)
            plt.gca().invert_yaxis()
            # Add text:
            plt.title(' '.join(("YZ-plane, x =",str(self.rays.x_node_labels[plane_idx]),"km")))
        else:
            print('Error: Plane option', plane, 'does not exist. Exiting.')
            sys.exit()
        plt.colorbar(label='$Q_'+self.inv.seismic_phase_to_use+'$')
        plt.xlabel('X (km)')
        plt.ylabel('Z (km)')
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

    def __init__(self, x_node_labels, y_node_labels, z_node_labels, vel_grid, n_threads=1):
        """Function to initialise checkerboard class. 
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


#----------------------------------------------- End: Define main functions -----------------------------------------------


#----------------------------------------------- Run script -----------------------------------------------
if __name__ == "__main__":
    # Add main script run commands here...

    print("Finished")
