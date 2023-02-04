#!/usr/bin/env python
# coding: utf-8

# import packages
import warnings
import inspect
from pathlib import Path
import numpy as np

alpha = Path(inspect.stack()[-1].filename)

'''change the following values to perform simulations with different parameters'''

# insert here if you want to run simulation in 2D or in 3D:
# input either 2D or 3D as a string
sim_dimensions = '3D'
# center of the spherocylinder
center_x, center_y, center_z = 0, 0, 0
# length and widht of the spherocylinder
length, width = 2.25, 0.9
# diffusion coefficient for the simulation in um^2/s
D = 20
# number of particles to perform the simulation
N = 25
# total time for the simulation to run in seconds
total_time = 2
# time step for the simulation to evolve in seconds
time_step = 0.1e-3
# delta_t to pair the start and end position
delta_t = 1.5e-3
# pixel size of the map in um (side of the square)
map_res = 0.1
# initial guess for diffusion
p0 = 1
# minimum number of displacements per pixel
disp_SMDM_minimum = 40
# maximum number of simulations for reconstructed diffusion
max_simulations = 10
# type of simulation to run
# choose between: diffusion, aggregation, interaction, two_components
sim_type = 'diffusion'
# slowdown factor: from 0 to 1. Used for aggregation, interaction and two components simulations.
# Diffusion of the slower specie is going to be slowdown_factor*D
slowdown_factor = 0.5

path_to_add0 = '/files'

# remove runtime warnings. array([1,2,np.nan]) > 0 outputs T, T, F, which is correct, but raises a runtime_warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":

    # import functions created for the program
    import billiard_simulation_multi.all_functions_2D as run_2D
    import billiard_simulation_multi.all_functions_3D as run_3D
            
    print('simulation for ' + str(D) + ' as diffusion and ' + str(N) + ' as particles.')
    print()

    if sim_dimensions == '2D':
        print('starting protocol for 2D simulation')
        print()
        center = np.array([center_x, center_y])
        final_map = run_2D.start_program(alpha, center, length, width, D, N, total_time, time_step, delta_t, map_res, p0, disp_SMDM_minimum, max_simulations, sim_type, slowdown_factor, 0, path_to_add0)
        
    elif sim_dimensions == '3D':
        print('starting protocol for 3D simulation')
        print()
        center = np.array([center_x, center_y, center_z])
        final_map = run_3D.start_program(alpha, center, length, width, D, N, total_time, time_step, delta_t, map_res, p0, disp_SMDM_minimum, max_simulations, sim_type, slowdown_factor, 0, path_to_add0)
        
    else:
        print('Error, wrong dimensionality used as input. Check for typo.')