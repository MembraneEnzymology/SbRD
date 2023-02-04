# import packages
import warnings
import time
import numpy as np

# remove runtime warnings. array([1,2,np.nan]) > 0 outputs T, T, F, which is correct, but raises a runtime_warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

'''change the following values to perform simulations with different parameters'''

# dimension (choose between 2 or 3 for 2D or 3D simulation)
dimension = 3
# center of the spherocylinder
center_x, center_y, center_z = 0, 0, 0
# length and widht of the spherocylinder
length, width = 4, 2
radius = width / 2
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
map_res = 0.05
# initial guess for diffusion
p0 = 1
# minimum number of displacements per pixel
disp_SMDM_minimum = 10
# folder where to save output
subpath = 'files'
# random seed for smoldyn
seed = int(time.time())

if __name__ == "__main__":
    
    if dimension == 2:
        center = np.array([center_x, center_y])
        import scripts.all_functions_2D as run
        bounce_map = run.start_program(center, length, width, D, N, total_time, time_step, delta_t, map_res, p0, disp_SMDM_minimum, subpath, seed)
        
    elif dimension == 3:
        center = np.array([center_x, center_y, center_z])
        import scripts.all_functions_3D as run
        bounce_map = run.start_program(center, length, width, D, N, total_time, time_step, delta_t, map_res, p0, disp_SMDM_minimum, subpath, seed)
    else:
        print('wrong value for dimension')
        