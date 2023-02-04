#!/usr/bin/env python
# coding: utf-8

import os
import time
import multiprocessing as mp
import smoldyn
import smoldyn._smoldyn as S
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt

### initial paramters ###

# get seed and path --> path needs to be substituted with a function to get a list of all paths, out_path needs to be redifined, can be given as input
seed = int(time.time())

# parameters for simulations
delta_t = 1.5e-3
time_step = 0.1e-3
disp_SMDM_minimum = 10
max_simulations = 10
outlier_correction = True
k_value = 3


### start of functions ###

# given the center, the radius and the length of the cell, this function outputs important coordinates
def critical_points_cell(center, radius, length):

    cx = center[0]
    cy = center[1]
    cz = center[2]
    lx = - length/2 + cx
    clx = - length/2 + radius + cx
    crx = length/2 - radius + cx
    rx = length/2 + cx
    top = radius + cy
    bot = - radius + cy
    pos = radius + cz
    neg = - radius + cz

    return cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg

# create random z_values inside a spherocylinder, based on the x,y position
def get_z_value(center, r, length, x, y, seed1):
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, r, length)
    
    mask_left = x < clx
    mask_right = x > crx
    
    arg = np.where(mask_left, r**2 - (x - clx)**2 - (y - cy)**2, np.where(mask_right, r**2 - (x - crx)**2 - (y - cy)**2, r**2 - (y - cy)**2))
    
    rng = np.random.default_rng(seed=seed1)
    z = np.where(arg >= 0, cz + rng.choice([-1, 1], len(x))*rng.uniform(0, np.sqrt(arg), len(x)), 0)

    return z

# function to run diffusion simulation with precise positioning
def smoldyn_billiard(center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, N, D, total_time, time_step, coord_x, coord_y, coord_z, intersection, seed1, main_path = 0):
    
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    
    # this is important for running multiple simulations in parallel
    if main_path == 0:
        # create simulation environment defining the bottom left corner and the top right
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True)

    else:        
        # create simulation environment defining the bottom left corner and the top right
        smoldyn.smoldyn.__top_model_file__ = main_path
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True, path_name=main_path)
    
    # define random seed. Important because otherwise all simulations run in parallel are exactly the same.
    S.Simulation.setRandomSeed(s, int(seed1))
    s.setFlags('q')
    
    # add a molecular specie with a diffusion coefficient D
    A = s.addSpecies("A", difc = D, color = "green")

    # create the different parts of the billiard
    s_right = smoldyn.Hemisphere(center=[crx, cy, cz], radius=radius, vector=[-1, 0, 0], slices=1000, stacks=1000, name='s_right')
    s_left = smoldyn.Hemisphere(center=[clx, cy, cz], radius=radius, vector=[1, 0, 0], slices=1000, stacks=1000, name='s_left')
    s_center = smoldyn.Cylinder(start=[clx, cy, cz], end=[crx, cy, cz], radius=radius, slices=1000, stacks=1000, name='s_center')
        
    # create the billiard by merging together the different parts
    billiard = s.addSurface("billiard", panels=[s_left, s_center, s_right])

    # define action of reflection for the billiard walls
    billiard.setAction('both', [A], "reflect")

    # if there is an intersection due to cell division, create a disk to separate cells
    if type(intersection) != str:
        
        # create a disk passing through intersection points and parallel to z axis
        # disk reflects particles. Since there are no particles between disk and cell pole,
        # disk will just act as a new boundary
        
        # coordinates of intersection points
        x_int_1, y_int_1, x_int_2, y_int_2 = intersection
        
        # center of disk for intersection
        disk_center_x = (x_int_1 + x_int_2)/2
        disk_center_y = (y_int_1 + y_int_2)/2
        disk_center_z = cz
        
        # radius of disk for intersection
        disk_radius = 3*radius  # three times the radius just to make sure that I cover the hole intersection
        
        # find normal to the disk
        normal_vector_x = y_int_2 - y_int_1
        normal_vector_y = - x_int_2 + x_int_1
        normal_vector_z = 0
        
        # adding a disk raises a bound errror on Windows or OSX, but works fine with Linux
        # see: https://issueantenna.com/repo/ssandrews/Smoldyn/issues/118
        division = smoldyn.Disk(center=[disk_center_x, disk_center_y, disk_center_z], radius=disk_radius, vector=[normal_vector_x, normal_vector_y, normal_vector_z], name='division_panel')

        septum = s.addSurface('septum', panels = [division])
        
        septum.setAction('both', [A], 'reflect')
    
    seed0 = seed1
    
    # create a compartment inside the billiard and place N molecules randomly in the compartment
    for i in range(N):

        s.addMolecules(A, 1, pos=[coord_x[i], coord_y[i], coord_z[i]])
        
        seed0 += 1
    
    # add info about the molecules position at eacht time step in an output to return at the end of the function (so it's not necessary to read the file every time)
    s.addOutputData('mydata')
    s.addCommand(cmd = "molpos A mydata", cmd_type = "E")

    # run the simulation
    s.run(total_time, dt = time_step, display = False, overwrite = True, log_level = 5, quit_at_end = True)
    
    # retrieve output data after running the simulation
    data = s.getOutputData('mydata', 0)
    
    # convert output in numpy array
    data = np.array(data)
    
    return data

# probability density function for diffusion without background correction
def diff_func(dists, t, D):
    
    k = 4*D*t
    num = (2*dists/k)*np.exp(-(dists**2)/(k))
    
    return num

# function to perfom the fitting and get a diffusion value for a pixel, starting from an initial guess p0
def final_fitting_function(x, t, p0, disp_SMDM_minimum):
    
    # raise error
    if p0 <= 0:
        
        print('Diffusion coefficient cannot be <= 0')
        return

    # if there are not enough data in the cell, don't start the fitting routine
    if len(x) <= disp_SMDM_minimum:
        return 0
    
    else:
        
        # transpose the data in the cell to have all the x in first row, all the y in second etc
        x = np.array(x).transpose()
        
        # calculate squared distance
        dists = np.sqrt((x[2] - x[0])**2 + (x[3] - x[1])**2)
        
        # free memory
        x = None
        
        # pack distances and time
        xy = [dists, t]
        
        # free memory
        dists = None
        
        # function for minimization, first variable is the one used for fitting
        def find_log(D, xy):
    
            # unpack values
            dists, t = xy
        
            # calculate logarithm and return sum of log --> less expensive than calculating the function and computing the multiplication
            my_log = np.log(diff_func(dists, t, D))

            return -np.sum(my_log)
        
        # apply the minimize method
        result = minimize(find_log, p0, xy, method = 'Nelder-Mead')
        
        #return the first and only element of the array x stored in result
        return result.x[0]


### set of functions to recreate diffusion map ###

# function to recursively obtain reconstructed diffusion. D is the fitting parameter
def find_difference_per_pixel_map(D, xy):

    # unpack all variables
    p0, row, column, map_res, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, intersection, main_path, seed = xy

    # if D = 0, there is no diffusion in that pixel
    if p0 == 0:

        return 0
    
    # find important points of the cell
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
    
    # a bin is a square that has as bottom-let corner the coordinate x_bin[0], y_bin[0], as top-right corner the coordinate x_bin[1], y_bin[1]
    # I need to get indexes for start_x and start_y that fall within the pixel

    index_x = np.where((start_x_map >= left_axis + column*map_res) & (start_x_map < left_axis + (column + 1)*map_res))[0]
    index_y = np.where((start_y_map >= bot_axis + row*map_res) & (start_y_map < bot_axis + (row + 1)*map_res))[0]

    real_index = np.intersect1d(index_x, index_y)

    # find x, y, z values that fall within the analyzed pixel
    start_x_map = start_x_map[real_index]
    start_y_map = start_y_map[real_index]
    start_z_map = start_z_map[real_index]

    real_index = None
    
    # calculate how many points are in the pixel
    N = len(start_x_map)
    
    # if there is less than a threshold number of points, don't analyze the pixel
    if N < disp_SMDM_minimum:
        
        return 0

    # define maximum number of particles. This is due to a limitation of Smoldyn. The complexity of 
    # Smoldyn simulation grows as O^2, while a for loop complexity is linear. However there is some
    # overhead in starting Smoldyn, so running 50000 simulation with 1 particles is less advantageous
    # than running 50 simulation with 1000 particles, but more advantageous than running a single simulation
    # of 50000 particles. Since the random seed used for Smaldyn is always the same, this method allows
    # to speed up the simulations routine
    max_particles = 1500

    # number of loops to to run with 1500 particles
    loops = N//max_particles

    # number of particles that are left out from the loops
    q = N % max_particles

    output_start = []
    output_end = []

    # divide the starting positions in multiple arrays
    if q == 0:

        start_x = np.array_split(start_x_map, loops)
        start_y = np.array_split(start_y_map, loops)
        start_z = np.array_split(start_z_map, loops)

    else:

        start_x = np.array_split(start_x_map, loops+1)
        start_y = np.array_split(start_y_map, loops+1)
        start_z = np.array_split(start_z_map, loops+1)

        particles_number = len(start_x[-1])

        simulation1 = smoldyn_billiard(center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, particles_number, D, delta_t, time_step, start_x[-1], start_y[-1], start_z[-1], intersection, seed, main_path)

        output_start.append(simulation1[0, 1:])
        output_end.append(simulation1[-1, 1:])

    start_x_map = None
    start_y_map = None
    start_z_map = None

    # start simulation routine for each array
    for i in range(loops):

        particles_number = len(start_x[i])

        simulation1 = smoldyn_billiard(center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, particles_number, D, delta_t, time_step, start_x[i], start_y[i], start_z[i], intersection, seed, main_path)

        # get only start and final positions of the simulation
        output_start.append(simulation1[0, 1:])
        output_end.append(simulation1[-1, 1:])

    simulation1 = None
    start_x = None
    start_y = None
    start_z = None

    output_start = np.concatenate(output_start)
    output_end = np.concatenate(output_end)
    
    # create map
    x_elements = output_start[::3].copy().ravel()
    y_elements = output_start[1::3].copy().ravel()

    x_elements_jump = output_end[::3].copy().ravel()
    y_elements_jump = output_end[1::3].copy().ravel()

    output_start = None
    output_end = None

    # reorder elements in the pixel in the same way as the SMdM maps
    pixel = np.array([x_elements, y_elements, x_elements_jump, y_elements_jump]).T

    x_elements = None
    y_elements = None
    x_elements_jump = None
    y_elements_jump = None

    # find diffusion coefficient in the pixel based on the simulated displacements
    result = final_fitting_function(pixel, delta_t, D, disp_SMDM_minimum)

    pixel = None

    # calculate suqared error between the output diffusion coefficient of the simulation
    # and the diffusion coefficient obtained via SMdM
    squared_error = (p0 - result)**2

    return squared_error


# wrapper for the minimization function
def find_best_D_per_pixel_map(x, xy):

    # get info about which pixel we are analyzing
    row = x[0]
    column = x[1]
    
    # unpack all variables
    map0, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, intersection, main_path, seed = xy

    # get map resolution
    map_res = (right_axis - left_axis) / map0.shape[1]

    # find info about diffusion coefficient in that pixel
    p0 = map0.iat[row, column]

    # free memory
    map0 = None

    # change random seed
    seed = seed + row + column

    # pack again all variables together
    xy = [p0, row, column, map_res, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, intersection,main_path, seed]

    # free memory
    start_x_map = None
    start_y_map = None
    start_z_map = None

    # minimize squared error between p0 and the final output D, changing the input D at every iteration
    result = minimize(find_difference_per_pixel_map, p0, xy, method='Nelder-Mead', options={'xatol': 0.01})

    return result.x[0]

# function that takes in all the info necessary to reconstruct the diffusion with simulations
# length and width of the cell, an size of the frame are crucial to position the simulated cell properly
def final_function_map(map0, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, intersection, main_path, seed):
    
    # get z positions based on an x,y starting position in the spherocylinder
    start_z_map = get_z_value(center, radius, length, start_x_map, start_y_map, seed)

    # pack everything into a single variable
    xy = [map0, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, intersection, main_path, seed]
    
    # free memory
    start_x_map = None
    start_y_map = None
    start_z_map = None

    # get number of rows and columns
    rows = map0.shape[0]
    columns = map0.shape[1]
    
    # free memory
    map0 = None

    # create a maps of indexes based on the number of rows and columns.
    # This will correspond to a pixel map, and the applymap function will allow to analyze each pixel individually
    indexes = [[(i, j) for j in range(columns)] for i in range(rows)]
    index_map = pd.DataFrame(data=[*indexes], index=range(rows), columns=range(columns))

    indexes = None

    # start minimization routin per pixel
    index_map = index_map.applymap(lambda x: find_best_D_per_pixel_map(x, xy))

    return index_map


### set of functions to recreate diffusion in cell regions ###

# function to recursively obtain reconstructed diffusion. D is the fitting parameter
def find_difference_per_pixel_regions(D, xy):

    # unpack all variables
    p0, column, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map,  x_left_pole, x_right_pole, intersection, main_path, seed = xy

    # if D = 0, there is no diffusion in that pixel
    if p0 == 0:

        return 0
    
    # find important points of the cell
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
    
    
    # find what regions we are analyzing based on the column value
    if column == 0:
        real_index = np.where(start_x_map < x_left_pole)[0]

    elif column == 1:
        real_index = np.where((start_x_map >= x_left_pole) & (start_x_map <= x_right_pole))[0]

    elif column == 2:
        real_index = np.where(start_x_map > x_right_pole)[0]

    else:

        print('error')
        return

    # find x, y, z values that fall within the analyzed pixel
    start_x_map = start_x_map[real_index]
    start_y_map = start_y_map[real_index]
    start_z_map = start_z_map[real_index]

    real_index = None
    
    # calculate how many points are in the pixel
    N = len(start_x_map)
    
    # if there is less than a threshold number of points, don't analyze the pixel
    if N < disp_SMDM_minimum:
        
        return 0

    # define maximum number of particles. This is due to a limitation of Smoldyn. The complexity of
    # Smoldyn simulation grows as O^2, while a for loop complexity is linear. However there is some
    # overhead in starting Smoldyn, so running 50000 simulation with 1 particles is less advantageous
    # than running 50 simulation with 1000 particles, but more advantageous than running a single simulation
    # of 50000 particles. Since the random seed used for Smaldyn is always the same, this method allows
    # to speed up the simulations routine
    max_particles = 1500

    # number of loops to to run with 1500 particles
    loops = N//max_particles

    # number of particles that are left out from the loops
    q = N % max_particles

    output_start = []
    output_end = []

    # divide the starting positions in multiple arrays
    if q == 0:

        start_x = np.array_split(start_x_map, loops)
        start_y = np.array_split(start_y_map, loops)
        start_z = np.array_split(start_z_map, loops)

    else:

        start_x = np.array_split(start_x_map, loops+1)
        start_y = np.array_split(start_y_map, loops+1)
        start_z = np.array_split(start_z_map, loops+1)

        particles_number = len(start_x[-1])

        simulation1 = smoldyn_billiard(center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, particles_number, D, delta_t, time_step, start_x[-1], start_y[-1], start_z[-1], intersection, seed, main_path)

        output_start.append(simulation1[0, 1:])
        output_end.append(simulation1[-1, 1:])

    start_x_map = None
    start_y_map = None
    start_z_map = None

    # start simulation routine for each array
    for i in range(loops):

        particles_number = len(start_x[i])

        simulation1 = smoldyn_billiard(center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, particles_number, D, delta_t, time_step, start_x[i], start_y[i], start_z[i], intersection, seed, main_path)

        # get only start and final positions of the simulation
        output_start.append(simulation1[0, 1:])
        output_end.append(simulation1[-1, 1:])

    simulation1 = None
    start_x = None
    start_y = None
    start_z = None

    output_start = np.concatenate(output_start)
    output_end = np.concatenate(output_end)
    
    # create map
    x_elements = output_start[::3].copy().ravel()
    y_elements = output_start[1::3].copy().ravel()

    x_elements_jump = output_end[::3].copy().ravel()
    y_elements_jump = output_end[1::3].copy().ravel()

    output_start = None
    output_end = None

    # reorder elements in the pixel in the same way as the SMdM maps
    pixel = np.array([x_elements, y_elements, x_elements_jump, y_elements_jump]).T

    x_elements = None
    y_elements = None
    x_elements_jump = None
    y_elements_jump = None

    # find diffusion coefficient in the pixel based on the simulated displacements
    result = final_fitting_function(pixel, delta_t, D, disp_SMDM_minimum)

    pixel = None

    # calculate suqared error between the output diffusion coefficient of the simulation
    # and the diffusion coefficient obtained via SMdM
    squared_error = (p0 - result)**2

    return squared_error


# wrapper for the minimization function
def find_best_D_per_pixel_regions(x, xy):
    
    # unpack all variables
    map0, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map,  x_left_pole, x_right_pole, intersection, main_path, seed = xy

    # get info about which region we are analyzing
    column = x
    
    # find info about diffusion coefficient in the region
    p0 = map0[column]

    # free memory
    map0 = None

    # change random seed
    seed = seed + column

    # pack again all variables together
    xy = [p0, column, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map,  x_left_pole, x_right_pole, intersection, main_path, seed]

    # free memory
    start_x_map = None
    start_y_map = None
    start_z_map = None

    # minimize squared error between p0 and the final output D, changing the input D at every iteration
    result = minimize(find_difference_per_pixel_regions, p0, xy, method='Nelder-Mead', options={'xatol': 0.01})

    return result.x[0]


# function that takes in all the info necessary to reconstruct the diffusion with simulations
# length and width of the cell, an size of the frame are crucial to position the simulated cell properly
def final_function_regions(map0, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, x_left_pole, x_right_pole, intersection, main_path, seed):
    
    # get z positions based on an x,y starting position in the spherocylinder
    start_z_map = get_z_value(center, radius, length, start_x_map, start_y_map, seed)

    # pack everything into a single variable
    xy = [map0, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, x_left_pole, x_right_pole, intersection, main_path, seed]
    
    # free memory
    start_x_map = None
    start_y_map = None
    start_z_map = None
    
    map0 = None

    # create an array of indexes to determine different regions
    # 0 = left, 1 = center, 2 = right
    indexes = [[0,1,2]]
    index_map = pd.DataFrame(data=indexes, index=range(1), columns=range(3))

    # start minimization routine per region
    index_map = index_map.applymap(lambda x: find_best_D_per_pixel_regions(x, xy))

    return index_map


def process(smoldyn_path, csv_path, destination_path):
    # read the csv containing the info  about all cells in a fov
    df0 = pd.read_csv(csv_path, header = 1)
    
    
    # destination folder / name of csv file
    # this is used as the base for all output files
    base_output_path = os.path.join(destination_path, os.path.basename(csv_path)[:-4])
    
    
    # get number of rows (number of cells)
    rows = df0.shape[0]
    
    dataset = []

    # analyze each cell separately
    for row in range(rows):
        
        cell_number = row + 1
        
        print('starting cell %d' % cell_number)

        # get important data from cell
        D_left = df0.iloc[row,3]
        n_left = df0.iloc[row,5]
        D_center = df0.iloc[row,6]
        n_center = df0.iloc[row,8]
        D_right = df0.iloc[row,9]
        n_right = df0.iloc[row,11]
        length = df0.iloc[row,12]
        width = df0.iloc[row,13]
        bin_size = df0.iloc[row,14]
        left_axis = df0.iloc[row,15]
        bot_axis = df0.iloc[row,16]
        n_columns = df0.iloc[row,17]
        n_rows = df0.iloc[row,18]
        right_axis = left_axis + bin_size*n_columns
        top_axis = bot_axis + bin_size*n_rows
        neg_axis = bot_axis
        pos_axis = top_axis
        x0 = df0.iloc[row,19]
        x0 = np.array([float(value) for value in x0.split(' ')])
        y0 = df0.iloc[row,20]
        y0 = np.array([float(value) for value in y0.split(' ')])
        
        left_is_new_pole = df0.iloc[row, 22]
        x_left_pole = df0.iloc[row, 23]
        x_right_pole = df0.iloc[row, 24]
        
        intersection = df0.iloc[row,25]
        
        if intersection != 'None':
            
            intersection = np.array([float(value) for value in intersection.split(' ')])

        diff_map_0 = df0.iloc[row,26]
        diff_map_0 = np.array([float(value) for value in diff_map_0.split(' ')])
        diff_map_0 = diff_map_0.reshape(n_rows, n_columns)
        
        diff_map_0 = pd.DataFrame(diff_map_0)
        
        center = [0,0,0]
        radius = width/2
        
        map_regions = [D_left, D_center, D_right]
        
        
        # start multiprocessing routine to simulate diffusion in the analyzed cell
        # each processor will perform a simulation with a different random seed
        # all simulations will be then averaged together. This is to remove bias based on 
        # Smoldyn simulations and on z position of the particle
        cores = mp.cpu_count()
        
        # limit the maximum of simulation to max_simulations
        if cores > max_simulations:
            cores = max_simulations

        start = time.time()
        
        # start diffusion simulation for regions
        df_regions = [(map_regions, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, x0, y0, x_left_pole, x_right_pole, intersection, smoldyn_path, int(start/1000) + i) for i in range(cores)]
        
        print('starting multiprocessing routine for regions')

        # multiprocessing routine
        p = mp.Pool(cores)
        df_regions = p.starmap(final_function_regions, df_regions)
        p.terminate()
        
        print('cell %d done for regions' % cell_number)
        
        # merge all simulations together
        for i in range(len(df_regions)):

            df_regions[i] = df_regions[i].to_numpy()

        # merge all maps together
        final_map_regions = np.mean(df_regions, axis=0)
        final_map_regions = final_map_regions.tolist()
        
        # get new diffusion in the center
        D_center_new = final_map_regions[0][1]
        
        # get difference between new and old D center (used later in delta_map)
        D_center_delta = D_center_new - D_center
        
        # get info about previous diffusions in regions, reconstructed diffusions and if left pole is new or old
        final_comparison = [*map_regions, *final_map_regions[0]]
        final_comparison.append(left_is_new_pole)

        # append info to dataset
        dataset.append(final_comparison)
        
        # start diffusion simulation for maps
        df = [(diff_map_0, center, radius, length, left_axis, right_axis, bot_axis, top_axis, neg_axis, pos_axis, time_step, delta_t, disp_SMDM_minimum, x0, y0, intersection, smoldyn_path, int(start/1000) + i) for i in range(cores)]
        
        print('starting multiprocessing routine for map')

        # multiprocessing routine
        p = mp.Pool(cores)
        df = p.starmap(final_function_map, df)
        p.terminate()
        
        # merge all maps together
        for i in range(len(df)):

            df[i] = df[i].to_numpy()
                           
        final_map = np.mean(df, axis=0)
        
        # apply outlier correction in the same way as SMdM
        if outlier_correction:
            lower_boundary = np.nanquantile(final_map, 0.25) - k_value * (np.nanquantile(final_map, 0.75) - np.nanquantile(final_map, 0.25))
            upper_boundary = np.nanquantile(final_map, 0.75) + k_value * (np.nanquantile(final_map, 0.75) - np.nanquantile(final_map, 0.25))
            
            final_map = np.where((final_map < lower_boundary) | (final_map > upper_boundary), np.nanmedian(final_map), final_map)

        # convert final maps in dataframes
        final_map = pd.DataFrame(final_map)
        
        # save final maps
        out_map_name = base_output_path + "_cell_%d_final_map_sbrd.csv" % cell_number
        final_map.to_csv(out_map_name)
        
        # save figure of the reconstructed maps
        plt_path = out_map_name[:-8] + '.pdf'
        
        # create delta maps: reconstructed - SMdM
        delta_map = final_map.to_numpy() - diff_map_0.to_numpy()
        
        # convert maps into dataframes
        delta_map = pd.DataFrame(delta_map)
        
        # save delta_maps
        delta_map_name = base_output_path + '_cell_%d_delta_map_sbrd.csv' % cell_number
        delta_map.to_csv(delta_map_name)
        
        
        # do all plotting here
        D_unit = "\u03BCm\u00B2s\u207B\u00B9"
        
        cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
        
        l_ = np.round(length, 2)  # for plot name
        r_ = np.round(radius, 2)
        
        theta_r = np.linspace(-np.pi/2, np.pi/2, 50)
        theta_l = np.linspace(np.pi/2, 3*np.pi/2, 50)

        x_r = crx + radius*np.cos(theta_r)
        x_l = clx + radius*np.cos(theta_l)

        y_r = radius*np.sin(theta_r)
        y_l = radius*np.sin(theta_l)
        
        x1, x2, y1, y2 = left_axis, right_axis, bot_axis, top_axis
        
        fig, axes = plt.subplots(squeeze = False, nrows = 4, figsize=(8, 18))
        fig.suptitle('Cell %d' % cell_number)
        
        axes[0,0].plot(x_r, y_r, color = 'blue')
        axes[0,0].plot(x_l, y_l, color = 'blue')
        axes[0,0].plot([clx, crx], [top, top], color = 'blue')
        axes[0,0].plot([clx, crx], [bot, bot], color = 'blue')
        axes[0,0].plot([x_left_pole, x_left_pole],[bot, top], color = 'red')
        axes[0,0].plot([x_right_pole, x_right_pole],[bot,top], color = 'red')
        axes[0,0].text(x_left_pole, width/2, "%.2f" % x_left_pole)
        axes[0,0].text(x_right_pole, width/2, "%.2f" % x_right_pole)
        
        if type(intersection) != str:
            
            int_x1, int_y1, int_x2, int_y2 = intersection
            axes[0,0].scatter(int_x1, int_y1, color = 'r')
            axes[0,0].scatter(int_x2, int_y2, color = 'r')
            axes[0,0].plot([int_x1, int_x2], [int_y1, int_y2], color = 'r')
            axes[0,0].text(int_x1, int_y1, "%.2f, %.2f" % (int_x1, int_y1))
            axes[0,0].text(int_x2, int_y2, "%.2f, %.2f" % (int_x2, int_y2))

        axes[0,0].axis('equal')
        axes[0,0].set_title('length = ' + str(l_) + ', radius = ' + str(r_))
        axes[0,0].set_ylabel(r'$(\mu m)$')
        
        image1 = axes[1,0].imshow(diff_map_0.to_numpy(), origin="lower", cmap='viridis', extent=(x1, x2, y1, y2))
        axes[1,0].plot(x_r, y_r, color = 'blue')
        axes[1,0].plot(x_l, y_l, color = 'blue')
        axes[1,0].plot([clx, crx], [top, top], color = 'blue')
        axes[1,0].plot([clx, crx], [bot, bot], color = 'blue')
        axes[1,0].plot([x_left_pole, x_left_pole], [bot, top], color='red')
        axes[1,0].plot([x_right_pole, x_right_pole], [bot, top], color='red')
        
        if type(intersection) != str:
            
            int_x1, int_y1, int_x2, int_y2 = intersection
            axes[1,0].scatter(int_x1, int_y1, color = 'r')
            axes[1,0].scatter(int_x2, int_y2, color = 'r')
            axes[1,0].plot([int_x1, int_x2], [int_y1, int_y2], color = 'r')
            
        axes[1,0].text(-length/2, 0, "%.2f%s" % (final_comparison[0], D_unit), size="large", weight="bold")
        axes[1,0].text(-width/4, 0, "%.2f%s" % (final_comparison[1], D_unit), size="large", weight="bold")
        axes[1,0].text(x_right_pole, 0, "%.2f%s" % (final_comparison[2], D_unit), size="large", weight="bold")
        axes[1,0].text(-length/2, -0.1, 'n = ' + str(n_left), size="large", weight="bold")
        axes[1,0].text(-width/4, -0.1, 'n = ' + str(n_center), size="large", weight="bold")
        axes[1,0].text(x_right_pole, -0.1, 'n = ' + str(n_right), size="large", weight="bold")
        
        image2 = axes[2,0].imshow(final_map.to_numpy(), origin="lower", cmap='viridis', extent=(x1, x2, y1, y2))
        axes[2,0].plot(x_r, y_r, color = 'blue')
        axes[2,0].plot(x_l, y_l, color = 'blue')
        axes[2,0].plot([clx, crx], [top, top], color = 'blue')
        axes[2,0].plot([clx, crx], [bot, bot], color = 'blue')
        axes[2,0].plot([x_left_pole, x_left_pole], [bot, top], color='red')
        axes[2,0].plot([x_right_pole, x_right_pole], [bot, top], color='red')
        
        if type(intersection) != str:
            
            int_x1, int_y1, int_x2, int_y2 = intersection
            axes[2,0].scatter(int_x1, int_y1, color = 'r')
            axes[2,0].scatter(int_x2, int_y2, color = 'r')
            axes[2,0].plot([int_x1, int_x2], [int_y1, int_y2], color = 'r')
            
        axes[2,0].plot([x_left_pole, x_left_pole],[bot, top], color = 'red')
        axes[2,0].plot([x_right_pole, x_right_pole],[bot,top], color = 'red')
        axes[2,0].text(-length/2, 0, "%.2f%s" % (final_comparison[3], D_unit), size="large", weight="bold")
        axes[2,0].text(-width/4, 0, "%.2f%s" % (final_comparison[4], D_unit), size="large", weight="bold")
        axes[2,0].text(x_right_pole, 0, "%.2f%s" % (final_comparison[5], D_unit), size="large", weight="bold")
        axes[2,0].text(-length/2, -0.1, 'n = ' + str(n_left), size="large", weight="bold")
        axes[2,0].text(-width/4, -0.1, 'n = ' + str(n_center), size="large", weight="bold")
        axes[2,0].text(x_right_pole, -0.1, 'n = ' + str(n_right), size="large", weight="bold")
        
        image3 = axes[3,0].imshow(delta_map.to_numpy(), origin="lower", cmap='bwr', extent=(x1, x2, y1, y2), vmin = -5*D_center_delta, vmax = 5*D_center_delta)
        axes[3,0].plot(x_r, y_r, color = 'blue')
        axes[3,0].plot(x_l, y_l, color = 'blue')
        axes[3,0].plot([clx, crx], [top, top], color = 'blue')
        axes[3,0].plot([clx, crx], [bot, bot], color = 'blue')
        axes[3,0].plot([x_left_pole, x_left_pole], [bot, top], color='red')
        axes[3,0].plot([x_right_pole, x_right_pole], [bot, top], color='red')
        
        if type(intersection) != str:
            
            int_x1, int_y1, int_x2, int_y2 = intersection
            axes[3,0].scatter(int_x1, int_y1, color = 'r')
            axes[3,0].scatter(int_x2, int_y2, color = 'r')
            axes[3,0].plot([int_x1, int_x2], [int_y1, int_y2], color = 'r')
            
        images = [image1, image2, image3]
        
        titles = ["SMdM", "SBRD-SMdM", "Delta Map"]
        
        for i in range(3):
            axes[i+1,0].set_title(titles[i])
            axes[i+1,0].set_ylabel(r'$(\mu m)$')
            cbar = fig.colorbar(images[i], ax=axes[i+1,0])
            cbar.set_label(r'$D$ $(\mu m^2/s)$')
            
        axes[-1,0].set_xlabel(r'$(\mu m)$')
        
        # for delta image
        cbar.set_label(r'$\Delta$ $D$ $(\mu m^2/s)$')
        
        plt_path = out_map_name[:-18] + 'maps_comparison.pdf'
        plt.savefig(plt_path, dpi=300, format='pdf')
        plt.close('all')
        
        
        
        print('cell %d done for map' % cell_number)
        
        
    # create dataframe for comparison of regions and save it
    columns_names = ['D_left', 'D_center', 'D_right', 'D_left_SBRD', 'D_center_SBRD', 'D_right_SBRD', 'left_is_new']
    table_comparison = pd.DataFrame(dataset, index = None, columns = columns_names)
    out_name = base_output_path + '_regions_comparison.csv'
    table_comparison.to_csv(out_name)
    
    return

# function to wrap all previous functions together
def start_program(smoldyn_path, source_path, destination_path):
     for fn in os.listdir(source_path):
        if fn.endswith(".csv"):
            csv_path = os.path.join(source_path, fn)
            process(smoldyn_path, csv_path, destination_path)