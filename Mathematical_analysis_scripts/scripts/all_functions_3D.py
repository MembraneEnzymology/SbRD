#!/usr/bin/env python
# coding: utf-8

# Periodic boundaries with surfaces
import os
import pandas as pd
import numpy as np
import smoldyn
import smoldyn._smoldyn as S
import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

### start of functions ###

# get important points of the billiard
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



def smoldyn_billiard(center, radius, length, N, D, total_time, time_step, seed1, path, main_path = 0):
    
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    neg_axis = neg - 0.5
    pos_axis = pos + 0.5
    
    # define length of the horizontal part of the billiard
    horizontal_size = [crx - clx]
    
    if main_path == 0:

        # create simulation environment defining the bottom left corner and the top right
        s = smoldyn.Simulation(low = [left_axis, bot_axis, neg_axis], high = [right_axis, top_axis, pos_axis], log_level = 5, quit_at_end = True)
        
    else:
        
        # create simulation environment defining the bottom left corner and the top right
        smoldyn.smoldyn.__top_model_file__ = main_path
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True, path_name=main_path)
    
    S.Simulation.setRandomSeed(s, int(seed1))
    s.setFlags('q')
    
    # add a molecular specie with a diffusion coefficient D
    A = s.addSpecies("A", difc = D, color = "green")

    # create the different parts of the billiard
    s_right = smoldyn.Hemisphere(center = [crx, cy, cz], radius = radius, vector = [-1 ,0, 0], slices = 1000, stacks = 1000, name = 's_right')
    s_left = smoldyn.Hemisphere(center = [clx, cy, cz], radius = radius, vector = [1, 0, 0], slices = 1000, stacks = 1000, name = 's_left')
    s_center = smoldyn.Cylinder(start = [clx, cy, cz], end = [crx, cy, cz], radius = radius, slices = 1000, stacks = 1000, name = 's_center')

    # create the billiard by merging together the different parts
    billiard = s.addSurface("billiard", panels=[s_left, s_center, s_right])

    # define action of reflection for the billiard walls
    billiard.setAction('both', [A], "reflect")
    billiard.setStyle('both', color = 'white', thickness = 1.0)

    # create a compartment inside the billiard and place N molecules randomly in the compartment
    cell = s.addCompartment('cell', surface = billiard, point = [cx, cy, cz])
    cell.simulation.addCompartmentMolecules('A', N, 'cell')
    
    # define file name
    fname = '3D_billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(rx-lx) + '_diam_' + str(top-bot) + '_D_' + str(D) + '_N_' + str(N) + '.npz'
    fpath = path + '\\' + fname
    
    # add info about the molecules position at eacht time step in an output to return at the end of the function (so it's not necessary to read the file every time)
    s.addOutputData('mydata')
    s.addCommand(cmd = "molpos A mydata", cmd_type = "E")

    # run the simulation
    s.run(total_time, dt = time_step, display = False, overwrite = True, log_level = 5, quit_at_end = True)
    
    # retrieve output data after running the simulation
    data = s.getOutputData('mydata', 0)
    
    # convert output in numpy array
    data = np.array(data)
    
    return data, fpath, fname

# save simulation file
def save_data(data, fpath):
    
    np.savez_compressed(fpath, data = data)
    
    return

# create displacements map
def create_map(center, radius, length, array, map_res, delta_t, time_step, D, N, path):
    
    # rows to skip are given by delta_t/time_step
    jump = int(delta_t/time_step)
    
    x_elements = array[:-jump, 1::3].copy().ravel()
    y_elements = array[:-jump, 2::3].copy().ravel()

    x_elements_jump = array[jump:, 1::3].copy().ravel()
    y_elements_jump = array[jump:, 2::3].copy().ravel()
    
    array = None
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
    
    # define number of pixels in x and y axes
    x_pixels = int(np.ceil((rx - lx)/map_res))
    y_pixels = int(np.ceil((top - bot)/map_res))
    
    # define bin edges in x and y axes
    x_bins = np.array([lx + i*map_res for i in range(x_pixels + 1)])
    y_bins = np.array([bot + i*map_res for i in range(y_pixels + 1)])

    # identify which x_elements fall between which bins and return the indices of the x_elements in the order of the bins
    # AKA Return the indices of the bins to which each value in input array belongs.
    x_indices = np.digitize(x_elements, x_bins)
    y_indices = np.digitize(y_elements, y_bins)
    
    x_indices = x_indices.ravel() - 1
    y_indices = y_indices.ravel() - 1
    
    # create empty dataframe with y_pixel rows and x_pixel columns
    df = pd.DataFrame(data = [], index = range(y_pixels), columns = range(x_pixels))

    # substitute nan with empty list to append values
    df = df.applymap(lambda x: [])

    # iterate through each row of the database
    for i in range(len(x_indices)):
            
        df.iat[y_indices[i], x_indices[i]].append(([x_elements[i], y_elements[i], x_elements_jump[i], y_elements_jump[i]]))
            
    # define file name
    fname = '3D_billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(rx-lx) + '_diam_' + str(top-bot) + '_D_' + str(D) + '_N_' + str(N) + '_map_df.csv.zip'
    fpath = path + '\\' + fname
    
    return df, fpath, fname

# save displacements map
def save_df(df, fpath, fname):
    
    df.to_csv(fpath)
    
    return fname

# gaussian distribution for displacements
def diff_func(dists, t, D):
    
    k = 4*D*t
    
    return (1/(np.pi*k))*np.exp(-(dists**2)/(k))

# fitting function to obtain diffusion coefficient
def final_fitting_function(x, t, p0, disp_SMDM_minimum):
    
    if p0 <= 0:
        
        print('Diffusion coefficient cannot be <= 0')
        return

    # if there are not enough data in the cell, don't start the fitting routine
    if len(x) <= disp_SMDM_minimum:
        return 0
    
    else:
        
        # transpose the data in the cell to have all the x in first row, all the y in second etc
        x = np.array(x).transpose()
        
        # assign the data to variables
        x0 = x[0]
        y0 = x[1]
        x1 = x[2]
        y1 = x[3]
        
        # calculate squared distance
        dists = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        # pack distance and time
        xy = [dists, t]
        
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
    
# create map of fitted values
def create_map_fit(df, t, p0, disp_SMDM_minimum):
    
    df = df.applymap(lambda x: final_fitting_function(x, t, p0, disp_SMDM_minimum))
    map_fit_array = df.to_numpy()
    map_fit_array = map_fit_array.ravel()
    
    indexes = np.where(map_fit_array == 0)[0]
    map_fit_array = np.delete(map_fit_array, indexes)
    
    average = np.mean(map_fit_array)
    
    return df, average

# save map fit
def save_map_fit(map_fit, fpath, fname):
    
    file_path = fpath[:-10] + 'diff_first_fit.csv.zip'
    file_name = fname[:-10] + 'diff_first_fit.csv.zip'
    
    map_fit.to_csv(file_path)
    
    return file_name


############ FIND INTERSECTION POINT ON THE CIRCUMFERENCE ###############

# define positive equation to use with fsolve
def eq_pos_func_arr(yc, args):

    r, x0, y0, x1, y1 = args
    args = None

    # we are interested only in real results
    xc = np.where((r**2 - yc**2) >= 0, np.sqrt(r**2 - yc**2, where=(r**2 - yc**2) >= 0), 0)

    a0 = yc - y0
    b0 = x0 - xc
    a1 = yc - y1
    b1 = x1 - xc

    x0 = None
    y0 = None
    x1 = None
    y1 = None

    d0 = np.where((a0**2 + b0**2) >= 0, np.sqrt(a0**2 + b0**2, where=(a0**2 + b0**2) >= 0), 0)
    d1 = np.where((a1**2 + b1**2) >= 0, np.sqrt(a1**2 + b1**2, where=(a1**2 + b1**2) >= 0), 0)

    eq_pos = (a0*xc + b0*yc)*d1 + (a1*xc + b1*yc)*d0

    return eq_pos


# define negative equation to use with fsolve
def eq_neg_func_arr(yc, args):

    r, x0, y0, x1, y1 = args
    args = None

    # we are interested only in real results
    xc = np.where((r**2 - yc**2) >= 0, - np.sqrt(r**2 - yc**2, where=(r**2 - yc**2) >= 0), 0)

    a0 = yc - y0
    b0 = x0 - xc
    a1 = yc - y1
    b1 = x1 - xc

    x0 = None
    y0 = None
    x1 = None
    y1 = None

    d0 = np.where((a0**2 + b0**2) >= 0, np.sqrt(a0**2 + b0**2, where=(a0**2 + b0**2) >= 0), 0)
    d1 = np.where((a1**2 + b1**2) >= 0, np.sqrt(a1**2 + b1**2, where=(a1**2 + b1**2) >= 0), 0)

    eq_neg = (a0*xc + b0*yc)*d1 + (a1*xc + b1*yc)*d0

    return eq_neg


# function to create a set of possible solutions and then get (almost) all solutions
# from a domain. This is to overcome limitations of fsolve method
def my_solver_arr(args, domain, equation_pos = eq_pos_func_arr, equation_neg = eq_neg_func_arr, step_size = 1e-2, tolerance = 1e-8):
    
    r, x0, y0, x1, y1, xa, ya = args
    args = None
    
    x0 = x0 - xa
    y0 = y0 - ya
    x1 = x1 - xa
    y1 = y1 - ya
    
    # define points in the domain
    yc = np.arange(domain[0], domain[1] + step_size, step_size)
    
    # x0 has shape 1xN, yc has shape 1xM
    # I want to test each x0, y0, x1, y1 with each xc, yc
    
    x0_t = np.repeat([x0], len(yc), axis = 0).T.ravel()
    y0_t = np.repeat([y0], len(yc), axis = 0).T.ravel()
    x1_t = np.repeat([x1], len(yc), axis = 0).T.ravel()
    y1_t = np.repeat([y1], len(yc), axis = 0).T.ravel()
    
    yc_t = np.repeat([yc], len(x0), axis = 0).ravel()
    
    # when comparing now I will have a flattened matrix of N rows and M columns
    # each column represents a value of xc, yc, each row a value of x0, y0, x1, y1
    
    rows = len(x0)
    columns = len(yc)
    
    x0 = None
    y0 = None
    x1 = None
    y1 = None
    yc = None
    
    # positive case
    
    xc_pos = np.where((r**2 - yc_t**2) >= 0, np.sqrt(r**2 - yc_t**2, where=(r**2 - yc_t**2) >= 0), 0)
    
    a0_pos = yc_t - y0_t
    b0_pos = x0_t - xc_pos
    a1_pos = yc_t - y1_t
    b1_pos = x1_t - xc_pos
    
    d0_pos = np.where((a0_pos**2 + b0_pos**2) >= 0, np.sqrt(a0_pos**2 + b0_pos**2, where=(a0_pos**2 + b0_pos**2) >= 0), 0)
    d1_pos = np.where((a1_pos**2 + b1_pos**2) >= 0, np.sqrt(a1_pos**2 + b1_pos**2, where=(a1_pos**2 + b1_pos**2) >= 0), 0)

    eq_pos = (a0_pos*xc_pos + b0_pos*yc_t)*d1_pos + (a1_pos*xc_pos + b1_pos*yc_t)*d0_pos
    
    a0_pos = None
    b0_pos = None
    a1_pos = None
    b1_pos = None
    d0_pos = None
    d1_pos = None
    xc_pos = None
    
    # check if any of the results is close to zero
    index_zeros_pos = np.where(np.isclose(eq_pos, 0, rtol = 0, atol = tolerance))[0]
    
    # check if any result crosses zero, it needs to be done row by row
    eq_pos = np.reshape(eq_pos, (rows, columns))
    index_zeros_cross = np.where(eq_pos >= 0, 1, 0)
    index_cross_x, index_cross_y = np.where(index_zeros_cross[:, :-1] != index_zeros_cross[:, 1:])
    index_zeros_cross = None
    index_cross = (index_cross_x, index_cross_y)
    index_cross_plus_one = (index_cross_x, index_cross_y+1)
    
    # find index closest to zero for resulsts that cross zero
    cross_indexes = np.where(np.abs(eq_pos[index_cross_x, index_cross_y]) < np.abs(eq_pos[index_cross_x, index_cross_y+1]), index_cross, index_cross_plus_one)
    
    index_cross = None
    index_cross_plus_one = None
    eq_pos = None
    
    # find to which element of a flattened array the xy index correspond to
    # e.g.: 5th element of the 2nd row --> you need to multiply the number
    # of columns by row value, and then add the column value
    cross_indexes_flattened = cross_indexes[0]*columns + cross_indexes[1]
    
    # combine index of values close to zero with index of values crossing zero
    final_index = np.unique(np.concatenate((index_zeros_pos, cross_indexes_flattened)))
    index_zeros_pos = None
    cross_indexes_flattened = None
    
    # create nan mask except where you want to keep the value
    mask = np.empty(len(yc_t))
    mask[:] = np.nan
    mask[final_index] = 0
    
    yc_t_pos = np.where(mask == 0, yc_t, np.nan)
    
    mask = None

    # find if I have zeros close to each other, it is possible that they are
    # corresponding to a single real zero, so I want to remove all of them except one
    yc_t_pos = np.reshape(yc_t_pos, (rows, columns))
    compare_index_pos = np.where(np.isclose(yc_t_pos[:, :-1], yc_t_pos[:, 1:], rtol=0, atol= 1.5*step_size))
    yc_t_pos[compare_index_pos[0], compare_index_pos[1]] = np.nan
    
    compare_index_pos = None
    
    # find positions where I have zeros for yc in the domain
    # which are the non nan element.
    # I want to know how many zeros I have for each start-end separately
    # My matrix has start-end as rows, reflection points as columns,
    # Hence I need to sum column-wise
    mask_sum = np.sum(np.where(np.isnan(yc_t_pos[:,:]), 1, 0), axis=1)
    
    # If a row is made of 1, it means that each circumference point is a reflection point,
    # which happen only if start and end are in the center of the circle
    # In that case I want to substitute every value with nan and then store -1 
    # as first value only
    check_R_pos = np.where(mask_sum == columns)[0]
    
    mask_sum = None
    
    yc_t_pos[check_R_pos,:] = np.nan
    
    # Now I can reflatten yc_t and apply fsolve to the points where I have zeros
    yc_t_pos = yc_t_pos.ravel()
    
    ind = np.where(~np.isnan(yc_t_pos))[0]

    result_pos_y = np.empty(len(yc_t_pos))
    result_pos_y[:] = np.nan
        
    result_pos_y[ind] = fsolve(equation_pos, yc_t_pos[ind], [r, x0_t[ind], y0_t[ind], x1_t[ind], y1_t[ind]])
    result_pos_x = np.where((r**2 - result_pos_y**2) >= 0, np.sqrt(r**2 - result_pos_y**2, where=(r**2 - result_pos_y**2) >= 0), 0)
    
    yc_t_pos = None
    
    ind = np.where(np.isnan(result_pos_y))[0]
    
    result_pos_x[ind] = np.nan
    
    ind = None
    
    # negative case
    
    xc_neg = np.where((r**2 - yc_t**2) >= 0, - np.sqrt(r**2 - yc_t**2, where=(r**2 - yc_t**2) >= 0), 0)
    
    a0_neg = yc_t - y0_t
    b0_neg = x0_t - xc_neg
    a1_neg = yc_t - y1_t
    b1_neg = x1_t - xc_neg
    
    d0_neg = np.where((a0_neg**2 + b0_neg**2) >= 0, np.sqrt(a0_neg**2 + b0_neg**2, where=(a0_neg**2 + b0_neg**2) >= 0), 0)
    d1_neg = np.where((a1_neg**2 + b1_neg**2) >= 0, np.sqrt(a1_neg**2 + b1_neg**2, where=(a1_neg**2 + b1_neg**2) >= 0), 0)

    eq_neg = (a0_neg*xc_neg + b0_neg*yc_t)*d1_neg + (a1_neg*xc_neg + b1_neg*yc_t)*d0_neg
    
    a0_neg = None
    b0_neg = None
    a1_neg = None
    b1_neg = None
    d0_neg = None
    d1_neg = None
    xc_neg = None
    
    # check if any of the results is close to zero
    index_zeros_neg = np.where(np.isclose(eq_neg, 0, rtol = 0, atol = tolerance))[0]
    
    # check if any result crosses zero, it needs to be done row by row
    eq_neg = np.reshape(eq_neg, (rows, columns))
    index_zeros_cross = np.where(eq_neg >= 0, 1, 0)
    index_cross_x, index_cross_y = np.where(index_zeros_cross[:, :-1] != index_zeros_cross[:, 1:])
    index_zeros_cross = None
    index_cross = (index_cross_x, index_cross_y)
    index_cross_plus_one = (index_cross_x, index_cross_y+1)
    
    # find index closest to zero for resulsts that cross zero
    cross_indexes = np.where(np.abs(eq_neg[index_cross_x, index_cross_y]) < np.abs(eq_neg[index_cross_x, index_cross_y+1]), index_cross, index_cross_plus_one)
    
    index_cross = None
    index_cross_plus_one = None
    eq_neg = None
    
    # find to which element of a flattened array the xy index correspond to
    # e.g.: 5th element of the 2nd row --> you need to multiply the number
    # of columns by row value, and then add the column value
    cross_indexes_flattened = cross_indexes[0]*columns + cross_indexes[1]
    
    # combine index of values close to zero with index of values crossing zero
    final_index = np.unique(np.concatenate((index_zeros_neg, cross_indexes_flattened)))
    index_zeros_neg = None
    cross_indexes_flattened = None
    
    # create nan mask except where you want to keep the value
    mask = np.empty(len(yc_t))
    mask[:] = np.nan
    mask[final_index] = 0
    
    yc_t_neg = np.where(mask == 0, yc_t, np.nan)
    
    yc_t = None
    mask = None
    
    yc_t_neg = np.reshape(yc_t_neg, (rows, columns))
    compare_index_neg = np.where(np.isclose(yc_t_neg[:, :-1], yc_t_neg[:, 1:], rtol=0, atol= 1.5*step_size))
    yc_t_neg[compare_index_neg[0], compare_index_neg[1]] = np.nan
    
    compare_index_neg = None

    # find positions where I have zeros for yc in the domain
    # which are the non nan element.
    # I want to know how many zeros I have for each start-end separately
    # My matrix has start-end as rows, reflection points as columns,
    # Hence I need to sum column-wise
    mask_sum = np.sum(np.where(np.isnan(yc_t_neg[:,:]), 1, 0), axis=1)
    
    # If a row is made of 1, it means that each circumference point is a reflection point,
    # which happen only if start and end are in the center of the circle
    # In that case I want to substitute every value with nan and then store -1 
    # as first value only
    check_R_neg = np.where(mask_sum == columns)[0]
    
    mask_sum = None
    
    yc_t_neg[check_R_neg,:] = np.nan
    
    # Now I can reflatten yc_t and apply fsolve to the points where I have zeros
    yc_t_neg = yc_t_neg.ravel()
    
    ind = np.where(~np.isnan(yc_t_neg))[0]
    
    result_neg_y = np.empty(len(yc_t_neg))
    result_neg_y[:] = np.nan
    
    result_neg_y[ind] = fsolve(equation_neg, yc_t_neg[ind], [r, x0_t[ind], y0_t[ind], x1_t[ind], y1_t[ind]])
    
    yc_t_neg = None
    x0_t = None
    y0_t = None
    x1_t = None
    y1_t = None
        
    result_neg_x = np.where((r**2 - result_neg_y**2) >= 0, - np.sqrt(r**2 - result_neg_y**2, where=(r**2 - result_neg_y**2) >= 0), 0)
    
    ind = np.where(np.isnan(result_neg_y))
    
    result_neg_x[ind] = np.nan
    
    ind = None
     
    # remove duplicates
    # find values and indices that are equal between x_pos and x_neg
    # x_pos and x_neg must have opposite sign, unless y +/- r
    
    x_neg_ind = np.where(np.isclose(result_neg_x, result_pos_x, rtol = 0, atol = tolerance))[0]
    
    # remove values from x_neg and from y_neg adn substitute them with np.nan
    result_neg_y[x_neg_ind] = np.nan
    result_neg_x[x_neg_ind] = np.nan
    
    x_neg_ind = None
    
    # reshape results array
    result_pos_x = np.reshape(result_pos_x, (rows, columns))
    result_pos_y = np.reshape(result_pos_y, (rows, columns))
    result_neg_x = np.reshape(result_neg_x, (rows, columns))
    result_neg_y = np.reshape(result_neg_y, (rows, columns))
    
    # merge positive and negative case
    result_x = np.concatenate((result_pos_x, result_neg_x), axis = 1)
    result_y = np.concatenate((result_pos_y, result_neg_y), axis = 1)
    
    result_pos_x = None
    result_pos_y = None
    result_neg_x = None
    result_neg_y = None
    
    # check if I have too many results
    mask_sum = np.sum(np.where(np.isnan(result_x), 0, 1), axis=1)

    check = mask_sum <= 4

    mask_sum = None

    if check.all() != True:

        print('Something wrong\nSome values have more than 4 reflection points')
        return np.nan, np.nan, np.nan
    
    # get index for which result is R
    R_index = np.intersect1d(check_R_pos, check_R_neg)
    
    # see if points of reflection are indeed on the circumference
    check = np.isclose(result_x**2 + result_y**2, r**2, rtol = 0, atol = tolerance) | np.isnan(result_x)
    
    if check.all() != True:
        
        print('Warning: some reflection points are not on the circumference\nRemoving...')
        result_x[~check] = np.nan
        result_y[~check] = np.nan
            
    result_x = result_x + xa
    result_y = result_y + ya
    
    return result_x, result_y, R_index


# exclude reflections from the outside of the circle
def get_internal_ref_arr(x0, y0, x1, y1, xc, yc, r, tolerance=1e-6):

    # xc has number of columns equal to the xc points
    # number of rows equal to the x0 points
    shape = np.shape(xc)

    x0_t = np.repeat([x0], shape[1], axis=0).T
    y0_t = np.repeat([y0], shape[1], axis=0).T
    x1_t = np.repeat([x1], shape[1], axis=0).T
    y1_t = np.repeat([y1], shape[1], axis=0).T

    x0 = None
    y0 = None
    x1 = None
    y1 = None

    index = np.where(~np.isnan(xc))

    dist_p0pc = np.sqrt((x0_t - xc)**2 + (y0_t - yc)**2)
    dist_p1pc = np.sqrt((x1_t - xc)**2 + (y1_t - yc)**2)

    dist_tot = dist_p0pc + dist_p1pc

    dist_p0pc = None
    dist_p1pc = None

    # check if either p0 or p1 are outside the circle
    check_p0 = x0_t[index[0], index[1]]**2 + y0_t[index[0], index[1]]**2 < r**2
    check_p1 = x1_t[index[0], index[1]]**2 + y1_t[index[0], index[1]]**2 < r**2

    check_p0p1 = (check_p0 == False) | (check_p1 == False)

    check_p0 = None
    check_p1 = None

    minima = np.amin(dist_tot, axis=1)
    min_array = np.repeat([minima], shape[1], axis=0).T

    check_min = dist_tot[index[0], index[1]] == min_array[index[0], index[1]]

    minima = None
    min_array = None

    dist_p0p1 = np.sqrt((x0_t-x1_t)**2 + (y0_t-y1_t)**2)

    check_dist = np.isclose(dist_p0p1[index[0], index[1]], dist_tot[index[0], index[1]], rtol=0, atol=tolerance)

    xc[index[0], index[1]] = np.where(((check_p0p1 & check_min) | check_dist), np.nan, xc[index[0], index[1]])
    yc[index[0], index[1]] = np.where(((check_p0p1 & check_min) | check_dist), np.nan, yc[index[0], index[1]])

    xc = xc.T
    yc = yc.T

    return xc, yc


# get reflections only for the proper side
def check_side(xc, yc, x_alpha, x_zero):

    # right semi circle
    if x_alpha > x_zero:

        xc = np.where(xc > x_alpha, xc, np.nan)
        yc = np.where(xc > x_alpha, yc, np.nan)

    # left semi circle
    else:

        xc = np.where(xc < x_alpha, xc, np.nan)
        yc = np.where(xc < x_alpha, yc, np.nan)

    return xc, yc

# get length of the bouncing path
def get_distance_circle(x0, y0, x1, y1, xc, yc):
    
    distance = np.sqrt((x0 - xc)**2 + (y0 - yc)**2) + np.sqrt((x1 - xc)**2 + (y1 - yc)**2)
    
    return distance


############## FIND INTERSECTION WITH THE LINE ################

def get_reflection_point(x0, y0, x1, y1, clx, crx, line):

    y1_r = 2*line - y1

    mask = y1_r != y0

    m = np.where(mask, np.divide((x1 - x0), (y1_r - y0), where=mask), 0)

    x_int = m*(line - y0) + x0

    mask = (x_int >= clx) & (x_int <= crx)

    x1_final = np.where(mask, x1, np.nan)
    y1_final = np.where(mask, y1_r, np.nan)

    return x1_final, y1_final

def get_distance_line(x0, y0, x1_r, y1_r):
    
    distance = np.sqrt((x0 - x1_r)**2 + (y0 - y1_r)**2)
    
    return distance


####### MERGE DISTANCES ARRAYS TOGETHER ########################

def merge_arrays(d_c_l, d_c_r, d_l_t, d_l_b):
    
    distances = np.vstack((d_c_l, d_c_r, d_l_t, d_l_b))
    
    # # check if I have too many results
    # mask_sum = np.sum(np.where(np.isnan(distances), 0, 1), axis=0)

    # check = mask_sum <= 4

    # mask_sum = None

    # if check.all() != True:

    #     print('Something wrong\nSome values have more than 4 reflection points')
    #     return np.nan, np.nan
    
    return distances

# diffusion function for bouncing
def diff_func_bouncing(dists_0, dists_bounce, r_index_left, r_index_right, t, D):

    k = 4*D*t
    
    p_r0 = (1/(np.pi*k))*np.exp(-(dists_0**2)/(k), where = ~np.isnan(dists_0))
    p_r1 = (1/(np.pi*k))*np.exp(-(dists_bounce**2)/(k), where = ~np.isnan(dists_bounce))
    
    p_R_left = np.empty(len(dists_0))
    p_R_left[:] = np.nan
    p_R_right = p_R_left.copy()
    
    p_R_left[r_index_left] = (1/k)*np.exp((-(2*dists_0[r_index_left])**2)/(k))
    p_R_right[r_index_right] = (1/k)*np.exp((-(2*dists_0[r_index_right])**2)/(k))
    
    p0_final = np.vstack((p_r0, p_r1, p_R_left, p_R_right))
    
    p_r0 = None
    p_r1 = None
    p_R_left = None
    p_R_right = None
    
    p0_final = np.where(np.isnan(p0_final), 0, p0_final)
    
    p0_sum = np.sum(p0_final, axis = 0)
    
    return p0_sum


def final_fitting_function_bounce(x, t, p0, disp_SMDM_minimum, center, radius, length):
    
    if p0 <= 0:
        
        print('Diffusion coefficient cannot be <= 0')
        return

    # if there are not enough data in the cell, don't start the fitting routine
    if len(x) <= disp_SMDM_minimum:
        return 0
    
    else:
        
        # transpose the data in the cell to have all the x in first row, all the y in second etc
        x = np.array(x).transpose()
        
        # assign the data to variables
        x0 = x[0]
        y0 = x[1]
        x1 = x[2]
        y1 = x[3]
        
        x = None
        
        # calculate squared distance
        dists_0 = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        # calculate distances with bounce
        cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
        
        domain = [-radius, radius]
        
        args_left = [radius, x0, y0, x1, y1, clx, cy]
        
        bounce_x_left, bounce_y_left, r_index_left = my_solver_arr(args_left, domain)
        args_left = None
        
        bounce_x_left, bounce_y_left = get_internal_ref_arr(x0, y0, x1, y1, bounce_x_left, bounce_y_left, radius)
        bounce_x_left, bounce_y_left = check_side(bounce_x_left, bounce_y_left, clx, cx)
        
        distance_left = get_distance_circle(x0, y0, x1, y1, bounce_x_left, bounce_y_left)
        
        bounce_x_left = None
        bounce_y_left = None
        
        args_right = [radius, x0, y0, x1, y1, crx, cy]
        
        bounce_x_right, bounce_y_right, r_index_right = my_solver_arr(args_right, domain)
        args_right = None
        
        bounce_x_right, bounce_y_right = get_internal_ref_arr(x0, y0, x1, y1, bounce_x_right, bounce_y_right, radius)
        bounce_x_right, bounce_y_right = check_side(bounce_x_right, bounce_y_right, crx, cx)
        
        distance_right = get_distance_circle(x0, y0, x1, y1, bounce_x_right, bounce_y_right)
        
        bounce_x_right = None
        bounce_y_right = None
        
        bounce_x_top, bounce_y_top = get_reflection_point(x0, y0, x1, y1, clx, crx, top)
        distance_top = get_distance_line(x0, y0, bounce_x_top, bounce_y_top)
        
        bounce_x_top = None
        bounce_y_top = None
        
        bounce_x_bot, bounce_y_bot = get_reflection_point(x0, y0, x1, y1, clx, crx, bot)
        distance_bot = get_distance_line(x0, y0, bounce_x_bot, bounce_y_bot)
        
        bounce_x_bot = None
        bounce_y_bot = None
        
        x0 = None
        y0 = None
        x1 = None
        y1 = None
        
        dists_1 = merge_arrays(distance_left, distance_right, distance_top, distance_bot)
        dists_1 = np.array(dists_1)
        
        # pack distance and time
        xy = [dists_0, dists_1, r_index_left, r_index_right, t]
        
        dists_0 = None
        dists_1 = None
        r_index_left = None
        r_index_right = None
        
        def find_log(D, xy):
    
            # unpack values
            dists_0, dists_1, r_index_left, r_index_right, t = xy
        
            # calculate logarithm and return sum of log --> less expensive than calculating the function and computing the multiplication
            my_log = np.log(diff_func_bouncing(dists_0, dists_1, r_index_left, r_index_right, t, D))

            return -np.sum(my_log)
        
        # apply the minimize method
        result = minimize(find_log, p0, xy, method = 'Nelder-Mead')
        
        #return the first and only element of the array x stored in result
        return result.x[0]
    
# create map of fitted values
def create_map_fit_bounce(df, t, p0, disp_SMDM_minimum, center, radius, length):
    
    df = df.applymap(lambda x: final_fitting_function_bounce(x, t, p0, disp_SMDM_minimum, center, radius, length))
    map_fit_array = df.to_numpy()
    map_fit_array = map_fit_array.ravel()
    
    indexes = np.where(map_fit_array == 0)[0]
    map_fit_array = np.delete(map_fit_array, indexes)
    
    indexes = None
    
    average = np.mean(map_fit_array)
    
    return df, average

# save map fit
def save_map_fit_bounce(map_fit, fpath, fname):
    
    file_path = fpath[:-10] + 'diff_bounce_fit.csv.zip'
    file_name = fname[:-10] + 'diff_bounce_fit.csv.zip'
    
    map_fit.to_csv(file_path)
    
    return file_name


def save_map_fit_delta(map_fit, fpath, fname):

    file_path = fpath[:-10] + 'delta_map.csv.zip'
    file_name = fname[:-10] + 'delta_map.csv.zip'

    map_fit.to_csv(file_path)

    return file_name


# function to wrap all previous functions together
def start_program(center, length, width, D, N, total_time, time_step, delta_t, map_res, p0, disp_SMDM_minimum, subpath, seed):
    
    radius = width / 2
    
    path = os.getcwd()
    path += '//' + subpath
    
    print('starting initial simulation')
    simulation_file, sim_file_path, sim_file_name = smoldyn_billiard(center, radius, length, N, D, total_time, time_step, seed, path)
    print('simulation finished')
    print('saving the simulation file...')
    save_data(simulation_file, sim_file_path)
    print('data saved in Files with name ' + sim_file_name)
    print()
    
    print('Creating simulation map...')
    simulation_map, sim_map_path, sim_map_name = create_map(center, radius, length, simulation_file, map_res, delta_t, time_step, D, N, path)
    print('Simulation map created.')
    print('saving the simulation map...')
    
    # free some memory
    simulation_file = None
    
    map_name = save_df(simulation_map, sim_map_path, sim_map_name)
    print('data saved in Files with name ' +  map_name)
    print()
    print('Creating the diffusion map...')
    diffusion_map, diffusion_average = create_map_fit(simulation_map, delta_t, p0, disp_SMDM_minimum)
    print('Diffusion map created.')
    print('saving the diffusion map...')
    diff_map_name = save_map_fit(diffusion_map, sim_map_path, sim_map_name)
    print('data saved in Files with name ' + diff_map_name)

    
    print('starting routine to get diffusion coefficient by taking into account one bounce')
    bounce_map, bounce_avg = create_map_fit_bounce(simulation_map, delta_t, p0, disp_SMDM_minimum, center, radius, length)
    
    print('analysis finished. Saving the new diffusion map...')
    fname = save_map_fit_bounce(bounce_map, sim_map_path, sim_map_name)
    print('data saved in Files with name ' + fname)
    
    print('creating the delta map')
    delta_map = bounce_map.to_numpy() - diffusion_map.to_numpy()
    delta_avg = diffusion_average - bounce_avg
    
    delta_map = pd.DataFrame(delta_map)
    
    print('saving the delta diffusion map...')
    name = save_map_fit_delta(delta_map, sim_map_path, sim_map_name)
    print('data saved in Files with name ' + name)
    
    print('Saving the plots the plots...')
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    
    x1, x2, y1, y2 = left_axis, right_axis, bot_axis, top_axis
    
    fig, axes = plt.subplots(3, 1, figsize=(13, 16))
    fig.suptitle('D = ' + str(D) + ', N = ' + str(N) + ', length = ' + str(length) + ', width = ' + str(width))
    
    image1 = axes[0].imshow(diffusion_map.to_numpy(), origin="lower", cmap='PuOr_r', extent=(x1, x2, y1, y2), vmin = 0.5*diffusion_average, vmax = 1.5*diffusion_average)
    image2 = axes[1].imshow(bounce_map.to_numpy(), origin="lower", cmap='PuOr_r', extent=(x1, x2, y1, y2), vmin = 0.5*bounce_avg, vmax = 1.5*bounce_avg)
    image3 = axes[2].imshow(delta_map.to_numpy(), origin="lower", cmap='bwr', extent=(x1, x2, y1, y2), vmin = -3*delta_avg, vmax = 3*delta_avg)
    images = [image1, image2, image3]
    
    titles = ["SMdM", "Bounce-billiard", "Delta Map"]
    
    for i in range(3):
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(r'$\mu m$')
        cbar = fig.colorbar(images[i], ax=axes[i])
        cbar.set_label(r'$D$ $(\mu m^2/s)$')
        
    axes[-1].set_xlabel(r'$\mu m$')
    
    # for delta image
    cbar.set_label(r'$\Delta D$ $(\mu m^2/s)$')
    
    plt_path = sim_map_path[:-10] + 'maps_comparison.pdf'
    plt_name = sim_map_name[:-10] + 'maps_comparison.pdf'
    plt.savefig(plt_path, dpi=300, format='pdf')
    plt.close('all')
    
    print('plot saved in Files with name ' + plt_name)
    
    print()

    
    return