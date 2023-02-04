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
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt

### initial paramters ###

seed = int(time.time())
path = os.getcwd()


### start of functions ###

# function to get coordinates of important points inside the spherocylinder
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

# function to generate random z values within the sphercylinder, given the x,y coordinates of a point
def get_z_value(center, r, length, x, y, seed1):
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, r, length)
    
    mask_left = x < clx
    mask_right = x > crx
    
    arg = np.where(mask_left, r**2 - (x - clx)**2 - (y - cy)**2, np.where(mask_right, r**2 - (x - crx)**2 - (y - cy)**2, r**2 - (y - cy)**2))
    
    rng = np.random.default_rng(seed=seed1)
    z = np.where(arg >= 0, cz + rng.uniform(-np.sqrt(arg), np.sqrt(arg), len(x)), 0)

    return z


# function to run the simulation with normal diffusion
def smoldyn_billiard_diffusion(sim_type, center, radius, length, N, D, total_time, time_step, seed1, i, path, main_path = 0):
    
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    neg_axis = neg - 0.5
    pos_axis = pos + 0.5

    if main_path == 0:

        # create simulation environment defining the bottom left corner and the top right
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True)

    else:

        # create simulation environment defining the bottom left corner and the top right
        smoldyn.smoldyn.__top_model_file__ = main_path
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True, path_name=main_path)
    
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

    # create a compartment inside the billiard and place N molecules randomly in the compartment
    cell = s.addCompartment('cell', surface = billiard, point = [cx, cy, cz])
    cell.simulation.addCompartmentMolecules('A', N, 'cell')
    
    # define file name
    fname = str(i) + '_3D_' + sim_type + '_' + 'billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(
        rx-lx) + '_diam_' + str(top-bot) + '_depth_' + str(pos-neg) + '_D_' + str(D) + '_N_' + str(N) + '.npz'
    fpath = path + '/' + fname
    
    # add info about the molecules position at eacht time step in an output to return at the end of the function (so it's not necessary to read the file every time)
    s.addOutputData('mydata')
    s.addCommand(cmd = "molpos A mydata", cmd_type = "E")

    # run the simulation
    s.run(total_time, dt=time_step, display=False,
          overwrite=True, log_level=5, quit_at_end=True)
    
    # retrieve output data after running the simulation
    data = s.getOutputData('mydata', 0)
    
    # convert output in numpy array
    data = np.array(data)
    
    return data, fpath, fname


# functions to run simulation for simulating aggregation at a cell pole
def generate_random_molecules_cell_aggregation(cy, cz, lx, clx, crx, rx, top, pos, radius, seed1, N):
    
    rng = np.random.default_rng(seed=seed1)
    x = rng.uniform(lx, rx, N)
    
    mask_left = x < clx
    mask_right = x > crx
    
    arg_y = np.where(mask_left, radius**2 - (x - clx)**2, np.where(mask_right, radius**2 - (x - crx)**2, top))
    
    rng = np.random.default_rng(seed=seed1 + 1)
    # sign = rng.choice([-1, 1])
    y = np.where(arg_y >= 0, cy + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_y), N), 0)
    
    arg_z = np.where(mask_left, radius**2 - (x - clx)**2 - (y - cy)**2, np.where(mask_right, radius**2 - (x - crx)**2 - (y - cy)**2, radius**2 - (y - cy)**2))
    
    rng = np.random.default_rng(seed=seed1 + 2)
    # sign = rng.choice([-1, 1])
    z = np.where(arg_z >= 0, cz + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_z), N), 0)
    
    return x, y, z

def generate_random_molecules_pole_aggregation(cy, cz, cx, radius, seed1, N):
    
    rng = np.random.default_rng(seed=seed1)
    x = rng.uniform(-radius + cx, radius + cx, N)
    
    arg_y = radius**2 - (x - cx)**2
    
    rng = np.random.default_rng(seed=seed1 + 1)
    # sign = rng.choice([-1, 1])
    y = np.where(arg_y >= 0, cy + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_y), N), 0)
    
    arg_z = radius**2 - (x - cx)**2 - (y - cy)**2
    
    rng = np.random.default_rng(seed=seed1 + 2)
    # sign = rng.choice([-1, 1])
    z = np.where(arg_z >= 0, cz + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_z), N), 0)
    
    return x, y, z
    
    

def smoldyn_billiard_aggregation(sim_type, center, radius, length, N, D, slowdown_factor, total_time, time_step, seed1, i, path, main_path = 0):
    
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    neg_axis = neg - 0.5
    pos_axis = pos + 0.5

    if main_path == 0:

        # create simulation environment defining the bottom left corner and the top right
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True)

    else:

        # create simulation environment defining the bottom left corner and the top right
        smoldyn.smoldyn.__top_model_file__ = main_path
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True, path_name=main_path)
    
    S.Simulation.setRandomSeed(s, int(seed1))
    s.setFlags('q')

    # add a molecular specie with a diffusion coefficient D
    A = s.addSpecies("A", difc = D, color = "green")
    B = s.addSpecies("B", difc = slowdown_factor*D, color="red")

    # create the different parts of the billiard
    s_right = smoldyn.Hemisphere(center = [crx, cy, cz], radius = radius, vector = [-1 ,0, 0], slices = 1000, stacks = 1000, name = 's_right')
    s_left = smoldyn.Hemisphere(center = [clx, cy, cz], radius = radius, vector = [1, 0, 0], slices = 1000, stacks = 1000, name = 's_left')
    s_center = smoldyn.Cylinder(start = [clx, cy, cz], end = [crx, cy, cz], radius = radius, slices = 1000, stacks = 1000, name = 's_center')

    sphere_left = smoldyn.Sphere(center=[(lx+clx)/2, cy, cz], radius=radius/2, slices=1000, stacks=1000, name='sphere_left')

    # create the billiard by merging together the different parts
    billiard = s.addSurface("billiard", panels=[s_left, s_center, s_right])
    left_pole = s.addSurface("left_pole", panels=[sphere_left])

    # define action of reflection for the billiard walls
    billiard.setAction('both', [A, B], "reflect")

    left_pole.setAction('both', [A, B], "reflect")
    
    x, y, z = generate_random_molecules_cell_aggregation(cy, cz, lx, clx, crx, rx, top, pos, radius, seed1, N)
    
    sphere_center_x = (lx + clx)/2
    sphere_radius = radius/2
    
    x_p, y_p, z_p = generate_random_molecules_pole_aggregation(cy, cz, sphere_center_x, sphere_radius, seed1+1, N)
    
    check = (x - sphere_center_x)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2
    
    seed0 = seed1+2
    
    while len(np.where(check)[0]) > 0:
            
        inds = np.where(check)[0]
            
        x = np.delete(x, inds)
        y = np.delete(y, inds)
        z = np.delete(z, inds)
        
        new_N = len(inds)
        
        x1, y1, z1 = generate_random_molecules_cell_aggregation(cy, cz, lx, clx, crx, rx, top, pos, radius, seed0, new_N)
        
        x = np.concatenate((x, x1))
        y = np.concatenate((y, y1))
        z = np.concatenate((z, z1))
        
        check = (x - sphere_center_x)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2
        
        seed0 += 1
    
    x = np.concatenate((x, x_p))
    y = np.concatenate((y, y_p))
    z = np.concatenate((z, z_p))
    
    check = (x - sphere_center_x)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2

    for i in range(len(x)):
        
        if check[i] == False:
            
            A.addToSolution(1, pos = [x[i], y[i], z[i]])
                           
        else:
            
            B.addToSolution(1, pos = [x[i], y[i], z[i]])
    
    # define file name
    fname = str(i) + '_3D_' + sim_type + '_' + 'billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(
        rx-lx) + '_diam_' + str(top-bot) + '_depth_' + str(pos-neg) + '_D_' + str(D) + '_N_' + str(N) + '.npz'
    fpath = path + '/' + fname
    
    # add info about the molecules position at eacht time step in an output to return at the end of the function (so it's not necessary to read the file every time)
    s.addOutputData('mydata')
    s.addCommand(cmd = "molpos all mydata", cmd_type = "E")

    # run the simulation
    s.run(total_time, dt=time_step, display=False,
          overwrite=True, log_level=5, quit_at_end=True)
    
    # retrieve output data after running the simulation
    data = s.getOutputData('mydata', 0)
    
    # convert output in numpy array
    data = np.array(data)
    
    return data, fpath, fname

# functions to run simulation for simulating interaction between molecules and component located in the cell pole
def generate_random_molecules_cell_interaction(cy, cz, lx, clx, crx, rx, top, pos, radius, seed1, N):
    
    rng = np.random.default_rng(seed=seed1)
    x = rng.uniform(lx, rx, N)
    
    mask_left = x < clx
    mask_right = x > crx
    
    arg_y = np.where(mask_left, radius**2 - (x - clx)**2, np.where(mask_right, radius**2 - (x - crx)**2, top))
    
    rng = np.random.default_rng(seed=seed1 + 1)
    # sign = rng.choice([-1, 1])
    y = np.where(arg_y >= 0, cy + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_y), N), 0)
    
    arg_z = np.where(mask_left, radius**2 - (x - clx)**2 - (y - cy)**2, np.where(mask_right, radius**2 - (x - crx)**2 - (y - cy)**2, radius**2 - (y - cy)**2))
    
    rng = np.random.default_rng(seed=seed1 + 2)
    # sign = rng.choice([-1, 1])
    z = np.where(arg_z >= 0, cz + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_z), N), 0)
    
    return x, y, z


def smoldyn_billiard_func(sim_type, center, radius, length, N, D, slowdown_factor, total_time, time_step, seed1, path, x=[], y=[], z=[],  main_path=0):
    
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    neg_axis = neg - 0.5
    pos_axis = pos + 0.5

    if main_path == 0:

        # create simulation environment defining the bottom left corner and the top right
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True)

    else:

        # create simulation environment defining the bottom left corner and the top right
        smoldyn.smoldyn.__top_model_file__ = main_path
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True, path_name=main_path)
    
    S.Simulation.setRandomSeed(s, int(seed1))
    s.setFlags('q')

    # add a molecular specie with a diffusion coefficient D
    A = s.addSpecies("A", difc = D, color = "green")
    B = s.addSpecies("B", difc = slowdown_factor*D, color="red")

    # create the different parts of the billiard
    s_right = smoldyn.Hemisphere(center = [crx, cy, cz], radius = radius, vector = [-1 ,0, 0], slices = 1000, stacks = 1000, name = 's_right')
    s_left = smoldyn.Hemisphere(center = [clx, cy, cz], radius = radius, vector = [1, 0, 0], slices = 1000, stacks = 1000, name = 's_left')
    s_center = smoldyn.Cylinder(start = [clx, cy, cz], end = [crx, cy, cz], radius = radius, slices = 1000, stacks = 1000, name = 's_center')

    # sphere_left = smoldyn.Sphere(center=[(lx+clx)/2, cy, cz], radius=radius/2, slices=1000, stacks=1000, name='sphere_left')

    # create the billiard by merging together the different parts
    billiard = s.addSurface("billiard", panels=[s_left, s_center, s_right])
    # left_pole = s.addSurface("left_pole", panels=[sphere_left])

    # define action of reflection for the billiard walls
    billiard.setAction('both', [A, B], "reflect")
    
    sphere_center_x = (lx + clx)/2
    sphere_radius = radius/2
    
    if len(x) == 0:
    
        x, y, z = generate_random_molecules_cell_interaction(cy, cz, lx, clx, crx, rx, top, pos, radius, seed1, N)
        
    check = (x - sphere_center_x)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2
    
    for i in range(len(x)):

        if check[i] == False:

            A.addToSolution(1, pos=[x[i], y[i], z[i]])

        else:

            B.addToSolution(1, pos=[x[i], y[i], z[i]])
    
    # define file name
    fname = str(i) + '_3D_' + sim_type + '_' + 'billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(
        rx-lx) + '_diam_' + str(top-bot) + '_depth_' + str(pos-neg) + '_D_' + str(D) + '_N_' + str(N) + '.npz'
    fpath = path + '/' + fname
    
    # add info about the molecules position at eacht time step in an output to return at the end of the function (so it's not necessary to read the file every time)
    s.addOutputData('mydata')
    s.addCommand(cmd = "molpos all mydata", cmd_type = "E")

    # run the simulation
    s.run(total_time, dt=time_step, display=False,
          overwrite=True, log_level=5, quit_at_end=True)
    
    # retrieve output data after running the simulation
    data = s.getOutputData('mydata', 0)
    
    # convert output in numpy array
    data = np.array(data)
    
    return data, fpath, fname


def smoldyn_billiard_interaction(sim_type, center, radius, length, N, D, slowdown_factor, total_time, time_step, seed1, path):
    
    steps = int(total_time // time_step)
    
    res_0, fpath, fname = smoldyn_billiard_func(sim_type, center, radius, length, N, D, slowdown_factor, time_step, time_step, seed1, path)
    
    res = res_0[0, 1:]
    x0 = res[-3::-3]
    y0 = res[-2::-3]
    z0 = res[-1::-3]
    
    arr = np.array([x0, y0, z0]).T.ravel()
    
    result = np.concatenate(([0], arr))
    
    data = [result]
    
    res = res_0[-1, 1:]
    x0 = res[-3::-3]
    y0 = res[-2::-3]
    z0 = res[-1::-3]

    arr = np.array([x0, y0, z0]).T.ravel()
    
    result = np.concatenate(([time_step], arr))
    
    data.append(result)
    
    seed0 = seed1 + 1
    
    for i in range(steps):
        
        result, fpath, fname = smoldyn_billiard_func(sim_type, center, radius, length, N, D, slowdown_factor, time_step, time_step, seed0, path, x0, y0, z0)
        
        t = (i+2)*time_step
        
        result = result[-1, 1:]
        x0 = result[-3::-3]
        y0 = result[-2::-3]
        z0 = result[-1::-3]
        
        arr = np.array([x0, y0, z0]).T.ravel()
        
        result = np.concatenate(([t], arr))
        
        data.append(result)
        
        seed0 += 1
        
    data = np.vstack(data)
    
    x = data[:, 1::3]
    y = data[:, 2::3]
    z = data[:, 3::3]
    
    x = x[::-1]
    y = y[::-1]
    z = z[::-1]
    
    data[:, 1::3] = x
    data[:, 2::3] = y
    data[:, 3::3] = z
    
    return data, fpath, fname

# function to run the simulation for simulating a 2 components diffusion
def generate_random_molecules_cell_two_comps(cy, cz, lx, clx, crx, rx, top, pos, radius, seed1, N):
    
    rng = np.random.default_rng(seed=seed1)
    x = rng.uniform(lx, rx, N)
    
    mask_left = x < clx
    mask_right = x > crx
    
    arg_y = np.where(mask_left, radius**2 - (x - clx)**2, np.where(mask_right, radius**2 - (x - crx)**2, top))
    
    rng = np.random.default_rng(seed=seed1 + 1)
    # sign = rng.choice([-1, 1])
    y = np.where(arg_y >= 0, cy + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_y), N), 0)
    
    arg_z = np.where(mask_left, radius**2 - (x - clx)**2 - (y - cy)**2, np.where(mask_right, radius**2 - (x - crx)**2 - (y - cy)**2, radius**2 - (y - cy)**2))
    
    rng = np.random.default_rng(seed=seed1 + 2)
    # sign = rng.choice([-1, 1])
    z = np.where(arg_z >= 0, cz + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_z), N), 0)
    
    return x, y, z

def generate_random_molecules_pole_two_comps(cy, cz, cx, radius, seed1, N):
    
    rng = np.random.default_rng(seed=seed1)
    x = rng.uniform(-radius + cx, radius + cx, N)
    
    arg_y = radius**2 - (x - cx)**2
    
    rng = np.random.default_rng(seed=seed1 + 1)
    # sign = rng.choice([-1, 1])
    y = np.where(arg_y >= 0, cy + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_y), N), 0)
    
    arg_z = radius**2 - (x - cx)**2 - (y - cy)**2
    
    rng = np.random.default_rng(seed=seed1 + 2)
    # sign = rng.choice([-1, 1])
    z = np.where(arg_z >= 0, cz + rng.choice([-1, 1], N)*rng.uniform(0, np.sqrt(arg_z), N), 0)
    
    return x, y, z
    
    

def smoldyn_billiard_two_comps(sim_type, center, radius, length, N, D, slowdown_factor, total_time, time_step, seed1, i, path, main_path = 0):
    
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    neg_axis = neg - 0.5
    pos_axis = pos + 0.5

    if main_path == 0:

        # create simulation environment defining the bottom left corner and the top right
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True)

    else:

        # create simulation environment defining the bottom left corner and the top right
        smoldyn.smoldyn.__top_model_file__ = main_path
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[
                               right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True, path_name=main_path)
    
    S.Simulation.setRandomSeed(s, int(seed1))
    s.setFlags('q')

    # add a molecular specie with a diffusion coefficient D
    A = s.addSpecies("A", difc = D, color = "green")
    B = s.addSpecies("B", difc = slowdown_factor*D, color="red")

    # create the different parts of the billiard
    s_right = smoldyn.Hemisphere(center = [crx, cy, cz], radius = radius, vector = [-1 ,0, 0], slices = 1000, stacks = 1000, name = 's_right')
    s_left = smoldyn.Hemisphere(center = [clx, cy, cz], radius = radius, vector = [1, 0, 0], slices = 1000, stacks = 1000, name = 's_left')
    s_center = smoldyn.Cylinder(start = [clx, cy, cz], end = [crx, cy, cz], radius = radius, slices = 1000, stacks = 1000, name = 's_center')

    sphere_left = smoldyn.Sphere(center=[(lx+clx)/2, cy, cz], radius=radius/2, slices=1000, stacks=1000, name='sphere_left')

    # create the billiard by merging together the different parts
    billiard = s.addSurface("billiard", panels=[s_left, s_center, s_right])
    left_pole = s.addSurface("left_pole", panels=[sphere_left])

    # define action of reflection for the billiard walls
    billiard.setAction('both', [A, B], "reflect")

    left_pole.setAction('both', [B], "reflect")
    
    x, y, z = generate_random_molecules_cell_two_comps(cy, cz, lx, clx, crx, rx, top, pos, radius, seed1, N)
    
    sphere_center_x = (lx + clx)/2
    sphere_radius = radius/2
    
    x_p, y_p, z_p = generate_random_molecules_pole_two_comps(cy, cz, sphere_center_x, sphere_radius, seed1+1, N//10)
    
    check = (x - sphere_center_x)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2
    
    seed0 = seed1+2
    
    while len(np.where(check)[0]) > 0:
            
        inds = np.where(check)[0]
            
        x = np.delete(x, inds)
        y = np.delete(y, inds)
        z = np.delete(z, inds)
        
        new_N = len(inds)
        
        x1, y1, z1 = generate_random_molecules_cell_two_comps(cy, cz, lx, clx, crx, rx, top, pos, radius, seed0, new_N)
        
        x = np.concatenate((x, x1))
        y = np.concatenate((y, y1))
        z = np.concatenate((z, z1))
        
        check = (x - sphere_center_x)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2
        
        seed0 += 1
    
    x = np.concatenate((x, x_p))
    y = np.concatenate((y, y_p))
    z = np.concatenate((z, z_p))
    
    check = (x - sphere_center_x)**2 + (y - cy)**2 + (z - cz)**2 <= sphere_radius**2

    for i in range(len(x)):
        
        if check[i] == False:
            
            A.addToSolution(1, pos = [x[i], y[i], z[i]])
                           
        else:
            
            B.addToSolution(1, pos = [x[i], y[i], z[i]])
    
    # define file name
    fname = str(i) + '_3D_' + sim_type + '_' + 'billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(
        rx-lx) + '_diam_' + str(top-bot) + '_depth_' + str(pos-neg) + '_D_' + str(D) + '_N_' + str(N) + '.npz'
    fpath = path + '/' + fname
    
    # add info about the molecules position at eacht time step in an output to return at the end of the function (so it's not necessary to read the file every time)
    s.addOutputData('mydata')
    s.addCommand(cmd = "molpos all mydata", cmd_type = "E")

    # run the simulation
    s.run(total_time, dt=time_step, display=False,
          overwrite=True, log_level=5, quit_at_end=True)
    
    # retrieve output data after running the simulation
    data = s.getOutputData('mydata', 0)
    
    # convert output in numpy array
    data = np.array(data)
    
    return data, fpath, fname

# function to run the simulation with precise positioning
def smoldyn_billiard_precise(center, radius, length, N, D, total_time, time_step, coord_x, coord_y, coord_z, seed1, main_path = 0):
    
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    neg_axis = neg - 0.5
    pos_axis = pos + 0.5
    
    if main_path == 0:

        # create simulation environment defining the bottom left corner and the top right
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True)

    else:

        # create simulation environment defining the bottom left corner and the top right
        smoldyn.smoldyn.__top_model_file__ = main_path
        s = smoldyn.Simulation(low=[left_axis, bot_axis, neg_axis], high=[right_axis, top_axis, pos_axis], log_level=5, quit_at_end=True, path_name=main_path)
    
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

# function to save the data
def save_data(data, fpath):
    
    np.savez_compressed(fpath, data = data)
    
    return

# function to create a map of displacements given a resolution, the simulation time and the delta_t
def create_map(sim_type, center, radius, length, array, map_res, delta_t, time_step, D, N, i, path):
    
    # rows to skip are given by delta_t/time_step
    jump = int(delta_t/time_step)
    
    x_elements = array[:-jump, 1::3].copy().ravel()
    y_elements = array[:-jump, 2::3].copy().ravel()

    x_elements_jump = array[jump:, 1::3].copy().ravel()
    y_elements_jump = array[jump:, 2::3].copy().ravel()
    
    array = None
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
    
    # define number of pixels in x and y axes
    x_pixels = int((rx - lx)/map_res) + 1
    y_pixels = int((top - bot)/map_res) + 1
    
    # define bin edges in x and y axes
    x_bins = np.array([lx + i*map_res for i in range(x_pixels + 1)])
    y_bins = np.array([bot + i*map_res for i in range(y_pixels + 1)])

    # identify which x_elements fall between which bins and return the indices of the x_elements in the order of the bins
    # AKA Return the indices of the bins to which each value in input array belongs.
    x_indices = np.digitize(x_elements, x_bins) - 1
    y_indices = np.digitize(y_elements, y_bins) - 1
    
    x_indices = x_indices.ravel()
    y_indices = y_indices.ravel()
    
    # create empty dataframe with y_pixel rows and x_pixel columns
    df = pd.DataFrame(data = [], index = range(y_pixels), columns = range(x_pixels))

    # substitute nan with empty list to append values
    df = df.applymap(lambda x: [])

    # iterate through each row of the database
    for j in range(len(x_indices)):

        df.iat[y_indices[j], x_indices[j]].append(([x_elements[j], y_elements[j], x_elements_jump[j], y_elements_jump[j]]))
            
    # define file name
    fname = str(i) +'_3D_' + sim_type + '_' + 'billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(rx-lx) + '_diam_' + str(top-bot) + '_depth_' + str(pos-neg) + '_D_' + str(D) + '_N_' + str(N) + '_map_df.csv.zip'
    fpath = path + '/' + fname
        
    return df, fpath, fname


def create_map_regions(sim_type, center, radius, length, array, delta_t, time_step, D, N, i, path):
    
    # rows to skip are given by delta_t/time_step
    jump = int(delta_t/time_step)
    
    x_elements = array[:-jump, 1::3].copy().ravel()
    y_elements = array[:-jump, 2::3].copy().ravel()

    x_elements_jump = array[jump:, 1::3].copy().ravel()
    y_elements_jump = array[jump:, 2::3].copy().ravel()
    
    array = None
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
    
    left_pole = x_elements < clx
    cell_center = ((x_elements >= clx) & (x_elements <= crx))
    right_pole = x_elements > crx
    
    left_region = [x_elements[left_pole], y_elements[left_pole], x_elements_jump[left_pole], y_elements_jump[left_pole]]
    center_region = [x_elements[cell_center], y_elements[cell_center], x_elements_jump[cell_center], y_elements_jump[cell_center]]
    right_region = [x_elements[right_pole], y_elements[right_pole], x_elements_jump[right_pole], y_elements_jump[right_pole]]
    
    values = [left_region, center_region, right_region]
    
    df = pd.DataFrame(data = [], index = range(1), columns = range(3))
    df = df.applymap(lambda x: [])
    
    for j in range(3):
        
        df.iat[0,j].append(values[j])
    
    # define file name
    fname = str(i) + '_3D_' + sim_type + '_' + 'billiard_cx_' + str(cx) + '_cy_' + str(cy) + '_cz_' + str(cz) + '_len_' + str(rx-lx) + '_diam_' + str(top-bot) + '_depth_' + str(pos-neg) + '_D_' + str(D) + '_N_' + str(N) + '_regions_df.csv.zip'
    fpath = path + '/' + fname
    
    return df, fpath, fname
    
    
# function to save the displacements map
def save_df(df, fpath, fname):
    
    df.to_csv(fpath)
    
    return fname

# probability density function for the diffusion
def diff_func(dists, t, D):
    
    k = 4*D*t
    num = (2*dists/k)*np.exp(-(dists**2)/(k))
    
    return num


# function to fit the diffusion function and obtain the diffusion coefficient
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
        
        # calculate squared distance
        dists = np.sqrt((x[2] - x[0])**2 + (x[3] - x[1])**2)
        
        x = None
        
        # pack distance and time
        xy = [dists, t]
        
        dists = None
        
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
    
# function for multiprocessing
def multifunction(a, t, p0, disp_SMDM_minimum):
    
    a = a.applymap(lambda x: final_fitting_function(x, t, p0, disp_SMDM_minimum))
    
    return a
    
# create map of fitted values
def create_map_fit(df, t, p0, disp_SMDM_minimum):
    
    # create routine for multiprocessing
    cores = mp.cpu_count()
    
    # split dataframe in smaller sets and then pack each of them in tuple
    df = np.array_split(df, cores)
    df = [(a, t, p0, disp_SMDM_minimum) for a in df]
    
    # multiprocessing routine
    p = mp.Pool(cores)
    df = p.starmap(multifunction, df)
    p.terminate()
    
    # merge back smaller dataframe
    df = pd.concat(df)
    
    # get average diff_coeff
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

# start fitting routine to get D
def find_difference_per_pixel_map(D, xy):

    # unpack all variables
    p0, row, column, map_res, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, main_path, seed = xy

    # if initial diffusion is zero, return zero
    if p0 == 0:

        return 0
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
    
    # a bin is a square that has as bottom-let corner the coordinate x_bin[0], y_bin[0], as top-right corner the coordinate x_bin[1], y_bin[1]
    # I need to get indexes for start_x and start_y that fall within the pixel

    index_x = np.where((start_x_map >= lx + column*map_res) & (start_x_map < lx + (column + 1)*map_res))[0]
    index_y = np.where((start_y_map >= bot + row*map_res) & (start_y_map < bot + (row + 1)*map_res))[0]

    real_index = np.intersect1d(index_x, index_y)

    start_x_map = start_x_map[real_index]
    start_y_map = start_y_map[real_index]
    start_z_map = start_z_map[real_index]

    real_index = None
    
    # if there are not enough displacements per pixel, return 0
    N = len(start_x_map)
    
    if N < disp_SMDM_minimum:
        
        return 0

    jump = int(delta_t/time_step)

    total_time_ = time_step*jump

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

        simulation1 = smoldyn_billiard_precise(center, radius, length, particles_number, D, total_time_, time_step, start_x[-1], start_y[-1], start_z[-1], seed, main_path)

        output_start.append(simulation1[0, 1:])
        output_end.append(simulation1[-1, 1:])

    start_x_map = None
    start_y_map = None
    start_z_map = None

    # start simulation routine for each array
    for i in range(loops):

        particles_number = len(start_x[i])

        simulation1 = smoldyn_billiard_precise(center, radius, length, particles_number, D, total_time_, time_step, start_x[i], start_y[i], start_z[i], seed, main_path)

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


# wrapper to start the fitting routine
def find_best_D_per_pixel_map(x, xy):

    # get info about which pixel we are analyzing
    row = x[0]
    column = x[1]
    map0, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, map_res, main_path, seed = xy

    # get the diffusion coefficient of that pixel
    p0 = map0.iat[row, column]

    map0 = None

    # change the randoms seed used by Smoldyn, so simulation will be different for every pixel
    seed = seed + row + column

    xy = [p0, row, column, map_res, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, main_path, seed]

    start_x_map = None
    start_y_map = None
    start_z_map = None

    # fitting function
    result = minimize(find_difference_per_pixel_map, p0, xy, method='Nelder-Mead', options={'xatol': 0.01})

    return result.x[0]


# wrap previous functions together
def final_function_map(map0, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, map_res, main_path, seed):
    
    # get z values inside spherocylinder given x,y positions
    start_z_map = get_z_value(center, radius, length, start_x_map, start_y_map, seed)

    xy = [map0, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, map_res, main_path, seed]
    
    start_x_map = None
    start_y_map = None
    start_z_map = None

    rows = map0.shape[0]
    columns = map0.shape[1]
    
    map0 = None

    # create a map of indexes (pixels) to use the applymap function and generate a result for each pixel
    indexes = [[(i, j) for j in range(columns)] for i in range(rows)]

    index_map = pd.DataFrame(data=[*indexes], index=range(rows), columns=range(columns))

    indexes = None

    # start fitting routine
    index_map = index_map.applymap(lambda x: find_best_D_per_pixel_map(x, xy))

    return index_map


# start fitting routine to get D
def find_difference_per_pixel_regions(D, xy):

    # unpack all variables
    p0, column, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, main_path, seed = xy

    # if initial diffusion is zero, return zero
    if p0 == 0:

        return 0
    
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)
    
    # find what regions we are analyzing based on the column value
    if column == 0:
        real_index = np.where(start_x_map < clx)[0]

    elif column == 1:
        real_index = np.where((start_x_map >= clx) & (start_x_map <= crx))[0]

    elif column == 2:
        real_index = np.where(start_x_map > crx)[0]

    else:

        print('error')
        return

    start_x_map = start_x_map[real_index]
    start_y_map = start_y_map[real_index]
    start_z_map = start_z_map[real_index]

    real_index = None
    
    # if there are not enough displacements per pixel, return 0
    N = len(start_x_map)
    
    if N < disp_SMDM_minimum:
        
        return 0

    jump = int(delta_t/time_step)

    total_time_ = time_step*jump

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

        simulation1 = smoldyn_billiard_precise(center, radius, length, particles_number, D, total_time_, time_step, start_x[-1], start_y[-1], start_z[-1], seed, main_path)

        output_start.append(simulation1[0, 1:])
        output_end.append(simulation1[-1, 1:])

    start_x_map = None
    start_y_map = None
    start_z_map = None

    # start simulation routine for each array
    for i in range(loops):

        particles_number = len(start_x[i])

        simulation1 = smoldyn_billiard_precise(center, radius, length, particles_number, D, total_time_, time_step, start_x[i], start_y[i], start_z[i], seed, main_path)

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

# wrapper to start the fitting routine
def find_best_D_per_pixel_regions(x, xy):

    # get info about which pixel we are analyzing
    map0, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, main_path, seed = xy
    
    column = x

    # get the diffusion coefficient of that pixel
    p0 = map0.iat[0,column]

    map0 = None

    # change the randoms seed used by Smoldyn, so simulation will be different for every pixel
    seed = seed + column

    xy = [p0, column, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, main_path, seed]

    start_x_map = None
    start_y_map = None
    start_z_map = None

    # fitting function
    result = minimize(find_difference_per_pixel_regions, p0, xy, method='Nelder-Mead', options={'xatol': 0.01})

    return result.x[0]

# wrap previous functions together
def final_function_regions(map0, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, main_path, seed):
    
    # get z values inside spherocylinder given x,y positions
    start_z_map = get_z_value(center, radius, length, start_x_map, start_y_map, seed)

    xy = [map0, center, radius, length, time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, start_z_map, main_path, seed]
    
    start_x_map = None
    start_y_map = None
    start_z_map = None
    
    map0 = None
    
    # create an array of indexes to determine different regions
    # 0 = left, 1 = center, 2 = right
    indexes = [[0,1,2]]
    index_map = pd.DataFrame(data=indexes, index=range(1), columns=range(3))

    indexes = None

    # start fitting routine
    index_map = index_map.applymap(lambda x: find_best_D_per_pixel_regions(x, xy))

    return index_map

# function to wrap all previous functions together
def start_program(main_path, center, length, width, D, N, total_time, time_step, delta_t, map_res, p0, disp_SMDM_minimum, max_simulations, sim_type, slowdown_factor, i, path_to_add, path=path, seed=seed):
    
    radius = width/2
    
    path += path_to_add
    
    print('starting initial simulation')
    if sim_type == 'diffusion':
        print('simulation type: normal diffusion')
        simulation_file, sim_file_path, sim_file_name = smoldyn_billiard_diffusion(sim_type, center, radius, length, N, D, total_time, time_step, seed, i, path)
    elif sim_type == 'aggregation':
        print('simulation type: aggregation')
        simulation_file, sim_file_path, sim_file_name = smoldyn_billiard_aggregation(sim_type, center, radius, length, N, D, slowdown_factor, total_time, time_step, seed, i, path)
    elif sim_type == 'interaction':
        print('simulation type: interaction')
        simulation_file, sim_file_path, sim_file_name = smoldyn_billiard_interaction(sim_type, center, radius, length, N, D, slowdown_factor, total_time, time_step, seed, i, path)
    elif sim_type == 'two_components':
        print('simulation type: two components diffusion')
        simulation_file, sim_file_path, sim_file_name = smoldyn_billiard_two_comps(sim_type, center, radius, length, N, D, slowdown_factor, total_time, time_step, seed, i, path)
    else:
        print('ERROR: wrong type of simulation used as input. Check for typos')
        return
        
    print('simulation finished')
    print('saving the simulation file...')
    save_data(simulation_file, sim_file_path)
    print('data saved in Files with name ' + sim_file_name)
    print()
    
    # get all the x,y starting positions of every particle
    jump = int(delta_t/time_step)

    start_x_map = simulation_file[:-jump, 1::3].copy().ravel()
    start_y_map = simulation_file[:-jump, 2::3].copy().ravel()

    
    print('Creating simulation map...')
    simulation_map, sim_map_path, sim_map_name = create_map(sim_type, center, radius, length, simulation_file, map_res, delta_t, time_step, D, N, i, path)
    print('Simulation map created.')
    print('saving the simulation map...')
    map_name = save_df(simulation_map, sim_map_path, sim_map_name)
    print('data saved in Files with name ' +  map_name)
    print()
    
    print('Creating simulation map for regions...')
    simulation_map_regions, sim_map_path_regions, sim_map_name_regions = create_map_regions(sim_type, center, radius, length, simulation_file, delta_t, time_step, D, N, i, path)
    print('Simulation map created.')
    
    # free some memory
    simulation_file = None
    
    print('saving the simulation map...')
    map_name_regions = save_df(simulation_map_regions, sim_map_path_regions, sim_map_name_regions)
    print('data saved in Files with name ' +  map_name_regions)
    print()
    
    print('Creating the diffusion map...')
    diffusion_map, diffusion_average = create_map_fit(simulation_map, delta_t, p0, disp_SMDM_minimum)
    print('Diffusion map created.')
    print('saving the diffusion map...')
    diff_map_name = save_map_fit(diffusion_map, sim_map_path, sim_map_name)
    print('data saved in Files with name ' + diff_map_name)
    print()
    
    # free some memory
    simulation_map = None
    
    print('Creating the diffusion map for regions...')
    diffusion_map_regions, diffusion_average_regions = create_map_fit(simulation_map_regions, delta_t, p0, disp_SMDM_minimum)
    print('Diffusion map created.')
    print('saving the diffusion map...')
    diff_map_name_regions = save_map_fit(diffusion_map_regions, sim_map_path_regions, sim_map_name_regions)
    print('data saved in Files with name ' + diff_map_name_regions)
    print()
    
    # free some memory
    simulation_map_regions = None
    
    # start recursive routine to get reconstructed diffusion maps
    cores = mp.cpu_count()
    
    # limit number of simulations
    if cores > max_simulations:
        cores = max_simulations

    # routine for reconstructed regions
    start = time.time()

    df_regions = [(diffusion_map_regions, center, radius, length,
           time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, main_path, int(start) + j) for j in range(cores)]

    print('start routine for simulation based reconstructed diffusion for regions')
    p = mp.Pool(cores)
    df_regions = p.starmap(final_function_regions, df_regions)
    p.terminate()
    
    # average all the simulations
    for j in range(len(df_regions)):
        df_regions[j] = df_regions[j].to_numpy()

    final_map_regions = np.mean(df_regions, axis=0)
    
    final_map_regions = pd.DataFrame(final_map_regions)
    
    print('recursive routine terminated for regions')
    print('saving the final diffusion regions...')
    final_path_regions = sim_map_path_regions[:-14] + 'final_regions_df.csv.zip'
    final_name_regions = sim_map_name_regions[:-14] + 'final_regions_df.csv.zip'
    final_map_regions.to_csv(final_path_regions)
    print('data saved in Files with name ' + final_name_regions)
    print()
    
    # routine for reconstructed map
    start = time.time()

    df = [(diffusion_map, center, radius, length,
           time_step, delta_t, disp_SMDM_minimum, start_x_map, start_y_map, map_res, main_path, int(start) + j) for j in range(cores)]

    # free memory
    start_x_map = None
    start_y_map = None

    print('start routine for simulation based reconstructed diffusion for map')
    p = mp.Pool(cores)
    df = p.starmap(final_function_map, df)
    p.terminate()

    # average all the simulations
    for j in range(len(df)):
        df[j] = df[j].to_numpy()

    final_map = np.mean(df, axis=0)

    final_average = final_map.ravel()
    indexes = np.where(final_average == 0)[0]
    final_average = np.delete(final_average, indexes)
    final_avg = np.mean(final_average)
    final_stdev = np.std(final_average)
    final_map = pd.DataFrame(final_map)

    print('recursive routine terminated for map')
    print('saving the final diffusion map...')
    final_path = sim_map_path[:-14] + 'final_map_df.csv.zip'
    final_name = sim_map_name[:-14] + 'final_map_df.csv.zip'
    final_map.to_csv(final_path)
    print('data saved in Files with name ' + final_name)
    print()
    
    # create delta maps: reconstructed - SMdM
    delta_map = final_map.to_numpy() - diffusion_map.to_numpy()
    
    delta_avg = np.mean(final_map.to_numpy()) - np.mean(diffusion_map.to_numpy())
    
    # convert maps into dataframes
    delta_map = pd.DataFrame(delta_map)
    
    print('saving the delta diffusion map...')
    delta_path = sim_map_path[:-14] + 'delta_map_df.csv.zip'
    delta_name = sim_map_name[:-14] + 'delta_map_df.csv.zip'
    delta_map.to_csv(delta_path)
    print('data saved in Files with name ' + delta_name)
    
    print('saving the diffusion plots...')
    diffusion_map = pd.DataFrame(diffusion_map)

    # do all plotting here
    # get shape limits
    cx, cy, cz, lx, clx, crx, rx, top, bot, pos, neg = critical_points_cell(center, radius, length)

    # set limits of axes
    left_axis = lx - 0.5
    right_axis = rx + 0.5
    bot_axis = bot - 0.5
    top_axis = top + 0.5
    neg_axis = neg - 0.5
    pos_axis = pos + 0.5
    
    x1, x2, y1, y2 = left_axis, right_axis, bot_axis, top_axis
    
    fig, axes = plt.subplots(3, 1, figsize=(13, 16))
    fig.suptitle('D = ' + str(D) + ', N = ' + str(N) + ', length = ' + str(length) + ', width = ' + str(width))
    
    image1 = axes[0].imshow(diffusion_map.to_numpy(), origin="lower", cmap='PuOr_r', extent=(x1, x2, y1, y2), vmin = 0.5*diffusion_average, vmax = 1.5*diffusion_average)
    image2 = axes[1].imshow(final_map.to_numpy(), origin="lower", cmap='PuOr_r', extent=(x1, x2, y1, y2), vmin = 0.5*final_avg, vmax = 1.5*final_avg)
    image3 = axes[2].imshow(delta_map.to_numpy(), origin="lower", cmap='bwr', extent=(x1, x2, y1, y2), vmin = -3*delta_avg, vmax = 3*delta_avg)
    images = [image1, image2, image3]
    
    titles = ["SMdM", "SBRD-SMdM", "Delta Map"]
    
    for j in range(3):
        axes[j].set_title(titles[j])
        axes[j].set_ylabel(r'$\mu m$')
        cbar = fig.colorbar(images[j], ax=axes[j])
        cbar.set_label(r'$D$ $(\mu m^2/s)$')
        
    axes[-1].set_xlabel(r'$\mu m$')
    
    # for delta image
    cbar.set_label(r'$\Delta D$ $(\mu m^2/s)$')
    
    plt_path = sim_map_path[:-14] + 'maps_comparison.pdf'
    plt_name = sim_map_name[:-14] + 'maps_comparison.pdf'
    plt.savefig(plt_path, dpi=300, format='pdf')
    plt.close('all')
    
    print('plot saved in Files with name ' + plt_name)
    
    print()
    
    print('generating figure for error analysis')

    final_standardized = (final_average - final_avg)/final_stdev

    def gaussian(x, mean, stdev):

        return (1/(stdev*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x - mean)/stdev)**2)

    gaussian_plot = gaussian(np.linspace(np.amin(final_average), np.amax(
        final_average), 100), np.mean(final_average), np.std(final_average))
    gaussian_standardized = gaussian(np.linspace(np.amin(final_standardized), np.amax(
        final_standardized), 100), 0, 1)

    plt_name = final_name[:-8] + '_analysis.png'
    plt_path = path + '/' + plt_name
    fig = plt.figure(figsize=(10, 10))
    axs1 = fig.add_subplot(221)
    axs2 = fig.add_subplot(222)
    axs3 = fig.add_subplot(223)
    axs4 = fig.add_subplot(224)

    fig.suptitle('ANALYSIS OF DIFFUISON HOMOGENEITY')
    axs1.hist(final_average, bins='doane', density=True,
                histtype='step', color='blue', label='diffusion distribution')
    axs1.hist(final_average, bins='doane', density=True,
                histtype='stepfilled', color='blue', alpha=0.3)
    axs1.plot(np.linspace(np.amin(final_average), np.amax(final_average), 100),
                gaussian_plot, linewidth=4, color='orange', label='gaussian plot')
    axs1.legend()
    axs1.set_xlabel('diffusion coefficient $(\mu m ^ 2/s)$')
    axs1.set_ylabel('probability density')
    axs1.set_title('Histogram and Gaussian')
    axs2.boxplot(final_average)
    axs2.set_xticklabels('')
    axs2.set_xticks([])
    axs2.set_title('boxplot')
    axs2.set_ylabel('diffusion coefficient $(\mu m ^ 2/s)$')
    axs3.hist(final_standardized, bins='doane', density=True,
                histtype='step', color='blue')
    axs3.hist(final_standardized, bins='doane', density=True,
                histtype='stepfilled', color='blue', alpha=0.3)
    axs3.plot(np.linspace(np.amin(final_standardized), np.amax(final_standardized), 100),
                gaussian_standardized, linewidth=4, color='orange')
    axs3.set_xlabel('standardized diffusion coefficient')
    axs3.set_ylabel('probability density')
    axs3.set_title('Standardized Histogram and Gaussian')
    sm.qqplot(final_standardized, line='45', ax=axs4)
    axs4.set_title('q-q plot')
    plt.savefig(plt_path, dpi=300, format='png')
    plt.close('all')
    print('plot saved in Files with name ' + plt_name)
    print()
    
    return
