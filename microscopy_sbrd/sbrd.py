#!/usr/bin/env python
# coding: utf-8

# import packages
import warnings
import inspect
from pathlib import Path
import os

# NOTE: Smoldyn runs openGL "under water". If you run this code remotely (e.g. on a server) you will need to enable X11 forwarding

# this is important to allow paralelization of smoldyn
alpha = Path(inspect.stack()[-1].filename)

# remove runtime warnings. array([1,2,np.nan]) > 0 outputs T, T, F, which is correct, but raises a runtime_warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

path = os.getcwd()
path = path[:-4]

# TODO: set this to the desired paths!
# source_path = path + 'measurements/output/dima'
# destination_path = path + 'measurements/sbrd_output/dima'

source_path = path + 'measurements/output/antibiotics/erythromycin'
destination_path = path + 'measurements/sbrd_output/antibiotics/erythromycin'

# this is important for multiprocessing in Windows
if __name__ == "__main__":

    # import functions created for the program
    import simulation_multiprocessing.all_functions as run

    # make sure the destination folder exists
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
        
    final_map = run.start_program(alpha, source_path, destination_path)