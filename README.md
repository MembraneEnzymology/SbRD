# SbRD

This is a repository of code developed in the Membrane Enzymology group of the University of Groningen for the analysis of protein diffusion in Escherichia coli cytoplasm using Simulation-based Reconstructed Diffusion (SbRD).

If you are using code from this package please refer to it using the following doi : [![DOI](https://zenodo.org/badge/597394159.svg)](https://zenodo.org/badge/latestdoi/597394159)

The repository is organized into four parts :
1. The mathematical part contains all the scripts used to find a numerical solution to the problem of finding a bouncing point inside a billiard for multiple particles with known start and end points. Particles positions are generated via simulations using Smoldyn (https://www.smoldyn.org)
2. The simulation part contains all the code for performing diffusion simulations with Smoldyn (https://www.smoldyn.org) and to perform Single Molecule displacement Mapping (SMdM), followed by Simulation-based Reconstructed Diffusion (SbRD) on these simulations. Simulations and SMdM analysis are performed as explained in our work: DOI: 10.1126/sciadv.abo5387 
3. The cell detection part consists in a Jupyter notebook that allows to perform cell detection of microscopy data and to fit billiards around the detected cells. The observed displacements are then analyzed via SMdM to obtain diffusion maps. Analysis of microscopy data is performed as explained in our work: DOI: 10.1126/sciadv.abo5387
4. The SbRD part contains all the code to perdorm Simulation-based Reconstructed Diffusion on microscopy data.
5. The analysis notebook is used to performed data analysis on all the acquired datasets.

## mathematical analysis ##
* Run program.py : In this script the user can change the values that are used to performe the diffusion simulations with Smoldyn.
* scripts
    * all_functions_2D.py & all_functions_3D.py : In these scripts are contained all the functions necessary to perform a simulation and to analyze it with a mathematical method that approximates the bounces of particles off of the confinement region.

## simulation part ##
* sbrd_simulation.py : In this script the user can change the values that are used to performe the diffusion simulations with Smoldyn and the parameters used for SbRD analysis
* billiard_simulation_multi
  * all_functions_2D.py & all_functions_3D.py : In these scripts are contained all the functions necessary to perform a simulation and to analyze it with SbRD. NOTE: this code uses multiprocessing.

## Cell detection ##
* cell_detection_billiard.ipynb : This notebook allows to perform cell detection, billiard fitting and SMdM analysis of microscopy data acquired as described in DOI: 10.1126/sciadv.abo5387. Parameters in the first cell of the notebook can be changed to adjust thresholds for sensitivity, dividing cells length, and more.

## SbRD microscopy ##
* sbrd.py : script used to start the sbrd process. Here, input and output paths can be changed.
* simulation_multiprocessing
  * all_functions.py : This script allows to perform SbRD on all the desired cells
  
## data analysis ##
* analysis_notebook.ipynb : This notebook contains all the scripts used to preform the data analysis in our paper "Simulation-based Reconstructed Diffusion provides accurate values of mobility in confined spaces, unveiling the effect of aging in Escherichia coli"

## Used Python Packages ##
* [numpy] (http://www.numpy.org/)
* [pandas] (https://pandas.pydata.org/)
* [matplotlib] (http://matplotlib.org/)
* [seaborn] (https://seaborn.pydata.org/)
* [scipy] (https://www.scipy.org/)
* [h5py] (https://www.h5py.org/)
* [tifffile] (https://pypi.org/project/tifffile/)
* [storm_analysis] (https://github.com/ZhuangLab/storm-analysis)
* [smoldyn] (https://github.com/ssandrews/Smoldyn)
* [psycopg2] (https://pypi.org/project/psycopg2/)
* [PyDAQmx] (https://pypi.org/project/PyDAQmx/)
* [statsmodels] (https://www.statsmodels.org/stable/index.html)
* [scikit-image] (https://scikit-image.org/)
