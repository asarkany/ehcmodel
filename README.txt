# ehcmodel

This repository contains the source code for the paper

Lőrincz, A., Sárkány, A. (2016). Semi-Supervised Learning of Cartesian Factors: A Top-Down Model of the Entorhinal Hippocampal Complex.

submitted to Frontiers in Psychology.

The two main software components are
I) Software implemented in the Unity3d framework to generate the observations in the arena
II) Python software for data analysis

===============================================
I) ehcmodel_dbgen

Requirements:
- latest version of Unity (tested with 5.1.2f)

Usage
1) Load the project in Unity
2) Set the parameters of the db generation in Controller.cs (number of boxes,size of the arena, field of view etc.)
3) Run from the Editor
4) Data files are generated in the output directory

II) ehcmodel
This Python package contains the scripts to
a) train and evaluate an autoencoder forming place cell representation on its hidden layer
b) learn a metric-like grid structure from the place cell representation

Requirements
- Python 2.7
- GPU with 4GB memory
- 24GB RAM (for some parameters)
- Python packages: theano, sklearn, matplotlib

Usage

a)
    1) Add the ehcmodel package to the PYTHONPATH
    2) Set paths in ehcmodel/config.cfg
    3a) For a single run you can use ehcmodel/factors/run.py . Set the parameters and run the script.
    3b) test_suite.py contains experiments we did. You can try those too.
    4) In your plot directory (set in config.cfg) you can find the evaluation plots.
    
b)
    1) Add the ehcmodel package to the PYTHONPATH
    2) Set paths in ehcmodel/config.cfg
    3) Set the parameters in ehcmodel/metric/grid_analysis.py and run the script.
    4) Look for the resulting plots in the plot directory.
    
License: BSD 3-clause, attached.

