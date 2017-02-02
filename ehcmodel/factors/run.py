# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE

from ehcmodel.factors.train_ae import optimize
from ehcmodel.factors.eval_ae import eval_on_grid
from ehcmodel.factors.experiment import run_experiment

# This is the demo script for factor learning i.e. the learning of place cell representation from
# idiothetic information and the known head cell representation.

#Parameters of the autoencoder training can be set here. For any given parameter multiple
# values can be given (in the form of lists) to setup multiple experiments at once.
param_grid = {
    'dir_name' : 'demo', #name of the folder
    'description' : "", #description string
    'sampling_rate' : [4], #how to sample to full database's location coordinates. 1 means use the full database. 2 means
                           #take every 2nd location on the x and every 2nd location on y axis and so on.
    'n_code' : [30], #dimension of the inne representation, number of formed place cells
    'n_coding_layer' : [1], #number of en- and decodeing layers of the autoencoder. 1 means no intermediate layers between input and representation
    'sparsity_mode' : [('lifetime','all')], # what type of sparsity to use in the nonlinearity of the middle layer: 'spatial' or ('lifetime','all')
    'sparsity_k' : [1./15], # k parameter of sparsity. For spaital sparsity it is and integer representing the number of nonzero components. For lifetime sparsity it is a float representing the ratio of nonzero samples.
    'dropout_p' : [None], # Dropout can be added for the encoding layers with this parameter. Float or None
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 #Type of preprocess. A tuple with arbitrary number ofelements which indicates preprocesing steps, executed in the given order. Possible preprocessing steps: 
    # ('retina',k): k indicates the angle for a single viewing angle, 20 means 20 + 2*4 degree which overlaps (=28)
    'view_angle' : [220],#[20,60,100,140,180,220,260,300,340], #size of the non_masked part ov the view. Must be divisible by the retina k parameter.
    'turn_angle' : [20], #while sampling rate samples the locations coordinates, the turn_angle aprameter can sample the directions. Any multiple of 10 degrees can be used.
    'minibatch_size': ['fullbatch'], #for minibatch training: integer of minibatch size. for fullbatch training: 'fullbatch' string
    'reconst_mask': [True], #bool, whether to use a mask in the loss function
    'early_stop': True, #bool, whether to use early stopping with 500 epoch early stopping patience (can be modified from source)
    'n_epochs' : 100, #max number of epochs, if early_stop is used then reaching it is not guaranteed.
    }
    
run_experiment(optimize,**param_grid) #run the training of the autoencoder

eval_funcs = [
            #'activations',
            'summed_activations',
            'spatial_coverage_plot',
            'epoch_metrics',
            #'use_cache',
            ]

#Evaluate the trained model respect to place cell formation in the inner representation.
#Plots generated for Figure 3 and 4. Images can be found in <dir_name>/all/bin_lin_wta andz_lin respectively
eval_on_grid(eval_funcs,**param_grid)
