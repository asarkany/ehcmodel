# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE

from ehcmodel.factors.train_ae import optimize
from ehcmodel.factors.eval_ae import eval_on_grid
from ehcmodel.factors.experiments import run_experiment

checkpoints_dir = 'checkpoints0725'
global_param_grid = {
    'early_stop': True,
    'n_epochs' : 10000,
    }

def test_test(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test_test',
    'description' : "test test",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'n_epochs' : 10,
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [140],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    'early_stop': False,
    'n_epochs' : 10,
    })
    
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)


def test1(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test1',
    'description' : "test1",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test1_early_stop(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test1_early_stop',
    'description' : "test1 early stop",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'n_epochs' : 10000,
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    'early_stop': True,
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test2(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test2',
    'description' : "test2",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))

def test3(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test3',
    'description' : "test3",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [60],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test4(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test4',
    'description' : "test4",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',60),'normalize','sort_by_angles'),(('retina',60),'sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [60,180,300],
    'turn_angle' : [60],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test5(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test5',
    'description' : "test5",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [('normalize','sort_by_angles'),('sort_by_angles',)], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))  

def test6(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test6',
    'description' : "test6",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./360,1./60,1./30,2./30,8./30,15./30,25./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test7(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test7',
    'description' : "test7",
    'sampling_rate' : [4],
    'n_code' : [1,15,30,60,120],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))  
        
        
def test8(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test8',
    'description' : "test8",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1,2,4],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [0.5],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,140,260],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360})) 
        
def test9(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test9',
    'description' : "test9",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [2,4],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None,0.2,0.5,0.7,0.9],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,140,260],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))

def test10(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test10',
    'description' : "test10",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [False],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test11(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test11',
    'description' : "test11",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test_multilayer(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test_multilayer',
    'description' : "test_multilayer",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [4],
    'n_epochs' : 10000,
    'sparsity_mode' : [('lifelong','groupwise')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None,0.5],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize','sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [140],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    'early_stop': True,
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        
def test16(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test16',
    'description' : "test16",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),)], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [False],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))

def test17(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test17',
    'description' : "test17",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'sort_by_angles')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [60],#[20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test18(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test18',
    'description' : "test18",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./15],#[1./60,1./15],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def debug_test18(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test18',
    'description' : "test18",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./15],#[1./60,1./15],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [220],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    #'n_epochs' : 100,
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        #run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        #eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test19(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test19',
    'description' : "test19",
    'sampling_rate' : [4],
    'n_code' : [15,60],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))


        
def test22(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test22',
    'description' : "test22",
    'sampling_rate' : [4],
    'n_code' : [15,60],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [('normalize',),(None,)], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test23(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test23',
    'description' : "test23",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [2],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),)], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test24(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test24',
    'description' : "test24",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [2],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test24_relu(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test24_relu',
    'description' : "test24_relu just for evaluation!!!!",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [2],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test25(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test25',
    'description' : "test25",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./30],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [False],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test28(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test28',
    'description' : "test28",
    'sampling_rate' : [4],
    'n_code' : [15],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./15],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
        
def test29(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test29',
    'description' : "test29",
    'sampling_rate' : [4],
    'n_code' : [60],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./60],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
        
def test31(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test31',
    'description' : "test31",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : ['spatial'],
    'sparsity_k' : [1],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
        
def test33(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test33',
    'description' : "test33",
    'sampling_rate' : [4],
    'n_code' : [60],
    'n_coding_layer' : [2],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./60],#[1./60,1./15],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [220],#[20,60,100,140,180,220,260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))

def test34(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test34',
    'description' : "test34",
    'sampling_rate' : [4],
    'n_code' : [90],
    'n_coding_layer' : [1],
    'sparsity_mode' : [('lifelong','all')],
    'sparsity_k' : [1./90],
    'dropout_p' : [None],
    'y_overlap' : ['noy'],
    'preprocess' : [(('retina',20),'normalize')], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [260,300,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [True],
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360}))
                
def test_de1(mode = 'train',eval_funcs = None):
    param_grid = dict(global_param_grid,**{
    'dir_name' : checkpoints_dir+'/test_de1',
    'description' : "test_de1",
    'sampling_rate' : [4],
    'n_code' : [30],
    'n_coding_layer' : [1],
    'sparsity_mode' : ['spatial'],
    'sparsity_k' : [1],#[1./60,1./15],
    'dropout_p' : [None],
    'y_overlap' : [('degrade',6)], #,('degrade',6)
    'preprocess' : [('normalize',)], # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : [180,260,20,100,340],
    'turn_angle' : [20],
    'xcov_lambda' : 0,
    'minibatch_size': ['fullbatch'],
    'reconst_mask': [False],
    #'n_epochs' : 2,
    })
    if mode == 'train':    
        run_experiment(optimize,**param_grid)
        run_experiment(optimize,**dict(param_grid,**{'view_angle':360,'turn_angle':360,'y_overlap':'onehot'}))
    elif mode == 'eval':
        eval_on_grid(eval_funcs,**param_grid)
        eval_on_grid(eval_funcs,**dict(param_grid,**{'view_angle':360,'turn_angle':360,'y_overlap':'onehot'}))
        
        

