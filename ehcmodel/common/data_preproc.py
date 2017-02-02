# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE
from collections import Counter
import math
import numpy as np
import scipy
import os
from scipy.io import loadmat
import theano
import theano.tensor as T
import random
import subprocess
import glob

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ehcmodel
from ehcmodel.common.kdl_template import projection_layer,np_tanh_fan,softplus,linear,convert_to_one_hot


import ConfigParser
config = ConfigParser.ConfigParser()
config.read(os.path.join(os.path.split(ehcmodel.__file__)[0],"config.cfg"))
data_root = config.get('paths','data')#'/home/redragon/projects/factorization/sparse_ae/SciPy2015/data/'
plot_root = config.get('paths','plot')

es_glob = lambda : {'lab1':{'num_boxes' : 30, 'num_angles' :12, 'step_size':0.2,'view_angle':60},
        'lab2_60v_30t':{'num_boxes' : 50, 'num_angles' :12, 'step_size':0.1,'view_angle':60},
        'lab2_fewer_60v_30t':{'num_boxes' : 20, 'num_angles' :12, 'step_size':0.1,'view_angle':60},
        'lab2_big_60v_30t':{'num_boxes' : 150, 'num_angles' :12, 'step_size':0.1,'view_angle':60},
        'lab2_tri01':{'num_boxes' : 150, 'num_angles' :12, 'step_size':0.1,'view_angle':60},
        'test0':{'num_boxes' : 2, 'num_angles' :12, 'step_size':0.1,'view_angle':60},
        'test1':{'num_boxes' : 3, 'num_angles' :12, 'step_size':0.1,'view_angle':60},
        'test2':{'num_boxes' : 3, 'num_angles' :12, 'step_size':0.1,'view_angle':60},
        'lab2_big_60v_10t':{'num_boxes' : 150, 'num_angles' :36, 'step_size':0.1,'view_angle':60},
        'lab2_big_60v_10t_t':{'num_boxes' : 150, 'num_angles' :36, 'step_size':0.1,'view_angle':60},
        'lab2_big_10v_10t_t':{'num_boxes' : 150, 'num_angles' :36, 'step_size':0.1,'view_angle':10},
        'lab2_big_20v_10t':{'num_boxes' : 150, 'num_angles' :36, 'step_size':0.1,'view_angle':20},
        'lab2_big_20v_10t_t':{'num_boxes' : 150, 'num_angles' :36, 'step_size':0.1,'view_angle':20},
        'lab2f_big_20v_10t':{'num_boxes' : 150, 'num_angles' :36, 'step_size':0.1,'view_angle':20},
        }
        
def load_boxworld_raw(exp_name,preprocess=None):
    data_path = os.path.join(data_root,exp_name)
    mdata = loadmat(os.path.join(data_path,'MDATA.mat'))
    mdata = mdata['Mdata']
    posori = loadmat(os.path.join(data_path,'posori.mat'))
    posori = posori['A'].T
    #posori_new = np.hstack([(posori[:,2]-min(posori[:,2])).reshape(-1,1),
    #                        (posori[:,0]-min(posori[:,0])).reshape(-1,1),
    #                        posori[:,4].reshape(-1,1)])
    posori_new = np.hstack([(posori[:,0]-min(posori[:,0])).reshape(-1,1),
                            (posori[:,2]-min(posori[:,2])).reshape(-1,1),
                            posori[:,4].reshape(-1,1)])
    #posori_new[:,0] = max(posori_new[:,0]) - posori_new[:,0]
    posori = posori_new
    #es = loadmat(os.path.join(data_path,'exp_settings.mat'))
    es =es_glob()[exp_name]
    if preprocess != None:
        posori = posori[::es['num_angles'],:]
        es['num_angles'] = 1
    es['num_samples'] = posori.shape[0]
    es['num_places'] = es['num_samples']/es['num_angles']
    es['turn_angle']= int(round(360/es['num_angles']))
    return mdata,posori,es
    
def read_boxpos(exp_name):
    with open(os.path.join(data_root,exp_name,'cube_positions_and_colors.txt'),'rb') as f:
        return np.array( [[float(val)  for val in line.split(" ")] for line in f.read().split('\n')[:-1]])

#Load data
def load_boxworld_data(exp_name,preprocess = None):
    data_path = os.path.join(data_root,exp_name)
    mdata = loadmat(os.path.join(data_path,'MDATA.mat'))
    mdata = mdata['Mdata']
    #es = loadmat(os.path.join(data_path,'exp_settings.mat'))
    es =es_glob()[exp_name]
    X = mdata[0:es['num_boxes'],:].T
    y = mdata[es['num_boxes']:,:].T
    es['num_samples'] = X.shape[0]
    es['num_places'] = es['num_samples']/es['num_angles']
    ##################
    #estimate 360
    
    if preprocess != None:
        view_angle_idx = range(6-int(round(preprocess/360.0*es['num_angles']))/2+1,int(round(preprocess/360.0*es['num_angles']))/2+6)
        newX = np.zeros((es['num_places']*len(view_angle_idx),X.shape[1]))
        for vaii,vai in enumerate(view_angle_idx):
            aX = X[vai::es['num_angles']]
            newX[vaii::len(view_angle_idx)] =aX
        X = (newX.T.reshape(es['num_boxes'],-1,len(view_angle_idx)).sum(axis=2).T>0).astype(int)
        y = y[::es['num_angles']]
        #X = np.repeat(X,es['num_angles'],axis=0)
        es['num_angles'] = 1
    es['turn_angle']= int(round(360/es['num_angles']))
    es['num_samples'] = X.shape[0]
    
    
    assert(X.shape[0] == y.shape[0])
    y = convert_to_num_classes(y)
    boxpos = read_boxpos(exp_name)
    
    boxangles_bycos = [np.degrees(math.acos((scipy.spatial.distance.cosine([box[2],box[0]],[1,0])*-1)+1)) for box in boxpos]
    boxangles = np.array([angle if x>0 else 360-angle for angle,x in zip(boxangles_bycos,boxpos[:,0])])
    
    ordered_box_indices = np.argsort(boxangles)
    #print(zip(boxangles,boxangles_bycos,boxpos[:,0]))
    X = X[:,ordered_box_indices]
    #print(boxangles[ordered_box_indices])
    assert (not np.any(np.isnan(X)))
    return X,y,es

#SPlit for train-test
def split_data(X,y,posori,mode,train_ratio,es,random_state):
    if mode == 'simplesplit':
        perm = random_state.permutation(X.shape[0])
        last_train_idx = math.floor(train_ratio*X.shape[0])
        X_train = X[perm[:last_train_idx],:]
        y_train = y[perm[:last_train_idx],:]
        posori_train = posori[perm[:last_train_idx],:]
        X_test = X[perm[last_train_idx:],:]
        y_test = y[perm[last_train_idx:],:]
        posori_test = posori[perm[last_train_idx:],:]
    elif mode == 'placesplit':
        #Precondition samples are ordered by locations so all angles for a specific location are after each other
        perm = random_state.permutation(es['num_places'])
        last_train_place_idx = int(math.floor(train_ratio*es['num_places']))
        perm2 = []
        for i in perm:
            perm2.extend(range(i*es['num_angles'],(i+1)*es['num_angles']))
        X_train = X[perm2[:last_train_place_idx*es['num_angles']],:]
        y_train = y[perm2[:last_train_place_idx*es['num_angles']]]
        posori_train = posori[perm2[:last_train_place_idx*es['num_angles']],:]
        X_test = X[perm2[last_train_place_idx*es['num_angles']:],:]
        y_test = y[perm2[last_train_place_idx*es['num_angles']:]]
        posori_test = posori[perm2[last_train_place_idx*es['num_angles']:],:]
    elif mode == 'notest':
        X_train = X
        y_train = y
        posori_train = posori
        X_test = None
        y_test = None
        posori_test = None
    assert (not np.any(np.isnan(X_train)))
    assert (X_test == None or not np.any(np.isnan(X_test)))
    return [X_train,y_train,posori_train],[X_test,y_test,posori_test],None

#Sample the locations 
def sample_dataset(X,y,posori,es,sampling = 1):
    if sampling>1:
        es = dict(es)
        #Preconditions
        # samples are ordered by locations so all angles for a specific location are after each other
        # all location ha a sample
        
        
        room_x,room_y = np.max(np.round(posori[:,0:2]/es['step_size']),axis=0).astype(int)+1
        print(room_x,room_y)
        assert posori.shape[0] == es['num_angles']*room_x*room_y
        posori=posori.reshape(room_x,room_y,es['num_angles'],3)
        posori = posori[::sampling,::sampling]
        
        X = X.reshape(room_x,room_y,es['num_angles'],es['num_boxes'])
        X = X[::sampling,::sampling]
        y = y.reshape(room_x,room_y,es['num_angles'])
        y = y[::sampling,::sampling]
        
        posori = posori.reshape(-1,3)
        X = X.reshape(-1,es['num_boxes'])
        y = y.reshape(-1)
        #assert es['num_places'] == X.shape[0]/es['num_angles']*sampling*sampling
        es['num_samples'] =X.shape[0]
        es['num_places'] = es['num_samples']/es['num_angles']
        es['step_size'] = es['step_size']*sampling
    assert (not np.any(np.isnan(X)))
    return X,y,posori,es

def sort_by_angles(X,y,posori,es):
    angles = posori[:,2].round().astype(int)
    X_n = None
    y_n = None
    posori_n = None
    for angle in np.arange(es['num_angles'])*360/es['num_angles']:
        anglesn = angles == angle
        X_angle = X[anglesn]
        y_angle = y[anglesn]
        posori_angle = posori[anglesn,:]
        assert len(np.unique(y_angle)) == 1
        if X_n != None and y_n != None:
            X_n =np.vstack([X_n,X_angle])
            y_n =np.vstack([y_n,y_angle[:,np.newaxis]])
            posori_n = np.vstack([posori_n,posori_angle])
        else:
            X_n = X_angle
            y_n = y_angle[:,np.newaxis]
            posori_n = posori_angle
    assert (not np.any(np.isnan(X_n)))        
    return X_n,np.squeeze(y_n),posori_n

def augment_angles(X,y,posori,es,new_num_angles,new_view_angle):
    es = dict(es)
    #First generate the min_view_angle perspectives
    #turn_angle = 360/es['num_angles']
    """
    min_view_angle = turn_angle*2
    
    X = X.reshape(-1,es['num_angles'],es['num_boxes'])
    Xw = np.array(X,copy=True).astype(bool)
    X_n = None
    for ai in range(0,es['num_angles']):
        angle = ai*turn_angle
        first = (ai-es['view_angle']/turn_angle/2+1)%es['num_angles']
        second = (ai+es['view_angle']/turn_angle/2-1)%es['num_angles']
        print(angle)
        print((first-es['view_angle']/turn_angle/2)%es['num_angles'],(first+es['view_angle']/turn_angle/2)%es['num_angles'])
        print((second-es['view_angle']/turn_angle/2)%es['num_angles'],(second+es['view_angle']/turn_angle/2)%es['num_angles'])
        print((ai-es['view_angle']/turn_angle/2+1)%es['num_angles'],(ai+es['view_angle']/turn_angle/2-1)%es['num_angles'])
        X_angle = np.logical_and(Xw[:,(ai-es['view_angle']/turn_angle/2+1)%es['num_angles'],:],Xw[:,(ai+es['view_angle']/turn_angle/2-1)%es['num_angles'],:])
        assert(np.sum(X_angle)>0)
        print(np.sum(X_angle,axis=0))
        print(np.sum(Xw[:,(ai)%es['num_angles'],:].reshape(-1,es['num_boxes']),axis=0))
        sumdiff = np.sum(X_angle,axis=0)-np.sum(Xw[:,(ai)%es['num_angles'],:].reshape(-1,es['num_boxes']),axis=0)
        print(sumdiff*(sumdiff>0).astype(int))
        assert np.all(X_angle<= Xw[:,(ai)%es['num_angles'],:].reshape(-1,es['num_boxes']))
        X_n = np.vstack([X_n,X_angle]) if X_n !=None else X_angle
    Xw = np.swapaxes(X_n.reshape(es['num_angles'],-1,es['num_boxes']),0,1)
    """
    if not (new_view_angle>=es['view_angle'] and new_view_angle % es['turn_angle'] ==0 and new_view_angle <=360):
        raise ValueError(str(new_view_angle) + " is not valid as new view angle")
    if not es['num_angles']%new_num_angles ==0 or new_num_angles>es['num_angles']:
        raise ValueError(str(new_num_angles) + " is not valid as new num angle")
    X = X.reshape(-1,es['num_angles'],es['num_boxes'])
    Xw = np.array(X,copy=True).astype(bool)
    X_n_l = []
    for ai in range(0,es['num_angles']):
        angle = ai*es['turn_angle']
        from_angle = 1+(angle-(new_view_angle/2))/es['turn_angle']
        to_angle = (angle+(new_view_angle/2))/es['turn_angle']
        X_angle = None
        for ai2 in range(from_angle,to_angle,1):
            X_angle = np.logical_or(X_angle,Xw[:,ai2%es['num_angles'],:]) if X_angle != None else Xw[:,ai2%es['num_angles'],:]
        
        X_angle = X_angle.astype(int)
        
        if new_view_angle > es['view_angle']:
            assert np.all(X_angle>= X[:,ai%es['num_angles'],:].reshape(-1,es['num_boxes']))
        else:
            assert np.array_equal(X_angle,X[:,ai%es['num_angles'],:].reshape(-1,es['num_boxes']))
            
        X_n_l.append(X_angle) 
    X_n = np.vstack(X_n_l)
    Xw = np.swapaxes(X_n.reshape(es['num_angles'],-1,es['num_boxes']),0,1)    
    
    
    Xw = Xw[:,::es['num_angles']/new_num_angles,:].reshape(-1,es['num_boxes'])
    posori = np.copy(posori)
    posori = posori[::es['num_angles']/new_num_angles,:]
    es['num_angles'] =new_num_angles
    y = np.squeeze(np.repeat(np.arange(es['num_angles']).reshape(-1,1),es['num_places'],axis=1).T.reshape(-1,1))
    es['num_samples'] = es['num_angles']*es['num_places']
    es['view_angle'] = new_view_angle
    es['turn_angle']= int(round(360/es['num_angles']))
    assert (not np.any(np.isnan(Xw)))
    return Xw.astype("float32"),y,posori,es
    
#Create the retina representation
def retina_X(X,y,posori,es,bin_angle,view_angle,turn_angle):
    X_bin,y_bin,posori_bin,es_bin = augment_angles(X,y,posori,es,360/bin_angle,bin_angle)
    
    es = dict(es)
        
    num_angles = 360/turn_angle
    #print(es_bin['num_angles'],bin_angle,turn_angle,view_angle)
    Xw = X_bin.reshape(-1,es_bin['num_angles'],es['num_boxes'])
    assert (not np.any(np.isnan(Xw)))
    rX = []
    for pi in range(len(Xw)):
        
        for ai1 in range(0,es_bin['num_angles'])[::es_bin['num_angles']/num_angles]:
            angle = ai1*bin_angle
            #print(angle)
            from_angle = 1+(angle-(view_angle/2))/bin_angle
            to_angle = 1+(angle+(view_angle/2))/bin_angle
            r = np.zeros(es_bin['num_angles']*es['num_boxes'])
            #print(range(from_angle,to_angle,1))
            for ai2 in range(from_angle,to_angle,1):
                r[(ai2%es_bin['num_angles'])*es['num_boxes']:((ai2%es_bin['num_angles'])+1)*es['num_boxes']] = Xw[pi,ai2%es_bin['num_angles'],:]
            rX.append(r)
    rX = np.vstack(rX)
    _,_,posori,es = augment_angles(X,y,posori,es,360/turn_angle,view_angle)
    y = y_bin[::turn_angle/bin_angle]
    assert(rX.shape==(es['num_samples'] ,es_bin['num_angles']*es['num_boxes']))
    assert(len(y) == len(rX))
    assert (not np.any(np.isnan(rX)))
    return rX.astype("float32"),y,posori,es
        
    
def test_augment_angles():
    
    """
    X_all1,y_all1,es_all1 = load_boxworld_data('lab2_big_60v_10t')
    X_all2,y_all2,es_all2 = load_boxworld_data('lab2_big_60v_30t')
    X,y,es = augment_angles(X_all1,y_all1,es_all1,36,60)
    assert np.array_equal(X,X_all1)
    assert np.array_equal(y,y_all1)
    assert es == es_all1
    
    X,y,es = augment_angles(X_all1,y_all1,es_all1,12,60)
    assert np.array_equal(X,X_all2)
    assert np.array_equal(y,y_all2)
    assert es == es_all2

    X,y,es = augment_angles(X_all2,y_all2,es_all2,12,60)
    assert np.array_equal(X,X_all2)
    assert np.array_equal(y,y_all2)
    assert es == es_all2
    """
    """
    X,y,es = augment_angles(X_all3,y_all3,es_all3,36,60)
    assert np.array_equal(X,X_all1)
    assert np.array_equal(y,y_all1)
    assert es == es_all1
    """
    """
    X,y,es = augment_angles(X_all3,y_all3,es_all3,12,60)
    assert np.array_equal(X,X_all2)
    assert np.array_equal(y,y_all2)
    assert es == es_all2
    """
    for ds in ['lab2_big_20v_10t','lab2_big_20v_10t_t']:
        X_all3,y_all3,es_all3 = load_boxworld_data(ds)
        
        X,y,es = augment_angles(X_all3,y_all3,es_all3,36,20)
        assert np.array_equal(X,X_all3)
        assert np.array_equal(y,y_all3)
        assert es == es_all3
        
        
        Xs,ys,ess = augment_angles(X_all3,y_all3,es_all3,36,60)
        X,y,es = augment_angles(X_all3,y_all3,es_all3,36,20)
        X,y,es = augment_angles(X,y,es,36,60)
        assert np.array_equal(X,Xs)
        assert np.array_equal(y,ys)
        assert es == ess

        X2,y2,es2 = augment_angles(X_all3,y_all3,es_all3,36,20)
        for a in range(40,361,20):
            X1,y1,es1 = X2,y2,es2
            X2,y2,es2 = augment_angles(X_all3,y_all3,es_all3,36,a)
            assert np.sum(X1)<np.sum(X2)
            assert np.all(np.sum(X1,axis=1)<=np.sum(X2,axis=1))
            assert np.all(y1==y2)
            assert es1['view_angle']<es2['view_angle']

    #this is only relevant for _t
    X,_,_ = augment_angles(X_all3,y_all3,es_all3,36,360)
    assert np.sum(X)==np.prod(X.shape)
    
def show_inputs(posori,es,orig_step_size = 0.1,size=None):
    xs,ys = np.round(posori[:,0:2]/orig_step_size).T.astype(int)
    if size==None:    
        xsize = max(xs)+1
        ysize = max(ys)+1
    else:
        xsize,ysize = size

    act_2d = np.zeros([xsize,ysize])
    for x,y in zip(xs,ys):
        act_2d[x,y] += 1
    
    plt.figure()
    im = plt.imshow(act_2d,interpolation='none',clim=(0,es['num_angles']))
    plt.colorbar(im)
    return act_2d
        
    
def convert_to_num_classes(y):
    new_y = []
    for i in range(y.shape[0]):
        new_y.append(np.where(y[i,:])[0][0])
    return np.array(new_y)
    
def convert_to_overlap_one_hot(itr,n_classes,k,mode = 'lin'):
    one_hot = np.zeros((len(itr),n_classes))
    if mode == 'lin':
        for i,c in enumerate(itr):
            for j in range(0,k):
                one_hot[i,((c+j) % n_classes)] = 1.-float(j)/float(k)
                one_hot[i,((c-j) % n_classes)] = 1.-float(j)/float(k)
    elif isinstance(mode,int):
        for i,c in enumerate(itr):
            for j in range(0,k):
                one_hot[i,((c+j) % n_classes)] = mode
                one_hot[i,((c-j) % n_classes)] = mode
    return one_hot

def normalize_X(X,group_num = 1):
    ###############
    assert (not np.any(np.isnan(X)))
    if group_num>1:
        assert X.shape[1] % group_num == 0
        num_samples = X.shape[0]
        X = X.reshape(X.shape[0],group_num,-1)
        print(X.shape)
        X = X / np.expand_dims(np.linalg.norm(X,axis = 2),2) #normalas
        X = X.reshape(num_samples,X.shape[1]*X.shape[2]) 
    else:
        X = X / np.expand_dims(np.linalg.norm(X,axis = 1),1) #normalas
    X[np.isnan(X)] = 0
    return X

def concat_X(X, y,es):
    tmp_X = np.zeros((X.shape[0],X.shape[1]*es['num_angles']), dtype=np.float32)
    #tmp_mask = np.zeros((X.shape[0],X_.shape[1]*es['num_angles']), dtype=np.float32)
    tmp_y = convert_to_one_hot(y, n_classes=es['num_angles'] )
    for j in range(0,X.shape[0]):
        tmp_X[j,:] = np.kron(tmp_y[j], X[j,:])
    #    tmp_mask[j,:] = np.kron(tmp_y[j], np.ones_like(X[j,:]))
    return tmp_X#, tmp_mask

"""
def discretize(posori,es):
    minx = min(posori[:,0])
    minz = min(posori[:,2])
    maxx = max(posori[:,0])
    maxz = max(posori[:,2])
       
    xs = ((posori[:,0] - minx) / es['step_size']).round().astype(int)
    zs = ((posori[:,2] - minz) / es['step_size']).round().astype(int)

    zs = max(zs)-zs
    return zs,xs
""" 

def test_dicretize():
    exp_name = 'lab2_big_20v_10t'
    _,posori_all,es_all = load_boxworld_raw(exp_name)
    #xs,ys = discretize(posori_all,es_all)
    X_all,y_all,es_all = load_boxworld_data(exp_name)
    disc_posori = np.round(posori_all[:,[0,2]]/es_all['step_size'])
    disc_places = disc_posori[::es_all['num_angles']]
    assert np.array_equal( np.kron(disc_places,np.ones((es_all['num_angles'],1))) , disc_posori)
    discr_idxs = {tuple(r):i for i,r in enumerate(disc_places)}
    disc_posori[::es_all['num_angles']]
    return disc_places,disc_posori,discr_idxs
    
    

    
def ksparse(x,sparsity_k):
    return T.switch(T.ge(x,T.min(T.sort(x,axis=1)[:,-sparsity_k:],axis=1,keepdims=True)),x,0)
    #return T.switch(T.ge(x, T.min(T.sort(x,axis=1)[:,:-sparsity_k-1])), x, 0)
       
def groupksparse(x,sparsity_k,num_groups):
    group_sizes = [(T.shape(x)[-1]/num_groups)]*num_groups
    group_norms_sqr = T.sum(T.sqr(T.split(x, group_sizes, n_splits=num_groups, axis=-1)), axis=-1)
    group_norms_argsort = T.argsort(group_norms_sqr.T, axis=-1)

    groups_to_keep = group_norms_argsort[:,-sparsity_k:]

    group_mask = T.zeros_like(group_norms_argsort)
    group_mask = T.set_subtensor(group_mask[T.arange(group_norms_argsort.shape[0]).reshape((-1, 1)), groups_to_keep], 1)
    x_mask = T.extra_ops.repeat(group_mask, group_sizes, axis=-1)

    return T.switch(x_mask, x, 0)
    
        
def groupklifetimesparse(x,sparsity_perc,num_groups):
    sparsity_k = T.iround((T.shape(x)[0]/num_groups)*sparsity_perc)
    group_sizes = [(T.shape(x)[0]/num_groups)]*num_groups
    groups = T.split(x, group_sizes, n_splits=num_groups, axis=0)
    if num_groups == 1:
        groups = groups.dimshuffle('x',0,1)
    groups_argsort = T.argsort(groups, axis=1)
    
    groups_to_keep = groups_argsort[:,-sparsity_k:,:]
    
    group_mask = T.zeros_like(groups_argsort)
    d1=T.slinalg.kron(T.arange(groups_argsort.shape[0]).reshape((-1,1)), T.ones_like(groups_to_keep[0,:,:]).reshape((-1,1)))
    d2=groups_to_keep.reshape((-1,1))
    d3=T.slinalg.kron(T.ones_like(groups_to_keep[:,:,0]).reshape((-1,1)), T.arange(groups_argsort.shape[2]).reshape((-1,1)))
    group_mask = T.set_subtensor(group_mask[d1, d2, d3], 1)
    
    x_mask = group_mask.reshape(T.shape(x))
    
    return T.switch(x_mask, x, 0)
#T.or_
#T.and_

def wta_np(x):
    x =np.array(x,copy=True)
    mask = np.zeros_like(x)
    mask[np.arange(x.shape[0]),np.argmax(x,axis=1)] = 1
    x[np.logical_not(mask.astype(bool))] = 0
    return x
    
def llsparsity_np(x,sp):
    
    x =np.array(x,copy=True)
    return x * (np.argsort(np.argsort(x,axis=0),axis=0)>x.shape[0]*(1-sp)).astype(int)

def wta(X):
    #return X.argsort(axis=1) - X.argsort(axis=1).max(axis=1,keepdims=True)
    #X2 = X.argsort(axis=1).argsort(axis=1)
    #return (X2-X2.max(axis=1,keepdims=True)>=0)
    return X * ((X - X.max(axis=1,keepdims=True))>=0)
    


def test_wta_layer():
    X = theano.tensor.dmatrix('X')
    wta_f = wta(X)
    f = theano.function([X],wta_f)
    #M = np.random.rand(5,10)
    M = np.zeros((5,10))
    return f(M),M
    
def test_k_sparse():   
    X = theano.tensor.dmatrix('X')
    ksparse_f = ksparse(softplus(X),2)
    f = theano.function([X],ksparse_f)
    M = np.random.rand(5,10)
    return f(M),M
    
def test_concat_mask():
    X_sym = T.matrix()
    y_sym = T.matrix()
    concat_mask_sym=T.extra_ops.repeat(y_sym, 10, axis=1)
    
    conc_f = theano.function([X_sym,y_sym],[concat_mask_sym],on_unused_input='warn')
    return conc_f

#    q=((z-T.slinalg.kron(T.ones_like(T.arange(es['num_angles']).reshape((-1,1))), z[0:(z.shape[0]/es['num_angles']),:]))**2).sum(axis=1).mean()
#    q_d=((z_d-T.slinalg.kron(T.ones_like(T.arange(es['num_angles']).reshape((-1,1))), z_d[0:(z.shape[0]/es['num_angles']),:]))**2).sum(axis=1).mean()

