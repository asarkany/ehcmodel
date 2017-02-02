# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE

from ehcmodel.common.kdl_template import *
from ehcmodel.common.data_preproc import *
from ehcmodel.common.exp_preproc import exp_preprocess, gen_save_path

import numpy as np
import scipy
from sklearn.decomposition import PCA
from skimage.feature import peak_local_max
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
                                               
from pprint import pprint
import itertools
import shutil
import math

import pdb

#load the data. If data can't fit in the GPU memory than use 2 batches
def load_z_lin(X,fullpath):

    try:
        checkpoint_dict = load_checkpoint(fullpath )
    except Exception as e:
        print (fullpath+" couldnt be loaded\n"+str(e))
        raise ValueError
    encode_linear_function = checkpoint_dict["encode_linear_function"]
    try:
        z_lin, = encode_linear_function(X)  
    except MemoryError:
        z_lin = []
        num_rows = X.shape[0]
        num_p = 2
        for pi in range(num_p):
           z_lin.append(encode_linear_function(X[pi*num_rows/num_p:(pi+1)*num_rows/num_p])[0])
        z_lin = np.vstack(z_lin)
        assert z_lin.shape[0] == X.shape[0]
    return z_lin

#create sparse tensor representation from coordinate rperesentation
def to_tensor(table,posori,es):
    discr_pos = np.round(posori[:,0:2]/es['step_size']).astype(int)
    xsize,ysize = np.max(discr_pos,axis=0)+1
    angles = (posori[:,2]/es['turn_angle']).round().astype(int)
    asize = np.max(angles)+1
    
    out_t = np.empty((asize,xsize,ysize,table.shape[1]))
    out_t[:] = np.nan
    for i in range(posori.shape[0]):
        out_t[angles[i],discr_pos[i,0],discr_pos[i,1]] = table[i]

    return out_t 

#create all trajectories in the arena
def generate_vectorized_eval_dataset(activations,es,posori,traj_step_size,traj_length,plot_dir_path):
    posori_t = to_tensor(posori,posori,es)
    x = np.round(posori_t[:,:,:,0]/es['step_size']).astype(int)
    y = np.round(posori_t[:,:,:,1]/es['step_size']).astype(int)
    angle = posori_t[:,:,:,2]
    
    add_step_dim = lambda t:np.repeat(t[:,:,:,np.newaxis],traj_length,-1)
    x,y,angle = map(add_step_dim,[x,y,angle])
    #angle = a*es['turn_angle']
    s = np.zeros_like(x)
    s[:,:,:] = np.arange(traj_length)
    
    px = x*es['step_size']
    py = y*es['step_size']
    px1 = px+np.cos(np.radians(-angle+90))*np.where((s-1)<0,0,(s-1))*traj_step_size
    py1 = py+np.sin(np.radians(-angle+90))*np.where((s-1)<0,0,(s-1))*traj_step_size
    px2 = px+np.cos(np.radians(-angle+90))*s*traj_step_size
    py2 = py+np.sin(np.radians(-angle+90))*s*traj_step_size
    #x1 = np.round(px1/es['step_size']).astype(int)
    #y1 = np.round(py1/es['step_size']).astype(int)
    x2 = np.round(px2/es['step_size']).astype(int)
    y2 = np.round(py2/es['step_size']).astype(int)
    
    valid_mask = np.logical_and(np.logical_and(np.min(posori_t[:,:,:,0])<= px2,px2 <= np.max(posori_t[:,:,:,0])),
                           np.logical_and(np.min(posori_t[:,:,:,1])<= py2,py2 <= np.max(posori_t[:,:,:,1])))

    invalid_seqs = np.logical_not(np.all(valid_mask,axis=3))
    valid_mask2 = np.array(valid_mask,copy=True)
    valid_mask2[invalid_seqs] = False
    assert np.all(valid_mask2[np.logical_not(invalid_seqs)])
    assert not np.any(valid_mask2[invalid_seqs])

    assert np.all(valid_mask2<=valid_mask) and not np.array_equal(valid_mask,valid_mask2)
    valid_mask = valid_mask2
    assert np.array_equal(np.all(valid_mask,axis=-1),np.any(valid_mask,axis=-1))   

    #activations_t = to_tensor(activations,posori,es)
    activations_traj_t = np.empty((activations.shape[:3]+(traj_length,activations.shape[3])))
    activations_traj_t[:] = np.nan
    
    for ai,xi,yi,si in zip(*np.nonzero(valid_mask)):
            activations_traj_t[ai,xi,yi,si]= activations[ai,x2[ai,xi,yi,si],y2[ai,xi,yi,si]]
    
    #print('Plotting')
    plt.figure()
    ax = plt.axes()
    for ai,xi,yi,si in zip(*np.nonzero(valid_mask)):
        idx_tpl = (ai,xi,yi,si)
        if xi % 5 ==0 and yi % 5 == 0 and ai in np.arange(es['num_angles'])[::6]:
            ax.arrow(px1[idx_tpl], py1[idx_tpl], px2[idx_tpl]-px1[idx_tpl], py2[idx_tpl]-py1[idx_tpl], 
                 head_width=0.05, head_length=0.1, fc='k', ec='k')
    
    plt.xlim(np.min(px),np.max(px))
    plt.ylim(np.min(py),np.max(py))
                
    #plt.show()
    if not os.path.isdir(os.path.join(plot_dir_path,'etc')):
        os.makedirs(os.path.join(plot_dir_path,'etc'))
    plt.savefig(os.path.join(plot_dir_path,'etc','some_eval_paths2.png'))
    plt.close()
    
    valid_mask = np.all(valid_mask,axis=-1)
    assert np.array_equal(np.logical_not(np.all(np.isnan(activations_traj_t),axis=(-1,-2))),valid_mask)
    return activations_traj_t,valid_mask

#plot a sample
def plot_ds_sample(dss,k,plot_dir_path,dir_name,random_state=np.random.RandomState(1999),save_npy=False):
    assert np.all(np.array([ds.shape[:3] for ds in dss])==dss[0].shape[:3])
    assert np.all(np.array([np.array_equal(np.all(np.isnan(ds),axis=(-1,-2)),np.all(np.isnan(dss[0]),axis=(-1,-2))) for ds in dss]))
    print('place cell activations '+dir_name)
    if os.path.isdir(os.path.join(plot_dir_path,'placecellseq',dir_name)):
        shutil.rmtree(os.path.join(plot_dir_path,'placecellseq',dir_name))
    os.makedirs(os.path.join(plot_dir_path,'placecellseq',dir_name))
    randf = lambda d:random_state.randint(d,size=k)

    
    #for ai,xi,yi in zip(*map(randf,dss[0].shape[:3])):
    idx_tuples = zip(*np.nonzero(np.logical_not(np.all(np.isnan(dss[0]),axis=(-1,-2)))))
    for ai,xi,yi in np.array(idx_tuples)[random_state.choice(len(idx_tuples),k,replace=False)]:
        fig = plt.figure(figsize=(3, 6))
        for j,ds in enumerate(dss):
            ax = fig.add_subplot(len(dss),1,j+1)
            #ds = ds.reshape(xsize,ysize,len(np.unique(angles)),-1)
            #ax.imshow(ds[x,y,a].reshape(-1,act_dim).T,interpolation = 'none',cmap=plt.get_cmap('hot_r'))
            im = ax.imshow(ds[ai,xi,yi].T,interpolation = 'none',cmap=plt.get_cmap('hot_r'))
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.01)

            #plt.colorbar(im, cax=cax)
            plt.colorbar(im)
            if j== 1:
                plt.yticks([0,5])
            ax.set_aspect(2)
        #plt.savefig(plot_dir_path+'/placecellseq/'+'_'.join([str(x),str(y),str(a)]))
        plt.savefig(os.path.join(plot_dir_path,'placecellseq',dir_name,'_'.join(map(str,[ai,xi,yi]))))
        if save_npy:
            np.save(os.path.join(plot_dir_path,'placecellseq',dir_name,'_'.join(map(str,[ai,xi,yi]))+'.npy'),
            [ds[ai,xi,yi].T for ds in dss])
        plt.close()

def convert_to_imshowxy(m):
    return np.flipud(m.T)

#plot trajectories
def plot_evaluation(act_traj,num_subplots,es,plot_dir_path,dir_name,angles = [],highlight = False,shift = False,traj_distance = None,add_grid = False,save_npy=False):
    #act_traj dims
    #0: different angles
    #1: subplots (e.g. prediction from different starting points)
    #2,3: image in 2d
    # 4: different components
    print('plotting '+dir_name)
    if shift and traj_distance == None:
        raise ValueError('Shifted plotting is selected but no traj_distance is provided')
    if angles ==[]:
        angles = np.arange(es['num_angles'])*es['turn_angle']
    else:
        if len(angles) != act_traj.shape[0]:
            raise ValueError('len(angles) is not the same as act_traj.shape[0]: '+str((len(angles),act_traj.shape[0])))
    act_dim = act_traj.shape[-1]
    
    if not os.path.isdir(os.path.join(plot_dir_path,dir_name)):
        os.makedirs(os.path.join(plot_dir_path,dir_name))

    #pdb.set_trace()
    #clim = (np.nanmean(act_traj,axis=(0,1,2,3))-np.nanstd(act_traj,axis=(0,1,2,3))*2,np.nanmean(act_traj,axis=(0,1,2,3))+np.nanstd(act_traj,axis=(0,1,2,3))*2)
    
    clim = (np.nanmean(act_traj,axis=(0,1,2,3))-0.05,np.nanmean(act_traj,axis=(0,1,2,3))+0.05)
    clim_kwars = [{'clim':(clim[0][di],clim[1][di])} if highlight else {} for di in range(act_dim)]
    
    #clim_kwars = [{'clim':(0,2.07474988699)} for di in range(act_dim)]
    clim_kwars = [{'clim':(np.nanmin(act_traj),np.nanmax(act_traj))} for di in range(act_dim)]
    print(clim_kwars)
    for ai,angle in enumerate(angles):
        #print(angle)
        for plot_group_id in range(int(math.ceil(act_traj.shape[1]/float(num_subplots)))):
            
            if shift:
                x_shift = np.round(np.cos(np.radians(-angle+90))*(traj_distance/2)/es['step_size']).astype(int)
                y_shift = np.round(np.sin(np.radians(-angle+90))*(traj_distance/2)/es['step_size']).astype(int)
                shift_f = lambda m: np.roll(np.roll(m,x_shift,axis=0),y_shift,axis=1)
            else:
                shift_f = lambda m: m
                
            for di in range(act_dim):
                fig = plt.figure()
                for spi in range(plot_group_id*num_subplots,
                                min(act_dim,(plot_group_id+1)*num_subplots)):  
                    #print(ai,plot_group_id,spi,di)
                    ax = fig.add_subplot(math.ceil(math.sqrt(num_subplots)),math.ceil(math.sqrt(num_subplots)),spi-plot_group_id*num_subplots+1)
                    plt.axis('off')
                    
                    im = ax.imshow(convert_to_imshowxy(shift_f(act_traj[ai,spi,:,:,di])),interpolation='none',**clim_kwars[di])
                    #plt.colorbar(im)
                   
                    if add_grid:
                        if dir_name == "spin_pred":
                            pdb.set_trace()
                        coordinates = peak_local_max(convert_to_imshowxy(act_traj[ai,spi,:,:,di]).T, min_distance=4,exclude_border=False,
                                    labels = np.logical_not(np.isnan(np.flipud(act_traj[ai,spi,:,:,di].T).T)).astype(int)  )
                        if len(coordinates)>0:
                            x,y = zip(*coordinates)
                            #pdb.set_trace()
                            try:
                                tri = Delaunay(coordinates)
                                plt.triplot(x, y, tri.simplices.copy(),'ko-')
                            except QhullError:
                                plt.plot(x,y,'ko')
                    
                    #plt.show()
                    #pdb.set_trace()
                
                #plt.show()
                plt.savefig(os.path.join(plot_dir_path,dir_name,str(di)+'_'+str(angle).zfill(3)+'_'+str(plot_group_id+1)+'.png'))
                if save_npy:
                    np.save(os.path.join(plot_dir_path,dir_name,str(di)+'_'+str(angle).zfill(3)+'_'+str(plot_group_id+1)+'.npy'),
                    convert_to_imshowxy(act_traj[ai,spi,:,:,di]))
                plt.close()
    
    for plot_group_id in range(int(math.ceil(act_traj.shape[1]/float(num_subplots)))):
        for di in range(act_dim):    
            fig = plt.figure()
            for spi in range(plot_group_id*num_subplots,
                            min(act_traj.shape[1],(plot_group_id+1)*num_subplots)): 
                #print(spi)
                ax = fig.add_subplot(math.ceil(math.sqrt(num_subplots)),math.ceil(math.sqrt(num_subplots)),spi-plot_group_id*num_subplots+1)
                plt.axis('off')
                im = ax.imshow(convert_to_imshowxy(np.nanmean(act_traj[:,spi,:,:,di],axis=0)),interpolation='none',**clim_kwars[di])
                plt.colorbar(im)

            plt.savefig(os.path.join(plot_dir_path,dir_name,'mean'+'_'+str(di)+'_'+str(plot_group_id+1)+'.png'))
            if save_npy:
                    np.save(os.path.join(plot_dir_path,dir_name,'mean'+'_'+str(di)+'_'+str(plot_group_id+1)+'.npy'),
                    convert_to_imshowxy(np.nanmean(act_traj[:,spi,:,:,di],axis=0)))
            plt.close()
            
    for plot_group_id in range(int(math.ceil(act_traj.shape[1]/float(num_subplots)))):
        for di in range(act_dim):    
            fig = plt.figure()
            for spi in range(plot_group_id*num_subplots,
                            min(act_traj.shape[1],(plot_group_id+1)*num_subplots)): 
                #print(spi)
                ax = fig.add_subplot(math.ceil(math.sqrt(num_subplots)),math.ceil(math.sqrt(num_subplots)),spi-plot_group_id*num_subplots+1)
                plt.axis('off')
                im = ax.imshow(convert_to_imshowxy(np.nanstd(act_traj[:,spi,:,:,di],axis=0)),interpolation='none')
                plt.colorbar(im)

            plt.savefig(os.path.join(plot_dir_path,dir_name,'std'+'_'+str(di)+'_'+str(plot_group_id+1)+'.png'))
            plt.close()

#build input-output mapping that can be used to train prediction
def build_prediction_db(act_traj_ds,history_length,pred_length):
    traj_length = act_traj_ds.shape[3]
    act_dim = act_traj_ds.shape[4]
    
    X = np.empty(act_traj_ds.shape[:3]+(history_length,act_dim))
    X[:] = np.nan
    Y = np.empty(act_traj_ds.shape[:3]+(pred_length,act_dim))
    Y[:] = np.nan
    
    traj_length = act_traj_ds.shape[3]
    for i in zip(*np.nonzero(np.logical_not(np.all(np.isnan(act_traj_ds),axis=(-1,-2))))): 
        assert len(i) ==3 # i is tuple of (ai,xi,yi)
        #pdb.set_trace()
        for j in range(history_length,traj_length-pred_length+1):
            X[i] = act_traj_ds[i][(j-history_length):j]
            Y[i] = act_traj_ds[i][j:j+pred_length] 
   
    return X,Y

#Predict with PLS on its own output
def pls_predict(act_traj_ds,pls,history_length,pred_length,spin_pred_length,keep_orig_data=True):
    act_dim = len(pls.x_mean_)/history_length
    
    pred_act_traj_ds = np.empty(act_traj_ds.shape[:3]+(spin_pred_length,act_dim))
    pred_act_traj_ds[:] = np.nan
    latent_act_traj_ds = np.empty(act_traj_ds.shape[:3]+(spin_pred_length,pls.n_components))
    latent_act_traj_ds[:] = np.nan
    
    pred_act_traj_ds[:,:,:,:history_length] = act_traj_ds[:,:,:,:history_length]
    
    
    np.testing.assert_equal(np.all(np.isnan(act_traj_ds),axis=(3,4)),np.any(np.isnan(act_traj_ds),axis=(3,4)))
    idx_tuples = set(zip(*np.nonzero(np.logical_not(np.all(np.isnan(act_traj_ds),axis=(3,4))))[:3]))
    print(len(idx_tuples))
    for idx_tuple in sorted(list(idx_tuples)):
        latent_act_traj_ds[idx_tuple][:history_length] = 0
        for i in range(history_length,spin_pred_length):
            pred_act_traj_ds[idx_tuple][i] = pls.predict(pred_act_traj_ds[idx_tuple][(i-history_length):i].reshape(1,-1)).reshape(-1)[:act_dim]
            
            latent_act_traj_ds[idx_tuple][i] = pls.transform(pred_act_traj_ds[idx_tuple][(i-history_length):i].reshape(1,-1))
    
    np.testing.assert_equal(np.all(np.isnan(pred_act_traj_ds),axis=(3,4)),np.any(np.isnan(pred_act_traj_ds),axis=(3,4)))
    np.testing.assert_equal(np.all(np.isnan(latent_act_traj_ds),axis=(3,4)),np.any(np.isnan(latent_act_traj_ds),axis=(3,4)))
    
    if not keep_orig_data:
        pred_act_traj_ds[:,:,:,:history_length] = np.nan
        latent_act_traj_ds[:,:,:,:history_length] = np.nan
    
    return pred_act_traj_ds,latent_act_traj_ds

#Predict with pseudoinverse on its own output
def max_pinv_predict(act_traj_ds,pinv_m,history_length,pred_length,spin_pred_length,max_pinv_comp_i,keep_orig_data=True):
    assert pred_length == 1
    act_dim = act_traj_ds.shape[-1]
    
    pred_act_traj_ds = np.empty(act_traj_ds.shape[:3]+(spin_pred_length,act_dim))
    pred_act_traj_ds[:] = np.nan
       
    pred_act_traj_ds[:,:,:,:history_length] = act_traj_ds[:,:,:,:history_length]
    
    
    np.testing.assert_equal(np.all(np.isnan(act_traj_ds),axis=(3,4)),np.any(np.isnan(act_traj_ds),axis=(3,4)))
    idx_tuples = set(zip(*np.nonzero(np.logical_not(np.all(np.isnan(act_traj_ds),axis=(3,4))))[:3]))
    print(len(idx_tuples))
    for idx_tuple in sorted(list(idx_tuples)):
        for i in range(history_length,spin_pred_length):
            #pdb.set_trace()
            pred_act_traj_ds[idx_tuple][i][0] =  np.dot(pred_act_traj_ds[idx_tuple][(i-history_length):i,max_pinv_comp_i].T,pinv_m)
            pred_act_traj_ds[idx_tuple][i][1:] = 0
               
    np.testing.assert_equal(np.all(np.isnan(pred_act_traj_ds),axis=(3,4)),np.any(np.isnan(pred_act_traj_ds),axis=(3,4)))
    
    if not keep_orig_data:
        pred_act_traj_ds[:,:,:,:history_length] = np.nan
    
    return pred_act_traj_ds,np.array(pred_act_traj_ds,copy=True)[:,:,:,:,0][:,:,:,:,np.newaxis]

#Create image-like representation from data
def act_traj_to_2d(act_traj,posori_t,es,traj_step_size,history_length):
    #if angles ==[]:
    #    angles = np.arange(es['num_angles'])*es['turn_angle']
    #else:
    #    if len(angles) != act_traj.shape[0]:
    #        raise ValueError('len(angles) is not the same as act_traj.shape[0]: '+str((len(angles),act_traj.shape[0])))       
    traj_length = act_traj.shape[-2]
    act_dim = act_traj.shape[-1]
    x = np.round(posori_t[:,:,:,0]/es['step_size']).astype(int)
    y = np.round(posori_t[:,:,:,1]/es['step_size']).astype(int)
    angle = posori_t[:,:,:,2]
    
    add_step_dim = lambda t:np.repeat(t[:,:,:,np.newaxis],traj_length,-1)
    x,y,angle = map(add_step_dim,[x,y,angle])
    #angle = a*es['turn_angle']
    s = np.zeros_like(x)
    s[:,:,:] = np.arange(traj_length)
    
    px = x*es['step_size']
    py = y*es['step_size']
    px2 = px+np.cos(np.radians(-angle+90))*s*traj_step_size
    py2 = py+np.sin(np.radians(-angle+90))*s*traj_step_size
    x2 = np.round(px2/es['step_size']).astype(int)
    y2 = np.round(py2/es['step_size']).astype(int)
    
    minx2 = np.min(x2)
    miny2 = np.min(y2)
    maxx2 = np.max(x2)
    maxy2 = np.max(y2)
    x2 = x2-minx2
    y2 = y2 -miny2
    
    traj_coords = np.stack([x2,y2],axis=4)
    #np.testing.assert_equal(np.all(np.isnan(act_traj[:,:,:,:history_length]),axis=(3,)),
    #                        np.any(np.isnan(act_traj[:,:,:,:history_length]),axis=(3,)))
    
    act_traj_2d = np.empty(act_traj.shape[:3]+(np.max(x2)+1,np.max(y2)+1,act_dim))
    act_traj_2d[:] = np.nan
       
    idx_tuples = set(zip(*np.nonzero(np.logical_not(np.all(np.isnan(act_traj),axis=(3,))))[:3]))
        
    for idx_tuple in sorted(list(idx_tuples)):
        #print(idx_tuple)
        act_traj_2d[idx_tuple][zip(*traj_coords[idx_tuple])] = act_traj[idx_tuple]
        
    return act_traj_2d,(minx2,miny2),(maxx2,maxy2)

#build plots together 
def build_2d_plot_tensors(act_traj_ds,history_length,angles,seeds,posori_all_s_t,es_all_s,traj_step_size):
    xsize = act_traj_ds.shape[2]
    ysize = act_traj_ds.shape[3]
    act_dim = act_traj_ds.shape[-1]
    act_traj_ds_2d_m = {}    
    mins = np.empty((len(angles),len(seeds),2))
    mins[:] = np.nan
    maxs = np.empty((len(angles),len(seeds),2))
    maxs[:] = np.nan
    for ai,angle in enumerate(angles):
        posori_a = posori_all_s_t[angle][np.newaxis]
        for si in range(len(seeds)):
            act_traj_ds_2d,mins_,maxs_ = act_traj_to_2d(act_traj_ds[ai,si][np.newaxis],posori_a,es_all_s,traj_step_size,history_length)
            assert act_traj_ds_2d.shape[:3] == (1,xsize,ysize) and act_traj_ds_2d.shape[5] == act_dim

            act_traj_ds_2d_m_ = np.nanmean(np.nanmean(act_traj_ds_2d,axis=1),axis=1)[0]
            assert len(act_traj_ds_2d_m_.shape) == 3 and act_traj_ds_2d_m_.shape[2] == act_dim
            
            act_traj_ds_2d_m[ai,si] = act_traj_ds_2d_m_
            mins[ai,si] = mins_
            maxs[ai,si] = maxs_
    
    allmax = np.max(maxs,axis=(0,1))
    allmin = np.min(mins,axis=(0,1))  
    size = allmax-allmin
    
    act_traj_ds_2d_m2 = np.empty((len(angles), len(seeds),size[0]+1,size[1]+1,act_dim))
    act_traj_ds_2d_m2[:] = np.nan
    #pdb.set_trace()
    for ai,si in act_traj_ds_2d_m.keys():
        act_traj_ds_2d_m_ = act_traj_ds_2d_m[ai,si]
        mins_ = mins[ai,si]
        maxs_ = maxs[ai,si]
        act_traj_ds_2d_m2_ = np.empty((size[0]+1,size[1]+1,act_dim))
        act_traj_ds_2d_m2_[:] = np.nan
               
        i,j = act_traj_ds_2d_m_.shape[:2]
        #pdb.set_trace()
        act_traj_ds_2d_m2_[mins_[0]-allmin[0]:maxs_[0]-allmin[0]+1,
                                mins_[1]-allmin[1]:maxs_[1]-allmin[1]+1] = act_traj_ds_2d_m_
        act_traj_ds_2d_m2[ai,si] = act_traj_ds_2d_m2_
    #pdb.set_trace()

    assert act_traj_ds_2d_m2.shape == (len(angles),len(seeds),size[0]+1,size[1]+1,act_dim)
    return act_traj_ds_2d_m2    
    

