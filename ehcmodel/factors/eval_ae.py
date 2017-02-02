# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE

from ehcmodel.common.kdl_template import *
from ehcmodel.common.data_preproc import *
from ehcmodel.common.exp_preproc import exp_preprocess, gen_save_path

import os
import inspect

from collections import defaultdict
import itertools
import pickle
import time
from pprint import pprint
from sklearn.decomposition import FastICA


exp_name = 'lab2_big_20v_10t'
_,posori_all,es_all = load_boxworld_raw(exp_name)
X_all,y_all,es_all = load_boxworld_data(exp_name);
X_all = X_all.astype("float32").astype(theano.config.floatX)
y_all = y_all.astype("int32")

cmap_name = 'hot_r'
#generate each experiment parameters and run eval_grid_point
def eval_on_grid(eval_funcs,**kwargs):
    sub_exp_num = len(glob.glob(plot_root+kwargs['dir_name']+'/eval_results*.pkl'))

    changing_param_names = [k for k in kwargs.keys() if isinstance(kwargs[k],list) and len(kwargs[k])>1]
    not_changing_param_names = [k for k in kwargs.keys() if not (isinstance(kwargs[k],list) and len(kwargs[k])>1)]
    assert 'dir_name' in not_changing_param_names
    
    grid_comps = []
    for k,v in kwargs.iteritems():
        if isinstance(v,list):
            grid_comps.append(v)
        else:
            grid_comps.append([v])
    print(grid_comps)
    not_changing_params = dict(kwargs)
    for k in changing_param_names:
        del not_changing_params[k]
    
    results =defaultdict(dict)  
  
    for grid_vertex in itertools.product(*grid_comps):
        params = dict(zip(kwargs.keys(),grid_vertex))
        changing_params = dict(params)
        for k in not_changing_param_names:
            del changing_params[k]
        starttime = time.time()
        values = eval_grid_point(eval_funcs,**params)
        results['eval_time'][tuple(changing_params.values())] = (time.time()-starttime)
        for key in values.keys():
            results[key][tuple(changing_params.values())] = values[key]
        pprint(dict(results))
    if not os.path.isdir(plot_root+kwargs['dir_name']):
        os.makedirs(plot_root+kwargs['dir_name'])
    pprint((not_changing_params,changing_params.keys(),dict(results)),open(plot_root+kwargs['dir_name']+'/eval_results'+str(sub_exp_num)+'.txt','w'))

    pickle.dump((not_changing_params,changing_params.keys(),dict(results)),open(plot_root+kwargs['dir_name']+'/eval_results'+str(sub_exp_num)+'.pkl','w'))
    return results
        
def eval_grid_point(eval_methods,dir_name,description,n_code,n_coding_layer,sparsity_mode,sparsity_k,dropout_p,preprocess,
                    sampling_rate,minibatch_size,view_angle,turn_angle,reconst_mask,early_stop,n_epochs):
    
    frame = inspect.currentframe()
    _, _, _, values = inspect.getargvalues(frame)
    
    del values['frame']
    del values['eval_methods']
    random_state = np.random.RandomState(1999)
    use_cache = 'use_cache' in eval_methods
    plot_types = ['z','z_lin','z_lin_wta','bin_lin_wta','z_lin_thres','bin_lin_thres','z_lin_ll1','z_lin_ll1_bin']
    
    Z_PLOT_TYPE = 0
    Z_LIN_PLOT_TYPE = 1
    Z_LIN_WTA_PLOT_TYPE = 2
    FIRE_LIN_WTA_PLOT_TYPE = 3
    Z_LIN_THRES_PLOT_TYPE = 4
    
    if use_cache:
        save_path = gen_save_path(values)
        values = {}
    else:
        save_path, X,Y,posori,_,_,_,es  = \
                                exp_preprocess(X_all = X_all,y_all=y_all,posori_all=posori_all,es_all=es_all,
                                               split=False,random_state=random_state,**values)                               
        checkpoint_dict = load_checkpoint(dir_name+'/'+save_path+".pkl" )
        try:
            checkpoint_dict = load_checkpoint(dir_name+'/'+save_path+".pkl" )
        except Exception as e:
            print (dir_name+'/'+save_path+".pkl couldnt be loaded\n"+str(e))
            return {}
        if n_coding_layer== 'ica':
            fica = checkpoint_dict["fica_object"]
            z_lin_all = fica.transform(X)
            z_all = fica.transform(X)
        else:
            test_function = checkpoint_dict["test_function"]
            encode_function = checkpoint_dict["encode_function"]
            encode_linear_function = checkpoint_dict["encode_linear_function"]
            #decode_function = checkpoint_dict["decode_function"]
            #predict_function = checkpoint_dict["predict_function"]
          
            z_lin_all, = encode_linear_function(X)
            z_all, = encode_function(X)
     
        
        values = {}
        dead_neuron_map = np.sum(z_all>0,axis=0)<=0
        dead_neuron_count = np.sum(dead_neuron_map)
        
        values['dead_neuron_count'] = dead_neuron_count
        values['num_samples'] = X.shape[0]
        
        #xs,ys = discretize(posori,es)
        #xs = xs / sampling_rate
        #ys = ys / sampling_rate
        
        discr_posori = np.round(posori[:,0:2]/es['step_size']).astype(int)
        xsize,ysize = np.max(discr_posori,axis=0)+1
        
        angles = posori[:,2].round().astype(int)
    
    
    
            
        fit_angles = (np.arange(es['num_angles'])*360/es['num_angles'])#[::36]#[0]
        act_2d = np.zeros([len(plot_types),len(fit_angles),z_all.shape[1],xsize,ysize])
        for ai,angle in enumerate(fit_angles):
            angle_mask = angles == angle
            discr_places = discr_posori[angle_mask]
            discr_place_inv_idx = {tuple(r):i for i,r in enumerate(discr_places)}
            X_angle = X[angle_mask]
            
            
            for pi,plot_type in enumerate(plot_types):
                if plot_type in ['z'] :
                    #z, = encode_function(X_angle)
                    z = z_all[angle_mask]
                elif plot_type == 'z_lin':
                    #z, = encode_linear_function(X_angle)
                    z = z_lin_all[angle_mask]
                elif plot_type in ['z_lin_wta'] :
                    z = wta_np(z_lin_all)[angle_mask]
                elif plot_type in ['bin_lin_wta']:
                    #z = (z_all[angle_mask]>0).astype(int)
                    z = (wta_np(z_lin_all)[angle_mask]>0).astype(int)
                elif plot_type == 'z_lin_thres':
                    z = z_lin_all[angle_mask]
                    z_max = np.max(z,axis=0)
                    mask =np.kron(z_max*0.8,np.ones((z.shape[0],1)))
                    z[z<mask] = 0
                elif plot_type == 'bin_lin_thres':
                    z = z_lin_all[angle_mask]
                    z_max = np.max(z,axis=0)
                    mask =np.kron(z_max*0.5,np.ones((z.shape[0],1)))
                    z[z<mask] = 0
                    z = (z>0).astype(int)
                elif plot_type == 'z_lin_ll1':
                    z = llsparsity_np(z_lin_all,1./z.shape[1])[angle_mask]
                elif plot_type == 'z_lin_ll1_bin':
                    z = (llsparsity_np(z_lin_all,1./z.shape[1])[angle_mask]>0).astype(int)
                    
                    
                #pprint(discr_place_inv_idx)                                     
                for x1 in range(0,xsize):
                    for y1 in range(0,ysize):
                        #print(x1,y1)
                        if not ((x1,y1) in discr_place_inv_idx):
                            print(str((x1,y1))+' is blocked')
                            continue
                        zxy = z[discr_place_inv_idx[(x1,y1)]]
                        act_2d[pi,ai,:,x1,y1] = zxy
        #pickle.dump([z_all,z_lin_all,yoh],open('zzzz0301.pkl','wb'))
        mins = np.min(z_lin_all,axis=0).tolist()
        maxs = np.max(z_lin_all,axis=0).tolist()
        minmax = ({'mins':mins,'maxs':maxs})
        lims = np.max(np.array([minmax['mins'],minmax['maxs']]),axis=0)
        #print(lims)

    if 'activations' in eval_methods:
         
        dir_path = plot_root+dir_name+'/'+save_path+"/activations/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        
        for ai,angle in enumerate(fit_angles):
            print(angle)
            for pi,plot_type in enumerate(plot_types[:4]):
                if not os.path.isdir(dir_path+plot_type+"/"):
                    os.makedirs(dir_path+plot_type+"/")
                fig = plt.figure()
                subplotdata = []
                for di in range(z_all.shape[1]):    
                    #print(di)
                    ax = fig.add_subplot(math.ceil(math.sqrt(z_all.shape[1])),math.ceil(math.sqrt(z_all.shape[1])),di+1)
                    plt.axis('off')
                    im = ax.imshow(act_2d[pi,ai,di],interpolation='none')
                    subplotdata.append(act_2d[pi,ai,di])
                    """
                    if plot_type == 'fire_lin_wta':
                        im = ax.imshow(act_2d[pi,ai,di],interpolation='none',clim=(-1,1))
                    elif lims[di]>0:
                        im = ax.imshow(act_2d[pi,ai,di],interpolation='none',clim=(-lims[di],lims[di]))
                    else:
                        im = ax.imshow(act_2d[pi,ai,di],interpolation='none',clim=(-1,1))
                    """
                    plt.colorbar(im)
                
                
                plt.savefig(dir_path+plot_type+"/"+plot_type+"_"+str(angle).zfill(3)+'.png')
                pickle.dump(subplotdata,open(dir_path+plot_type+"/z"+plot_type+"_"+str(angle).zfill(3)+'.pkl','w'))    
                #plt.show()
                plt.close()
                #print(act_2d)
    
    if 'summed_activations' in eval_methods:
        print('Plotting summed activations')
        dir_path = plot_root +dir_name+ "/all/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
                       
        for pi,plot_type in enumerate(plot_types):#zip([3],['sum']):#zip([0,1],['sum','firesum']):
            if not os.path.isdir(dir_path+plot_type+"/"):
                os.makedirs(dir_path+plot_type+"/")
            fig = plt.figure()
            if use_cache:
                cached_d2plot = pickle.load(open(dir_path+plot_type+"/zsum_act_"+save_path+'.pkl','r'))
            subplotdata = []
            for di in range(n_code):
                ax = fig.add_subplot(math.ceil(math.sqrt(n_code)),math.ceil(math.sqrt(n_code)),di+1)
                plt.axis('off')
                if use_cache:
                    d2plot = cached_d2plot[di]
                else:
                    d2plot = np.sum(act_2d[pi,:,di],axis=0)
                    
                if np.max(d2plot)>0:
                    im = ax.imshow(d2plot,interpolation='none',clim=(0,np.max(d2plot)),cmap=plt.get_cmap(cmap_name))
                else:
                    im = ax.imshow(d2plot,interpolation='none',cmap=plt.get_cmap(cmap_name))
                subplotdata.append(d2plot)
                plt.colorbar(im)
           
            plt.savefig(dir_path+plot_type+"/sum_act_"+save_path+'.png')
            if not use_cache:
                pickle.dump(subplotdata,open(dir_path+plot_type+"/zsum_act_"+save_path+'.pkl','w'))
            plt.close()

    if 'spatial_coverage_plot' in eval_methods:
        print('Plotting spatial coverage')
        dir_path = plot_root +dir_name+"/all/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        for pi,plot_type in enumerate(plot_types):#zip([0,1],['sum','firesum']):
            if not os.path.isdir(dir_path+plot_type+"/"):
                os.makedirs(dir_path+plot_type+"/")
            fig = plt.figure()
            plt.axis('off')
            if use_cache:
                d2plot = pickle.load(open(dir_path+plot_type+"/zcov_"+save_path+'.pkl','r'))
            else:
                if plot_type == 'bin_lin_wta':
                    d2plot = np.max(np.sum(act_2d[pi,:],axis=0),axis=0) 
                else:
                    d2plot = np.sum(np.sum(act_2d[pi,:],axis=0),axis=0) 
            #im = plt.imshow(d2plot,interpolation='none',clim=(0,np.max(d2plot)))
            if np.max(d2plot)>0:
                im = plt.imshow(d2plot,interpolation='none',clim=(0,np.max(d2plot)),cmap=plt.get_cmap(cmap_name))
            else:
                im = plt.imshow(d2plot,interpolation='none',cmap=plt.get_cmap(cmap_name))
            plt.colorbar(im)
            plt.savefig(dir_path+plot_type+"/cov_"+save_path+".png")
            if not use_cache:
                pickle.dump(d2plot,open(dir_path+plot_type+"/zcov_"+save_path+'.pkl','w'))     
            plt.close()
                       
            
    if 'epoch_metrics' in eval_methods:
        dir_path = plot_root+dir_name+'/'+save_path+"/epoch_metrics/"       
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            
        train_epoch_results = checkpoint_dict["train_epoch_results"]
        val_epoch_results = defaultdict(list,checkpoint_dict["val_epoch_results"])
        val_epoch_results['cost']= val_epoch_results['val_cost']

        #print (train_epoch_results.keys())
        #print (val_epoch_results.keys())
        values['last_val_cost'] = val_epoch_results['cost'][-1]
        values['last_train_cost'] = train_epoch_results['cost'][-1]
        values['num_epochs'] = len(train_epoch_results['cost'])

        for k in train_epoch_results:
            plt.figure()
            plt.plot(train_epoch_results[k],label = 'train')
            
            if k in val_epoch_results:
                assert len(val_epoch_results[k])-1 % len(train_epoch_results[k])
                plt.plot(range(0,len(train_epoch_results[k]),len(train_epoch_results[k])/(len(val_epoch_results[k])-1)),
                                val_epoch_results[k][1:],label = 'val')
            plt.title(str(k))
            plt.legend()
            plt.savefig(dir_path+str(k)+'.png')  
            plt.close()
            

    return values
    
    
    

