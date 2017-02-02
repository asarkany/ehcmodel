# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE
from ehcmodel.common.kdl_template import *
from ehcmodel.common.data_preproc import *
from ehcmodel.common.exp_preproc import exp_preprocess, gen_save_path
from methods import to_tensor,load_z_lin,generate_vectorized_eval_dataset, \
                            plot_ds_sample, plot_evaluation, build_prediction_db,pls_predict, max_pinv_predict, build_2d_plot_tensors

from pprint import pprint
import itertools
import shutil
import math
import csv
import pdb
import pickle

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

exp_name = 'lab2_big_20v_10t'
_,posori_all,es_all = load_boxworld_raw(exp_name)
X_all,y_all,es_all = load_boxworld_data(exp_name);
X_all = X_all.astype("float32").astype(theano.config.floatX)
y_all = y_all.astype("int32")

random_state = np.random.RandomState(1999)

values = {
    'dir_name' : 'demo',
    'description' : '',
    'n_code' : 30,
    'sparsity_k' : 1./15,#[1./60,1./15],  
    'n_coding_layer' : 1,
    'sampling_rate' : 4,
    'sparsity_mode' : ('lifelong','all'),
    'dropout_p' : None,
    'preprocess' : (('retina',20),'normalize'), # retina: 20,60,100,140,180,220,260,300,340 
    'view_angle' : 220,
    'turn_angle' : 20,
    'minibatch_size': 'fullbatch',
    'reconst_mask': True,
    'early_stop': True,
    'n_epochs' : 100,
    }


train_alg = "pls" #"pls" "max_pinv" # whether to use pseudoinverse on a single component or PLS regression
train_mode = "angle" #"all" "angle" #train with all data, or train each angle separately

do_spinning = True #perform prediction outside the arena
plot_samples = True #plot some sample data

do_plot_normalize = True #normalize each 'paralelgram' on the plots
do_highlight = False
do_add_grid = False
do_save_plots_npy = True
do_save_npy = True

max_pinv_comp_i = 0 #if pseudoinverse is used, which component should be included in the magnitude ordered vector (0 means that the largest)

ROWWISE = 0
COLWISE = 1
norm_mode = None

sampling_rate = 1
spin_pred_length = 100*4/sampling_rate #how many step should be predicted outside the arena
bottleneck_size = 1 # size of the inner PLS representation



save_path, X_all_s,y_all_s,posori_all_s,_,_,_,es_all_s  = \
                                exp_preprocess(X_all = X_all,y_all=y_all,posori_all=posori_all,es_all=es_all,
                                               split=False,random_state=random_state,**values)
values['sampling_rate'] = sampling_rate                                  
_, X_all_s,y_all_s,posori_all_s,_,_,_,es_all_s  = \
                                exp_preprocess(X_all = X_all,y_all=y_all,posori_all=posori_all,es_all=es_all,
                                               split=False,random_state=random_state,**values)
#X_all_s,y_all_s,posori_all_s,es_all_s = X_all,y_all,posori_all,es_all
posori_all_s_t = to_tensor(posori_all_s,posori_all_s,es_all_s)


plot_dir_path = plot_root


#def traj_pca_analysis(plot_dir_path,traj_step_size,traj_length):
traj_step_size = es_all_s['step_size']


#plot_dir_path += str(traj_step_size)+'_'+str(traj_length)+'/'

if not os.path.isdir(plot_dir_path):
        os.makedirs(plot_dir_path)

with open (os.path.join(plot_dir_path,'source_data.txt'),'wb') as f:
    f.write(values['dir_name']+'/'+save_path)
    
activations = load_z_lin(X=X_all_s,fullpath=values['dir_name']+'/'+save_path+".pkl")
del X_all_s
del X_all
del y_all_s
del y_all
activations = to_tensor(activations,posori_all_s,es_all_s)
#pdb.set_trace()
#plot_evaluation(activations_t,0,posori_all_s,es_all_s,plot_dir_path)
act_dim = activations.shape[3]
#pdb.set_trace()


asize,xsize,ysize =activations.shape[:3]


fields =  ['sampling','history_length','angle']
fields.extend(['c'+str(ci) for ci in range(bottleneck_size)])
fields.extend(['min_pred','max_pred','min_latent','max_latent'])
svalue_file = open(os.path.join(plot_dir_path,'svalues.csv'), 'w')
svalue_writer = csv.DictWriter(svalue_file,fields)
svalue_writer.writeheader()
history_lengths = np.array( [25])*4/sampling_rate
pred_length = 1

angles = [0,1,2,3,4,14,15,16,17]#np.arange(es_all_s['num_angles'])#[::2]
seeds = [0]#[0,ysize/2,ysize-2]#range(0,ysize,10)
    
for history_length in history_lengths:
    plot_dir_path_hl = os.path.join(plot_dir_path,train_alg+"_"+str(history_length)+"_"+str(pred_length))
    plot_dir_path_hl += '_n' if do_plot_normalize else ''
    plot_dir_path_hl += '_h' if do_highlight else ''

    if not os.path.isdir(plot_dir_path_hl):
        os.makedirs(plot_dir_path_hl)
        
    if train_alg in ['pls']:
        plss = []

    if train_alg == "max_pinv":
        pinv_ms = []
        bottleneck_size = 1
        
    #generate all trajectories that are in the arena and can be used for training
    act_traj_ds,valid_traj_mask = generate_vectorized_eval_dataset(
                                        activations = activations,
                                        es = es_all_s,
                                        posori = posori_all_s,
                                        traj_step_size = traj_step_size,
                                        traj_length = history_length+pred_length,
                                        plot_dir_path = plot_dir_path_hl)
    assert act_traj_ds.shape == (asize,xsize,ysize,history_length+pred_length,act_dim)
    assert valid_traj_mask.shape == (asize,xsize,ysize)
    valid_traj_mask2 = np.logical_not(np.all(np.isnan(act_traj_ds),axis=(-1,-2)))
    np.testing.assert_equal(valid_traj_mask,valid_traj_mask2)
    ordered_act_traj_ds = np.sort(act_traj_ds,axis=-1)[:,:,:,:,::-1]
    assert ordered_act_traj_ds.shape == (asize,xsize,ysize,history_length+pred_length,act_dim)
    
    #Create input-output mapping for training
    print('Building prediction db')
    X,Y = build_prediction_db(act_traj_ds = ordered_act_traj_ds,
                              history_length = history_length,
                              pred_length = pred_length)
    assert X.shape == (asize,xsize,ysize,history_length,act_dim)
    assert Y.shape == (asize,xsize,ysize,pred_length,act_dim)

    if plot_samples:
        plot_ds_sample([ordered_act_traj_ds[angles][:,:,:,:,:10]],50,plot_dir_path_hl,'train_ordered')
        plot_ds_sample([act_traj_ds[angles]],50,plot_dir_path_hl,'train')
        plot_ds_sample([act_traj_ds[angles],ordered_act_traj_ds[angles][:,:,:,:,:10]],50,plot_dir_path_hl,'train_comp',save_npy=do_save_plots_npy)
                                  
    #pdb.set_trace()
    pred_act_traj_ds = []
    latent_act_traj_ds = []
    stepk_pred_act_traj_ds = []
    stepk_latent_act_traj_ds = []
    ordered_normed_act_traj_ds = []
    
    for ai,angle in enumerate(angles):
        print(angle)
        
        if train_mode == "angle":
            #Extracting and reshaping (e.g. flattening) training data
            X_nt = X[angle][valid_traj_mask[angle]].reshape(-1,history_length*act_dim)
            Y_nt = Y[angle][valid_traj_mask[angle]].reshape(-1,pred_length*act_dim)
        elif train_mode == "all":
            #for angleindependent learning
            X_nt = X[valid_traj_mask].reshape(-1,history_length*act_dim)
            Y_nt = Y[valid_traj_mask].reshape(-1,pred_length*act_dim)
        
        XY_nt = np.hstack([X_nt,Y_nt])
        #Normalizing
        if norm_mode != None:
            if norm_mode == ROWWISE:
                mean_ = np.mean(XY_nt,axis=1).reshape(-1,1)
                XY_nt -= np.repeat(mean_,XY_nt.shape[1],axis=1)
                length_ = np.linalg.norm(XY_nt,2,axis=1).reshape(-1,1)
                XY_nt /= np.repeat(length_,XY_nt.shape[1],axis=1)
                np.testing.assert_allclose(np.linalg.norm(XY_nt,2,axis=1),np.ones(XY_nt.shape[0]))
            elif norm_mode == COLWISE:
                mean_ = np.mean(XY_nt,axis=0).reshape(1,-1)
                XY_nt -= np.repeat(mean_,XY_nt.shape[0],axis=0)
                std_ = np.std(XY_nt,axis=0).reshape(1,-1)
                std_[std_==0] = 1
                XY_nt /= np.repeat(std_,XY_nt.shape[0],axis=0)
                np.testing.assert_allclose(np.std(XY_nt,axis=0)[(std_!=1).flatten()],np.ones(XY_nt.shape[1])[(std_!=1).flatten()])
            X_nt = XY_nt[:,:history_length*act_dim]
            Y_nt = XY_nt[:,history_length*act_dim:]
        #pdb.set_trace()  
        assert X_nt.shape[1] == history_length*act_dim
        assert Y_nt.shape[1] == pred_length*act_dim
        assert XY_nt.shape[1] == (history_length+pred_length)*act_dim
        csvrow = {'sampling':sampling_rate,
                    'history_length':history_length,
                    'angle':angle*es_all_s['turn_angle']} 
        if train_alg in ["pls"]:
            #Training PLS
            pls = PLSRegression(n_components=bottleneck_size,scale=False)
            print('Fitting PLS on '+str(X_nt.shape[0])+' samples with history_length= '+str(history_length)+'and pred_length= '+str(pred_length))
            pls.fit(X_nt, Y_nt)
            plss.append(pls)
            #s_x_mean = np.sort(pls.x_mean_)[::-1]
            #s_x_std = np.sort(pls.x_std_)[::-1]
            #print(s_x_mean[s_x_mean!=0],s_x_std[s_x_std!=0])
            #print(pls.y_mean_,pls.y_std_)
            #pls.coef_ = pls.coef_/np.linalg.norm(pls.coef_,2)
            coef_svalues = np.linalg.svd(pls.coef_)[1][:bottleneck_size]
            xrot_svalues = np.linalg.svd(pls.x_rotations_)[1]#[:bottleneck_size]
            print(coef_svalues)
            print( xrot_svalues)
            
            for ci in range(bottleneck_size):
                csvrow.update({'c'+str(ci):coef_svalues[ci]})
                      
        elif train_alg == "max_pinv":
            #X_nt = X_nt[:,::act_dim]
            #Y_nt = Y_nt[:,::act_dim]
            X_nt_pinv = np.linalg.pinv(X_nt[:,max_pinv_comp_i::act_dim])
            pinv_m = np.dot(X_nt_pinv,Y_nt[:,max_pinv_comp_i::act_dim])
            pinv_ms.append(pinv_m)
            #pdb.set_trace()
            
        if train_mode == "all":    
            #for angleindependent learning
            X_nt = X[angle][valid_traj_mask[angle]].reshape(-1,history_length*act_dim)
            Y_nt = Y[angle][valid_traj_mask[angle]].reshape(-1,pred_length*act_dim)
            XY_nt = np.hstack([X_nt,Y_nt])
        
        ordered_normed_act_traj_ds_ = np.empty_like(ordered_act_traj_ds[0])
        ordered_normed_act_traj_ds_[:] = np.nan
        ordered_normed_act_traj_ds_[valid_traj_mask[angle]] = XY_nt.reshape(XY_nt.shape[0],history_length+pred_length,act_dim)
        ordered_normed_act_traj_ds.append(ordered_normed_act_traj_ds_)
        assert ordered_normed_act_traj_ds_.shape == (xsize,ysize,history_length+pred_length,act_dim)

    ordered_normed_act_traj_ds = np.stack(ordered_normed_act_traj_ds,axis=0)
    assert ordered_normed_act_traj_ds.shape == (len(angles),xsize,ysize,history_length+pred_length,act_dim)
    if plot_samples:
        plot_ds_sample([ordered_normed_act_traj_ds[:,:,:,:,:10]],50,plot_dir_path_hl,'train_ordered_normed')
    #pdb.set_trace()
    for ai,angle in enumerate(angles):
        print(angle)
        ##############################
        #  do_spinning prediction seeded by original data but "spinned" on generated inputs
        ##############################
        if do_spinning:
            pred_act_traj_ds_ = []
            latent_act_traj_ds_ = []
            
            for y in seeds:
                #seed_act_traj = np.array(ordered_act_traj_ds[angle][np.newaxis],copy=True)
                #seed_act_traj[:,:,:y] = np.nan
                #seed_act_traj[:,:,y+1:] = np.nan
                #pred_act_traj_ds__, latent_act_traj_ds__ = pls_predict(seed_act_traj,pls,history_length,spin_pred_length)
                
                #print(y)
                if train_alg in ['pls']:
                    pred_act_traj_ds__, latent_act_traj_ds__ = pls_predict(ordered_normed_act_traj_ds[ai,:,y][np.newaxis,:,np.newaxis],plss[ai],history_length,pred_length,spin_pred_length,keep_orig_data=False)
                elif train_alg == 'max_pinv':
                    pred_act_traj_ds__, latent_act_traj_ds__ = max_pinv_predict(ordered_normed_act_traj_ds[ai,:,y][np.newaxis,:,np.newaxis],pinv_ms[ai],history_length,pred_length,spin_pred_length,max_pinv_comp_i,keep_orig_data=False)
                #pdb.set_trace() #(pred_act_traj_ds__.shape,(1,xsize,1,spin_pred_length,act_dim))
                assert pred_act_traj_ds__.shape == (1,xsize,1,spin_pred_length,act_dim)
                assert latent_act_traj_ds__.shape == (1,xsize,1,spin_pred_length,bottleneck_size)
                pred_act_traj_ds_.append(pred_act_traj_ds__[0])
                latent_act_traj_ds_.append(latent_act_traj_ds__[0])
            pred_act_traj_ds_=np.stack(pred_act_traj_ds_,axis=0)
            latent_act_traj_ds_=np.stack(latent_act_traj_ds_,axis=0)
            assert pred_act_traj_ds_.shape == (len(seeds),xsize,1,spin_pred_length,act_dim)
            assert latent_act_traj_ds_.shape == (len(seeds),xsize,1,spin_pred_length,bottleneck_size)
            pred_act_traj_ds.append(pred_act_traj_ds_)
            latent_act_traj_ds.append(latent_act_traj_ds_)
            
            lims = {}
            lims['max_pred'] = np.nanmax(pred_act_traj_ds_)
            lims['min_pred'] = np.nanmin(pred_act_traj_ds_)
            lims['max_latent'] = np.nanmax(latent_act_traj_ds_)
            lims['min_latent'] = np.nanmin(latent_act_traj_ds_)
            print(lims)
            csvrow.update(lims)
        svalue_writer.writerow(csvrow)
        #pdb.set_trace()
        
        
    #pdb.set_trace()
    
    es_l = dict(es_all_s)
    es_l['num_angles'] = len(angles)
    es_l['turn_angle'] = 360/es_l['num_angles']
    #posori_l = posori_all_s_t[[angles]]
    
    ordered_activations = np.sort(activations,axis=-1)[:,:,:,::-1]
    plot_evaluation(ordered_activations[:,np.newaxis,:,:,:10],1,es_l,plot_dir_path_hl,'ordered_orig',highlight = do_highlight,add_grid=do_add_grid,save_npy=do_save_plots_npy )
    
       
    
    plot_evaluation(activations[:,np.newaxis],1,es_l,plot_dir_path_hl,'orig',highlight = do_highlight,save_npy=do_save_plots_npy)
    
    #pdb.set_trace()
    plot_dir_path_hl = os.path.join(plot_dir_path_hl,str(bottleneck_size))
    if not os.path.isdir(plot_dir_path_hl):
        os.makedirs(plot_dir_path_hl)
        
    if do_spinning:
        print('do_spinning:')
        pred_act_traj_ds = np.stack(pred_act_traj_ds,axis=0)
        latent_act_traj_ds = np.stack(latent_act_traj_ds,axis=0)
        #pdb.set_trace()
        assert pred_act_traj_ds.shape == (len(angles),len(seeds),xsize,1,spin_pred_length,act_dim)
        assert latent_act_traj_ds.shape == (len(angles),len(seeds),xsize,1,spin_pred_length,bottleneck_size)
      
        
        if plot_samples:
            plot_ds_sample([pred_act_traj_ds.reshape((len(angles)*len(seeds),)+pred_act_traj_ds.shape[2:])[:,:,:,:,:10],
                        latent_act_traj_ds.reshape((len(angles)*len(seeds),)+latent_act_traj_ds.shape[2:])],
                        50,plot_dir_path_hl,'pred_lat')    
        #pdb.set_trace()
        #pred_act_traj_ds = act_traj_to_2d(pred_act_traj_ds[:,:,ysize/2][:,:,np.newaxis],posori_l,es_l,traj_step_size)
        #latent_act_traj_ds = act_traj_to_2d(latent_act_traj_ds[:,:,ysize/2][:,:,np.newaxis],posori_l,es_l,traj_step_size)
        
        if do_plot_normalize:
            pred_act_traj_ds_maxs =  np.nanmax(pred_act_traj_ds,axis=(2,3,4))
            pred_act_traj_ds_maxs =  np.repeat(pred_act_traj_ds_maxs[:,:,np.newaxis],pred_act_traj_ds.shape[4],axis=2)
            pred_act_traj_ds_maxs =  np.repeat(pred_act_traj_ds_maxs[:,:,np.newaxis],pred_act_traj_ds.shape[3],axis=2)
            pred_act_traj_ds_maxs =  np.repeat(pred_act_traj_ds_maxs[:,:,np.newaxis],pred_act_traj_ds.shape[2],axis=2)

            pred_act_traj_ds_mins =  np.nanmin(pred_act_traj_ds,axis=(2,3,4))
            pred_act_traj_ds_mins =  np.repeat(pred_act_traj_ds_mins[:,:,np.newaxis],pred_act_traj_ds.shape[4],axis=2)
            pred_act_traj_ds_mins =  np.repeat(pred_act_traj_ds_mins[:,:,np.newaxis],pred_act_traj_ds.shape[3],axis=2)
            pred_act_traj_ds_mins =  np.repeat(pred_act_traj_ds_mins[:,:,np.newaxis],pred_act_traj_ds.shape[2],axis=2)

            latent_act_traj_ds_maxs =  np.nanmax(latent_act_traj_ds,axis=(2,3,4))
            latent_act_traj_ds_maxs =  np.repeat(latent_act_traj_ds_maxs[:,:,np.newaxis],latent_act_traj_ds.shape[4],axis=2)
            latent_act_traj_ds_maxs =  np.repeat(latent_act_traj_ds_maxs[:,:,np.newaxis],latent_act_traj_ds.shape[3],axis=2)
            latent_act_traj_ds_maxs =  np.repeat(latent_act_traj_ds_maxs[:,:,np.newaxis],latent_act_traj_ds.shape[2],axis=2)

            latent_act_traj_ds_mins =  np.nanmin(latent_act_traj_ds,axis=(2,3,4))
            latent_act_traj_ds_mins =  np.repeat(latent_act_traj_ds_mins[:,:,np.newaxis],latent_act_traj_ds.shape[4],axis=2)
            latent_act_traj_ds_mins =  np.repeat(latent_act_traj_ds_mins[:,:,np.newaxis],latent_act_traj_ds.shape[3],axis=2)
            latent_act_traj_ds_mins =  np.repeat(latent_act_traj_ds_mins[:,:,np.newaxis],latent_act_traj_ds.shape[2],axis=2)

            pred_act_traj_ds = (pred_act_traj_ds-pred_act_traj_ds_mins)/(pred_act_traj_ds_maxs-pred_act_traj_ds_mins)
            latent_act_traj_ds = (latent_act_traj_ds-latent_act_traj_ds_mins)/(latent_act_traj_ds_maxs-latent_act_traj_ds_mins)
        if do_save_npy:
            np.save(os.path.join(plot_dir_path_hl,'pred_act_traj_ds.npy'),
            pred_act_traj_ds)
        #pdb.set_trace()
        print('Transforming to 2D representation')
        pred_act_traj_ds_2d_m2 = build_2d_plot_tensors(pred_act_traj_ds,history_length,angles,seeds,posori_all_s_t,es_all_s,traj_step_size)
        latent_act_traj_ds_2d_m2 = build_2d_plot_tensors(latent_act_traj_ds,history_length,angles,seeds,posori_all_s_t,es_all_s,traj_step_size)

        
        #pdb.set_trace()
        

        print('Plotting')
        plot_evaluation(pred_act_traj_ds_2d_m2[:,:,:,:,:10],1,es_l,plot_dir_path_hl,'spin_pred',highlight = do_highlight,add_grid=do_add_grid,save_npy=do_save_plots_npy)
        plot_evaluation(latent_act_traj_ds_2d_m2,1,es_l,plot_dir_path_hl,'spin_latent',highlight = do_highlight,add_grid=do_add_grid,save_npy=do_save_plots_npy)
        
              
    if train_alg == 'pls':
        pickle.dump(plss,open(os.path.join(plot_dir_path_hl,'plss.pkl'),'wb'))
        pickle.dump(ordered_normed_act_traj_ds,open(os.path.join(plot_dir_path_hl,'dss.pkl'),'wb'))
    #pdb.set_trace()
                              
svalue_file.close()


    






    



