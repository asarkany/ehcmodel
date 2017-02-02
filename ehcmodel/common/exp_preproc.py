# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE
import inspect
from ehcmodel.common.nipg import *
from ehcmodel.common.kdl_template import convert_to_one_hot
from sklearn.decomposition import PCA,FastICA

from pprint import pprint
from collections import defaultdict
import itertools
import time

def gen_save_path(values):
    
    #pprint(values)
    args = values.keys()
    strvalues = {}
    for arg in args:
        if arg in ['dir_name','description','split']:
            continue
        if isinstance(values[arg],tuple):
            strvalues[arg] = "("+"_".join(p[0] if isinstance(p,str) else str(p) for p in values[arg])+")"
        elif isinstance(values[arg],str):
            strvalues[arg] = values[arg][0]
        elif values[arg] == None:
            strvalues[arg] = 'N'
        else:
            strvalues[arg] = str(values[arg])
    
    ordered_args = ['sparsity_mode','sparsity_k','n_code','n_coding_layer','dropout_p','preprocess','sampling_rate',
        'view_angle','turn_angle','reconst_mask','minibatch_size','early_stop','n_epochs']
    save_path = "_".join(strvalues[arg] for arg in ordered_args)
    return save_path

#Preprocess data respect to the experiment settings
def exp_preprocess(dir_name,description,n_code,n_coding_layer,sparsity_k,sparsity_mode,dropout_p,preprocess,sampling_rate,
                            view_angle,turn_angle,minibatch_size,reconst_mask,early_stop,n_epochs,
                            X_all,y_all,posori_all,es_all,split,random_state):
    #global X_all,y_all,posori_all,es_all
    frame = inspect.currentframe()
    _, _, _, values = inspect.getargvalues(frame)
    for a in ['frame','X_all','y_all','posori_all','es_all','split','random_state']:
        del values[a]
    
    save_path = gen_save_path(values)
    print('Sampling grid_points by '+str(sampling_rate)+'in both x and y directions')
    X_all_s,y_all_s,posori_all_s,es = sample_dataset(X_all,y_all,posori_all,es_all,sampling_rate)
    if view_angle == None or turn_angle == None:
        raise ValueError('Please specifiy turn angle and view_angle!')
    
    retina = [p for p in preprocess if isinstance(p,tuple) and p[0] =='retina']
    retina = None if len(retina)==0 else retina[0]
    if retina == None:
        print('Simple '+str(es['num_boxes'])+' input')
        print('Augmenting db to '+str(turn_angle)+' d turn angle and '+str(view_angle)+' d view_angle')
        X_all_s,y_all_s,posori_all_s,es = augment_angles(X_all_s,y_all_s,posori_all_s,es,360/turn_angle,view_angle)
    
    if retina != None and 'normalize' in preprocess:
        assert preprocess.index(retina)<preprocess.index('normalize')

    show_inputs(posori_all_s,es)
    for p in preprocess:
        if p == 'ica':
            fica = FastICA(n_components=30)
            X_all_s = fica.fit_transform(X_all_s.astype("float32"))
        elif p == 'pca':
            pca = PCA(n_components=30)
            X_all_s = pca.fit_transform(X_all_s.astype("float32"))
        elif p == 'inverse':
            X_all_s = np.logical_not(X_all_s).astype(int)
        elif p == 'normalize':
            if retina == None:
                print('Normalizing')
                X_all_s = normalize_X(X_all_s)         
            else:
                print('Normalizing for retina input')
                X_all_s = normalize_X(X_all_s,360/retina[1])         
        elif p =='sort_by_angles':
            pass
        elif p == 'concat':
            X_all_s =concat_X(X_all_s, y_all_s,es)
        elif isinstance(p,tuple):
            if p[0] == 'retina':
                print('Creating retina input')
                if p[1]> turn_angle or p[1]>view_angle or turn_angle % p[1] != 0 or view_angle %p[1] != 0:
                    raise ValueError('Retina bin angle should be smaller and a divisor of turn_angle and view_angle'+str((p[1],turn_angle,view_angle)))
                X_all_s,y_all_s,posori_all_s,es = retina_X(X_all_s,y_all_s,posori_all_s,es, p[1],view_angle,turn_angle)
        elif p==None:
            pass
        else:
            raise ValueError('\"'+p +'\" preprocessing is not implemented, Isnt it a typo?')
    
    print('\nFinal: (before train-valid separation)')
    print(es)
    print()
    if split:
        train,valid,test = split_data(X_all_s, y_all_s,posori_all_s,mode='placesplit',train_ratio=0.9,es=es,random_state=random_state)
        X_train = train[0].astype("float32").astype(theano.config.floatX)
        y_train = train[1].astype("int32")
        posori_train = train[2]
        X_val = valid[0].astype("float32").astype(theano.config.floatX)
        y_val = valid[1].astype("int32")
        posori_val = valid[2]
        show_inputs(posori_train,es)
        show_inputs(posori_val,es)
        #plt.show()
        plt.close()
    else:
        X_train = X_all_s
        y_train = y_all_s
        posori_train = posori_all_s
        X_val,y_val,posori_val = (None,None,None)
        
    if 'sort_by_angles' in preprocess:
        X_train,y_train,posori_train = sort_by_angles(X_train,y_train,posori_train,es)
        X_val,y_val,posori_val = sort_by_angles(X_val,y_val,posori_val,es) if split else (None,None,None)
    

    print('Making yoh for reconst-mask for retina input')
    k = 1+view_angle/retina[1]/2
    yoh_train = convert_to_overlap_one_hot(y_train, n_classes=360/retina[1],k=k,mode=1).astype("float32")
    yoh_val = convert_to_overlap_one_hot(y_val, n_classes=360/retina[1],k=k,mode=1).astype("float32") if split else None
           
    #assert not np.all(X_m_train<X_train) #np.all(X_m_train.astype(int)>=X_train) 
    #assert(X_val == None or not np.all(X_m_val<X_val))
    
    if 'normalize' in preprocess and retina != None:
        #print (np.abs(np.sum(X_train.reshape(X_train.shape[0],-1,es['num_boxes'])**2,axis=2)-1))
        assert (np.all(np.abs(np.sum(X_train.reshape(X_train.shape[0],-1,es['num_boxes'])**2,axis=2)-1) 
                [np.sum(X_train.reshape(X_train.shape[0],-1,es['num_boxes'])**2,axis=2)>0] <=1e-3  ))
        #print(np.sum(X_train**2,axis=1))
    print(X_train[0])
    print(yoh_train[0:2])
    #if split:
    #    print(X_val[0])
    #    print(yoh_val[0:2])

    return save_path, X_train,yoh_train,posori_train,X_val,yoh_val,posori_val,es

