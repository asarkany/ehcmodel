# Author: Andras Sarkany and Zoltan Milacski (based on the framework of Kyle Kastner)
# License: BSD 3-clause
# Copyright (c) 2017, ELTE
from ehcmodel.common.kdl_template import *
from ehcmodel.common.data_preproc import *
from ehcmodel.common.exp_preproc import exp_preprocess

from pprint import pprint

import inspect
import theano.tensor as T

exp_name = 'lab2_big_20v_10t'
_,posori_all,es_all = load_boxworld_raw(exp_name)
X_all,y_all,es_all = load_boxworld_data(exp_name)
print('Full db:')
print(es_all)

show_inputs(posori_all,es_all)
plt.close()

positive_code_layer = relu_layer #relu_layer #leaky_relu_layer #softplus_layer
xcorr = True

val_epoch_results_all = defaultdict(list)
previous_epoch_results = None

    
def constant_sparsity(k):
    return lambda n_epoch:k

#Method to train the autoencoder and save the trained model 
def optimize(dir_name,description,n_code,n_coding_layer,sparsity_k,sparsity_mode,dropout_p,preprocess,sampling_rate,
                view_angle,turn_angle,minibatch_size,reconst_mask,early_stop,n_epochs):
    #global X_all,y_all,posori_all,es_all
    global val_epoch_results_all,early_stop_counter
    frame = inspect.currentframe()
    _, _, _, values = inspect.getargvalues(frame)
    del values['frame']
    pprint(values)
    random_state = np.random.RandomState(1999)
    
    #Preprocess the database
    save_path, X_train,yoh_train,posori_train,X_val,yoh_val,posori_val,es  = \
                            exp_preprocess(X_all = X_all,y_all=y_all,posori_all=posori_all,es_all=es_all,
                                           split=True,random_state=random_state,**values)
                                           
    save_path = save_path[:save_path.rfind('_')]
    if len([fn for fn in os.listdir(dir_name)
              if fn.startswith(save_path)])>0:
        pass
        #raise ValueError(dir_name+'/'+save_path+' already exists!')
        #print(dir_name+'/'+save_path+' already exists!')
        #return {'cost':-1,'box_match_perc':-1}
    save_path = (dir_name+"/"+save_path) 
    print(save_path)
    # random state so script is deterministic

    dropout_on_off = int(dropout_p != None)
    n_coding_layer= n_coding_layer-1
    #dropout_p_str=dropout_p
    #dropout_p = 0.1 if dropout_p==None else dropout_p
    
    #Build the network architecture
    # graph holds information necessary to build layers from parents    
    graph = OrderedDict()
    X_sym, y_sym = add_datasets_to_graph([X_train, yoh_train], ["X", "y"], graph)
    add_epoch_counter(graph)
    
    if sparsity_mode == 'spatial' or sparsity_mode==None:
        add_arrays_to_graph([0],['_k_sparsity'],graph)
    elif isinstance(sparsity_mode,tuple) and sparsity_mode[0] in ['lifetime','lifelong']:
        add_arrays_to_graph([0.],['_k_sparsity'],graph)
        
    n_epoch, = fetch_from_graph(['_n_epoch'], graph)
    k_sparsity_sym, = fetch_from_graph(['_k_sparsity'], graph)
    
    if sparsity_mode ==None:
        print('No sparsity')
    else:
        if sparsity_mode == 'spatial':
            print('Spatial sparsity')
            def _sparse(X):
                return ksparse(X,k_sparsity_sym)
        elif isinstance(sparsity_mode,tuple) and sparsity_mode[0] in ['lifetime','lifelong']:
            if sparsity_mode[1] == 'all':
                print('Lifetime sparsity')
                num_groups = 1
            def _sparse(X):
                return groupklifetimesparse(X,k_sparsity_sym,num_groups)
                
            
        print('Sparsity_k: '+str(sparsity_k))
        add_updates_to_graph([(k_sparsity_sym,constant_sparsity(sparsity_k)(n_epoch))],['constant_sparsity'],graph)
    
    #X_sym_d = dropout_layer([X_sym], "X_d", dropout_on_off, 0.2, random_state)
    X_sym_d = X_sym
    
    n_enc_layer = list(reversed(np.linspace(n_code,X_train.shape[1],n_coding_layer+2).astype(int)[1:1+n_coding_layer]))#[40, 40] 
    n_dec_layer = np.linspace(n_code,X_train.shape[1],n_coding_layer+2).astype(int)[1:1+n_coding_layer]#[40,40]
    encoding_layers = [[X_sym]]
    encoding_d_layers = [[X_sym_d]]
    for li in range(n_coding_layer):
        print('Adding '+str(n_coding_layer)+' encoding layers with ReLU and sizes +'+str(n_enc_layer) )
        new_layer,new_d_layer = [positive_code_layer(incoming, graph, 'l'+str(li+1)+'_enc', n_enc_layer[li],
                                    random_state,strict=strict)
                                for incoming,strict in [(encoding_layers[-1],True),(encoding_d_layers[-1],False)]]
                                
        encoding_layers.append([new_layer])
        encoding_d_layers.append([new_d_layer])
        if sparsity_mode != None:
            new_layer,new_d_layer = [nonlinearity_layer(incoming, graph, 'l'+str(li+1)+'_enc_sp'+str(sparsity_k), n_enc_layer[li],
                                        random_state,func = _sparse,strict=strict)
                                    for incoming,strict in [(encoding_layers[-1],True),(encoding_d_layers[-1],False)]]
                                    
            encoding_layers.append([new_layer])
            encoding_d_layers.append([new_d_layer])
        encoding_d_layers.append([dropout_layer(encoding_d_layers[-1], 'l'+str(li+1)+'_enc_d', dropout_on_off, dropout_p, random_state)])
            
    print('Adding hidden layer (z) with ReLU')
    z_lin,z_lin_d = [positive_code_layer(incoming, graph, 'z', n_code,
                            random_state,strict=strict)
                     for incoming,strict in [(encoding_layers[-1],True),(encoding_d_layers[-1],False)]]                        
    if sparsity_mode ==None:
        z = z_lin
        z_d = z_lin_d    
    else:
        z,z_d = [nonlinearity_layer(incoming, graph, 'z_sp'+str(sparsity_k), n_code,
                            random_state,func = _sparse,strict=strict)
                  for incoming,strict in [([z_lin],True),([z_lin_d],False)]]  
    

    decoding_layers = [[z]]
    decoding_d_layers = [[z_d]]
        
    for li in range(n_coding_layer):
        print('Adding '+str(n_coding_layer)+' decoding layers with ReLU and sizes +'+str(n_dec_layer) )
        new_layer,new_d_layer = [positive_code_layer(incoming, graph, 'l'+str(li+1)+'_dec', n_dec_layer[li],
                                    random_state,strict=strict)
                                 for incoming,strict in [(decoding_layers[-1],True),(decoding_d_layers[-1],False)]]
        decoding_layers.append([new_layer])
        decoding_d_layers.append([new_d_layer])
        if sparsity_mode != None:
            new_layer,new_d_layer = [nonlinearity_layer(incoming, graph, 'l'+str(li+1)+'_dec_sp'+str(sparsity_k), n_dec_layer[li],
                                        random_state,func=_sparse,strict=strict)
                                     for incoming,strict in [(decoding_layers[-1],True),(decoding_d_layers[-1],False)]]
            decoding_layers.append([new_layer])
            decoding_d_layers.append([new_d_layer])
        decoding_d_layers.append([dropout_layer(decoding_d_layers[-1], 'l'+str(li+1)+'_dec_d', dropout_on_off, dropout_p, random_state)])
    
    print('Adding linear output layer')                               
    out,out_d = [linear_layer(incoming, graph, 'out',  X_train.shape[1],
                                random_state,strict=strict)
                for incoming,strict in [(decoding_layers[-1],True),(decoding_d_layers[-1],False)]]
    
   
    print(graph.keys())
    
    sqrdiff=(X_sym-out)**2
    sqrdiff_d=(X_sym-out_d)**2
    
    if reconst_mask and ('concat' in preprocess or
        len([p for p in preprocess if isinstance(p,tuple) and p[0] =='retina'])>0):
        print('Applying reconstruction mask')
        concat_mask_sym=T.extra_ops.repeat(y_sym, es['num_boxes'], axis=1)
        sqrdiff=T.switch(concat_mask_sym,sqrdiff,0.)
        sqrdiff_d=T.switch(concat_mask_sym,sqrdiff_d,0.)       
    
    test_reconst_cost = sqrdiff.sum(axis=1).mean()
    train_reconst_cost = sqrdiff_d.sum(axis=1).mean()
    

    test_cost = test_reconst_cost
    train_cost = train_reconst_cost
    
    params, grads = get_params_and_grads(graph, train_cost)
    pre_epoch_funcs = get_pre_epoch_funcs(graph)

    learning_rate = 0.003
    opt = adam(params)
    updates = opt.updates(params, grads, learning_rate)

    # Checkpointing
    

    fit_function = theano.function([X_sym,y_sym], [train_cost,train_reconst_cost],
                                   updates=updates,on_unused_input='warn')
    test_function = theano.function([X_sym,y_sym], [test_cost,test_reconst_cost],on_unused_input='warn')
    encode_function = theano.function([X_sym], [z],
                                      on_unused_input='warn')
    encode_linear_function = theano.function([X_sym], [z_lin],
                                      on_unused_input='warn')
    # Need both due to tensor.switch, but only one should ever be used
    decode_function = theano.function([z,y_sym], [out],on_unused_input='warn')
    predict_function = theano.function([X_sym,y_sym], [out],on_unused_input='warn')
    predict_d_function = theano.function([X_sym,y_sym], [out_d],on_unused_input='warn')
    
    checkpoint_dict = {}
    checkpoint_dict["fit_function"] = fit_function
    checkpoint_dict["test_function"] = test_function
    checkpoint_dict["encode_function"] = encode_function
    checkpoint_dict["encode_linear_function"] = encode_linear_function
    checkpoint_dict["decode_function"] = decode_function
    checkpoint_dict["predict_function"] = predict_function
    checkpoint_dict["predict_d_function"] = predict_d_function
    #previous_epoch_results = None

    shuffle = True
    if 'sort_by_angles' in preprocess:
        shuffle = 'sort_by_angles' #False
    minibatch_size = X_train.shape[0] if minibatch_size == 'fullbatch' else minibatch_size

    val_epoch_results_all = defaultdict(list)
    early_stop_counter=0
    def status_func(epoch_number, epoch_results, status_points, early_stop_patience=500):
        global val_epoch_results_all
        global early_stop_counter
        global previous_epoch_results

        val_epoch_results = iterate_function(test_function, [X_val,yoh_val], X_val.shape[0],
                                     list_of_output_names=["val_cost","reconst_cost"],
                                     n_epochs=1,
                                     previous_epoch_results=None,
                                     shuffle=shuffle,
                                     random_state=random_state,
                                     es=es)
        for k,v in val_epoch_results.iteritems():
            val_epoch_results_all[k].extend(v)
        
        checkpoint_dict["train_epoch_results"] = epoch_results
        checkpoint_dict["val_epoch_results"] = dict(val_epoch_results_all)
            
        if early_stop:
            if epoch_number == 0 or (val_epoch_results_all['val_cost'][-1] <= min(val_epoch_results_all['val_cost'])):
                early_stop_counter = 0
                if epoch_number!=0:
                    checkpoint_status_func(save_path+'_'+str(n_epochs)+'.pkl', checkpoint_dict, epoch_results)
                    print_status_func(val_epoch_results)
                    print("Saving checkpoint based on validation score")
            else:
                print_status_func(epoch_results)
                print_status_func(val_epoch_results)
            if len(val_epoch_results_all['val_cost']) > early_stop_patience and not (val_epoch_results_all['val_cost'][-1] <= min(val_epoch_results_all['val_cost'][:-early_stop_patience])):
                early_stop_counter += 1
                print(early_stop_counter)
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping... Best validation score: "+str(min(val_epoch_results_all['val_cost'])))
                    return True
            print(early_stop_counter)
        else:
            if epoch_number >0:
                if epoch_number % 1000 == 0 or epoch_number == n_epochs:
                    checkpoint_status_func(save_path+'_'+str(epoch_number)+'.pkl', checkpoint_dict, epoch_results)
                print_status_func(epoch_results)
            print_status_func(val_epoch_results)
        return False
        
    #save_checkpoint(save_path+'_0.pkl', checkpoint_dict)
    epoch_results = iterate_function(fit_function, [X_train,yoh_train], minibatch_size,
                                     list_of_output_names=["cost","reconst_cost"],
                                     n_epochs=n_epochs,
                                     status_func=status_func,
                                     previous_epoch_results=None,
                                     shuffle=shuffle,
                                     random_state=random_state,graph = graph,pre_epoch_funcs = pre_epoch_funcs,es=es)
    

    checkpoint_dict = load_checkpoint(save_path+'_'+str(n_epochs)+'.pkl')
    test_function = checkpoint_dict["test_function"]
    encode_function = checkpoint_dict["encode_function"]
    decode_function = checkpoint_dict["decode_function"]
    #predict_function = checkpoint_dict["predict_function"]
    val_cost,val_reconst_cost = test_function(X_val,yoh_val)
    train_cost,train_reconst_cost = test_function(X_train,yoh_train)
    z, = encode_function(X_train)
    out, = decode_function(z,yoh_train)
    #print(predict_function(X_val[0:1,:],yoh_val[0:1,:])[0][0][0])
    #print(predict_d_function(X_val[0:1,:],yoh_val[0:1,:])[0][0][0])
    #box_match_perc = 1-np.sum(np.abs(X_train-(out > 0.5)))/np.prod(X_train.shape)
    print(es)
    
    return {'train_cost':train_cost,'train_reconst_cost':train_reconst_cost,
            'val_cost':val_cost,'val_reconst_cost':val_reconst_cost}

