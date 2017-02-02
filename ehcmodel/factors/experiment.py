# Author: Andras Sarkany
# License: BSD 3-clause
# Copyright (c) 2017, ELTE

import inspect
from ehcmodel.common.data_preproc import *
from ehcmodel.common.kdl_template import convert_to_one_hot
from sklearn.decomposition import PCA,FastICA

from pprint import pprint
from collections import defaultdict
import itertools
import time

# Generate the paramaters for each single experiment and run them.
def run_experiment(opt_fun,**kwargs):
    pprint(kwargs)
    #print(kwargs.keys())
    
    if not os.path.isdir(kwargs['dir_name']):
        os.makedirs(kwargs['dir_name'])
    if kwargs['description']!= "":
        with open(kwargs['dir_name']+'/description.txt','wb') as f:
            f.write(kwargs['description'])
        
        
    changing_params = [k for k in kwargs.keys() if isinstance(kwargs[k],list) and len(kwargs[k])>1]
    not_changing_params = [k for k in kwargs.keys() if not (isinstance(kwargs[k],list) and len(kwargs[k])>1)]
    grid_comps = []
    for k,v in kwargs.iteritems():
        if isinstance(v,list):
            grid_comps.append(v)
        else:
            grid_comps.append([v])
    #pprint(grid_comps)
    
    #print fixed params
    pkwargs = dict(kwargs)
    for k in changing_params:
        del pkwargs[k]
    
    
    results =defaultdict(dict)
    try:
        for grid_vertex in itertools.product(*grid_comps):
            params = dict(zip(kwargs.keys(),grid_vertex))
            pparams = dict(params)
            for k in not_changing_params:
                del pparams[k]
            #print(params)
            starttime = time.time()
            #run a single training
            res = opt_fun(**params)
            results['time'][tuple(pparams.values())] = "%0.0f" % (time.time()-starttime)
            for key in res.keys():
                results[key][tuple(pparams.values())] = "%0.5f" % np.asscalar(res[key])    
            pprint(pkwargs)
            print(changing_params)
            pprint(dict(results))
    finally:    
        with open(kwargs['dir_name']+'/results.txt','ab') as f:
            pprint(kwargs,f)
            pprint(pkwargs,f)
            f.write(str(changing_params)+'\n')
            pprint(dict(results),f)
            pprint('#'*40,f)
 
