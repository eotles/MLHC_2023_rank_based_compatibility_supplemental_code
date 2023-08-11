import pandas as pd
from joblib import Parallel, delayed
import json
import numpy as np
import os
import random
import shutil
import sklearn.metrics as sklearn_metrics
import tensorflow as tf

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
    
import compatibility_metrics as cm
import model_functions



# ----- UTILS

## ----- JSON
def save_json(fp, json_data):
    with open(fp, 'w') as f:
        json.dump(json_data, f)
        
        
def load_json(fp):
    with open(fp, 'r') as f:
        json_data = json.load(f)
    return json_data
    
## ----- PICKLE
def save_pickle(fp, pickle_data):
    with open(fp, 'wb') as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(fp):
    with open(fp, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


## ----- DATA
def save_data(data_dir, X, y):
    try:
        os.mkdir(data_dir)
    except:
        pass
    
    for k, v in {'X': X, 'y': y}.items():
        ds_fp = '{}/{}.p'.format(data_dir, k)
        save_pickle(ds_fp, v)
    
def load_data(data_dir):
    X_fp = '{}/X.p'.format(data_dir)
    X = load_pickle(X_fp)
    
    
    y_fp = '{}/y.p'.format(data_dir)
    y = load_pickle(y_fp)
        
    return (X,y)
    
    
## ----- DATASETS
def load_datasets(f_info, ds_names=None):
    X,y = load_data(f_info['data_dir'])
    
    splits = f_info['splits']
    
    if ds_names is None:
        ds_names = splits.keys()
    
    datasets = {}
    for ds_name in ds_names:
        ds_idxs = splits[ds_name]
        ds_X = X[ds_idxs]
        ds_y = y[ds_idxs]
        datasets[ds_name] = (ds_X,ds_y)
        
    return datasets


## ----- P_HATS
def load_p_hats(p_hat_dir, ds_names=None):
    
    if ds_names is None:
        ds_names = [fn.split('.')[0] for fn in os.listdir(p_hat_dir)]
    
    p_hats = {}
    for ds_name in ds_names:
        ds_fp = '{}/{}.p'.format(p_hat_dir, ds_name)
        p_hat_ds = load_pickle(ds_fp)
        p_hats[ds_name] = np.expand_dims(p_hat_ds, axis=-1)

    return p_hats



# ----- SETUP FUNCTIONS

def setup_candidate_f_os(shared_f_o_info, Cs):
    
    f_os_dir = '{}/candidate_f_os'.format(shared_f_o_info['rep_dir'])

    try:
        os.mkdir(f_os_dir)
    except:
        pass
    
    
    candidate_f_o_dirs = []
    for f_o_num, C in enumerate(Cs):
        f_o_dir = '{}/{:05d}'.format(f_os_dir, f_o_num)
        try:
            os.mkdir(f_o_dir)
        except:
            pass
        
        f_o_info = shared_f_o_info.copy()
        f_o_info['C'] = C
    
        f_o_model_info_fp = '{}/info.json'.format(f_o_dir)

        
        save_json(f_o_model_info_fp, f_o_info)
            
        candidate_f_o_dirs.append(f_o_dir)
        
    return candidate_f_o_dirs
    
    
    
def setup_candidate_f_us(shared_f_u_info,
                         Cs,
                         alphas,
                         n_engineered = 1,
                         n_engineered_resample = 1,
                         n_standard = 1,
                         n_standard_resample = 1):
    
    f_us_dir = '{}/f_us'.format(shared_f_u_info['rep_dir'])

    try:
        os.mkdir(f_us_dir)
    except:
        pass
    
    
    
    #f_u_engineered_list = [{'C': C, 'alpha': alpha, 'f_u_type': 'engineered'}
    #                       for C in Cs
    #                       for alpha in alphas]
    
    #f_u_standard_list = [{'C': C, 'alpha': 1, 'f_u_type': 'standard'} for C in Cs]
    
    
    f_u_params_dict = {}
    for i in range(n_engineered):
        params_list = [{'C': C, 'alpha': alpha,
                        'f_u_type': 'engineered'}
                       for C in Cs
                       for alpha in alphas]
        
        _ = len(f_u_params_dict)
        f_u_params_dict[_] = params_list
    
    
    for i in range(n_engineered_resample):
        splits = dict(shared_f_u_info['splits'])
        splits['ud'] = random.choices(splits['ud'], k=len(splits['ud']))
        params_list = [{'C': C, 'alpha': alpha,
                        'f_u_type': 'engineered', 'splits': splits}
                       for C in Cs
                       for alpha in alphas]
        
        
        _ = len(f_u_params_dict)
        f_u_params_dict[_] = params_list
        
        
    for i in range(n_standard):
        params_list = [{'C': C, 'alpha': 1,
                        'f_u_type': 'standard'}
                       for C in Cs]
        
        _ = len(f_u_params_dict)
        f_u_params_dict[_] = params_list
    
    
    for i in range(n_standard_resample):
        splits = dict(shared_f_u_info['splits'])
        splits['ud'] = random.choices(splits['ud'], k=len(splits['ud']))
        params_list = [{'C': C, 'alpha': 1,
                        'f_u_type': 'standard', 'splits': splits}
                       for C in Cs]
        
        
        _ = len(f_u_params_dict)
        f_u_params_dict[_] = params_list
        
    
    
    candidate_f_u_dirs = []
    for f_u_num, f_u_params_list in f_u_params_dict.items():
        for i, f_u_params in enumerate(f_u_params_list):
            f_u_dir = '{}/{:05d}_{:05d}'.format(f_us_dir, f_u_num, i)
            try:
                os.mkdir(f_u_dir)
            except:
                pass

            f_u_info = shared_f_u_info.copy()
            f_u_info.update(f_u_params)

            f_u_model_info_fp = '{}/info.json'.format(f_u_dir)

            
            save_json(f_u_model_info_fp, f_u_info)

            candidate_f_u_dirs.append(f_u_dir)
        
    return candidate_f_u_dirs
    
    
    
# ----- RUN MODEL

## ----- f_o
def run_f_o(f_o_dir):
    f_o_info_fp = '{}/info.json'.format(f_o_dir)
    f_o_model_dir = '{}/model'.format(f_o_dir)
    f_o_hist = '{}/fit_hist.csv'.format(f_o_dir)
    
    f_o_info = load_json(f_o_info_fp)
        
    if f_o_info['fit']:
        return
        
    datasets = load_datasets(f_o_info)
    X_od, y_od = datasets['od']
    X_oe, y_oe = datasets['oe']
    
    n_features = X_od.shape[1]
    
    f_o = model_functions.make_f_o(n_features=f_o_info['n_features'],
                   C=f_o_info['C'],
                   optimizer=f_o_info['optimizer']
                   )
        
        
    early_stopping_kwargs = f_o_info['early_stopping_kwargs']
    ESCallback = tf.keras.callbacks.EarlyStopping(**early_stopping_kwargs)

    
    f_o_fit = f_o.fit(X_od, y_od,
                      validation_data=(X_oe, y_oe),
                      callbacks=[ESCallback,
                                 tf.keras.callbacks.CSVLogger(f_o_hist)
                                ],
                      **f_o_info['fit_kwargs'],
                     )
    
    f_o.save(f_o_model_dir)
    f_o_info['fit'] = True
    
    
    p_hat_o_dir = '{}/p_hat_o'.format(f_o_dir)

    try:
        os.mkdir(p_hat_o_dir)
    except:
        pass

    auroc_os = {}
    for ds_name, ds in datasets.items():
        X_ds, y_ds = ds
        p_hat_o_ds_fp = '{}/{}.p'.format(p_hat_o_dir, ds_name)
        p_hat_o = f_o.predict(X_ds).squeeze()
        save_pickle(p_hat_o_ds_fp, p_hat_o)

        try:
            auroc = sklearn_metrics.roc_auc_score(y_ds, p_hat_o)
        except:
            auroc = None
        auroc_os[ds_name] = auroc


    f_o_performance_fp = '{}/performance.json'.format(f_o_dir)
    performance = {'auroc': auroc_os}
    
    save_json(f_o_performance_fp, performance)
    save_json(f_o_info_fp, f_o_info)



## ----- f_u


### ----- CALLBACKS

class RBCCallback(tf.keras.callbacks.Callback):
    # https://stackoverflow.com/a/61686136
    
    def __init__(self, train, validation=None):
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):
        
        X_train, y_train, p_hat_o_train = self.train
        p_hat_u_train = self.model.predict((X_train, y_train, p_hat_o_train))
        logs['RBC'] = cm.rbc_score(y_train, p_hat_o_train, p_hat_u_train)

        if (self.validation):
            X_val, y_val, p_hat_o_val = self.validation
            p_hat_u_val = self.model.predict((X_val, y_val, p_hat_o_val))
            logs['val_RBC'] = cm.rbc_score(y_val, p_hat_o_val, p_hat_u_val)


class WSCallback(tf.keras.callbacks.Callback):

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        
    '''
    def on_train_begin(self, logs={}):

        self.WC_scores = []
        self.val_WC_scores = []
        self.WC_score=0
        self.val_WC_score=0
    '''

    def on_epoch_end(self, epoch, logs={}):
        #print(logs)
        alpha = self.alpha
        WC_Score = alpha*logs['AUC'] + (1-alpha)*logs['RBC']    #stupid tensorflow auc -> AUC
        val_WC_Score = alpha*logs['val_AUC'] + (1-alpha)*logs['val_RBC']
        #self.WC_Scores.append(self.WC_Score)
        #self.val_WC_Scores.append(self.val_WC_Score)
        
        logs['WC_Score'] = WC_Score
        logs['val_WC_Score'] = val_WC_Score
        
        #print("— WC_score: %f — val_WC_score: %f" %(WC_Score, val_WC_Score))

            

def run_f_u(f_u_dir):

    f_u_info_fp = '{}/info.json'.format(f_u_dir)
    f_u_model_dir = '{}/model'.format(f_u_dir)
    f_u_hist = '{}/fit_hist.csv'.format(f_u_dir)
    
    
    f_u_info = load_json(f_u_info_fp)
        
    if f_u_info['fit']:
        return
    
    #f_o
    f_o_dir = '{}/f_o'.format(f_u_info['rep_dir'])
    
    f_o_model_dir = '{}/model'.format(f_o_dir)
    f_o = tf.keras.models.load_model(f_o_model_dir)
    f_o.trainable = False
    
    #load data and p_hat_os
    datasets = load_datasets(f_u_info, ds_names=['ud', 'ue', 'e'])
    p_hat_os = load_p_hats('{}/p_hat_o'.format(f_o_dir), ds_names=['ud', 'ue', 'e'])
    
    X_ud, y_ud = datasets['ud']
    X_ue, y_ue = datasets['ue']
    p_hat_o_ud = p_hat_os['ud']
    p_hat_o_ue = p_hat_os['ue']
    
    
    f_u = model_functions.make_f_u(n_features=f_u_info['n_features'],
                                alpha=f_u_info['alpha'],
                                C=f_u_info['C'],
                                optimizer=f_u_info['optimizer']
                               )
    
    if f_u_info['copy_f_o_weights']:
        f_o_weights = f_o.layers[1].get_weights()
        f_u.layers[3].set_weights(f_o_weights)
        

    ESCallback = tf.keras.callbacks.EarlyStopping(**f_u_info['early_stopping_kwargs'])

    
    f_u_fit = f_u.fit((X_ud, y_ud, p_hat_o_ud), y_ud,
                      validation_data=((X_ue, y_ue, p_hat_o_ue), y_ue),
                      callbacks=[RBCCallback(train=(X_ud, y_ud, p_hat_o_ud),
                                             validation=(X_ue, y_ue, p_hat_o_ue)),
                                 WSCallback(alpha=f_u_info['alpha']),
                                 ESCallback,
                                 tf.keras.callbacks.CSVLogger(f_u_hist)
                                ],
                      **f_u_info['fit_kwargs'],
                     )
    
    f_u.save(f_u_model_dir)
    f_u_info['fit'] = True
    
    
    p_hat_u_dir = '{}/p_hat_u'.format(f_u_dir)
    try:
        os.mkdir(p_hat_u_dir)
    except:
        pass
    
    auroc_us = {}
    rbcs = {}
    for ds_name, ds in datasets.items():
        X_ds, y_ds = ds
        p_hat_o_ds = p_hat_os[ds_name].squeeze()
        
        p_hat_u_ds_fp = '{}/{}.p'.format(p_hat_u_dir, ds_name)
        p_hat_u_ds = f_u.predict((X_ds, y_ds, p_hat_o_ds)).squeeze()
        save_pickle(p_hat_u_ds_fp, p_hat_u_ds)
        
        
        try:
            auroc_u_ds = sklearn_metrics.roc_auc_score(y_ds, p_hat_u_ds)
        except:
            auroc_u_ds = None
        auroc_us[ds_name] = auroc_u_ds
        
        try:
            rbc_ds = cm.rbc_score(y_ds, p_hat_o_ds, p_hat_u_ds)
        except:
            rbc_ds = None
        rbcs[ds_name] = rbc_ds
    
    f_u_performance_fp = '{}/performance.json'.format(f_u_dir)
    performance = {'auroc': auroc_us,
                   'rbc': rbcs}
    
    save_json(f_u_performance_fp, performance)
    save_json(f_u_info_fp, f_u_info)
    





# ---- UPDATE RESULTS FUNCTION

def update_res(experiment_dir):

    res = []
    for rep_num in os.listdir(experiment_dir):
        if rep_num == 'data':
            continue

        rep_dir = '{}/{}'.format(experiment_dir, rep_num)
        if not os.path.isdir(rep_dir):
            continue

        f_o_dir = '{}/f_o'.format(rep_dir)
        f_us_dir = '{}/f_us'.format(rep_dir)

        f_o_performance_fp = '{}/performance.json'.format(f_o_dir)
        f_o_info_fp = '{}/info.json'.format(f_o_dir)


        try:
            f_o_res = {'f_o_name': rep_num,
                       'f_o_rep': int(rep_num)
                      }
            
            f_o_info = load_json(f_o_info_fp)
            f_o_res['f_o_C'] = f_o_info['C']
            #f_o_res['f_o_info'] = f_o_info
            
            f_o_performance = load_json(f_o_performance_fp)

            for scorer, scorer_dict in f_o_performance.items():
                for ds_name, v in scorer_dict.items():
                    if scorer == 'auroc':
                        f_o_res['AUROC(f_o,{})'.format(ds_name)] = v
                    elif scorer == 'rbc':
                        f_o_res['RBC({})'.format(ds_name)] = v
                    else:
                        f_o_res['{}({})'.format(scorer, ds_name)] = v

        except:
            pass


        for f_u in os.listdir(f_us_dir):

            f_u_dir = '{}/{}'.format(f_us_dir, f_u)
            f_u_performance_fp = '{}/performance.json'.format(f_u_dir)
            f_u_info_fp = '{}/info.json'.format(f_u_dir)

            try:
                row = f_o_res.copy()
                row['f_u_name'] = f_u

                with open(f_u_info_fp, 'r') as f:
                    f_u_info = json.load(f)
                    alpha_col_str_template ='{:0.1f}'
                    row['f_u_alpha'] = alpha_col_str_template.format(f_u_info['alpha'])
                    row['f_u_C'] = f_u_info['C']
                    #row['f_u_info'] = f_u_info
                    row['f_u_type'] = f_u_info['f_u_type']
                
                f_u_performance = load_json(f_u_performance_fp)

                for scorer, scorer_dict in f_u_performance.items():
                    for ds_name, v in scorer_dict.items():
                        if scorer == 'auroc':
                            row['AUROC(f_u,{})'.format(ds_name)] = v
                        elif scorer == 'rbc':
                            row['RBC({})'.format(ds_name)] = v
                        else:
                            row['{}({})'.format(scorer, ds_name)] = v

                res.append(row)

            except:
                pass

    res_df = pd.DataFrame(res)
    res_df.to_csv('{}/res.csv'.format(experiment_dir), index=False)











# RUN
f_o_optimizer = f_u_optimizer = 'adam'

fit_kwargs={'epochs': 100, 'verbose': True}
f_o_fit_kwargs = f_u_fit_kwargs = fit_kwargs


early_stopping_kwargs = {
    #'monitor': 'val_WC_Score',
    'mode': 'max',
    'patience': 5,
    'restore_best_weights': True
}

f_o_early_stopping_kwargs = early_stopping_kwargs.copy()
f_o_early_stopping_kwargs['monitor'] = 'val_auc'

f_u_early_stopping_kwargs = early_stopping_kwargs.copy()
f_u_early_stopping_kwargs['monitor'] = 'val_WC_Score'


def setup(experiment_dir,
          X, y,
          Cs=[0], alphas=[0, 0.5, 1.0],
          n_reps=5,
          n_engineered=1,
          n_engineered_resample=0,
          n_standard=1,
          n_standard_resample=1,
          n_od=200,
          n_oe=200,
          n_ud=3000,
          n_ue=3000,
          f_o_optimizer=f_o_optimizer,
          f_u_optimizer=f_u_optimizer,
          f_o_fit_kwargs=f_o_fit_kwargs,
          f_u_fit_kwargs=f_u_fit_kwargs,
          f_o_early_stopping_kwargs=f_o_early_stopping_kwargs,
          f_u_early_stopping_kwargs=f_u_early_stopping_kwargs,
          ):
    
    experiment_info_fp = '{}/experiment_info.json'.format(experiment_dir)


    #try:
    os.mkdir(experiment_dir)
    
    #data
    data_dir = '{}/data'.format(experiment_dir)
    save_data(data_dir, X, y)
    
    
    n_instances, n_features = X.shape
    
    ds_n = {'od': n_od,
            'oe': n_od if n_oe is None else n_oe,
            'ud': n_ud,
            'ue': n_ud if n_ue is None else n_ue
           }
    ds_n['e'] = n_instances - sum(ds_n.values())

    
    
    experiment_info = {'n_instances': n_instances,
                       'n_features': n_features,
                       'n_reps': n_reps,
                       'ds_n': ds_n,
                       'n_engineered': n_engineered,
                       'n_engineered_resample': n_engineered_resample,
                       'n_standard': n_standard,
                       'n_standard_resample': n_standard_resample,
                       'Cs': list(Cs),
                       'alphas': list(alphas),
                       'f_o_optimizer': f_o_optimizer,
                       'f_u_optimizer': f_u_optimizer,
                       'f_o_fit_kwargs': f_o_fit_kwargs,
                       'f_u_fit_kwargs': f_u_fit_kwargs,
                       'f_o_early_stopping_kwargs': f_o_early_stopping_kwargs,
                       'f_u_early_stopping_kwargs': f_u_early_stopping_kwargs,
                       'complete': False
                      }
    save_json(experiment_info_fp, experiment_info)
    
    n_f_os = len(Cs)
    n_f_us = len(Cs)*(len(alphas)*(n_engineered + n_engineered_resample) + n_standard + n_standard_resample)

    print('{} replications, per replication we will train {} f_o and {} f_u models'.format(n_reps, n_f_os, n_f_us))
    print('{} total'.format(n_reps*(n_f_os+n_f_us)))
        
    #except:
    #    pass
        
        
def run(experiment_dir, n_jobs=5):

    data_dir = '{}/data'.format(experiment_dir)
    experiment_info_fp = '{}/experiment_info.json'.format(experiment_dir)
    experiment_info = load_json(experiment_info_fp)


    if not experiment_info['complete']:
        shared_rep_info = {'complete': False}

        for rep_num in range(experiment_info['n_reps']):

            print('rep: {}'.format(rep_num))
            rep_dir = '{}/{:05d}'.format(experiment_dir, rep_num)
            rep_info_dir = '{}/rep_info.json'.format(rep_dir)

            #replication directory
            try:
                os.mkdir(rep_dir)

                rep_info = dict(shared_rep_info)
                save_json(rep_info_dir, rep_info)


            except:
                with open(rep_info_dir, 'r') as f:
                    rep_info = json.load(f)


            if not rep_info['complete']:
                #f_o
                print('\tf_o')


                #split
                shared_info_fp = '{}/shared_info.json'.format(rep_dir)
                if not os.path.exists(shared_info_fp):
                    
                    idxs = [i for i in range(experiment_info['n_instances'])]
                    random.shuffle(idxs)
                    splits = {}
                    start = 0
                    for ds, n in experiment_info['ds_n'].items():
                        end = start + n
                        splits[ds] = idxs[start:end]
                        start = end


                    shared_info = {
                        'rep_dir': rep_dir,
                        'data_dir': data_dir,
                        'n_features': experiment_info['n_features'],
                        'splits': splits,
                        'fit': False
                    }
                    
                    save_json(shared_info_fp, shared_info)

                else:
                    shared_info = load_json(shared_info_fp)





                #train f_o
                f_o_dir = '{}/f_o'.format(rep_dir)
                if not os.path.isdir(f_o_dir):

                    shared_f_o_info = shared_info.copy()
                    shared_f_o_info['optimizer'] = experiment_info['f_o_optimizer']
                    shared_f_o_info['fit_kwargs'] = experiment_info['f_o_fit_kwargs']
                    shared_f_o_info['early_stopping_kwargs'] = experiment_info['f_o_early_stopping_kwargs']



                    # train candidate f_os
                    candidate_f_o_dirs = setup_candidate_f_os(shared_f_o_info,
                                                              experiment_info['Cs'])

                    Parallel(n_jobs=n_jobs, verbose=1)(
                        delayed(run_f_o)(
                            f_o_dir=f_o_dir
                        )
                        for f_o_dir in candidate_f_o_dirs
                    )

                    ## find best candidate f_o
                    max_auroc_oe_f_o_dir = None
                    max_auroc_oe = 0
                    for candidate_f_o_dir in candidate_f_o_dirs:
                        candidate_f_o_performance_fp = '{}/performance.json'.format(candidate_f_o_dir)
                        with open(candidate_f_o_performance_fp, 'r') as f:
                                candidate_f_o_performance = json.load(f)
                                candidate_f_o_auroc_oe = candidate_f_o_performance['auroc']['oe']
                                if candidate_f_o_auroc_oe>max_auroc_oe:
                                    max_auroc_oe = candidate_f_o_auroc_oe
                                    max_auroc_oe_f_o_dir = candidate_f_o_dir
                    ## copy to f_o
                    shutil.copytree(max_auroc_oe_f_o_dir, f_o_dir)


                else:
                    pass



                #f_u subdirectory
                print('\tf_u')
                f_us_dir = '{}/f_us'.format(rep_dir)

                try:
                    os.mkdir(f_us_dir)
                except:
                    pass


                #run f_us
                shared_f_u_info = shared_info.copy()
                shared_f_u_info['optimizer'] = experiment_info['f_u_optimizer']
                shared_f_u_info['fit_kwargs'] = experiment_info['f_u_fit_kwargs']
                shared_f_u_info['early_stopping_kwargs'] = experiment_info['f_u_early_stopping_kwargs']
                shared_f_u_info['copy_f_o_weights'] = True



                candidate_f_us = setup_candidate_f_us(shared_f_u_info,
                                                      experiment_info['Cs'],
                                                      experiment_info['alphas'],
                                                      n_engineered = experiment_info['n_engineered'],
                                                      n_engineered_resample = experiment_info['n_engineered_resample'],
                                                      n_standard = experiment_info['n_standard'],
                                                      n_standard_resample = experiment_info['n_standard_resample']
                                                     )

                Parallel(n_jobs=n_jobs, verbose=1)(
                    delayed(run_f_u)(
                        f_u_dir=f_u_dir
                    )
                    for f_u_dir in candidate_f_us
                )


                rep_info['complete'] = True
                save_json(rep_info_dir, rep_info)



            print('\t update res')
            update_res(experiment_dir)
            
            
        experiment_info['complete'] = True
        save_json(experiment_info_fp, experiment_info)

    else:
        print('update res')
        update_res(experiment_dir)






def setup_and_run(experiment_dir,
                  X, y,
                  Cs=[0], alphas=[0, 0.5, 1.0],
                  n_reps=5,
                  n_jobs=5,
                  n_engineered=1,
                  n_engineered_resample=0,
                  n_standard=1,
                  n_standard_resample=1,
                  n_od=200,
                  n_oe=200,
                  n_ud=3000,
                  n_ue=3000,
                  f_o_optimizer=f_o_optimizer,
                  f_u_optimizer=f_u_optimizer,
                  f_o_fit_kwargs=f_o_fit_kwargs,
                  f_u_fit_kwargs=f_u_fit_kwargs,
                  f_o_early_stopping_kwargs=f_o_early_stopping_kwargs,
                  f_u_early_stopping_kwargs=f_u_early_stopping_kwargs,

        ):
    
    setup(experiment_dir,
          X, y,
          Cs=Cs, alphas=alphas,
          n_reps=n_reps,
          n_engineered=n_engineered,
          n_engineered_resample=n_engineered_resample,
          n_standard=n_standard,
          n_standard_resample=n_standard_resample,
          n_od=n_od,
          n_oe=n_oe,
          n_ud=n_ud,
          n_ue=n_ue,
          f_o_optimizer=f_o_optimizer,
          f_u_optimizer=f_u_optimizer,
          f_o_fit_kwargs=f_o_fit_kwargs,
          f_u_fit_kwargs=f_u_fit_kwargs,
          f_o_early_stopping_kwargs=f_o_early_stopping_kwargs,
          f_u_early_stopping_kwargs=f_u_early_stopping_kwargs,
          )
          
          
    run(experiment_dir, n_jobs=n_jobs)








# ----- Hyperparameters (C and n) SEARCH

def hp_res(hp_experiment_dir):
    
    
    res = []
    for n_od in os.listdir(hp_experiment_dir):
        
        hp_n_od_experiment_dir = '{}/{}'.format(hp_experiment_dir, n_od)
        if not os.path.isdir(hp_n_od_experiment_dir):
            continue
        
        for rep_num in os.listdir(hp_n_od_experiment_dir):
            if rep_num == 'data':
                continue
            
            rep_dir = '{}/{}'.format(hp_n_od_experiment_dir, rep_num)
            if not os.path.isdir(rep_dir):
                continue
            
            candidate_f_os_dir = '{}/candidate_f_os'.format(rep_dir)
            
            for f_o in os.listdir(candidate_f_os_dir):
                f_o_dir = '{}/{}'.format(candidate_f_os_dir, f_o)
            
                f_o_performance_fp = '{}/performance.json'.format(f_o_dir)
                f_o_info_fp = '{}/info.json'.format(f_o_dir)
                
                _res = {'n_od': n_od,
                        'n_od_rep': rep_num,
                        'f_o_name': f_o,
                       }

                f_o_info = load_json(f_o_info_fp)
                _res['f_o_C'] = f_o_info['C']

                f_o_performance = load_json(f_o_performance_fp)

                for scorer, scorer_dict in f_o_performance.items():
                    for ds_name, v in scorer_dict.items():
                        if scorer == 'auroc':
                            _res['AUROC(f_o,{})'.format(ds_name)] = v

                res.append(_res)

    res_df = pd.DataFrame(res)
    res_df.to_csv('{}/res.csv'.format(hp_experiment_dir), index=False)






def run_hp(hp_experiment_dir,
           n_ods, n_ud, n_ue,
              X, y,
              Cs=[0],
              n_reps=5,
              n_jobs=5,
              n_engineered=1,
              n_engineered_resample=0,
              n_standard=1,
              n_standard_resample=1,
              f_o_optimizer=f_o_optimizer,
              f_u_optimizer=f_u_optimizer,
              f_o_fit_kwargs=f_o_fit_kwargs,
              f_u_fit_kwargs=f_u_fit_kwargs,
              f_o_early_stopping_kwargs=f_o_early_stopping_kwargs,
              f_u_early_stopping_kwargs=f_u_early_stopping_kwargs,
    ):
    
    os.mkdir(hp_experiment_dir)

    for n_od in n_ods:
        print(n_od)
        n_oe = n_od
        
        hp_n_od_experiment_dir = '{}/{}'.format(hp_experiment_dir, n_od)
        setup_and_run(hp_n_od_experiment_dir, X, y,
                                 Cs=Cs,
                                 alphas=[],
                                 n_reps=n_reps,
                                 n_jobs=n_jobs,
                                 n_engineered=0,
                                 n_engineered_resample=0,
                                 n_standard=0,
                                 n_standard_resample=0,
                                 n_od=n_od,
                                 n_oe=n_oe,
                                 n_ud=n_ud,
                                 n_ue=n_ue,
                                )
        
        
        hp_res(hp_experiment_dir)

