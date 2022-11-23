from hbdet.plot_utils import (plot_evaluation_metric, 
                              plot_model_results, 
                              plot_sample_spectrograms)
from hbdet.models import GoogleMod
from hbdet.funcs import get_labels_and_preds
from hbdet.tfrec import run_data_pipeline, spec
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time
import numpy as np
from hbdet.humpback_model_dir import front_end
import hbdet.global_config as conf

tfrec_path =[
    # 'Daten/Datasets/ScotWest_v4_2khz',
    # # 'Daten/Datasets/Mixed_v1_2khz',
    # # 'Daten/Datasets/Mixed_v2_2khz',
    # # 'Daten/Datasets/Benoit_v1_2khz',
    # 'Daten/Datasets/BERCHOK_SAMANA_200901_4',
    # 'Daten/Datasets/CHALLENGER_AMAR123.1',
    # 'Daten/Datasets/MELLINGER_NOVA-SCOTIA_200508_EmrldN',
    # 'Daten/Datasets/NJDEP_NJ_200903_PU182',
    # 'Daten/Datasets/SALLY_TUCKERS_AMAR088.1.16000',
    # 'Daten/Datasets/SAMOSAS_EL1_2021',
    # 'Daten/Datasets/SAMOSAS_N1_2021',
    # 'Daten/Datasets/SAMOSAS_S1_2021',
    'Daten/Datasets/Tolsta_2kHz_D2_2018'
    ]

train_dates = [
    # '2022-11-10_18',
    # '2022-05-00_00',
    # '2022-11-07_16',
    # '2022-11-07_21',
    # '2022-11-08_03',
    # '2022-11-09_03',
    # '2022-11-10_18',
    # '2022-11-21_17',
    # '2022-11-21_21',
    # '2022-11-22_00',
    '2022-11-22_17'
              ]

display_keys = [
    # 'data_path', 
    # 'batch_size', 
    'bool_SpecAug', 
    'bool_time_shift', 
    'bool_MixUps', 
    'init_lr', 
    'final_lr',    
    # 'weight_clipping', 
    ]

def get_info(date):
    keys = ['data_path', 'batch_size', 'epochs', 'load_weights', 
            'steps_per_epoch', 'f_score_beta', 'f_score_thresh', 
            'bool_SpecAug', 'bool_time_shift', 'bool_MixUps', 
            'weight_clipping', 'init_lr', 'final_lr', 'unfreezes', 
            'preproc blocks']    
    path = Path(f'trainings/{date}')
    f = pd.read_csv(path.joinpath('training_info.txt'), sep='\t')
    l, found = [], 0
    for key in keys:
        found = 0
        for s in f.values:
            if key in s[0]:
                l.append(s[0])
                found = 1
        if found == 0:
            l.append(f'{key}= nan')
    return {key: s.split('= ')[-1] for s, key in zip(l, keys)}


def create_overview_plot(train_dates, val_set, display_keys, model_class):
    info_dicts = [get_info(date) for date in train_dates]

    val_s = ''.join([Path(s).stem.split('_2khz')[0]+';' for s in val_set])
    string = str(
        # 'batch:{}; ' 
        't_aug:{}; ' 
        'mixup:{}; ' 
        'specaug:{}; ' 
        'lr_beg:{}; ' 
        'lr_end:{}; ' 
        # 'clip:{} ; '
        f'val: all')
    if conf.THRESH != 0.5:
        string += f' thr: {conf.THRESH}'


    labels = [string.format(*[d[k] for k in display_keys]) for d in info_dicts]

    training_runs = []
    for i, train in enumerate(train_dates):
        training_runs += list(Path(f'trainings/{train}').glob('unfreeze*'))
        for _ in range(len(list(Path(f'trainings/{train}').glob('unfreeze*')))):
            labels += labels[i]
    val_data = run_data_pipeline(val_set, 'val', return_spec=False)


    time_start = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    subfigs = fig.subfigures(2, 1)#, wspace=0.07, width_ratios=[1, 1])

    plot_model_results(train_dates, labels, fig=subfigs[0], legend=False)#, **info_dict)
    plot_evaluation_metric(model_class, training_runs, val_data, plot_labels=labels,
                            fig = subfigs[1], plot_pr=True, plot_cm=True, 
                            train_dates=train_dates, label=None)

    fig.savefig(f'trainings/{train_dates[-1]}/{time_start}_results_combo.png')

def create_incorrect_prd_plot(model_instance, train_date, val_data_path, **kwargs):
    training_run = Path(f'trainings/{train_date}').glob('unfreeze*')
    val_data = run_data_pipeline(val_data_path, 'val', return_spec=False)
    labels, preds = get_labels_and_preds(model_instance, training_run, 
                                         val_data, **kwargs)
    preds = preds.reshape([len(preds)])
    bin_preds = list(map(lambda x: 1 if x >= conf.THRESH else 0, preds))
    false_pos, false_neg = [], []
    for i in range(len(preds)):
        if bin_preds[i] == 0 and labels[i] == 1:
            false_neg.append(i)
        if bin_preds[i] == 1 and labels[i] == 0:
            false_pos.append(i)
            
    offset = min([false_neg[0], false_pos[0]])
    val_data = run_data_pipeline(val_data_path, 'val', return_spec=False, 
                                 return_meta=True)
    val_data = val_data.batch(1)
    val_data = val_data.map(lambda x, y, z, w: (spec()(x), y, z, w))
    val_data = val_data.unbatch()
    data = list(val_data.skip(offset))
    fp = [data[i-offset] for i in false_pos]
    fn = [data[i-offset] for i in false_neg]
    plot_sample_spectrograms(fn, dir = train_date,
                    name=f'False_Negative', plot_meta=True, **kwargs)
    plot_sample_spectrograms(fp, dir = train_date,
                    name=f'False_Positive', plot_meta=True, **kwargs)
   
    

# create_incorrect_prd_plot(GoogleMod, train_dates[0], tfrec_path)
for path in tfrec_path:
    create_overview_plot(train_dates, path, display_keys, GoogleMod)
