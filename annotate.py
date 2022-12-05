import time
from hbdet import models
from hbdet.funcs import (get_files, gen_annotations, 
                         get_dt_filename, 
                         remove_str_flags_from_predictions)
from hbdet import global_config as conf
import pandas as pd
import numpy as np
from pathlib import Path


class MetaData:
    def __init__(self):
        self.filename = 'filename'
        self.f_dt = 'date from timestamp'
        self.n_pred_col = 'number of predictions'
        self.avg_pred_col = 'average prediction value'
        self.n_pred08_col = 'number of predictions with thresh>0.8'
        self.n_pred09_col =  'number of predictions with thresh>0.9'
        self.time_per_file = 'computing time [s]'
        self.df = pd.DataFrame(columns=[self.filename, 
                                        self.f_dt, 
                                        self.n_pred_col, 
                                        self.avg_pred_col, 
                                        self.n_pred08_col, 
                                        self.n_pred09_col])
        
    def append_and_save_meta_file(self, file, annot, f_ind, time_start,
                                  relativ_path = conf.SOUND_FILES_SOURCE,
                                  computing_time = 'not calculated'):
        self.df.loc[f_ind, self.f_dt] = str(get_dt_filename(file).date())
        self.df.loc[f_ind, self.filename] = Path(file).relative_to(
                                                        relativ_path)
        self.df.loc[f_ind, self.n_pred_col] = len(annot)
        df_clean = remove_str_flags_from_predictions(annot)
        self.df.loc[f_ind, self.avg_pred_col] = np.mean(df_clean[conf.ANNOTATION_COLUMN])
        self.df.loc[f_ind, self.n_pred08_col] = len(df_clean.loc[df_clean[
                                                conf.ANNOTATION_COLUMN]>0.8])
        self.df.loc[f_ind, self.n_pred09_col] = len(df_clean.loc[df_clean[
                                                conf.ANNOTATION_COLUMN]>0.9])
        self.df.loc[f_ind, self.time_per_file] = computing_time
        self.df.to_csv(f'../generated_annotations/{time_start}/stats.csv')
    
def run_annotation(train_date=None):
    time_start = time.strftime('%Y-%m-%d_%H_%M', time.gmtime())
    files = get_files(location=conf.SOUND_FILES_SOURCE,
                      search_str='**/*wav')
    
    if not train_date:
            model = models.init_model()
    else:
        df = pd.read_csv('../trainings/20221124_meta_trainings.csv')
        row = df.loc[df['training_date'] == train_date]
        model_name = row.Model.values[0]
        keras_mod_name = row.keras_mod_name.values[0]
        model_class = getattr(models, model_name)
        
        model = models.init_model(model_instance=model_class, 
                        checkpoint_dir=f'../trainings/{train_date}/unfreeze_no-TF', 
                        keras_mod_name=keras_mod_name)
    mdf = MetaData()
    f_ind = 0
    for i, file in enumerate(files):    
        try:
            f_ind += 1
            start = time.time()
            annot = gen_annotations(file, model, mod_label=train_date, 
                                 time_start=time_start)
            computing_time = time.time() - start
            mdf.append_and_save_meta_file(file, annot, f_ind, time_start,
                                          computing_time=computing_time)

        except Exception as e:
            print(f"{file} couldn't be loaded, continuing with next file.\n", e)
            continue

def generate_stats():
    files = get_files(location=conf.ANNOTATION_SOURCE, search_str='**/*txt')
    mdf = MetaData()
    f_ind = 0
    for i, file in enumerate(files):    
        f_ind += 1
        annot = pd.read_csv(file, sep='\t')
        mdf.append_and_save_meta_file(file, annot, f_ind, 
                                        Path(conf.ANNOTATION_SOURCE).stem,
                                        relativ_path=conf.ANNOTATION_SOURCE)

if __name__ == '__main__':
    train_dates = [    
        # '2022-11-30_11',
        # '2022-12-01_02',
        # '2022-11-30_22',
        # '2022-11-30_01',
        # '2022-11-29_17',
        # '2022-11-29_19',
        # '2022-11-29_21',
        # '2022-11-29_22'
        '2022-11-30_01'
        ]

    for train_date in train_dates:
        start = time.time()
        run_annotation(train_date)
        end = time.time()
        print(end-start)