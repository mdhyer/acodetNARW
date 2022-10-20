import numpy as np
import tensorflow as tf
from pathlib import Path
from hbdet.tfrec import get_dataset
from hbdet.google_funcs import GoogleMod
import matplotlib.pyplot as plt


AUTOTUNE = tf.data.AUTOTUNE

def get_val_data(tfrec_path, batch_size, debug=False, **kwArgs):
    test_files = tf.io.gfile.glob(f"{tfrec_path}/val/*.tfrec")
    test_data = get_dataset(test_files, batch_size, AUTOTUNE = AUTOTUNE)

    if debug:
        return test_data.take(100), 100
    else:
        return test_data, len(list(test_data))
    
def get_val_labels(val_data, num_of_samples):
    return list(val_data.batch(num_of_samples))[0][1].numpy()

def init_model(checkpoint_dir, load_untrained_model=False, **kwArgs):
    g = GoogleMod(input_tensors='array')
    model = g.model
    if not load_untrained_model:
        checkpoints = list(checkpoint_dir.glob('cp-*.index'))
        checkpoints.sort()
        model.load_weights(str(checkpoints[-1]).replace('.index', ''))
    return model

def print_evaluation(val_data, model, batch_size):
    return model.evaluate(val_data, batch_size = batch_size, verbose =2)
    
def predict_values(val_data, model):
    return model.predict(x = val_data.batch(32))

def create_pr_curve(labels, preds):
    threshs=np.linspace(0, 1, num=100)[:-1]
    
    r = tf.keras.metrics.Recall(thresholds = list(threshs))
    r.update_state(labels, preds.reshape(len(preds)))
    recall_res = r.result().numpy()

    p = tf.keras.metrics.Precision(thresholds = list(threshs))
    p.update_state(labels, preds.reshape(len(preds)))
    precision_res = p.result().numpy()
    
    return recall_res, precision_res

def plot_pr_curve(ax, val_data, length, training_path, **kwArgs):
    model = init_model(training_path, **kwArgs)
    preds = predict_values(val_data, model)
    labels = get_val_labels(val_data, length)
    recall_res, precision_res = create_pr_curve(labels, preds)
    
    if 'load_untrained_model' in kwArgs:
        ax.plot(recall_res, precision_res, label='untrained_model')
    else:
        ax.plot(recall_res, precision_res, label=f'{training_path.stem}')
    return ax
    
    
def create_and_save_figure(tfrec_path, batch_size, train_date, 
                            debug = False, **kwargs):
    
    training_runs = Path(f'trainings/{train_date}').glob('unfreeze*')
    val_data, length = get_val_data(tfrec_path, batch_size, debug=debug)
    
    fig, ax = plt.subplots()

    info_string = ''
    for key, val in kwargs.items():
        info_string += f' | {key}: {val}'
    
    fig.suptitle(f'Precision and Recall Curve{info_string}')
    
    for i, run in enumerate(training_runs):
        if i == 0:
            ax = plot_pr_curve(ax, val_data, length, run, 
                               load_untrained_model=True)
        ax = plot_pr_curve(ax, val_data, length, run)
        print('creating pr curve for ', run.stem)
    
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')
    ax.legend()
    ax.grid(True)
    plt.savefig(f'{run.parent}/pr_curve.png')
    
if __name__ == '__main__':
    tfrec_path = 'tfrecords_0s_shift'
    train_date = '2022-09-21_20'
    batch_size = 32
    create_and_save_figure(tfrec_path, batch_size, train_date, 
                            init_lr='1e-2', end_lr='1e-5')
