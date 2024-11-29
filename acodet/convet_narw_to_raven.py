"""
Convert NARW start end labels to Raven table style.

Raven table:
Selection	Begin Time (s)	End Time (s)	High Freq (Hz)	Low Freq (Hz)	Prediction/Comments
1	11.6325	15.51	1000	50	0.71534884
2	15.51	19.3875	1000	50	0.78959155
3	23.265	27.142500000000002	1000	50	0.52791274
"""
import glob
import numpy as np

destination = r"D:\NARWData\RavenTables\Kirsebom"
directory = r'C:\Users\matth\Documents\KirsebomData\Predictions'

original_labels = glob.glob(directory + '/ground*.txt')

for lab in original_labels:
    labels = np.loadtxt(lab)

    if labels.ndim == 1 and len(labels) > 1:
        labels = [labels]

    with open(lab.replace(directory, destination), 'w') as f:
        f.write('Selection\tBegin Time (s)\tEnd Time (s)\tHigh Freq (Hz)\tLow Freq (Hz)\tPrediction/Comments\n')


        for i, label in enumerate(labels):
            try:
                f.write('%d\t%.02f\t%.02f\t%d\t%d\t1.00\n' % (i + 1, float(label[0]), float(label[1]), 1000, 50))
            except:
                print('here')
