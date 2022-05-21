from scipy import signal
from orion.benchmark import benchmark

SIGNALS = [
    'P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7',
    'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-3',
    'D-2', 'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5',
    'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9',
    'F-2', 'G-4', 'T-3', 'D-11',  'B-1', 'G-6', 'G-7', 'P-7',
    'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 'A-8', 'A-9', 'F-3', 'M-6',
    'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4',
    'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14',
    'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8'
]
SMAP=[  'P-1',  'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7',
    'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'A-1', 'D-1', 'P-3',
    'D-2', 'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5',
    'D-6', 'D-7', 'P-4', 'G-3',  'D-8', 'D-9',
    'F-2', 'G-4', 'T-3', 'D-11',  'G-6', 'G-7',
    'R-1', 'A-5', 'A-6',  'D-13', 'A-8', 'F-3']


MSL=['M-6',
    'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4',
    'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14',
    'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8']




append_train='-train'
append_test='-test'
append_new='-new'
training = [sub+append_train for sub in SMAP]
print(len(training))

pipelines = [
    'lstm_dynamic_threshold','tadgan','lstm_autoencoder'
]

import warnings

warnings.filterwarnings("ignore")

metrics = [
    'f1',
    'recall',
    'precision',
    'accuracy']
scores = benchmark(pipelines=pipelines, datasets=SIGNALS, metrics=metrics, rank='f1')
print(scores)