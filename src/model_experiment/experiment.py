import sys
import os
myDir = os.getcwd()
#sys.path.append(myDir + '/src')
#sys.path.append(myDir + '/Data')
sys.path.append('../')

for path in sys.path:
    print(path)

import pickle
import matplotlib.pyplot as plt
from dojo.train import train_model
import math
import numpy as np

#–----------------------------- SET UP -------------------------------------

# path to the data

met_path = '../../Data/metoffice_spot.csv'
open_path = '../../Data/openweather_data.csv'
energy_path = '../../Data/energy_onsite.csv'

# Variables to change
architectures = ['gru', 'lstm']
targets = ['w', 's']
look_back = [5, 10, 15, 20, 25, 30]
no_units = [8, 16, 32, 64, 128]
batch_size = [8, 16, 32, 64, 128]

# Static variables
val_split = 0.2
epochs = 100

# store the resutls in the dictionary
results = {}

#–----------------------------- EXPERIMENT -------------------------------------
print("START: Training begining...")
# total of 600 models to train and evaluate
i = 0
for a in architectures:
    for t in targets:
        for lb in look_back:
            for nu in no_units: 
                for bs in batch_size:
                    model_name = "{}_{}_nu-{}_lb-{}_bs-{}".format(a, t, nu, lb, bs)
                    model, eval = train_model(a, t, met_path, open_path, energy_path, lb, model_name, nu, epochs, val_split, bs)
                    model_eval = {
                        'mse' : eval[0],
                        'mae' : eval[1],
                        'msle' : eval[2]
                    }
                    results[model_name] = model_eval
                    
                    
                    i = i+1
                    print('Experiment - {:.2f}%% Complete'.format(math.floor(i/600)))

# save the results in a pickled file
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("END: Training finished...")








