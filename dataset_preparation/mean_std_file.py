import os
import pickle
import numpy as np
from tqdm import tqdm

"""
Script to generare the required meanstd file.
"""

dir = '/home/maja/Documents/Dataset_4_garnot/pixelset_without_clouds/'

means = np.zeros((24, 10))
stds = np.zeros((24, 10))
total = 0
for entry in tqdm(os.scandir(dir)):
    data = np.load(entry.path)
    if data.shape[2] != 0:
        total += 1
        means += np.mean(data, axis=2)
        stds += np.std(data, axis=2)
means /= total
stds /= total

out = (means, stds)

filename2 = '/home/maja/Documents/Dataset_4_garnot/S2-2019-T33TWM-meanstd.pkl'
outfile = open(filename2, 'wb')
pickle.dump(out, outfile)
outfile.close()

# filename1 = '/home/maja/ssd/S2-2017-T31TFM-meanstd.pkl'
# infile = open(filename1, 'rb')
# mysterious_file = pickle.load(infile)
# infile.close()