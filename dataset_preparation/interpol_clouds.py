import os
import pickle
import numpy as np
from tqdm import tqdm
import itertools
import copy

"""
Expensive implementation to interpolate cloudy pixels.
"""


in_dir = '/home/maja/Documents/Dataset_4_garnot/pixelset/DATA/'
out_dir = '/home/maja/Documents/Dataset_4_garnot/pixelset_without_clouds/'

# org = np.load('/home/maja/ssd/rc2020dataset/pixelset/DATA/1.npy')

thrs_band1 = 1500
thrs_band2 = 1500
thrs_band3 = 2200

entry_path = ''
for entry in tqdm(os.scandir(in_dir)):
    entry_path = entry.path
    entry_name = entry.name
    data = np.load(entry.path)
    data_copy = copy.deepcopy(data)
    time_dim, channel_dim, pixel_dim = data.shape
    valid_timesteps = {}
    for timestep, pixel in itertools.product(range(time_dim), range(pixel_dim)):
        pixel1 = data[timestep, 0, pixel]
        pixel2 = data[timestep, 1, pixel]
        pixel3 = data[timestep, 2, pixel]
        if pixel1 < thrs_band1 and pixel2 < thrs_band2 and pixel3 < thrs_band3:
            if pixel in valid_timesteps:
                valid_timesteps[pixel].append(timestep)
            else:
                valid_timesteps[pixel] = [timestep]

    # set first pixels to first valid timestep
    for pixel_iter, timesteps_val in valid_timesteps.items():
        if timesteps_val[0] != 0:
            # Get the data from the first valid timestep
            subst = data[timesteps_val[0], :, pixel_iter]
            # Repeat it as often as nescessary to fill everything until the first valid timestep
            subst2 = np.full((timesteps_val[0], channel_dim), subst)
            data_copy[:timesteps_val[0], :, pixel_iter] = subst2
        if timesteps_val[-1] != time_dim - 1:
            # Get the data from the last valid timestep
            subst = data[timesteps_val[-1], :, pixel_iter]
            # Repeat it as often as nescessary to fill everything until the last valid timestep
            subst2 = np.full(((time_dim - timesteps_val[-1] - 1), channel_dim), subst)
            data_copy[timesteps_val[-1]+1:, :, pixel_iter] = subst2
        for i in range(0, len(timesteps_val) - 1):
            curr_t = timesteps_val[i]
            next_t = timesteps_val[i + 1]
            time_dist = np.arange(1, (next_t - curr_t))
            if len(time_dist) > 0:
                d1 = data[next_t, :, pixel_iter].astype(int)
                d2 = data[curr_t, :, pixel_iter].astype(int)
                intermed1 = d1 - d2
                for td in time_dist:
                    intermed2 = (td / (len(time_dist) + 1)) * intermed1  # intermed so stacken, dass dim mit intermed2 zusammen passen
                    inter = data[curr_t, :, pixel_iter] + intermed2
                    subst2 = np.full((1, channel_dim), inter)
                    data_copy[curr_t+td:curr_t+1+td, :, pixel_iter] = subst2.astype('uint16')

    outfile = out_dir + entry_name
    np.save(outfile, data_copy)