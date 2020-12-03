import os
import json
import numpy as np
from tqdm import tqdm

"""
Approach to calculate the means and stds as mentioned in the paper.
"""


def preprocess_dataset(datapath, metapath):
    """
    Args:
    :param datapath (string):   Path to the .npy files of the dataset
    :param metapath (string):   Path to the lables.json file
    """
    with open(os.path.join(metapath + 'labels.json')) as f:
        labels_json = json.load(f)
    with open(os.path.join(metapath + 'geomfeat.json')) as f:
        geom_json = json.load(f)

    all_valid_elements = {}
    classes = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]

    for key, value in tqdm(labels_json['label_44class'].items()):
        if value in classes and os.path.exists(datapath + key + '.npy'):
            data_array = np.load(datapath + key + '.npy')
            if data_array.shape[2] >= 1:
                all_valid_elements[key] = classes.index(value)

                # data_array = np.round(data_array + np.random.randn(*data_array.shape) * 0.01 / 0.05) * 0.05
                data_mean = np.mean(data_array, axis=2)
                np.save('/home/maja/ssd/rc2020dataset/pixelset/STATS/means/' + key + '.npy', data_mean)
                data_std = np.std(data_array, axis=2)
                np.save('/home/maja/ssd/rc2020dataset/pixelset/STATS/stds/' + key + '.npy', data_std)

    with open(os.path.join(metapath + 'valid_elements.json'), 'w') as fp:
        json.dump(all_valid_elements, fp, indent=4)



# Run it
datapath = '/home/maja/ssd/rc2020dataset/pixelset/DATA/'
metapath = '/home/maja/ssd/rc2020dataset/pixelset/META/'

preprocess_dataset(datapath, metapath)
