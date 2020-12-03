import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn.preprocessing


class PixelSetDataset(Dataset):
    """ Dataset for the dataset of garnot2020satellite with the wrongly assumed dict for the labels"""

    def __init__(self, datapath, metapath, set_size, transforms=None):
        """
        Args:
        :param datapath (string):   Path to the .npy files of the dataset
        :param metapath (string):   Path to the lables.json file
        :param set_size (int):      Number of pixels drawn from all pixels of parcel
        :param transforms (callable, optional): Optional transform to be applied
                                                on a sample.
        """
        with open(os.path.join(metapath + 'labels.json')) as f:
            self.labels_json = json.load(f)
        with open(os.path.join(metapath + 'geomfeat.json')) as f:
            self.geom_json = json.load(f)

        self.datapath = datapath
        self.data_list = sorted(os.listdir(self.datapath))  # Just sorted by string, not number
        self.set_size = set_size
        self.transform = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        parcel_nr = self.data_list[idx]
        data_array = np.load(self.datapath + parcel_nr)
        parcel_nr = parcel_nr[:-4]
        label = self.labels_json['label_19class'][parcel_nr]
        geom = torch.FloatTensor(self.geom_json[parcel_nr])

        # Normalise data
        for i, timestep in enumerate(data_array):
            if timestep.shape[1] != 0:
                data_array[i, :, :] = sklearn.preprocessing.normalize(timestep, axis=0)


        # A set of set_size pixels is randomly drawn from all pixels N from the parcel.
        # If set_size > N an arbitrary pixel is repeated.
        data_array = np.moveaxis(data_array, -1, 0)
        pixels_in_parcel = len(data_array)
        np.random.shuffle(data_array)
        if pixels_in_parcel < self.set_size:
            if pixels_in_parcel == 0:
                data_array = np.zeros((1, 24, 10))
            difference = self.set_size - len(data_array)
            arbitrary_pixel = data_array[0]
            data_array_extension = np.repeat(arbitrary_pixel[np.newaxis, :, :], difference, axis=0)
            data_array = np.vstack((data_array, data_array_extension))
        else:
            data_array = data_array[:self.set_size]

        if self.transform:
            data_array = self.transform(data_array)

        data_array = np.round(data_array + np.random.randn(*data_array.shape) * 0.01, 2)

        sample = {'data': data_array.astype(dtype='float32'),
                  'label': label,
                  'geom': geom,
                  'pixels_in_parcel': pixels_in_parcel}

        return sample




# Test the dataset
# datapath = 'pixelset/DATA/'
# metapath = 'pixelset/META/'
#
# dat = PixelSetDataset(datapath, metapath, 12)
# print(dat[0]['data'])
# print('tst')
