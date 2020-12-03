import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class PixelSetToyDataset(Dataset):
    """ Dataset for the toy dataset of garnot2020satellite"""

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

        # A set of set_size pixels is randomly drawn from all pixels N from the parcel.
        # If set_size > N an arbitrary pixel is repeated.
        data_array = np.moveaxis(data_array, -1, 0)
        np.random.shuffle(data_array)
        if len(data_array) < self.set_size:
            difference = self.set_size - len(data_array)
            arbitrary_pixel = data_array[0]
            data_array_extension = np.repeat(arbitrary_pixel[np.newaxis, :, :], difference, axis=0)
            data_array = np.vstack((data_array, data_array_extension))
        else:
            data_array = data_array[:self.set_size]

        sample = {'data': data_array.astype(dtype='float32'), 'label': label, 'geom': geom}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Test the dataset
# datapath = 'toydata/DATA/'
# metapath = 'toydata/META/'
#
# dat = PixelSetToyDataset(datapath, metapath, 10)
# print(dat[0])
# print('tst')
